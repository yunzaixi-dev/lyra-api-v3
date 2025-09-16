"""
数据加载和预处理模块
处理ADE20K格式的语义分割数据
"""
import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import random
import collections


class ADE20KDataset(Dataset):
    """ADE20K语义分割数据集"""
    
    def __init__(
        self,
        data_dir: str,
        target_classes: List[str],
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (256, 256),
        mode: str = "train",
        filter_empty_samples: bool = True,
        shuffle_samples: bool = True
    ):
        self.data_dir = data_dir
        self.target_classes = target_classes
        self.transform = transform
        self.image_size = image_size
        self.split = "training" if mode == "train" else "validation"
        self.filter_empty_samples = filter_empty_samples
        self.shuffle_samples = shuffle_samples
        self.samples = []
        self.samples_by_class = collections.defaultdict(list)  # 按类别分组的样本
        
        # 创建类别到索引的映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}
        
        self._load_samples()
    
    def _load_samples(self):
        """加载所有样本 - 适应ADE20K标准目录结构，可选择筛选空样本"""
        # ADE20K标准路径: ADE20K/images/ADE/training/
        ade_path = os.path.join(self.data_dir, "images", "ADE", self.split)
        
        if not os.path.exists(ade_path):
            print(f"警告: 路径不存在 {ade_path}")
            return
        
        all_samples = []
        valid_samples = []
        processed_count = 0
        
        print(f"📋 开始加载 {self.split} 数据集...")
        
        # 遍历所有场景类别目录
        for scene_type in os.listdir(ade_path):
            scene_path = os.path.join(ade_path, scene_type)
            if not os.path.isdir(scene_path):
                continue
                
            # 遍历每个场景下的具体场景
            for scene_subdir in os.listdir(scene_path):
                scene_subdir_path = os.path.join(scene_path, scene_subdir)
                if not os.path.isdir(scene_subdir_path):
                    continue
                    
                # 加载该场景目录下的所有图片
                for file_name in os.listdir(scene_subdir_path):
                    if file_name.endswith('.jpg'):
                        base_name = file_name.replace('.jpg', '')
                        json_path = os.path.join(scene_subdir_path, f"{base_name}.json")
                        seg_path = os.path.join(scene_subdir_path, f"{base_name}_seg.png")
                        
                        if os.path.exists(json_path) and os.path.exists(seg_path):
                            sample = {
                                'image_path': os.path.join(scene_subdir_path, file_name),
                                'json_path': json_path,
                                'seg_path': seg_path,
                                'scene_class': scene_subdir,
                                'scene_type': scene_type
                            }
                            all_samples.append(sample)
                            
                            # 如果启用筛选，检查样本是否包含目标类别
                            if self.filter_empty_samples:
                                if self._sample_has_target_classes(sample):
                                    valid_samples.append(sample)
                            else:
                                valid_samples.append(sample)
                            
                            processed_count += 1
                            if processed_count % 1000 == 0:
                                print(f"⏳ 已处理 {processed_count} 个样本...")
        
        # 根据筛选设置选择最终样本
        self.samples = valid_samples
        
        # 如果启用洗牌，按类别分组样本
        if self.shuffle_samples:
            self._group_samples_by_class()
            self._shuffle_samples()
        
        # 输出统计信息
        total_samples = len(all_samples)
        valid_samples_count = len(valid_samples)
        
        print(f"✅ 数据加载完成:")
        print(f"  📊 总样本数: {total_samples:,}")
        if self.filter_empty_samples:
            filtered_count = total_samples - valid_samples_count
            filter_rate = (filtered_count / total_samples * 100) if total_samples > 0 else 0
            print(f"  🔍 筛选后样本数: {valid_samples_count:,}")
            print(f"  🗑️  过滤掉的空样本: {filtered_count:,} ({filter_rate:.1f}%)")
        else:
            print(f"  📝 使用所有样本: {valid_samples_count:,}")
        
        if self.shuffle_samples:
            print(f"  🔀 启用样本洗牌，按类别平衡分布")
    
    def _group_samples_by_class(self):
        """按类别分组样本，用于平衡洗牌"""
        print("🔍 按类别分析样本...")
        self.samples_by_class = collections.defaultdict(list)
        
        # 获取统一的ADE20K标注名称映射
        ade20k_name_to_target = self._get_ade20k_mapping()
        
        for sample in self.samples:
            try:
                # 加载标注文件分析类别
                annotation = self._load_annotation(sample['json_path'])
                objects = annotation.get('annotation', {}).get('object', [])
                
                sample_classes = set()
                for obj in objects:
                    obj_name = obj.get('name', '').strip()
                    if not obj_name:
                        continue
                    
                    # 查找匹配的目标类别
                    target_class = ade20k_name_to_target.get(obj_name)
                    if not target_class:
                        target_class = ade20k_name_to_target.get(obj_name.lower())
                    
                    if target_class and target_class in self.class_to_idx:
                        sample_classes.add(target_class)
                
                # 将样本添加到每个包含的类别中
                for class_name in sample_classes:
                    self.samples_by_class[class_name].append(sample)
                    
            except Exception as e:
                # 如果解析失败，将样本归为通用类别
                self.samples_by_class['background'].append(sample)
        
        # 输出每个类别的样本数量
        print("📊 各类别样本统计:")
        for class_name in sorted(self.samples_by_class.keys()):
            count = len(self.samples_by_class[class_name])
            print(f"  {class_name}: {count:,} 个样本")
    
    def _shuffle_samples(self):
        """使用洗牌算法重新排列样本，确保批次间类别平衡"""
        if not self.samples_by_class:
            print("⚠️  没有按类别分组的样本，使用简单随机洗牌")
            random.shuffle(self.samples)
            return
        
        print("🔀 执行平衡洗牌算法...")
        
        # 获取所有类别及其样本数量
        class_samples = [(class_name, samples) for class_name, samples in self.samples_by_class.items()]
        
        # 对每个类别内部洗牌
        for class_name, samples in class_samples:
            random.shuffle(samples)
        
        # 使用轮询方式重新排列样本，确保类别分布均匀
        shuffled_samples = []
        max_samples = max(len(samples) for _, samples in class_samples) if class_samples else 0
        
        for i in range(max_samples):
            for class_name, samples in class_samples:
                if i < len(samples):
                    shuffled_samples.append(samples[i])
        
        # 最后再次随机洗牌以避免完全规律的模式
        random.shuffle(shuffled_samples)
        
        self.samples = shuffled_samples
        print(f"✅ 洗牌完成，重排了 {len(shuffled_samples):,} 个样本")
    
    def shuffle_for_new_epoch(self):
        """为新的epoch重新洗牌"""
        if self.shuffle_samples and hasattr(self, 'samples_by_class'):
            self._shuffle_samples()
    
    def _sample_has_target_classes(self, sample: dict) -> bool:
        """检查样本是否包含目标类别"""
        try:
            # 加载标注文件
            annotation = self._load_annotation(sample['json_path'])
            objects = annotation.get('annotation', {}).get('object', [])
            
            # 获取统一的ADE20K标注名称映射
            ade20k_name_to_target = self._get_ade20k_mapping()
            
            # 检查是否有任何目标类别
            for obj in objects:
                obj_name = obj.get('name', '').strip()
                if not obj_name:
                    continue
                
                # 直接查找精确匹配
                target_class = ade20k_name_to_target.get(obj_name)
                
                # 如果没有精确匹配，尝试小写匹配
                if not target_class:
                    target_class = ade20k_name_to_target.get(obj_name.lower())
                
                if target_class and target_class in self.class_to_idx:
                    return True  # 找到至少一个目标类别
            
            return False  # 没有找到任何目标类别
            
        except Exception as e:
            # 如果处理失败，保守地保留样本
            return True
    
    def _get_ade20k_mapping(self) -> Dict[str, str]:
        """获取统一的ADE20K标注名称到目标类别的映射"""
        return {
            # Person类别
            'person, individual, someone, somebody, mortal, soul': 'person',
            'person': 'person',
            'individual': 'person',
            'human': 'person',
            'man': 'person',
            'woman': 'person',
            'child': 'person',
            'people': 'person',
            
            # Sky类别  
            'sky': 'sky',
            
            # Tree类别
            'tree': 'tree',
            'palm, palm tree': 'tree',
            'palm tree': 'tree',
            'palm': 'tree',
            
            # Rock类别
            'rock, stone': 'rock',
            'rock': 'rock',
            'stone': 'rock',
            'stones': 'rock',
            'boulder': 'rock',
            
            # Bush类别
            'shrub, bush': 'bush',
            'bush': 'bush',
            'shrub': 'bush',
            'bushes': 'bush',
            'ground shrubs': 'bush',
            
            # Grass类别
            'grass': 'grass',
            'lawn': 'grass',
            'turf': 'grass',
            
            # Dog类别
            'dog, domestic dog, canis familiaris': 'dog',
            'dog': 'dog',
            'dogs': 'dog',
            'puppy': 'dog',
            
            # Cat类别
            'cat': 'cat',
            'cats': 'cat',
            'kitten': 'cat',
            
            # Bird类别
            'bird': 'bird',
            'birds': 'bird',
            
            # Duck类别
            'duck': 'duck',
            'ducks': 'duck',
            
            # Clouds类别
            'cloud': 'clouds',
            'clouds': 'clouds',
            
            # Hill类别
            'hill': 'hill',
            'hills': 'hill',
            'mound': 'hill',
            
            # Leaf类别
            'leaf, leafage, foliage': 'leaf',
            'leaf': 'leaf',
            'leaves': 'leaf',
            'foliage': 'leaf',
            
            # River类别
            'river': 'river',
            'stream': 'river',
            'creek': 'river',
            'brook': 'river',
            
            # Lake类别
            'lake': 'lake',
            'pond': 'lake',
            'pond water': 'lake',
            'reservoir': 'lake',
            
            # Flower类别
            'flower': 'flower',
            'flowers': 'flower',
            'blossom': 'flower',
            'bloom': 'flower',
            'dried flowers': 'flower',
        }

    def _load_annotation(self, json_path: str) -> Dict:
        """加载标注文件"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except UnicodeDecodeError:
            try:
                # 尝试其他编码
                with open(json_path, 'r', encoding='latin-1') as f:
                    return json.load(f)
            except Exception:
                # 如果仍然失败，返回空标注
                return {'annotation': {'object': []}}
    
    def _create_mask(self, annotation: Dict, seg_image: np.ndarray) -> np.ndarray:
        """根据标注创建目标类别的掩码 - 针对ADE20K复杂标注名称优化"""
        h, w = seg_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 获取标注中的所有对象
        objects = annotation.get('annotation', {}).get('object', [])

        # 获取统一的ADE20K标注名称映射
        ade20k_name_to_target = self._get_ade20k_mapping()

        for obj in objects:
            obj_name = obj.get('name', '').strip()
            if not obj_name:
                continue

            # 直接查找精确匹配
            target_class = ade20k_name_to_target.get(obj_name)
            
            # 如果没有精确匹配，尝试小写匹配
            if not target_class:
                target_class = ade20k_name_to_target.get(obj_name.lower())

            if target_class and target_class in self.class_to_idx:
                class_idx = self.class_to_idx[target_class]

                # 获取多边形坐标
                polygon = obj.get('polygon', {})
                if 'x' in polygon and 'y' in polygon:
                    x_coords = polygon['x']
                    y_coords = polygon['y']

                    if len(x_coords) == len(y_coords) and len(x_coords) >= 3:
                        # 创建多边形掩码
                        points = np.array([[x, y] for x, y in zip(x_coords, y_coords)], dtype=np.int32)
                        cv2.fillPoly(mask, [points], class_idx)

        return mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 加载图像
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载分割图像（用于辅助）
        seg_image = cv2.imread(sample['seg_path'], cv2.IMREAD_GRAYSCALE)
        
        # 加载标注并创建掩码
        annotation = self._load_annotation(sample['json_path'])
        mask = self._create_mask(annotation, seg_image)
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # 确保掩码是长整型
            mask = mask.long()
        else:
            # 默认变换
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'scene_class': sample['scene_class'],
            'image_path': sample['image_path']
        }


def get_transforms(image_size: Tuple[int, int] = (256, 256), mode: str = "train") -> A.Compose:
    """获取数据增强变换"""
    
    if mode == "train":
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    else:  # val/test
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})


def create_dataloaders(
    data_dir: str,
    target_classes: List[str],
    batch_size: int = 8,
    image_size: Tuple[int, int] = (256, 256),
    train_ratio: float = 0.8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    
    # 创建数据集
    full_dataset = ADE20KDataset(
        data_dir=data_dir,
        target_classes=target_classes,
        transform=None,  # 暂时不用变换
        image_size=image_size
    )
    
    # 划分训练和验证集
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 为训练和验证集设置不同的变换
    train_dataset.dataset.transform = get_transforms(image_size, "train")
    val_dataset.dataset.transform = get_transforms(image_size, "val")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def visualize_sample(dataset: ADE20KDataset, idx: int = 0, save_path: Optional[str] = None):
    """可视化数据集样本"""
    sample = dataset[idx]
    
    image = sample['image']
    mask = sample['mask']
    
    # 反归一化图像
    if image.max() <= 1.0:
        image = image * 255.0
    
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(image.cpu().numpy().astype(np.uint8))
    axes[0].set_title("原图")
    axes[0].axis('off')
    
    # 掩码
    axes[1].imshow(mask.cpu().numpy(), cmap='tab20')
    axes[1].set_title("分割掩码")
    axes[1].axis('off')
    
    # 叠加图
    overlay = image.cpu().numpy().astype(np.uint8).copy()
    mask_colored = plt.cm.tab20(mask.cpu().numpy() / len(dataset.target_classes))[:, :, :3]
    overlay = cv2.addWeighted(overlay, 0.7, (mask_colored * 255).astype(np.uint8), 0.3, 0)
    axes[2].imshow(overlay)
    axes[2].set_title("叠加图")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    # 测试数据集
    target_classes = [
        "clouds", "person", "sky", "hill", "rock", "tree", "leaf",
        "river", "lake", "bush", "dog", "cat", "flower", "grass", "bird", "duck"
    ]
    
    dataset = ADE20KDataset(
        data_dir="images",
        target_classes=target_classes,
        transform=get_transforms(mode="train")
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"类别映射: {dataset.class_to_idx}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"样本键: {sample.keys()}")
        print(f"图像形状: {sample['image'].shape}")
        print(f"掩码形状: {sample['mask'].shape}")
        print(f"掩码唯一值: {torch.unique(sample['mask'])}")
