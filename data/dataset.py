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


class ADE20KDataset(Dataset):
    """ADE20K语义分割数据集"""
    
    def __init__(
        self,
        data_dir: str,
        target_classes: List[str],
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (256, 256),
        mode: str = "train"
    ):
        """
        Args:
            data_dir: 数据根目录路径
            target_classes: 目标类别列表
            transform: 数据增强变换
            image_size: 目标图像尺寸
            mode: 数据模式 ('train', 'val', 'test')
        """
        self.data_dir = data_dir
        self.target_classes = ["background"] + target_classes  # 添加背景类
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.target_classes)}
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        
        # 收集所有样本
        self.samples = []
        self._collect_samples()
        
        print(f"找到 {len(self.samples)} 个样本")
        print(f"目标类别: {self.target_classes}")
    
    def _collect_samples(self):
        """收集所有有效样本"""
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            for file_name in os.listdir(class_path):
                if file_name.endswith('.jpg'):
                    # 获取对应的标注文件
                    base_name = file_name.replace('.jpg', '')
                    json_path = os.path.join(class_path, f"{base_name}.json")
                    seg_path = os.path.join(class_path, f"{base_name}_seg.png")
                    
                    if os.path.exists(json_path) and os.path.exists(seg_path):
                        self.samples.append({
                            'image_path': os.path.join(class_path, file_name),
                            'json_path': json_path,
                            'seg_path': seg_path,
                            'scene_class': class_dir
                        })
    
    def _load_annotation(self, json_path: str) -> Dict:
        """加载标注文件"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_mask(self, annotation: Dict, seg_image: np.ndarray) -> np.ndarray:
        """根据标注创建目标类别的掩码"""
        h, w = seg_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 获取标注中的所有对象
        objects = annotation.get('annotation', {}).get('object', [])
        
        for obj in objects:
            obj_name = obj.get('name', '').lower()
            
            # 检查是否是我们关心的类别
            target_class = None
            for target in self.target_classes[1:]:  # 跳过background
                if target.lower() in obj_name or obj_name in target.lower():
                    target_class = target
                    break
            
            if target_class:
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
        ])
    else:  # val/test
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


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
