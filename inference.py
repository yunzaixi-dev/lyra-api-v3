"""
掩码生成模型推理脚本
用于对新图像进行语义分割预测
"""
import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

from models.segmentation_model import MaskGenerationModel, MultiScaleModel
from data.dataset import get_transforms
from utils.visualization import create_overlay


class SegmentationInference:
    """语义分割推理器"""
    
    def __init__(
        self, 
        model_path: str,
        device: str = 'auto',
        config_path: str = None
    ):
        """
        Args:
            model_path: 训练好的模型权重路径
            device: 推理设备 ('auto', 'cpu', 'cuda')
            config_path: 模型配置文件路径
        """
        self.device = self._get_device(device)
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # 默认配置
            self.config = {
                "model": {
                    "architecture": "deeplabv3plus",
                    "encoder_name": "resnet50",
                    "encoder_weights": "imagenet",
                    "in_channels": 3,
                    "multiscale": False
                },
                "data": {
                    "image_size": [256, 256]
                }
            }
        
        # 目标类别
        self.target_classes = [
            "background", "clouds", "person", "sky", "hill", "rock", 
            "tree", "leaf", "river", "lake", "bush", "dog", "cat", 
            "flower", "grass", "bird", "duck"
        ]
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 创建预处理变换
        self.transform = get_transforms(
            image_size=tuple(self.config['data']['image_size']),
            mode='val'
        )
        
        print(f"推理器初始化完成，使用设备: {self.device}")
        print(f"支持的类别: {self.target_classes}")
    
    def _get_device(self, device: str) -> torch.device:
        """获取推理设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载训练好的模型"""
        # 创建模型
        model = MaskGenerationModel(
            architecture=self.config['model']['architecture'],
            encoder_name=self.config['model']['encoder_name'],
            encoder_weights=None,  # 推理时不需要预训练权重
            in_channels=self.config['model']['in_channels'],
            classes=len(self.target_classes),
            activation=self.config['model'].get('activation', None)
        )
        
        # 如果使用多尺度模型
        if self.config['model'].get('multiscale', False):
            model = MultiScaleModel(
                model, 
                scales=self.config['model'].get('scales', [0.5, 1.0, 1.5])
            )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 处理不同格式的checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """预处理输入图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            (处理后的张量, 原始图像)
        """
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        # 应用变换
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # 添加batch维度
        
        return image_tensor.to(self.device), original_image
    
    def predict(
        self, 
        image_path: str, 
        return_probs: bool = False,
        threshold: float = 0.5
    ) -> Dict:
        """对单张图像进行预测
        
        Args:
            image_path: 图像路径
            return_probs: 是否返回概率图
            threshold: 二值化阈值
            
        Returns:
            预测结果字典
        """
        # 预处理
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # 推理
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1)
        
        # 转换为numpy
        prediction_np = prediction.squeeze().cpu().numpy()
        probs_np = probs.squeeze().cpu().numpy() if return_probs else None
        
        # 调整大小到原图尺寸
        original_size = (original_image.shape[1], original_image.shape[0])  # (width, height)
        prediction_resized = cv2.resize(
            prediction_np.astype(np.uint8), 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        # 生成各类别掩码
        class_masks = {}
        for i, class_name in enumerate(self.target_classes):
            if i == 0:  # 跳过背景
                continue
            mask = (prediction_resized == i).astype(np.uint8) * 255
            class_masks[class_name] = mask
        
        result = {
            'segmentation_map': prediction_resized,
            'class_masks': class_masks,
            'original_image': original_image,
            'image_path': image_path
        }
        
        if return_probs:
            probs_resized = []
            for i in range(probs_np.shape[0]):
                prob_resized = cv2.resize(probs_np[i], original_size)
                probs_resized.append(prob_resized)
            result['probabilities'] = np.stack(probs_resized)
        
        return result
    
    def predict_batch(
        self, 
        image_paths: List[str],
        return_probs: bool = False
    ) -> List[Dict]:
        """批量预测"""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, return_probs)
                results.append(result)
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {e}")
                results.append(None)
        return results
    
    def visualize_prediction(
        self, 
        result: Dict, 
        save_path: str = None,
        show_classes: List[str] = None,
        alpha: float = 0.3
    ):
        """可视化预测结果"""
        original_image = result['original_image']
        segmentation_map = result['segmentation_map']
        class_masks = result['class_masks']
        
        # 选择要显示的类别
        if show_classes is None:
            show_classes = [cls for cls in self.target_classes[1:] 
                          if class_masks[cls].sum() > 0]  # 只显示存在的类别
        
        # 计算子图布局
        num_classes = len(show_classes)
        cols = min(4, num_classes + 2)  # +2 for original and segmentation
        rows = (num_classes + 2 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1) if cols > 1 else np.array([[axes]])
        
        idx = 0
        
        # 原图
        if idx < rows * cols:
            row, col = idx // cols, idx % cols
            axes[row, col].imshow(original_image)
            axes[row, col].set_title('原图')
            axes[row, col].axis('off')
            idx += 1
        
        # 分割图
        if idx < rows * cols:
            row, col = idx // cols, idx % cols
            axes[row, col].imshow(segmentation_map, cmap='tab20')
            axes[row, col].set_title('语义分割')
            axes[row, col].axis('off')
            idx += 1
        
        # 各类别掩码
        for class_name in show_classes:
            if idx >= rows * cols:
                break
                
            row, col = idx // cols, idx % cols
            mask = class_masks[class_name]
            
            # 创建叠加效果
            overlay = original_image.copy()
            colored_mask = np.zeros_like(overlay)
            colored_mask[mask > 0] = [255, 0, 0]  # 红色高亮
            
            overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
            
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f'{class_name}')
            axes[row, col].axis('off')
            idx += 1
        
        # 隐藏多余的子图
        for i in range(idx, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_masks(self, result: Dict, output_dir: str):
        """保存各类别掩码为图像文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存分割图
        seg_map = result['segmentation_map']
        cv2.imwrite(str(output_dir / 'segmentation_map.png'), seg_map)
        
        # 保存各类别掩码
        for class_name, mask in result['class_masks'].items():
            if mask.sum() > 0:  # 只保存非空掩码
                cv2.imwrite(str(output_dir / f'{class_name}_mask.png'), mask)
        
        print(f"掩码文件保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='语义分割模型推理')
    parser.add_argument('--model', required=True, help='模型权重路径')
    parser.add_argument('--config', default=None, help='配置文件路径')
    parser.add_argument('--input', required=True, help='输入图像路径或目录')
    parser.add_argument('--output', default='inference_results', help='输出目录')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='推理设备')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    parser.add_argument('--save_masks', action='store_true', help='保存掩码文件')
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = SegmentationInference(
        model_path=args.model,
        device=args.device,
        config_path=args.config
    )
    
    # 准备输入图像列表
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_path.glob(f'*{ext}'))
            image_paths.extend(input_path.glob(f'*{ext.upper()}'))
        image_paths = [str(p) for p in image_paths]
    else:
        raise ValueError(f"输入路径不存在: {args.input}")
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 批量推理
    results = inferencer.predict_batch(image_paths)
    
    # 处理结果
    for i, (image_path, result) in enumerate(zip(image_paths, results)):
        if result is None:
            continue
            
        image_name = Path(image_path).stem
        
        # 可视化
        if args.visualize:
            vis_path = output_dir / f'{image_name}_visualization.png'
            inferencer.visualize_prediction(result, str(vis_path))
        
        # 保存掩码
        if args.save_masks:
            mask_dir = output_dir / f'{image_name}_masks'
            inferencer.save_masks(result, mask_dir)
        
        # 保存预测统计
        stats = {
            'image_path': image_path,
            'detected_classes': []
        }
        
        for class_name, mask in result['class_masks'].items():
            pixel_count = (mask > 0).sum()
            if pixel_count > 0:
                stats['detected_classes'].append({
                    'class_name': class_name,
                    'pixel_count': int(pixel_count),
                    'percentage': float(pixel_count / mask.size * 100)
                })
        
        # 保存统计信息
        stats_path = output_dir / f'{image_name}_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"推理完成，结果保存到: {output_dir}")


if __name__ == "__main__":
    main()
