"""
语义分割评估指标
"""
import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    """语义分割评估指标计算器"""
    
    def __init__(self, num_classes: int, ignore_index: int = -1):
        """
        Args:
            num_classes: 类别数量
            ignore_index: 忽略的类别索引
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """重置统计数据"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """更新统计数据
        
        Args:
            preds: 预测结果 [N, H, W]
            targets: 真实标签 [N, H, W]
        """
        # 转换为numpy并展平
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # 过滤忽略的索引
        if self.ignore_index != -1:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]
        
        # 确保预测值在有效范围内
        preds = np.clip(preds, 0, self.num_classes - 1)
        targets = np.clip(targets, 0, self.num_classes - 1)
        
        # 更新混淆矩阵
        cm = confusion_matrix(
            targets, preds, 
            labels=np.arange(self.num_classes)
        )
        self.confusion_matrix += cm
        self.total_samples += len(targets)
    
    def compute_pixel_accuracy(self) -> float:
        """计算像素准确率"""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / (total + 1e-8)
    
    def compute_accuracy(self) -> float:
        """计算准确率（别名）"""
        return self.compute_pixel_accuracy()
    
    def compute_class_accuracy(self) -> np.ndarray:
        """计算每个类别的准确率"""
        class_correct = np.diag(self.confusion_matrix)
        class_total = self.confusion_matrix.sum(axis=1)
        return class_correct / (class_total + 1e-8)
    
    def compute_class_iou(self) -> np.ndarray:
        """计算每个类别的IoU"""
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) - 
            intersection
        )
        return intersection / (union + 1e-8)
    
    def compute_miou(self) -> float:
        """计算平均IoU"""
        iou = self.compute_class_iou()
        return np.mean(iou)
    
    def compute_frequency_weighted_iou(self) -> float:
        """计算频率加权IoU"""
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        iou = self.compute_class_iou()
        return np.sum(freq * iou)
    
    def compute_dice_coefficient(self) -> np.ndarray:
        """计算每个类别的Dice系数"""
        intersection = np.diag(self.confusion_matrix)
        dice = (2.0 * intersection) / (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) + 1e-8
        )
        return dice
    
    def compute_mean_dice(self) -> float:
        """计算平均Dice系数"""
        dice = self.compute_dice_coefficient()
        return np.mean(dice)
    
    def get_confusion_matrix(self) -> np.ndarray:
        """获取混淆矩阵"""
        return self.confusion_matrix.copy()
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """计算所有指标"""
        return {
            'pixel_accuracy': self.compute_pixel_accuracy(),
            'mean_iou': self.compute_miou(),
            'frequency_weighted_iou': self.compute_frequency_weighted_iou(),
            'mean_dice': self.compute_mean_dice(),
        }
    
    def print_metrics(self, class_names: List[str] = None):
        """打印详细指标"""
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(self.num_classes)]
        
        print("=" * 50)
        print("语义分割评估指标")
        print("=" * 50)
        
        # 总体指标
        print(f"像素准确率: {self.compute_pixel_accuracy():.4f}")
        print(f"平均IoU: {self.compute_miou():.4f}")
        print(f"频率加权IoU: {self.compute_frequency_weighted_iou():.4f}")
        print(f"平均Dice: {self.compute_mean_dice():.4f}")
        
        print("\n" + "=" * 50)
        print("各类别详细指标")
        print("=" * 50)
        
        # 各类别指标
        class_acc = self.compute_class_accuracy()
        class_iou = self.compute_class_iou()
        class_dice = self.compute_dice_coefficient()
        
        print(f"{'类别':<15} {'准确率':<10} {'IoU':<10} {'Dice':<10}")
        print("-" * 50)
        
        for i in range(self.num_classes):
            name = class_names[i] if i < len(class_names) else f'Class_{i}'
            print(f"{name:<15} {class_acc[i]:<10.4f} {class_iou[i]:<10.4f} {class_dice[i]:<10.4f}")


def compute_batch_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    num_classes: int
) -> Dict[str, float]:
    """计算批次级别的快速指标
    
    Args:
        predictions: 预测结果 [B, H, W]
        targets: 真实标签 [B, H, W]
        num_classes: 类别数量
        
    Returns:
        包含各种指标的字典
    """
    # 展平张量
    preds_flat = predictions.view(-1)
    targets_flat = targets.view(-1)
    
    # 计算像素准确率
    correct = (preds_flat == targets_flat).float().sum()
    total = targets_flat.numel()
    pixel_acc = (correct / total).item()
    
    # 计算IoU
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds_flat == cls)
        target_cls = (targets_flat == cls)
        
        intersection = (pred_cls & target_cls).float().sum()
        union = (pred_cls | target_cls).float().sum()
        
        if union > 0:
            iou = (intersection / union).item()
        else:
            iou = 1.0  # 如果该类别在GT和预测中都不存在，认为IoU为1
        
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    
    return {
        'pixel_accuracy': pixel_acc,
        'mean_iou': mean_iou,
        'class_ious': ious
    }


def calculate_class_weights(
    dataloader, 
    num_classes: int, 
    device: torch.device = None
) -> torch.Tensor:
    """计算类别权重用于处理类别不平衡
    
    Args:
        dataloader: 数据加载器
        num_classes: 类别数量
        device: 设备
        
    Returns:
        类别权重张量
    """
    if device is None:
        device = torch.device('cpu')
    
    class_counts = torch.zeros(num_classes)
    
    print("正在统计类别分布...")
    for batch in dataloader:
        masks = batch['mask']
        
        for cls in range(num_classes):
            class_counts[cls] += (masks == cls).sum().item()
    
    # 计算权重（反比例）
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts)
    
    # 归一化权重
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("类别分布统计:")
    for i in range(num_classes):
        print(f"类别 {i}: {class_counts[i]:,.0f} 像素 ({100 * class_counts[i] / total_pixels:.2f}%)")
    
    print("\n类别权重:")
    for i in range(num_classes):
        print(f"类别 {i}: {class_weights[i]:.4f}")
    
    return class_weights.to(device)


if __name__ == "__main__":
    # 测试评估指标
    num_classes = 5
    batch_size = 2
    height, width = 64, 64
    
    # 生成随机预测和标签
    predictions = torch.randint(0, num_classes, (batch_size, height, width))
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # 创建评估器
    metrics = SegmentationMetrics(num_classes)
    metrics.update(predictions, targets)
    
    # 打印指标
    class_names = ['Background', 'Person', 'Car', 'Tree', 'Building']
    metrics.print_metrics(class_names)
    
    # 测试批次指标
    batch_metrics = compute_batch_metrics(predictions, targets, num_classes)
    print(f"\n批次指标: {batch_metrics}")
