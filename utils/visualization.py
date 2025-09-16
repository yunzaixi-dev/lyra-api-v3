"""
可视化工具模块
用于训练过程中的结果可视化
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import seaborn as sns
from torch.utils.data import DataLoader


def save_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
    num_samples: int = 5,
    class_names: List[str] = None
):
    """保存模型预测结果的可视化"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    samples_saved = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if samples_saved >= num_samples:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 获取预测结果
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # 转换为numpy
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = predictions.cpu().numpy()
            
            for i in range(images.shape[0]):
                if samples_saved >= num_samples:
                    break
                    
                # 反归一化图像
                img = images_np[i].transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                mask_gt = masks_np[i]
                mask_pred = preds_np[i]
                
                # 创建可视化
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # 原图
                axes[0, 0].imshow(img)
                axes[0, 0].set_title("原图")
                axes[0, 0].axis('off')
                
                # 真实掩码
                axes[0, 1].imshow(mask_gt, cmap='tab20', vmin=0, vmax=len(class_names) if class_names else 16)
                axes[0, 1].set_title("真实掩码")
                axes[0, 1].axis('off')
                
                # 预测掩码
                axes[0, 2].imshow(mask_pred, cmap='tab20', vmin=0, vmax=len(class_names) if class_names else 16)
                axes[0, 2].set_title("预测掩码")
                axes[0, 2].axis('off')
                
                # 叠加图（真实）
                overlay_gt = create_overlay(img, mask_gt)
                axes[1, 0].imshow(overlay_gt)
                axes[1, 0].set_title("原图+真实掩码")
                axes[1, 0].axis('off')
                
                # 叠加图（预测）
                overlay_pred = create_overlay(img, mask_pred)
                axes[1, 1].imshow(overlay_pred)
                axes[1, 1].set_title("原图+预测掩码")
                axes[1, 1].axis('off')
                
                # 差异图
                diff = (mask_gt != mask_pred).astype(np.uint8)
                axes[1, 2].imshow(diff, cmap='Reds')
                axes[1, 2].set_title("差异图 (红色=错误)")
                axes[1, 2].axis('off')
                
                plt.tight_layout()
                plt.savefig(save_dir / f'sample_{samples_saved:03d}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                samples_saved += 1
    
    print(f"保存了 {samples_saved} 个预测可视化到 {save_dir}")


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """创建图像和掩码的叠加效果"""
    # 确保图像在[0,1]范围内
    if image.max() <= 1.0:
        image = image.copy()
    else:
        image = image / 255.0
    
    # 创建彩色掩码
    num_classes = int(mask.max()) + 1
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    
    mask_colored = np.zeros((*mask.shape, 3))
    for i in range(num_classes):
        mask_colored[mask == i] = colors[i][:3]
    
    # 混合图像和掩码
    overlay = image * (1 - alpha) + mask_colored * alpha
    return np.clip(overlay, 0, 1)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_mious: List[float],
    save_path: str = None
):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs[:len(val_losses)], val_losses, 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # mIoU曲线
    ax2.plot(epochs[:len(val_mious)], val_mious, 'g-', label='验证mIoU', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.set_title('验证mIoU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    normalize: bool = True
):
    """绘制混淆矩阵"""
    if normalize:
        cm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-8)
        title = "归一化混淆矩阵"
        fmt = '.2f'
    else:
        cm = confusion_matrix
        title = "混淆矩阵"
        fmt = 'd'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'shrink': 0.8}
    )
    
    plt.title(title)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_class_metrics(
    class_names: List[str],
    class_ious: np.ndarray,
    class_accs: np.ndarray = None,
    save_path: str = None
):
    """绘制各类别指标对比图"""
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # IoU柱状图
    bars1 = ax.bar(x - width/2, class_ious, width, label='IoU', alpha=0.8, color='skyblue')
    
    # 准确率柱状图（如果提供）
    if class_accs is not None:
        bars2 = ax.bar(x + width/2, class_accs, width, label='Accuracy', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('类别')
    ax.set_ylabel('指标值')
    ax.set_title('各类别性能指标')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{class_ious[i]:.3f}', ha='center', va='bottom', fontsize=8)
    
    if class_accs is not None:
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{class_accs[i]:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_data_distribution(
    dataloader: DataLoader,
    class_names: List[str],
    save_path: str = None
):
    """可视化数据集中的类别分布"""
    class_counts = np.zeros(len(class_names))
    total_pixels = 0
    
    print("正在统计数据分布...")
    for batch in dataloader:
        masks = batch['mask'].numpy()
        
        for i in range(len(class_names)):
            class_counts[i] += (masks == i).sum()
        
        total_pixels += masks.size
    
    # 计算百分比
    class_percentages = class_counts / total_pixels * 100
    
    # 创建饼图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 饼图
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    wedges, texts, autotexts = ax1.pie(
        class_percentages, 
        labels=class_names, 
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax1.set_title('类别分布 (百分比)')
    
    # 柱状图
    bars = ax2.bar(class_names, class_percentages, color=colors)
    ax2.set_xlabel('类别')
    ax2.set_ylabel('百分比 (%)')
    ax2.set_title('类别分布 (柱状图)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, pct in zip(bars, class_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
    
    # 打印统计信息
    print("\n数据集类别分布:")
    print("-" * 40)
    for i, (name, count, pct) in enumerate(zip(class_names, class_counts, class_percentages)):
        print(f"{name:<15}: {count:>8,.0f} ({pct:>5.1f}%)")
    print("-" * 40)
    print(f"{'总计':<15}: {total_pixels:>8,.0f} (100.0%)")


def create_prediction_grid(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 16,
    save_path: str = None
):
    """创建预测结果网格图"""
    model.eval()
    samples_collected = 0
    images_list = []
    masks_gt_list = []
    masks_pred_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= num_samples:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            batch_size = images.shape[0]
            for i in range(min(batch_size, num_samples - samples_collected)):
                # 反归一化图像
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                images_list.append(img)
                masks_gt_list.append(masks[i].cpu().numpy())
                masks_pred_list.append(predictions[i].cpu().numpy())
                
                samples_collected += 1
                if samples_collected >= num_samples:
                    break
    
    # 创建网格图
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows * 3, cols, figsize=(cols * 4, rows * 12))
    
    for i in range(samples_collected):
        row = (i // cols) * 3
        col = i % cols
        
        # 原图
        if rows == 1:
            ax_img = axes[row, col] if cols > 1 else axes[row]
            ax_gt = axes[row + 1, col] if cols > 1 else axes[row + 1]
            ax_pred = axes[row + 2, col] if cols > 1 else axes[row + 2]
        else:
            ax_img = axes[row, col]
            ax_gt = axes[row + 1, col]
            ax_pred = axes[row + 2, col]
        
        ax_img.imshow(images_list[i])
        ax_img.set_title(f'原图 {i+1}')
        ax_img.axis('off')
        
        # 真实掩码
        ax_gt.imshow(masks_gt_list[i], cmap='tab20')
        ax_gt.set_title(f'真实掩码 {i+1}')
        ax_gt.axis('off')
        
        # 预测掩码
        ax_pred.imshow(masks_pred_list[i], cmap='tab20')
        ax_pred.set_title(f'预测掩码 {i+1}')
        ax_pred.axis('off')
    
    # 隐藏多余的子图
    for i in range(samples_collected, rows * cols):
        row = (i // cols) * 3
        col = i % cols
        
        if rows == 1:
            axes[row, col].axis('off') if cols > 1 else axes[row].axis('off')
            axes[row + 1, col].axis('off') if cols > 1 else axes[row + 1].axis('off')
            axes[row + 2, col].axis('off') if cols > 1 else axes[row + 2].axis('off')
        else:
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')
            axes[row + 2, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # 测试可视化工具
    print("可视化工具模块测试完成")
