#!/usr/bin/env python3
"""
测试修复后的数据集处理逻辑
验证各个类别的标注是否正确
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from data.dataset import ADE20KDataset, get_transforms

# 目标类别
TARGET_CLASSES = [
    'background', 'clouds', 'person', 'sky', 'hill', 'rock', 'tree', 
    'leaf', 'river', 'lake', 'bush', 'dog', 'cat', 'flower', 'grass', 
    'bird', 'duck'
]

def test_dataset_processing(data_dir: str = "ADE20K", num_samples: int = 50):
    """测试数据集处理逻辑"""
    
    print("🧪 开始测试修复后的数据集处理逻辑...")
    
    # 创建数据集实例
    dataset = ADE20KDataset(
        data_dir=data_dir,
        target_classes=TARGET_CLASSES,
        transform=None,  # 不使用变换，方便查看原始结果
        image_size=(256, 256),
        mode="train"
    )
    
    if len(dataset) == 0:
        print("❌ 数据集为空，请检查数据路径")
        return
    
    print(f"📊 数据集大小: {len(dataset)}")
    print(f"🎯 目标类别数: {len(TARGET_CLASSES)}")
    
    # 统计各类别在掩码中的出现情况
    class_pixel_counts = Counter()
    class_occurrence_counts = Counter()
    total_samples_tested = 0
    
    # 测试指定数量的样本
    test_samples = min(num_samples, len(dataset))
    print(f"📋 测试样本数: {test_samples}")
    
    for i in range(test_samples):
        try:
            sample = dataset[i]
            mask = sample['mask']
            
            # 统计该样本中每个类别的像素数
            unique_classes, counts = torch.unique(mask, return_counts=True)
            
            for class_idx, count in zip(unique_classes, counts):
                class_idx = class_idx.item()
                count = count.item()
                
                if class_idx < len(TARGET_CLASSES):
                    class_name = TARGET_CLASSES[class_idx]
                    class_pixel_counts[class_name] += count
                    
                    # 如果该类别在此样本中出现，记录一次
                    if count > 0 and class_idx > 0:  # 忽略背景类
                        class_occurrence_counts[class_name] += 1
            
            total_samples_tested += 1
            
            if (i + 1) % 10 == 0:
                print(f"⏳ 已测试 {i + 1}/{test_samples} 个样本...")
                
        except Exception as e:
            print(f"⚠️ 样本 {i} 处理失败: {e}")
            continue
    
    print(f"✅ 测试完成！成功处理 {total_samples_tested} 个样本")
    
    # 输出统计结果
    print_test_results(class_pixel_counts, class_occurrence_counts, total_samples_tested)
    
    # 创建可视化样本
    create_sample_visualizations(dataset, num_vis_samples=5)
    
    return class_pixel_counts, class_occurrence_counts

def print_test_results(pixel_counts: Counter, occurrence_counts: Counter, total_samples: int):
    """打印测试结果"""
    
    print(f"\n📈 类别检测统计结果:")
    print(f"{'类别名称':<12} {'样本数':<8} {'像素总数':<12} {'平均像素':<10} {'检测率':<8}")
    print("-" * 60)
    
    zero_detection_classes = []
    
    for class_name in TARGET_CLASSES[1:]:  # 跳过background
        sample_count = occurrence_counts.get(class_name, 0)
        pixel_count = pixel_counts.get(class_name, 0)
        avg_pixels = pixel_count / max(sample_count, 1)
        detection_rate = (sample_count / total_samples) * 100
        
        print(f"{class_name:<12} {sample_count:<8} {pixel_count:<12,} {avg_pixels:<10.1f} {detection_rate:<8.1f}%")
        
        if sample_count == 0:
            zero_detection_classes.append(class_name)
    
    # 分析结果
    print(f"\n🎯 检测分析:")
    if zero_detection_classes:
        print(f"  ❌ 未检测到的类别 ({len(zero_detection_classes)}个): {', '.join(zero_detection_classes)}")
    else:
        print(f"  ✅ 所有目标类别都被检测到！")
    
    detected_classes = len(TARGET_CLASSES) - 1 - len(zero_detection_classes)
    print(f"  📊 检测成功率: {detected_classes}/{len(TARGET_CLASSES)-1} ({(detected_classes/(len(TARGET_CLASSES)-1)*100):.1f}%)")
    
    # 检测频率最高的类别
    print(f"\n🔝 检测频率最高的5个类别:")
    for class_name, count in occurrence_counts.most_common(5):
        rate = (count / total_samples) * 100
        print(f"  {class_name}: {count}次 ({rate:.1f}%)")

def create_sample_visualizations(dataset: ADE20KDataset, num_vis_samples: int = 5):
    """创建样本可视化"""
    
    print(f"\n🖼️ 创建 {num_vis_samples} 个样本的可视化...")
    
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    # 选择有多个类别的样本进行可视化
    selected_samples = []
    for i in range(min(len(dataset), 100)):  # 检查前100个样本
        try:
            sample = dataset[i]
            mask = sample['mask']
            unique_classes = torch.unique(mask)
            
            # 选择包含多个类别的样本（不仅仅是背景）
            if len(unique_classes) >= 3:  # 至少3个类别（包括背景）
                selected_samples.append(i)
                if len(selected_samples) >= num_vis_samples:
                    break
        except:
            continue
    
    if not selected_samples:
        print("⚠️ 未找到包含多个类别的样本")
        return
    
    # 创建可视化
    fig, axes = plt.subplots(num_vis_samples, 3, figsize=(15, 5 * num_vis_samples))
    if num_vis_samples == 1:
        axes = [axes]
    
    for idx, sample_idx in enumerate(selected_samples):
        try:
            sample = dataset[sample_idx]
            image = sample['image']
            mask = sample['mask']
            
            # 转换图像格式用于显示
            if isinstance(image, torch.Tensor):
                if image.max() <= 1.0:
                    image = (image * 255).clamp(0, 255)
                if image.shape[0] == 3:  # CHW格式
                    image = image.permute(1, 2, 0)
                image = image.numpy().astype(np.uint8)
            
            # 原图
            axes[idx][0].imshow(image)
            axes[idx][0].set_title(f'原图 - 样本 {sample_idx}')
            axes[idx][0].axis('off')
            
            # 掩码
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
            im = axes[idx][1].imshow(mask_np, cmap='tab20', vmin=0, vmax=len(TARGET_CLASSES)-1)
            axes[idx][1].set_title('分割掩码')
            axes[idx][1].axis('off')
            
            # 叠加图
            colored_mask = plt.cm.tab20(mask_np / len(TARGET_CLASSES))[:, :, :3]
            overlay = image.astype(float) * 0.6 + colored_mask * 255 * 0.4
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            axes[idx][2].imshow(overlay)
            axes[idx][2].set_title('叠加效果')
            axes[idx][2].axis('off')
            
            # 显示该样本中包含的类别
            unique_classes = np.unique(mask_np)
            class_names = [TARGET_CLASSES[i] for i in unique_classes if i < len(TARGET_CLASSES)]
            print(f"  样本 {sample_idx} 包含类别: {', '.join(class_names)}")
            
        except Exception as e:
            print(f"⚠️ 样本 {sample_idx} 可视化失败: {e}")
    
    plt.tight_layout()
    save_path = "outputs/dataset_test_samples.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📸 样本可视化已保存到: {save_path}")

if __name__ == "__main__":
    # 测试数据集处理
    pixel_counts, occurrence_counts = test_dataset_processing(num_samples=50)
    
    print(f"\n💡 测试总结:")
    print(f"  修复后的数据处理逻辑已生效")
    print(f"  建议检查可视化结果确认类别标注的准确性")
    print(f"  如果所有目标类别都能被检测到，可以开始重新训练")
