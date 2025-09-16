#!/usr/bin/env python3
"""
分析ADE20K数据集中各类别的数据分布
确保目标类别有足够的训练数据
"""

import os
import json
from collections import defaultdict, Counter
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# 我们关心的15个核心类别
TARGET_CLASSES = [
    'background', 'clouds', 'person', 'sky', 'hill', 'rock', 'tree', 
    'leaf', 'river', 'lake', 'bush', 'dog', 'cat', 'flower', 'grass', 
    'bird', 'duck'
]

def analyze_ade20k_dataset(data_dir: str = "ADE20K"):
    """分析ADE20K数据集的类别分布"""
    
    # 统计数据
    class_counts = Counter()
    scene_counts = Counter()
    total_images = 0
    total_objects = 0
    
    # ADE20K训练集路径
    training_path = os.path.join(data_dir, "images", "ADE", "training")
    
    if not os.path.exists(training_path):
        print(f"错误: 路径不存在 {training_path}")
        return
    
    print("🔍 开始分析ADE20K数据集...")
    
    # 遍历所有场景类型
    for scene_type in os.listdir(training_path):
        scene_type_path = os.path.join(training_path, scene_type)
        if not os.path.isdir(scene_type_path):
            continue
            
        print(f"  分析场景类型: {scene_type}")
        
        # 遍历具体场景
        for scene_name in os.listdir(scene_type_path):
            scene_path = os.path.join(scene_type_path, scene_name)
            if not os.path.isdir(scene_path):
                continue
                
            scene_counts[scene_name] += 1
            
            # 分析该场景下的所有图片
            for filename in os.listdir(scene_path):
                if filename.endswith('.json'):
                    json_path = os.path.join(scene_path, filename)
                    
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        total_images += 1
                        
                        # 提取对象标注
                        objects = data.get('annotation', {}).get('object', [])
                        for obj in objects:
                            obj_name = obj.get('name', '').lower()
                            class_counts[obj_name] += 1
                            total_objects += 1
                            
                    except Exception as e:
                        continue
    
    print(f"\n📊 数据集统计:")
    print(f"总图片数: {total_images:,}")
    print(f"总对象数: {total_objects:,}")
    print(f"唯一场景数: {len(scene_counts)}")
    print(f"唯一对象类别数: {len(class_counts)}")
    
    # 分析目标类别的数据情况
    print(f"\n🎯 目标类别数据统计:")
    target_class_data = {}
    
    for target_class in TARGET_CLASSES:
        count = 0
        # 寻找相似的类别名
        for class_name, class_count in class_counts.items():
            if target_class.lower() in class_name.lower() or class_name.lower() in target_class.lower():
                count += class_count
        
        target_class_data[target_class] = count
        status = "✅" if count > 100 else "⚠️" if count > 10 else "❌"
        print(f"  {status} {target_class}: {count:,} 个对象")
    
    # 显示最常见的对象类别
    print(f"\n🔝 最常见的20个对象类别:")
    for class_name, count in class_counts.most_common(20):
        print(f"  {class_name}: {count:,}")
    
    # 显示最常见的场景
    print(f"\n🏞️ 最常见的20个场景:")
    for scene_name, count in scene_counts.most_common(20):
        print(f"  {scene_name}: {count:,} 张图片")
    
    # 生成可视化图表
    create_visualizations(target_class_data, class_counts, scene_counts)
    
    return target_class_data, class_counts, scene_counts

def create_visualizations(target_class_data, class_counts, scene_counts):
    """创建数据可视化图表"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 目标类别分布
    ax1 = axes[0, 0]
    classes = list(target_class_data.keys())
    counts = list(target_class_data.values())
    
    bars = ax1.bar(range(len(classes)), counts, color='skyblue', alpha=0.7)
    ax1.set_xlabel('目标类别')
    ax1.set_ylabel('对象数量')
    ax1.set_title('目标类别数据分布')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    # 2. 最常见类别（Top 15）
    ax2 = axes[0, 1]
    top_classes = dict(class_counts.most_common(15))
    ax2.barh(range(len(top_classes)), list(top_classes.values()), color='lightcoral', alpha=0.7)
    ax2.set_xlabel('对象数量')
    ax2.set_ylabel('类别')
    ax2.set_title('最常见的15个对象类别')
    ax2.set_yticks(range(len(top_classes)))
    ax2.set_yticklabels(list(top_classes.keys()))
    
    # 3. 最常见场景（Top 15）
    ax3 = axes[1, 0]
    top_scenes = dict(scene_counts.most_common(15))
    ax3.barh(range(len(top_scenes)), list(top_scenes.values()), color='lightgreen', alpha=0.7)
    ax3.set_xlabel('图片数量')
    ax3.set_ylabel('场景')
    ax3.set_title('最常见的15个场景')
    ax3.set_yticks(range(len(top_scenes)))
    ax3.set_yticklabels(list(top_scenes.keys()))
    
    # 4. 数据质量评估
    ax4 = axes[1, 1]
    sufficient_data = sum(1 for count in target_class_data.values() if count > 100)
    moderate_data = sum(1 for count in target_class_data.values() if 10 < count <= 100)
    insufficient_data = sum(1 for count in target_class_data.values() if count <= 10)
    
    labels = ['充足 (>100)', '中等 (10-100)', '不足 (≤10)']
    sizes = [sufficient_data, moderate_data, insufficient_data]
    colors = ['green', 'orange', 'red']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('目标类别数据质量分布')
    
    plt.tight_layout()
    plt.savefig('outputs/dataset_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 可视化图表已保存到: outputs/dataset_analysis.png")

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    # 分析数据集
    target_data, all_classes, scenes = analyze_ade20k_dataset()
    
    # 给出建议
    print(f"\n💡 建议:")
    low_data_classes = [cls for cls, count in target_data.items() if count < 50]
    
    if low_data_classes:
        print(f"  ⚠️  以下类别数据不足，可能影响训练效果:")
        for cls in low_data_classes:
            print(f"     - {cls} ({target_data[cls]} 个对象)")
        print(f"  🔧 建议: 考虑数据增强或调整类别权重")
    else:
        print(f"  ✅ 所有目标类别都有充足的训练数据！")
    
    print(f"\n🚀 数据集分析完成！可以开始训练了。")
