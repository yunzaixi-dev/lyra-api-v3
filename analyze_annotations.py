#!/usr/bin/env python3
"""
深度分析ADE20K标注文件中的对象名称
找出为什么某些类别训练不到的根本原因
"""

import os
import json
import random
from collections import Counter, defaultdict
from typing import Dict, List, Set
import matplotlib.pyplot as plt
import seaborn as sns

# 我们的目标类别
TARGET_CLASSES = [
    'background', 'clouds', 'person', 'sky', 'hill', 'rock', 'tree', 
    'leaf', 'river', 'lake', 'bush', 'dog', 'cat', 'flower', 'grass', 
    'bird', 'duck'
]

def load_annotation_safe(json_path: str) -> Dict:
    """安全加载标注文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        try:
            with open(json_path, 'r', encoding='latin-1') as f:
                return json.load(f)
        except Exception:
            return {'annotation': {'object': []}}
    except Exception:
        return {'annotation': {'object': []}}

def analyze_annotations(data_dir: str = "ADE20K", max_files: int = 1000):
    """分析标注文件中的对象名称"""
    
    print("🔍 开始深度分析ADE20K标注文件...")
    
    # 统计数据
    all_object_names = Counter()
    target_related_names = defaultdict(list)  # 记录与目标类别相关的所有名称
    sample_files = []  # 记录样本文件路径，用于后续详细检查
    total_files = 0
    processed_files = 0
    
    # ADE20K训练集路径
    training_path = os.path.join(data_dir, "images", "ADE", "training")
    
    if not os.path.exists(training_path):
        print(f"❌ 错误: 路径不存在 {training_path}")
        return
    
    # 收集所有JSON文件
    json_files = []
    for scene_type in os.listdir(training_path):
        scene_type_path = os.path.join(training_path, scene_type)
        if not os.path.isdir(scene_type_path):
            continue
            
        for scene_name in os.listdir(scene_type_path):
            scene_path = os.path.join(scene_type_path, scene_name)
            if not os.path.isdir(scene_path):
                continue
                
            for filename in os.listdir(scene_path):
                if filename.endswith('.json'):
                    json_files.append(os.path.join(scene_path, filename))
    
    total_files = len(json_files)
    print(f"📁 找到 {total_files} 个标注文件")
    
    # 随机采样分析（如果文件太多）
    if total_files > max_files:
        json_files = random.sample(json_files, max_files)
        print(f"🎲 随机采样 {max_files} 个文件进行分析")
    
    # 分析每个文件
    for json_path in json_files:
        try:
            annotation = load_annotation_safe(json_path)
            objects = annotation.get('annotation', {}).get('object', [])
            
            for obj in objects:
                obj_name = obj.get('name', '').lower().strip()
                if obj_name:
                    all_object_names[obj_name] += 1
                    
                    # 检查是否与目标类别相关
                    for target in TARGET_CLASSES[1:]:  # 跳过background
                        if (target.lower() in obj_name or 
                            obj_name in target.lower() or 
                            any(keyword in obj_name for keyword in get_keywords(target))):
                            target_related_names[target].append(obj_name)
            
            processed_files += 1
            if processed_files % 100 == 0:
                print(f"⏳ 已处理 {processed_files}/{len(json_files)} 个文件...")
                
        except Exception as e:
            continue
    
    print(f"✅ 分析完成！处理了 {processed_files} 个文件")
    
    # 输出分析结果
    print_analysis_results(all_object_names, target_related_names)
    
    # 创建可视化
    create_analysis_visualizations(all_object_names, target_related_names)
    
    return all_object_names, target_related_names

def get_keywords(target_class: str) -> List[str]:
    """获取目标类别的相关关键词"""
    keywords_map = {
        'sky': ['sky', 'heaven'],
        'tree': ['tree', 'trees', 'palm', 'pine', 'oak', 'maple', 'birch'],
        'grass': ['grass', 'lawn', 'turf'],
        'river': ['river', 'stream', 'creek', 'brook'],
        'bush': ['bush', 'shrub', 'bushes'],
        'bird': ['bird', 'birds', 'eagle', 'pigeon', 'sparrow', 'crow'],
        'duck': ['duck', 'ducks', 'mallard'],
        'dog': ['dog', 'dogs', 'puppy', 'canine'],
        'cat': ['cat', 'cats', 'kitten', 'feline'],
        'person': ['person', 'people', 'human', 'man', 'woman', 'child'],
        'clouds': ['cloud', 'clouds', 'cloudy'],
        'hill': ['hill', 'hills', 'mound'],
        'rock': ['rock', 'rocks', 'stone', 'stones', 'boulder'],
        'leaf': ['leaf', 'leaves', 'foliage'],
        'lake': ['lake', 'pond', 'reservoir'],
        'flower': ['flower', 'flowers', 'blossom', 'bloom'],
    }
    return keywords_map.get(target_class, [target_class])

def print_analysis_results(all_names: Counter, target_related: Dict):
    """打印分析结果"""
    
    print(f"\n📊 标注文件分析结果:")
    print(f"总计发现 {len(all_names)} 种不同的对象名称")
    print(f"总计标注对象 {sum(all_names.values())} 个")
    
    print(f"\n🔝 最常见的30个对象名称:")
    for name, count in all_names.most_common(30):
        print(f"  {name}: {count:,}")
    
    print(f"\n🎯 目标类别相关名称分析:")
    for target in TARGET_CLASSES[1:]:  # 跳过background
        related_names = set(target_related.get(target, []))
        if related_names:
            total_count = sum(all_names[name] for name in related_names)
            print(f"  ✅ {target}: 找到 {len(related_names)} 种相关名称，共 {total_count:,} 个对象")
            # 显示最常见的相关名称
            related_counts = [(name, all_names[name]) for name in related_names]
            related_counts.sort(key=lambda x: x[1], reverse=True)
            for name, count in related_counts[:5]:  # 只显示前5个
                print(f"    - {name}: {count:,}")
        else:
            print(f"  ❌ {target}: 未找到相关名称")
    
    # 检查可能被遗漏的类别
    print(f"\n🔍 可能相关但未匹配的对象名称:")
    unmatched_names = []
    for name, count in all_names.most_common(100):  # 检查前100个常见名称
        is_matched = False
        for target in TARGET_CLASSES[1:]:
            if name in target_related.get(target, []):
                is_matched = True
                break
        if not is_matched and count > 50:  # 只显示出现频率较高的
            unmatched_names.append((name, count))
    
    for name, count in unmatched_names[:20]:  # 显示前20个
        print(f"  ⚠️  {name}: {count:,}")

def create_analysis_visualizations(all_names: Counter, target_related: Dict):
    """创建分析可视化图表"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 最常见对象名称
    ax1 = axes[0, 0]
    top_names = dict(all_names.most_common(15))
    ax1.barh(range(len(top_names)), list(top_names.values()), color='skyblue', alpha=0.7)
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Object Names')
    ax1.set_title('Top 15 Most Common Object Names')
    ax1.set_yticks(range(len(top_names)))
    ax1.set_yticklabels(list(top_names.keys()))
    
    # 2. 目标类别覆盖情况
    ax2 = axes[0, 1]
    target_counts = []
    target_labels = []
    for target in TARGET_CLASSES[1:]:
        related_names = target_related.get(target, [])
        if related_names:
            total_count = sum(all_names[name] for name in related_names)
            target_counts.append(total_count)
        else:
            target_counts.append(0)
        target_labels.append(target)
    
    bars = ax2.bar(range(len(target_labels)), target_counts, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Target Classes')
    ax2.set_ylabel('Total Object Count')
    ax2.set_title('Target Class Coverage in Annotations')
    ax2.set_xticks(range(len(target_labels)))
    ax2.set_xticklabels(target_labels, rotation=45, ha='right')
    
    # 添加数值标签
    for bar, count in zip(bars, target_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(target_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=8)
    
    # 3. 类别匹配状态
    ax3 = axes[1, 0]
    matched_count = sum(1 for target in TARGET_CLASSES[1:] if target_related.get(target, []))
    unmatched_count = len(TARGET_CLASSES) - 1 - matched_count
    
    labels = ['Matched', 'Unmatched']
    sizes = [matched_count, unmatched_count]
    colors = ['green', 'red']
    
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Target Class Matching Status')
    
    # 4. 对象分布直方图
    ax4 = axes[1, 1]
    counts = list(all_names.values())
    ax4.hist(counts, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Object Count')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Object Counts')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    plt.savefig('outputs/annotation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 分析图表已保存到: outputs/annotation_analysis.png")

if __name__ == "__main__":
    # 分析标注文件
    all_names, target_related = analyze_annotations()
    
    print(f"\n💡 问题诊断:")
    zero_iou_classes = ['sky', 'tree', 'bush', 'dog', 'grass', 'bird', 'duck']
    
    print(f"  📋 IoU为0的类别分析:")
    for cls in zero_iou_classes:
        related_names = target_related.get(cls, [])
        if related_names:
            total_count = sum(all_names[name] for name in set(related_names))
            print(f"    {cls}: 找到相关标注 {total_count:,} 个")
        else:
            print(f"    {cls}: ❌ 完全没有找到相关标注")
    
    print(f"\n🔧 建议的修复方案:")
    print(f"  1. 更新数据处理逻辑中的名称映射表")
    print(f"  2. 添加更多同义词和变体名称")
    print(f"  3. 实现更智能的名称匹配算法")
    print(f"  4. 检查标注文件的完整性和格式")
