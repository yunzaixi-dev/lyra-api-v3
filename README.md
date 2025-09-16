# 掩码生成模型训练系统

基于ADE20K数据集的语义分割模型训练系统，支持识别15个自然场景元素。

## 项目概述

本项目实现了一个完整的语义分割训练和推理系统，能够识别以下15个元素：

- **自然景观**: clouds, sky, hill, rock, tree, leaf, river, lake, bush, grass, flower
- **动物**: person, dog, cat, bird, duck

## 目录结构

```text
lyra-api-v3/
├── config/                 # 配置文件
│   └── train_config.json  # 训练配置
├── data/                   # 数据处理模块
│   └── dataset.py         # 数据集加载器
├── images/                 # 训练数据（ADE20K格式）
│   ├── hill/              # 按场景分类的图像
│   ├── river/
│   ├── sky/
│   └── ...
├── models/                 # 模型定义
│   └── segmentation_model.py  # 分割模型架构
├── utils/                  # 工具模块
│   ├── metrics.py         # 评估指标
│   └── visualization.py   # 可视化工具
├── checkpoints/           # 模型检查点（训练后生成）
├── runs/                  # TensorBoard日志（训练后生成）
├── outputs/               # 输出结果（训练后生成）
├── train.py              # 训练脚本
├── inference.py          # 推理脚本
├── main.py               # 主程序入口
└── pyproject.toml        # 项目依赖配置
```

## 快速开始

### 1. 安装依赖

```bash
# 使用uv安装（推荐）
uv sync

# 或使用pip安装
pip install -e .
```

### 2. 训练模型

```bash
# 使用默认配置训练
python train.py

# 使用自定义配置
python train.py --config config/train_config.json

# 恢复训练
python train.py --resume checkpoints/latest_checkpoint.pth
```

### 3. 模型推理

```bash
# 对单张图像推理
python inference.py --model checkpoints/best_model.pth --input path/to/image.jpg --output results/ --visualize --save_masks

# 批量推理
python inference.py --model checkpoints/best_model.pth --input path/to/images/ --output results/ --visualize
```

## 模型架构

### 支持的模型

- **DeepLabV3+** (默认): 高精度语义分割模型
- **UNet**: 经典的U型网络结构
- **PSPNet**: 金字塔场景解析网络
- **FPN**: 特征金字塔网络

### 支持的编码器

- ResNet系列 (resnet18, resnet34, resnet50, resnet101, resnet152)
- EfficientNet系列 (efficientnet-b0 到 efficientnet-b7)
- RegNet系列
- 等等

## 训练配置

主要配置参数（`config/train_config.json`）：

```json
{
  "model": {
    "architecture": "deeplabv3plus",  // 模型架构
    "encoder_name": "resnet50",       // 编码器
    "encoder_weights": "imagenet"     // 预训练权重
  },
  "training": {
    "epochs": 50,                     // 训练轮数
    "batch_size": 8,                  // 批次大小
    "learning_rate": 0.001,           // 学习率
    "optimizer": "adam",              // 优化器
    "loss_type": "combined"           // 损失函数类型
  },
  "data": {
    "image_size": [256, 256],         // 输入图像尺寸
    "train_ratio": 0.8                // 训练集比例
  }
}
```

## 数据格式

项目支持ADE20K格式的数据，每个样本包含：

- `*.jpg`: 原始图像
- `*.json`: 标注文件（包含多边形标注）
- `*_seg.png`: 分割掩码图像

## 评估指标

模型使用以下指标进行评估：

- **像素准确率** (Pixel Accuracy): 整体像素分类准确率
- **平均IoU** (Mean IoU): 各类别IoU的平均值
- **Dice系数** (Dice Coefficient): 重叠度量指标
- **频率加权IoU** (Frequency Weighted IoU): 基于类别频率加权的IoU

## 可视化功能

### 训练过程可视化

- TensorBoard日志记录
- 训练曲线绘制
- 混淆矩阵可视化

### 预测结果可视化

- 原图与分割掩码叠加
- 各类别独立掩码显示
- 预测置信度热图

## 高级功能

### 多尺度预测

启用多尺度预测提高精度：

```json
{
  "model": {
    "multiscale": true,
    "scales": [0.5, 1.0, 1.5]
  }
}
```

### 损失函数组合

支持多种损失函数：

```json
{
  "training": {
    "loss_type": "combined",
    "ce_weight": 1.0,      // 交叉熵权重
    "dice_weight": 1.0,    // Dice损失权重
    "focal_weight": 0.5    // Focal损失权重
  }
}
```

### 类别权重平衡

自动计算类别权重处理数据不平衡：

```json
{
  "training": {
    "use_class_weights": true
  }
}
```

## 性能优化建议

1. **GPU加速**: 确保安装CUDA版本的PyTorch
2. **数据并行**: 增加`num_workers`参数
3. **混合精度**: 使用AMP加速训练
4. **模型选择**: 根据精度/速度需求选择合适的架构

## 故障排除

### 常见问题

1. **内存不足**: 减少`batch_size`或`image_size`
2. **CUDA错误**: 检查PyTorch和CUDA版本兼容性
3. **数据加载慢**: 增加`num_workers`或使用SSD存储

### 调试技巧

```bash
# 检查数据集
python -c "from data.dataset import ADE20KDataset; print('数据集测试正常')"

# 测试模型
python -c "from models.segmentation_model import MaskGenerationModel; print('模型测试正常')"

# 验证配置
python train.py --config config/train_config.json --help
```

## 许可证

本项目基于MIT许可证开源。

## 贡献指南

欢迎提交Issues和Pull Requests来改进项目！

## 更新日志

### v0.1.0 (2024-09-17)

- 初始版本发布
- 支持15个自然场景元素识别
- 完整的训练和推理流程
- 多种模型架构支持