#!/usr/bin/env python3
"""
使用训练好的模型对单张图片进行语义分割测试
"""

import os
import argparse
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from models.segmentation_model import MaskGenerationModel

# 目标类别（必须与训练时一致）
TARGET_CLASSES = [
    'background', 'clouds', 'person', 'sky', 'hill', 'rock', 'tree', 
    'leaf', 'river', 'lake', 'bush', 'dog', 'cat', 'flower', 'grass', 
    'bird', 'duck'
]

# 为可视化创建一个颜色映射
COLOR_MAP = plt.cm.get_cmap('tab20', len(TARGET_CLASSES))

def load_model(model_path: str, device: torch.device) -> MaskGenerationModel:
    """加载训练好的模型检查点"""
    print(f"加载模型从: {model_path}")
    model = MaskGenerationModel(
        architecture='deeplabv3plus',
        encoder_name='resnet50',
        in_channels=3,
        classes=len(TARGET_CLASSES)
    )
    # 加载状态字典，并忽略任何不匹配的键（例如 optimizer state）
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    print("模型加载成功！")
    return model

def preprocess_image(image_path: str, image_size: tuple = (256, 256)) -> torch.Tensor:
    """加载并预处理单张图片"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"图片文件未找到: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    processed_image = transform(image=image)['image']
    return processed_image.unsqueeze(0) # 增加 batch 维度

def visualize_prediction(image_path: str, prediction: np.ndarray, save_path: str):
    """可视化原图、预测掩码和叠加效果"""
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (prediction.shape[1], prediction.shape[0]))

    # 将颜色映射应用到预测掩码
    colored_mask = (COLOR_MAP(prediction / len(TARGET_CLASSES))[:, :, :3] * 255).astype(np.uint8)

    # 创建叠加图像
    overlay = cv2.addWeighted(original_image, 0.6, colored_mask, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title('原图')
    axes[0].axis('off')

    axes[1].imshow(colored_mask)
    axes[1].set_title('预测掩码')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('叠加效果')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n🎉 可视化结果已保存到: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="测试语义分割模型")
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='训练好的模型路径')
    parser.add_argument('--image_path', type=str, required=True, help='需要测试的图片路径')
    parser.add_argument('--output_path', type=str, default='outputs/test_result.png', help='输出结果保存路径')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256], help='模型输入的图片尺寸')
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载模型
    model = load_model(args.model_path, device)

    # 2. 加载并预处理图片
    input_tensor = preprocess_image(args.image_path, tuple(args.image_size)).to(device)

    # 3. 执行推理
    print("\n🚀 开始推理...")
    with torch.no_grad():
        output = model(input_tensor)
        # 对于DeepLabV3+，输出可能在 'out' 键中
        if isinstance(output, dict):
            logits = output['out']
        else:
            logits = output
        
        # 获取预测结果
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    print("推理完成!")

    # 4. 可视化结果
    visualize_prediction(args.image_path, prediction, args.output_path)

if __name__ == '__main__':
    main()
