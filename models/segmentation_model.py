"""
掩码生成模型架构
支持多种语义分割模型：UNet, DeepLabV3+, PSPNet等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Dict, List, Optional


class MaskGenerationModel(nn.Module):
    """掩码生成模型主类"""
    
    def __init__(
        self,
        architecture: str = "deeplabv3plus",
        encoder_name: str = "resnet50", 
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 16,  # 15个目标类别 + 1个背景
        activation: Optional[str] = None,
    ):
        super().__init__()
        
        self.num_classes = classes
        self.class_names = [
            "background", "clouds", "person", "sky", "hill", "rock", 
            "tree", "leaf", "river", "lake", "bush", "dog", "cat", 
            "flower", "grass", "bird", "duck"
        ]
        
        # 根据架构选择模型
        if architecture.lower() == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif architecture.lower() == "deeplabv3plus":
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif architecture.lower() == "pspnet":
            self.model = smp.PSPNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif architecture.lower() == "fpn":
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        else:
            raise ValueError(f"不支持的架构: {architecture}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.model(x)
    
    def predict_masks(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """预测掩码，返回每个类别的二值掩码"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            
            masks = {}
            for i, class_name in enumerate(self.class_names):
                if i == 0:  # 跳过背景类
                    continue
                class_mask = (probs[:, i] > threshold).float()
                masks[class_name] = class_mask
                
            return masks
    
    def get_segmentation_map(self, x: torch.Tensor) -> torch.Tensor:
        """获取分割图，返回每个像素的类别ID"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


class MultiScaleModel(nn.Module):
    """多尺度掩码生成模型"""
    
    def __init__(self, base_model: MaskGenerationModel, scales: List[float] = [0.5, 1.0, 1.5]):
        super().__init__()
        self.base_model = base_model
        self.scales = scales
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """多尺度前向传播"""
        original_size = x.shape[-2:]
        outputs = []
        
        for scale in self.scales:
            if scale != 1.0:
                h, w = int(original_size[0] * scale), int(original_size[1] * scale)
                scaled_x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            else:
                scaled_x = x
                
            output = self.base_model(scaled_x)
            
            if scale != 1.0:
                output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
                
            outputs.append(output)
            
        # 平均融合多尺度结果
        return torch.mean(torch.stack(outputs), dim=0)


class SegmentationLoss(nn.Module):
    """语义分割损失函数"""
    
    def __init__(
        self, 
        loss_type: str = "combined",
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.loss_type = loss_type
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Dice损失"""
        pred_softmax = torch.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred_softmax * target_onehot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor, alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Focal损失"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算总损失"""
        total_loss = 0.0
        
        if self.loss_type in ["ce", "combined"]:
            ce_loss = self.ce_loss(pred, target)
            total_loss += self.ce_weight * ce_loss
            
        if self.loss_type in ["dice", "combined"]:
            dice_loss = self.dice_loss(pred, target)
            total_loss += self.dice_weight * dice_loss
            
        if self.loss_type in ["focal", "combined"] and self.focal_weight > 0:
            focal_loss = self.focal_loss(pred, target)
            total_loss += self.focal_weight * focal_loss
            
        return total_loss


def create_model(config: Dict) -> MaskGenerationModel:
    """根据配置创建模型"""
    return MaskGenerationModel(
        architecture=config.get("architecture", "deeplabv3plus"),
        encoder_name=config.get("encoder_name", "resnet50"),
        encoder_weights=config.get("encoder_weights", "imagenet"),
        in_channels=config.get("in_channels", 3),
        classes=config.get("num_classes", 16),
        activation=config.get("activation", None)
    )


if __name__ == "__main__":
    # 测试模型
    model = MaskGenerationModel(
        architecture="deeplabv3plus",
        encoder_name="resnet50",
        classes=16
    )
    
    # 测试输入
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"类别数量: {model.num_classes}")
    print(f"类别名称: {model.class_names}")
