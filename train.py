"""
掩码生成模型训练脚本
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from datetime import datetime

from models.segmentation_model import MaskGenerationModel, SegmentationLoss, MultiScaleModel
from data.dataset import ADE20KDataset, create_dataloaders, get_transforms
from utils.metrics import SegmentationMetrics
from utils.visualization import save_predictions


class Trainer:
    """模型训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 目标类别
        self.target_classes = [
            "clouds", "person", "sky", "hill", "rock", "tree", "leaf",
            "river", "lake", "bush", "dog", "cat", "flower", "grass", "bird", "duck"
        ]
        
        # 创建模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 创建数据加载器
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # 创建损失函数和优化器
        self.criterion = self._create_loss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 创建评估指标
        self.metrics = SegmentationMetrics(num_classes=len(self.target_classes) + 1)
        
        # 创建日志记录
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # 训练状态
        self.epoch = 0
        self.best_miou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        
    def _create_model(self):
        """创建模型"""
        model = MaskGenerationModel(
            architecture=self.config['model']['architecture'],
            encoder_name=self.config['model']['encoder_name'],
            encoder_weights=self.config['model']['encoder_weights'],
            in_channels=self.config['model']['in_channels'],
            classes=len(self.target_classes) + 1,  # +1 for background
            activation=self.config['model'].get('activation', None)
        )
        
        # 是否使用多尺度模型
        if self.config['model'].get('multiscale', False):
            model = MultiScaleModel(model, scales=self.config['model'].get('scales', [0.5, 1.0, 1.5]))
            
        return model
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        return create_dataloaders(
            data_dir=self.config['data']['data_dir'],
            target_classes=self.target_classes,
            batch_size=self.config['training']['batch_size'],
            image_size=tuple(self.config['data']['image_size']),
            train_ratio=self.config['data']['train_ratio'],
            num_workers=self.config['data']['num_workers']
        )
    
    def _create_loss(self):
        """创建损失函数"""
        # 计算类别权重
        class_weights = None
        if self.config['training']['use_class_weights']:
            # 这里可以根据数据集统计来计算类别权重
            # 暂时使用均匀权重
            class_weights = torch.ones(len(self.target_classes) + 1).to(self.device)
            
        return SegmentationLoss(
            loss_type=self.config['training']['loss_type'],
            ce_weight=self.config['training']['ce_weight'],
            dice_weight=self.config['training']['dice_weight'],
            focal_weight=self.config['training']['focal_weight'],
            class_weights=class_weights
        )
    
    def _create_optimizer(self):
        """创建优化器"""
        if self.config['training']['optimizer'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=self.config['training']['momentum'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config['training']['optimizer']}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config['training']['scheduler'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif self.config['training']['scheduler'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['step_size'],
                gamma=self.config['training']['gamma']
            )
        elif self.config['training']['scheduler'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config['training']['gamma'],
                patience=self.config['training']['patience']
            )
        else:
            return None
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 记录到tensorboard
            global_step = self.epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], global_step)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # 统计损失
                total_loss += loss.item()
                
                # 计算预测结果
                predictions = torch.argmax(outputs, dim=1)
                
                # 更新指标
                self.metrics.update(predictions.cpu(), masks.cpu())
        
        # 计算平均损失和指标
        avg_loss = total_loss / len(self.val_loader)
        miou = self.metrics.compute_miou()
        acc = self.metrics.compute_accuracy()
        class_ious = self.metrics.compute_class_iou()
        
        # 记录指标
        self.val_losses.append(avg_loss)
        self.val_mious.append(miou)
        
        # 记录到tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, self.epoch)
        self.writer.add_scalar('Val/mIoU', miou, self.epoch)
        self.writer.add_scalar('Val/Accuracy', acc, self.epoch)
        
        # 记录各类别IoU
        for i, class_name in enumerate(['background'] + self.target_classes):
            if i < len(class_ious):
                self.writer.add_scalar(f'Val/IoU_{class_name}', class_ious[i], self.epoch)
        
        return avg_loss, miou, acc, class_ious
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_miou': self.best_miou,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious
        }
        
        # 保存最新检查点
        checkpoint_path = Path(self.config['checkpoint_dir']) / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = Path(self.config['checkpoint_dir']) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型，mIoU: {self.best_miou:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_miou = checkpoint['best_miou']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_mious = checkpoint['val_mious']
        
        print(f"加载检查点，epoch: {self.epoch}, best_miou: {self.best_miou:.4f}")
    
    def train(self):
        """主训练循环"""
        print("开始训练...")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.epoch, self.config['training']['epochs']):
            self.epoch = epoch
            
            # 为新epoch重新洗牌训练数据
            if hasattr(self.train_loader.dataset, 'shuffle_for_new_epoch'):
                self.train_loader.dataset.shuffle_for_new_epoch()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, miou, acc, class_ious = self.validate_epoch()
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(miou)
                else:
                    self.scheduler.step()
            
            # 打印结果
            print(f'\nEpoch {epoch+1}/{self.config["training"]["epochs"]}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'mIoU: {miou:.4f}')
            print(f'Accuracy: {acc:.4f}')
            
            # 打印各类别IoU
            print("各类别IoU:")
            for i, class_name in enumerate(['background'] + self.target_classes):
                if i < len(class_ious):
                    print(f'  {class_name}: {class_ious[i]:.4f}')
            
            # 保存检查点
            is_best = miou > self.best_miou
            if is_best:
                self.best_miou = miou
            
            self.save_checkpoint(is_best)
            
            # 保存预测可视化
            if (epoch + 1) % self.config['training']['vis_interval'] == 0:
                save_predictions(
                    self.model, self.val_loader, self.device,
                    save_dir=Path(self.config['output_dir']) / 'predictions' / f'epoch_{epoch+1}',
                    num_samples=5
                )
        
        print(f"训练完成！最佳mIoU: {self.best_miou:.4f}")
        self.writer.close()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='训练掩码生成模型')
    parser.add_argument('--config', type=str, default='config/train_config.json', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # 使用默认配置
        config = {
            "data": {
                "data_dir": "images",
                "image_size": [256, 256],
                "train_ratio": 0.8,
                "num_workers": 4
            },
            "model": {
                "architecture": "deeplabv3plus",
                "encoder_name": "resnet50",
                "encoder_weights": "imagenet",
                "in_channels": 3,
                "activation": None,
                "multiscale": False,
                "scales": [0.5, 1.0, 1.5]
            },
            "training": {
                "epochs": 100,
                "batch_size": 8,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "momentum": 0.9,
                "optimizer": "adam",
                "scheduler": "cosine",
                "step_size": 30,
                "gamma": 0.1,
                "patience": 10,
                "loss_type": "combined",
                "ce_weight": 1.0,
                "dice_weight": 1.0,
                "focal_weight": 0.0,
                "use_class_weights": False,
                "vis_interval": 10
            },
            "checkpoint_dir": "checkpoints",
            "log_dir": "runs",
            "output_dir": "outputs"
        }
    
    # 创建必要的目录
    for dir_name in ['checkpoint_dir', 'log_dir', 'output_dir']:
        os.makedirs(config[dir_name], exist_ok=True)
    
    # 保存配置
    config_save_path = Path(config['checkpoint_dir']) / 'config.json'
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 恢复训练
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
