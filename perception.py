# File: perception.py
"""Perception system implementation using PyTorch"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple
from config import ModelConfig

class PerceptionSystem(nn.Module):
    """Main perception system combining CNN backbone with FPN"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.backbone = self._create_backbone(config)
        self.fpn = FeaturePyramidNetwork(config.feature_channels)
        self.temporal_conv = TemporalConvNetwork()
    
    def _create_backbone(self, config: ModelConfig) -> nn.Module:
        """Creates the CNN backbone"""
        if config.cnn_backbone == "resnet152":
            model = models.resnet152(pretrained=True)
            return nn.Sequential(*list(model.children())[:-2])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the perception system"""
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        temporal_features = self.temporal_conv(fpn_features)
        return temporal_features

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network implementation"""
    
    def __init__(self, channels: List[int]):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, 256, 1)
            for in_ch in channels
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1)
            for _ in channels
        ])
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through FPN"""
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, x)]
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] += nn.functional.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest'
            )
        return [smooth_conv(lateral) for smooth_conv, lateral in zip(self.smooth_convs, laterals)]
