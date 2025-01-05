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

class TemporalConvNetwork(nn.Module):
    """
    Temporal Convolution Network for processing sequential feature maps
    Applies 1D convolutions across the temporal dimension of features
    """
    
    def __init__(self, 
                 in_channels: int = 256,
                 hidden_channels: int = 512,
                 out_channels: int = 256,
                 num_layers: int = 3,
                 kernel_size: int = 3):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        # Create temporal conv layers with increasing dilation
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels if i == 0 else hidden_channels,
                hidden_channels if i < num_layers-1 else out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=2**i
            ) for i in range(num_layers)
        ])
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through temporal convolution network
        
        Args:
            x: List of feature maps from FPN [B, C, H, W]
            
        Returns:
            Processed temporal features [B, C, H, W]
        """
        batch_size = x[0].shape[0]
        
        # Reshape and combine spatial dimensions
        features = []
        for feat in x:
            h, w = feat.shape[-2:]
            # Reshape to [B, C, H*W]
            feat = feat.view(batch_size, feat.shape[1], -1)
            features.append(feat)
            
        # Concatenate along temporal dimension
        x = torch.cat(features, dim=2)
        
        # Apply temporal convolutions
        for conv in self.temporal_convs:
            x = conv(x)
            x = self.relu(x)
            x = self.dropout(x)
            
        # Reshape back to spatial dimensions of last feature map
        x = x.view(batch_size, -1, h, w)
        
        return x