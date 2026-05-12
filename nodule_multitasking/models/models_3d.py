"""Multi-Task 3D Models for Nodule Segmentation and Classification.

This module implements multi-task learning architectures for 3D data that perform:
    1. Nodule Segmentation: Dense voxel-wise prediction
    2. Malignancy Classification: Volume-level classification

Note: 3D multi-task models are more computationally expensive than 2D.
"""

import torch
import torch.nn as nn
from typing import Tuple

SEG_N_CLASSES = 2  # 0: background, 1: nodule
CLS_N_CLASSES = 2  # 0: benign, 1: malignant


class MultiTaskModels3D:
    """Factory class for 3D multi-task models.
    
    This class provides 3D architectures for multi-task learning.
    Currently provides a basic implementation that can be extended.
    """

    class _MultiTask3DUNet(nn.Module):
        """Basic Multi-Task 3D U-Net.
        
        A simple 3D U-Net architecture with classification head.
        For production use, consider using MONAI's SwinUNETR or other
        advanced 3D architectures.
        
        Args:
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            base_filters: Number of filters in first layer (default: 32)
            dropout: Dropout rate for classification head (default: 0.3)
        """
        def __init__(
            self,
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            base_filters: int = 32,
            dropout: float = 0.3,
        ):
            super().__init__()
            
            # Encoder
            self.enc1 = self._conv_block(in_channels, base_filters)
            self.enc2 = self._conv_block(base_filters, base_filters * 2)
            self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
            self.enc4 = self._conv_block(base_filters * 4, base_filters * 8)
            
            self.pool = nn.MaxPool3d(2)
            
            # Bottleneck
            self.bottleneck = self._conv_block(base_filters * 8, base_filters * 16)
            
            # Decoder for segmentation
            self.upconv4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, 2, stride=2)
            self.dec4 = self._conv_block(base_filters * 16, base_filters * 8)
            
            self.upconv3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, 2, stride=2)
            self.dec3 = self._conv_block(base_filters * 8, base_filters * 4)
            
            self.upconv2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, 2, stride=2)
            self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)
            
            self.upconv1 = nn.ConvTranspose3d(base_filters * 2, base_filters, 2, stride=2)
            self.dec1 = self._conv_block(base_filters * 2, base_filters)
            
            # Segmentation head
            self.seg_head = nn.Conv3d(base_filters, seg_classes, 1)
            
            # Classification head
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(base_filters * 16, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
            """3D convolution block with BatchNorm and ReLU."""
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass for 3D multi-task learning.
            
            Args:
                x: Input tensor of shape (B, C, D, H, W)
            
            Returns:
                Tuple of:
                    - seg_output: Segmentation logits (B, seg_classes, D, H, W)
                    - cls_output: Classification logits (B, cls_classes)
            """
            # Encoder
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.pool(enc1))
            enc3 = self.enc3(self.pool(enc2))
            enc4 = self.enc4(self.pool(enc3))
            
            # Bottleneck
            bottleneck = self.bottleneck(self.pool(enc4))
            
            # Classification from bottleneck features
            cls_output = self.classification_head(bottleneck)
            
            # Decoder for segmentation
            dec4 = self.upconv4(bottleneck)
            dec4 = torch.cat([dec4, enc4], dim=1)
            dec4 = self.dec4(dec4)
            
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat([dec3, enc3], dim=1)
            dec3 = self.dec3(dec3)
            
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.dec2(dec2)
            
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.dec1(dec1)
            
            # Segmentation output
            seg_output = self.seg_head(dec1)
            
            return seg_output, cls_output

    @staticmethod
    def get_model(model_name: str = "unet3d", **kwargs):
        """
        Factory method to get 3D multi-task model.
        
        Args:
            model_name: Name of model architecture (currently only "unet3d")
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Multi-task 3D model instance
        """
        if model_name.lower() == "unet3d":
            return MultiTaskModels3D._MultiTask3DUNet(**kwargs)
        else:
            raise ValueError(
                f"Unknown 3D model: {model_name}. "
                f"Available models: ['unet3d']"
            )
