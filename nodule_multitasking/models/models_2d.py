"""Multi-Task 2D Models for Nodule Segmentation and Classification.

This module implements multi-task learning architectures that perform:
    1. Nodule Segmentation: Dense pixel-wise prediction
    2. Malignancy Classification: Image-level classification

Architecture: Shared encoder + Dual decoder/classifier heads
Based on segmentation-models-pytorch with custom classification heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from monai.networks.nets import SwinUNETR
from typing import Tuple

SEG_N_CLASSES = 2  # 0: background, 1: nodule
CLS_N_CLASSES = 2  # 0: benign, 1: malignant


class MultiTaskModels2D:
    """Factory class for 2D multi-task models.
    
    This class provides various multi-task architectures that combine
    segmentation and classification in a single model with shared encoder.
    """

    class _MultiTaskUNet(nn.Module):
        """Multi-Task U-Net with shared encoder.
        
        Architecture:
            Input → Shared Encoder → ┬→ Segmentation Decoder → Seg Mask
                                     └→ Classification Head → Malignancy
        
        The encoder features are used for both tasks:
        - Segmentation decoder: Full spatial resolution preserved
        - Classification head: Global pooling + FC layers
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet34", "efficientnet-b0")
            encoder_weights: Pretrained weights ("imagenet" or None)
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            dropout: Dropout rate for classification head (default: 0.3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            
            # Create base U-Net model for segmentation
            self.unet = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=seg_classes,
                activation=None,  # No activation, raw logits
            )
            
            # Get encoder output channels for classification head
            # Most encoders in smp have encoder.out_channels attribute
            if hasattr(self.unet.encoder, 'out_channels'):
                encoder_channels = self.unet.encoder.out_channels[-1]
            else:
                # Fallback: common encoder output sizes
                encoder_output_map = {
                    'resnet18': 512,
                    'resnet34': 512,
                    'resnet50': 2048,
                    'resnet101': 2048,
                    'efficientnet-b0': 1280,
                    'efficientnet-b1': 1280,
                    'efficientnet-b2': 1408,
                }
                encoder_channels = encoder_output_map.get(encoder_name, 512)
            
            # Classification head
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
                nn.Flatten(),
                nn.Linear(encoder_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass for multi-task learning.
            
            Args:
                x: Input tensor of shape (B, C, H, W)
            
            Returns:
                Tuple of:
                    - seg_output: Segmentation logits (B, seg_classes, H, W)
                    - cls_output: Classification logits (B, cls_classes)
            """
            # Extract encoder features
            features = self.unet.encoder(x)
            
            # Segmentation branch: pass features as list to decoder
            decoder_output = self.unet.decoder(features)
            seg_output = self.unet.segmentation_head(decoder_output)
            
            # Classification branch: use last encoder feature map (bottleneck)
            cls_output = self.classification_head(features[-1])
            
            return seg_output, cls_output

    class _MultiTaskUNetPlusPlus(nn.Module):
        """Multi-Task U-Net++ with shared encoder.
        
        Similar to MultiTaskUNet but using U-Net++ architecture for
        better feature aggregation through nested skip connections.
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet34", "efficientnet-b0")
            encoder_weights: Pretrained weights ("imagenet" or None)
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            dropout: Dropout rate for classification head (default: 0.3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            
            # Create base U-Net++ model for segmentation
            self.unetpp = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=seg_classes,
                activation=None,
            )
            
            # Get encoder output channels
            if hasattr(self.unetpp.encoder, 'out_channels'):
                encoder_channels = self.unetpp.encoder.out_channels[-1]
            else:
                encoder_output_map = {
                    'resnet18': 512, 'resnet34': 512, 'resnet50': 2048,
                    'resnet101': 2048, 'efficientnet-b0': 1280,
                }
                encoder_channels = encoder_output_map.get(encoder_name, 512)
            
            # Classification head
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(encoder_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning segmentation and classification outputs."""
            features = self.unetpp.encoder(x)
            
            # Segmentation
            decoder_output = self.unetpp.decoder(features)
            seg_output = self.unetpp.segmentation_head(decoder_output)
            
            # Classification
            cls_output = self.classification_head(features[-1])
            
            return seg_output, cls_output

    class _MultiTaskDeepLabV3Plus(nn.Module):
        """Multi-Task DeepLabV3+ with shared encoder.
        
        DeepLabV3+ with ASPP module for multi-scale feature extraction
        combined with classification head.
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet34", "resnet50")
            encoder_weights: Pretrained weights ("imagenet" or None)
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            dropout: Dropout rate for classification head (default: 0.3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            
            # Create base DeepLabV3+ model
            self.deeplabv3plus = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=seg_classes,
                activation=None,
            )
            
            # Get encoder output channels
            if hasattr(self.deeplabv3plus.encoder, 'out_channels'):
                encoder_channels = self.deeplabv3plus.encoder.out_channels[-1]
            else:
                encoder_output_map = {
                    'resnet18': 512, 'resnet34': 512, 'resnet50': 2048,
                }
                encoder_channels = encoder_output_map.get(encoder_name, 512)
            
            # Classification head
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(encoder_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning segmentation and classification outputs."""
            features = self.deeplabv3plus.encoder(x)
            
            # Segmentation
            decoder_output = self.deeplabv3plus.decoder(features)
            seg_output = self.deeplabv3plus.segmentation_head(decoder_output)
            
            # Classification
            cls_output = self.classification_head(features[-1])
            
            return seg_output, cls_output

    class _MultiTaskFPN(nn.Module):
        """Multi-Task Feature Pyramid Network.
        
        FPN with multi-scale features for segmentation and classification.
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet34", "resnet50")
            encoder_weights: Pretrained weights ("imagenet" or None)
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            dropout: Dropout rate for classification head (default: 0.3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            
            # Create base FPN model
            self.fpn = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=seg_classes,
                activation=None,
            )
            
            # Get encoder output channels
            if hasattr(self.fpn.encoder, 'out_channels'):
                encoder_channels = self.fpn.encoder.out_channels[-1]
            else:
                encoder_output_map = {
                    'resnet18': 512, 'resnet34': 512, 'resnet50': 2048,
                }
                encoder_channels = encoder_output_map.get(encoder_name, 512)
            
            # Classification head
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(encoder_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning segmentation and classification outputs."""
            features = self.fpn.encoder(x)
            
            # Segmentation
            decoder_output = self.fpn.decoder(features)
            seg_output = self.fpn.segmentation_head(decoder_output)
            
            # Classification
            cls_output = self.classification_head(features[-1])
            
            return seg_output, cls_output

    class _MultiTaskLinkNet(nn.Module):
        """Multi-Task LinkNet.
        
        Efficient encoder-decoder with skip connections plus classification head.
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet34", "resnet50")
            encoder_weights: Pretrained weights ("imagenet" or None)
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            dropout: Dropout rate for classification head (default: 0.3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            
            # Create base LinkNet model
            self.linknet = smp.Linknet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=seg_classes,
                activation=None,
            )
            
            # Get encoder output channels
            if hasattr(self.linknet.encoder, 'out_channels'):
                encoder_channels = self.linknet.encoder.out_channels[-1]
            else:
                encoder_output_map = {
                    'resnet18': 512, 'resnet34': 512, 'resnet50': 2048,
                }
                encoder_channels = encoder_output_map.get(encoder_name, 512)
            
            # Classification head
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(encoder_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning segmentation and classification outputs."""
            features = self.linknet.encoder(x)
            
            # Segmentation
            decoder_output = self.linknet.decoder(features)
            seg_output = self.linknet.segmentation_head(decoder_output)
            
            # Classification
            cls_output = self.classification_head(features[-1])
            
            return seg_output, cls_output

    class _MultiTaskMANet(nn.Module):
        """Multi-Task MA-Net.
        
        Multi-scale Attention Network for medical image segmentation with classification head.
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet34", "resnet50")
            encoder_weights: Pretrained weights ("imagenet" or None)
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            dropout: Dropout rate for classification head (default: 0.3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            
            # Create base MA-Net model
            self.manet = smp.MAnet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=seg_classes,
                activation=None,
            )
            
            # Get encoder output channels
            if hasattr(self.manet.encoder, 'out_channels'):
                encoder_channels = self.manet.encoder.out_channels[-1]
            else:
                encoder_output_map = {
                    'resnet18': 512, 'resnet34': 512, 'resnet50': 2048,
                }
                encoder_channels = encoder_output_map.get(encoder_name, 512)
            
            # Classification head
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(encoder_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning segmentation and classification outputs."""
            features = self.manet.encoder(x)
            
            # Segmentation
            decoder_output = self.manet.decoder(features)
            seg_output = self.manet.segmentation_head(decoder_output)
            
            # Classification
            cls_output = self.classification_head(features[-1])
            
            return seg_output, cls_output

    class _MultiTaskPSPNet(nn.Module):
        """Multi-Task PSPNet.
        
        Pyramid Scene Parsing Network with spatial pyramid pooling plus classification head.
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet34", "resnet50")
            encoder_weights: Pretrained weights ("imagenet" or None)
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            dropout: Dropout rate for classification head (default: 0.3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            
            # Create base PSPNet model
            self.pspnet = smp.PSPNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=seg_classes,
                activation=None,
            )
            
            # Get encoder output channels
            if hasattr(self.pspnet.encoder, 'out_channels'):
                encoder_channels = self.pspnet.encoder.out_channels[-1]
            else:
                encoder_output_map = {
                    'resnet18': 512, 'resnet34': 512, 'resnet50': 2048,
                }
                encoder_channels = encoder_output_map.get(encoder_name, 512)
            
            # Classification head
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(encoder_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning segmentation and classification outputs."""
            features = self.pspnet.encoder(x)
            
            # Segmentation
            decoder_output = self.pspnet.decoder(features)
            seg_output = self.pspnet.segmentation_head(decoder_output)
            
            # Classification
            cls_output = self.classification_head(features[-1])
            
            return seg_output, cls_output

    class _MultiTaskUperNet(nn.Module):
        """Multi-Task UperNet.
        
        Unified Perceptual Parsing Network for scene understanding with classification head.
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet34", "resnet50")
            encoder_weights: Pretrained weights ("imagenet" or None)
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            dropout: Dropout rate for classification head (default: 0.3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            
            # Create base UperNet model
            self.upernet = smp.UPerNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=seg_classes,
                activation=None,
            )
            
            # Get encoder output channels
            if hasattr(self.upernet.encoder, 'out_channels'):
                encoder_channels = self.upernet.encoder.out_channels[-1]
            else:
                encoder_output_map = {
                    'resnet18': 512, 'resnet34': 512, 'resnet50': 2048,
                }
                encoder_channels = encoder_output_map.get(encoder_name, 512)
            
            # Classification head
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(encoder_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning segmentation and classification outputs."""
            features = self.upernet.encoder(x)
            
            # Segmentation
            decoder_output = self.upernet.decoder(features)
            seg_output = self.upernet.segmentation_head(decoder_output)
            
            # Classification
            cls_output = self.classification_head(features[-1])
            
            return seg_output, cls_output

    class _MultiTaskSegformer(nn.Module):
        """Multi-Task Segformer.
        
        Segformer with transformer-based encoder for multi-scale features plus classification head.
        
        Args:
            encoder_name: Name of encoder (e.g., "mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5")
            encoder_weights: Pretrained weights ("imagenet" or None)
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            dropout: Dropout rate for classification head (default: 0.3)
        """
        def __init__(
            self,
            encoder_name: str = "mit_b2",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            
            # Create base Segformer model
            self.segformer = smp.Segformer(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=seg_classes,
                activation=None,
            )
            
            # Get encoder output channels
            # Segformer encoders have different output channels
            if hasattr(self.segformer.encoder, 'out_channels'):
                encoder_channels = self.segformer.encoder.out_channels[-1]
            else:
                # MiT (Mix Transformer) encoder output channels
                encoder_output_map = {
                    'mit_b0': 256,
                    'mit_b1': 512,
                    'mit_b2': 512,
                    'mit_b3': 512,
                    'mit_b4': 512,
                    'mit_b5': 512,
                }
                encoder_channels = encoder_output_map.get(encoder_name, 512)
            
            # Classification head
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(encoder_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning segmentation and classification outputs."""
            features = self.segformer.encoder(x)
            
            # Segmentation
            decoder_output = self.segformer.decoder(features)
            seg_output = self.segformer.segmentation_head(decoder_output)
            
            # Classification
            cls_output = self.classification_head(features[-1])
            
            return seg_output, cls_output

    class _MultiTaskSwinUNETR2D(nn.Module):
        """Multi-Task Swin UNETR for 2D images.
        
        Swin Transformer-based U-Net from MONAI with classification head.
        Uses hierarchical shifted window attention mechanism.
        
        Note: This uses MONAI's SwinUNETR with spatial_dims=2 for 2D images.
              SwinUNETR requires fixed input size.
        
        Args:
            img_size: Input image size (height, width). Default: (224, 224)
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            feature_size: Feature dimension size. Default: 24
            dropout: Dropout rate for classification head (default: 0.3)
            use_checkpoint: Use gradient checkpointing to save memory. Default: False
        """
        def __init__(
            self,
            img_size: tuple = (224, 224),
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            feature_size: int = 24,
            dropout: float = 0.3,
            use_checkpoint: bool = False,
        ):
            super().__init__()
            
            # Create 2D Swin UNETR model from MONAI
            self.swinunetr = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=seg_classes,
                feature_size=feature_size,
                use_checkpoint=use_checkpoint,
                spatial_dims=2,  # 2D for slice-based processing
            )
            
            # Classification head
            # SwinUNETR encoder output size is feature_size * 16 for the last layer in 2D
            # For feature_size=24: 24 * 16 = 384
            # For feature_size=48: 48 * 16 = 768
            encoder_channels = feature_size * 16
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(encoder_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning segmentation and classification outputs."""
            # SwinUNETR forward pass for segmentation
            # Get hidden features from encoder
            hidden_states_out = self.swinunetr.swinViT(x, self.swinunetr.normalize)
            
            # Segmentation output
            seg_output = self.swinunetr(x)
            
            # Classification from last encoder hidden state
            # hidden_states_out[4] is the last encoder output (before decoder)
            cls_output = self.classification_head(hidden_states_out[4])
            
            return seg_output, cls_output

    class _MultiTaskFCN(nn.Module):
        """Multi-Task FCN (Fully Convolutional Network).
        
        FCN from torchvision with classification head.
        Uses ResNet50 or ResNet101 backbone.
        
        Args:
            encoder_name: Encoder backbone ("resnet50" or "resnet101")
            encoder_weights: Ignored for FCN (always uses pretrained)
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            dropout: Dropout rate for classification head (default: 0.3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_weights: str = None,  # Ignored for FCN
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            
            import torchvision.models as models
            from torchvision.models.segmentation import FCN_ResNet50_Weights, FCN_ResNet101_Weights
            
            # Load pretrained FCN
            if encoder_name == "resnet50":
                self.fcn = models.segmentation.fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
            elif encoder_name == "resnet101":
                self.fcn = models.segmentation.fcn_resnet101(weights=FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
            else:
                raise ValueError(f"FCN only supports 'resnet50' or 'resnet101', got '{encoder_name}'")
            
            # Modify input layer for grayscale
            if in_channels != 3:
                self.fcn.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # Modify classifier for our classes
            self.fcn.classifier[4] = nn.Conv2d(512, seg_classes, kernel_size=1)
            if hasattr(self.fcn, 'aux_classifier'):
                self.fcn.aux_classifier[4] = nn.Conv2d(256, seg_classes, kernel_size=1)
            
            # Classification head - use backbone features
            encoder_channels = 2048  # Both resnet50 and resnet101 have 2048 output channels
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(encoder_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning segmentation and classification outputs."""
            # Get backbone features for classification
            features = self.fcn.backbone(x)
            
            # Segmentation output
            seg_output = self.fcn(x)['out']
            
            # Classification from bottleneck features
            cls_output = self.classification_head(features['out'])
            
            return seg_output, cls_output

    class _MultiTasknnUNet(nn.Module):
        """Multi-Task nnU-Net for 2D medical image segmentation and classification.
        
        Combines nnU-Net's self-configuring architecture for segmentation with
        a classification head that uses bottleneck features.
        
        Reference:
            Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. "nnU-Net: a self-configuring 
            method for deep learning-based biomedical image segmentation." 
            Nat Methods (2020). https://doi.org/10.1038/s41592-020-01008-z
        
        Args:
            in_channels: Number of input channels (default: 1)
            seg_classes: Number of segmentation classes (default: 2)
            cls_classes: Number of classification classes (default: 2)
            base_num_features: Base number of features (default: 32)
            num_pool: Number of pooling layers (default: 4 for 2D)
            dropout: Dropout rate for classification head (default: 0.3)
        """
        def __init__(
            self,
            in_channels: int = 1,
            seg_classes: int = SEG_N_CLASSES,
            cls_classes: int = CLS_N_CLASSES,
            base_num_features: int = 32,
            num_pool: int = 4,
            dropout: float = 0.3,
        ):
            super().__init__()
            
            from nnunet.network_architecture.generic_UNet import Generic_UNet
            
            # Create nnU-Net model for segmentation
            self.nnunet = Generic_UNet(
                input_channels=in_channels,
                base_num_features=base_num_features,
                num_classes=seg_classes,
                num_pool=num_pool,
                num_conv_per_stage=2,
                feat_map_mul_on_downscale=2,
                conv_op=nn.Conv2d,
                norm_op=nn.BatchNorm2d,
                norm_op_kwargs={'eps': 1e-5, 'affine': True},
                dropout_op=nn.Dropout2d,
                dropout_op_kwargs={'p': 0, 'inplace': True},
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
                deep_supervision=False,  # Disabled for multi-task
                dropout_in_localization=False,
                final_nonlin=lambda x: x,  # No final activation
                weightInitializer=None,
                pool_op_kernel_sizes=None,
                conv_kernel_sizes=None,
                upscale_logits=False,
                convolutional_pooling=False,
                convolutional_upsampling=False,
            )
            
            # Hook to capture bottleneck features
            self.bottleneck_features = None
            self._register_hook()
            
            # Auto-detect bottleneck feature channels with dummy forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, in_channels, 64, 64)
                _ = self.nnunet(dummy_input)
                if self.bottleneck_features is not None:
                    bottleneck_features = self.bottleneck_features.shape[1]
                else:
                    # Fallback to calculated value
                    bottleneck_features = base_num_features * (2 ** (num_pool - 1))
            
            # Classification head using detected bottleneck features
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(bottleneck_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, cls_classes),
            )
        
        def _register_hook(self):
            """Register forward hook to capture bottleneck features."""
            def hook_fn(module, input, output):
                self.bottleneck_features = output
            
            # Register hook on the last encoder conv block (bottleneck)
            # In Generic_UNet, this is conv_blocks_context[-1]
            if hasattr(self.nnunet, 'conv_blocks_context'):
                self.nnunet.conv_blocks_context[-1].register_forward_hook(hook_fn)
        
        def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass for multi-task learning.
            
            Args:
                x: Input tensor of shape (B, C, H, W)
            
            Returns:
                Tuple of:
                    - seg_output: Segmentation logits (B, seg_classes, H, W)
                    - cls_output: Classification logits (B, cls_classes)
            """
            # Forward through nnUNet (segmentation)
            seg_output = self.nnunet(x)
            
            # Use captured bottleneck features for classification
            if self.bottleneck_features is not None:
                cls_output = self.classification_head(self.bottleneck_features)
            else:
                # Fallback: use global pooling on segmentation output
                cls_output = self.classification_head(seg_output)
            
            return seg_output, cls_output

    @staticmethod
    def get_model(model_name: str, encoder_name: str = "resnet34", encoder_weights: str = "imagenet"):
        """
        Factory method to get multi-task model.
        
        Args:
            model_name: Name of model architecture ("unet", "unetplusplus", 
                       "deeplabv3plus", "fpn", "linknet", "manet", "pspnet",
                       "upernet", "segformer", "swinunetr", "fcn")
            encoder_name: Name of encoder backbone
                         - For CNN models: "resnet34", "resnet50", "efficientnet-b0", etc.
                         - For Segformer: "mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"
                         - For SwinUNETR: Not used (ignored)
            **kwargs: Additional arguments for model initialization
                     - For SwinUNETR: img_size=(224, 224), feature_size=24, use_checkpoint=False
            
        Returns:
            Multi-task model instance
            
        Note:
            - FCN only supports resnet50 and resnet101 encoders
            - SwinUNETR requires fixed input size (default: 224x224)
            - Segformer uses MiT (Mix Transformer) encoders
        """
        in_channels = 1
        model_name_lower = model_name.lower()
        
        if "unet" in model_name_lower and "upernet" not in model_name_lower and "unet++" not in model_name_lower and "swinunetr" not in model_name_lower and "swin_unetr" not in model_name_lower and "nnunet" not in model_name_lower:
            return MultiTaskModels2D._MultiTaskUNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels
            )
            
        elif "unet++" in model_name_lower:
            return MultiTaskModels2D._MultiTaskUNetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels
            )
        
        elif "deeplabv3+" in model_name_lower:
            return MultiTaskModels2D._MultiTaskDeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels
            )
        
        elif "fpn" in model_name_lower:
            return MultiTaskModels2D._MultiTaskFPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels
            )
        
        elif "manet" in model_name_lower:
            return MultiTaskModels2D._MultiTaskMANet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels
            )
        
        elif "linknet" in model_name_lower:
            return MultiTaskModels2D._MultiTaskLinkNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels
            )
        
        elif "pspnet" in model_name_lower:
            return MultiTaskModels2D._MultiTaskPSPNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels
            )
        
        elif "upernet" in model_name_lower:
            return MultiTaskModels2D._MultiTaskUperNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels
            )
        
        elif "dpt" in model_name_lower:
            return MultiTaskModels2D._MultiTaskDPT(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels
            )
            
        elif "segformer" in model_name_lower:
            return MultiTaskModels2D._MultiTaskSegformer(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels
            )
        
        elif "swinunetr" in model_name_lower or "swin_unetr" in model_name_lower:
            return MultiTaskModels2D._MultiTaskSwinUNETR2D(in_channels=in_channels)
        
        elif "fcn" in model_name_lower:
            # FCN from torchvision - only uses encoder_name (no encoder_weights)
            return MultiTaskModels2D._MultiTaskFCN(
                encoder_name=encoder_name,
                in_channels=in_channels
            )
        
        elif "nnunet" in model_name_lower:
            # nnU-Net - self-configuring architecture
            # encoder_name and encoder_weights are ignored
            return MultiTaskModels2D._MultiTasknnUNet(
                in_channels=in_channels,
            )
        
        else:
            raise ValueError(
                f"Model '{model_name}' not recognized. "
                f"Available models: unet, unet++, deeplabv3+, fpn, manet, linknet, pspnet, upernet, dpt, segformer, swinunetr, nnunet, fcn"
            )

