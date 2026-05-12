import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp
from torchvision.models.segmentation import FCN_ResNet50_Weights, FCN_ResNet101_Weights
from monai.networks.nets import SwinUNETR

SEG_N_CLASSES = 2  # 0: background, 1: nodule (benign or malignant)


class SegmentationModels2D:
    """Factory class for 2D segmentation models.
    
    This class provides various segmentation architectures optimized for
    binary nodule segmentation. Models follow a similar structure to
    ClassificationModels2D but adapted for dense prediction tasks.
    """

    class _UNet(nn.Module):
        """U-Net architecture wrapper using segmentation-models-pytorch.
        
        Wrapper around segmentation_models_pytorch.Unet for flexible encoder selection.
        Supports any encoder available in segmentation-models-pytorch.
        
        Args:
            encoder_name: Name of encoder architecture (e.g., "resnet18", "resnet34", "resnet50", 
                         "efficientnet-b0", "mobilenet_v2", etc.). Default: "resnet34"
            encoder_weights: Pretrained weights to use ("imagenet", None, etc.). Default: "imagenet"
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 2)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
        ):
            super().__init__()
            
            # Create U-Net model from segmentation-models-pytorch
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # No activation, raw logits for CrossEntropyLoss
            )
        
        def forward(self, x):
            return self.model(x)

    class _UNetPlusPlus(nn.Module):
        """U-Net++ architecture wrapper using segmentation-models-pytorch.
        
        Wrapper around segmentation_models_pytorch.UnetPlusPlus for flexible encoder selection.
        Supports any encoder available in segmentation-models-pytorch.
        
        Args:
            encoder_name: Name of encoder architecture (e.g., "resnet18", "resnet34", "resnet50", 
                         "efficientnet-b0", "mobilenet_v2", etc.). Default: "resnet34"
            encoder_weights: Pretrained weights to use ("imagenet", None, etc.). Default: "imagenet"
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 2)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
        ):
            super().__init__()
            
            # Create U-Net++ model from segmentation-models-pytorch
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # No activation, raw logits for CrossEntropyLoss
            )
        
        def forward(self, x):
            return self.model(x)

    class _DeepLabV3Plus(nn.Module):
        """DeepLabV3Plus wrapper using segmentation-models-pytorch.
        
        Wrapper around segmentation_models_pytorch.DeepLabV3Plus for flexible encoder selection.
        
        Args:
            encoder_name: Name of encoder architecture (e.g., "resnet50", "resnet101", 
                         "efficientnet-b3", etc.). Default: "resnet50"
            encoder_weights: Pretrained weights to use ("imagenet", None, etc.). Default: "imagenet"
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
        ):
            super().__init__()
            
            # Create DeepLabV3Plus model from segmentation-models-pytorch
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # No activation, raw logits for CrossEntropyLoss
            )
            
        def forward(self, x):
            return self.model(x)

    class _FPN(nn.Module):
        """Feature Pyramid Network (FPN) wrapper using segmentation-models-pytorch.
        
        Wrapper around segmentation_models_pytorch.FPN for flexible encoder selection.
        FPN is used as modern alternative to classic FCN.
        
        Args:
            encoder_name: Name of encoder architecture (e.g., "resnet50", "resnet101", 
                         "efficientnet-b3", etc.). Default: "resnet50"
            encoder_weights: Pretrained weights to use ("imagenet", None, etc.). Default: "imagenet"
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
        ):
            super().__init__()
            
            # Create FPN model from segmentation-models-pytorch
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # No activation, raw logits for CrossEntropyLoss
            )
            
        def forward(self, x):
            return self.model(x)

    class _MANet(nn.Module):
        """MA-Net wrapper using segmentation-models-pytorch.
        
        Multi-scale Attention Network for medical image segmentation.
        
        Args:
            encoder_name: Name of encoder architecture (e.g., "resnet50", "resnet101", 
                         "efficientnet-b3", etc.). Default: "resnet50"
            encoder_weights: Pretrained weights to use ("imagenet", None, etc.). Default: "imagenet"
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
        ):
            super().__init__()
            
            # Create MA-Net model from segmentation-models-pytorch
            self.model = smp.MAnet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # No activation, raw logits for CrossEntropyLoss
            )
            
        def forward(self, x):
            return self.model(x)

    class _LinkNet(nn.Module):
        """LinkNet wrapper using segmentation-models-pytorch.
        
        Efficient encoder-decoder architecture with skip connections.
        
        Args:
            encoder_name: Name of encoder architecture (e.g., "resnet50", "resnet101", 
                         "efficientnet-b3", etc.). Default: "resnet50"
            encoder_weights: Pretrained weights to use ("imagenet", None, etc.). Default: "imagenet"
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
        ):
            super().__init__()
            
            # Create LinkNet model from segmentation-models-pytorch
            self.model = smp.Linknet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # No activation, raw logits for CrossEntropyLoss
            )
            
        def forward(self, x):
            return self.model(x)

    class _PSPNet(nn.Module):
        """PSPNet wrapper using segmentation-models-pytorch.
        
        Pyramid Scene Parsing Network with spatial pyramid pooling.
        
        Args:
            encoder_name: Name of encoder architecture (e.g., "resnet50", "resnet101", 
                         "efficientnet-b3", etc.). Default: "resnet50"
            encoder_weights: Pretrained weights to use ("imagenet", None, etc.). Default: "imagenet"
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
        ):
            super().__init__()
            
            # Create PSPNet model from segmentation-models-pytorch
            self.model = smp.PSPNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # No activation, raw logits for CrossEntropyLoss
            )
            
        def forward(self, x):
            return self.model(x)

    class _UperNet(nn.Module):
        """UperNet wrapper using segmentation-models-pytorch.
        
        Unified Perceptual Parsing Network for scene understanding.
        
        Args:
            encoder_name: Name of encoder architecture (e.g., "resnet50", "resnet101", 
                         "efficientnet-b3", etc.). Default: "resnet50"
            encoder_weights: Pretrained weights to use ("imagenet", None, etc.). Default: "imagenet"
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
        ):
            super().__init__()
            
            # Create UperNet model from segmentation-models-pytorch  
            self.model = smp.UPerNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # No activation, raw logits for CrossEntropyLoss
            )
            
        def forward(self, x):
            return self.model(x)

    class _DPT(nn.Module):
        """DPT wrapper using segmentation-models-pytorch.
        
        Dense Prediction Transformer for dense prediction tasks.
        Note: Use Swin/BEiT encoders for best compatibility. Standard ViT may not support forward_intermediates.
        
        Args:
            encoder_name: Name of encoder architecture from DPT encoder list. 
                         Recommended: tu-swin_base_patch4_window7_224.ms_in22k_ft_in1k
            encoder_weights: Pretrained weights to use ("imagenet", None, etc.). Default: "imagenet"
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 2)
        """
        def __init__(
            self,
            encoder_name: str = "tu-swin_base_patch4_window7_224.ms_in22k_ft_in1k",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
        ):
            super().__init__()
            
            self.model = smp.DPT(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # No activation, raw logits for CrossEntropyLoss
            )
            
        def forward(self, x):
            return self.model(x)

    class _Segformer(nn.Module):
        """Segformer wrapper using segmentation-models-pytorch.
        
        Segformer for dense prediction tasks.
        
        Args:
            encoder_name: Name of encoder architecture (e.g., "resnet50", "resnet101", 
                         "efficientnet-b3", etc.). Default: "resnet50"
            encoder_weights: Pretrained weights to use ("imagenet", None, etc.). Default: "imagenet"
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
        ):
            super().__init__()
            
            # Create Segformer model from segmentation-models-pytorch
            self.model = smp.Segformer(  # Fallback to DeepLabV3Plus as Segformer might not be available
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # No activation, raw logits for CrossEntropyLoss
            )
            
        def forward(self, x):
            return self.model(x)

    class _SwinUNETR2D(nn.Module):
        """Swin UNETR wrapper using MONAI for 2D images.
        
        Swin Transformer-based U-Net for 2D medical image segmentation.
        Combines Swin Transformer encoder with CNN decoder from MONAI.
        
        Note: This uses MONAI's SwinUNETR with spatial_dims=2 for 2D images.
              Requires MONAI to be installed.
        
        Args:
            img_size: Input image size (height, width). Default: (224, 224)
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 2)
            feature_size: Feature dimension size. Default: 24
            use_checkpoint: Use gradient checkpointing to save memory. Default: False
        """
        def __init__(
            self,
            img_size: tuple = (224, 224),
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
            feature_size: int = 24,
            use_checkpoint: bool = False,
        ):
            super().__init__()
            
            # Create 2D Swin UNETR model from MONAI
            self.model = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=num_classes,
                feature_size=feature_size,
                use_checkpoint=use_checkpoint,
                spatial_dims=2,  # KEY: Use 2D instead of 3D
            )
        
        def forward(self, x):
            return self.model(x)

    class _nnUNet(nn.Module):
        """nnU-Net (No New U-Net) for 2D medical image segmentation.
        
        Self-configuring method for deep learning-based biomedical image segmentation.
        Uses the Generic_UNet architecture from the official nnU-Net implementation.
        
        Reference:
            Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. "nnU-Net: a self-configuring 
            method for deep learning-based biomedical image segmentation." 
            Nat Methods (2020). https://doi.org/10.1038/s41592-020-01008-z
        
        Args:
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 2)
            base_num_features: Base number of features (default: 32)
            num_pool: Number of pooling layers (default: 4 for 2D)
            num_conv_per_stage: Number of conv layers per stage (default: 2)
            deep_supervision: Use deep supervision (default: False for simplicity)
        """
        def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
            base_num_features: int = 32,
            num_pool: int = 4,
            num_conv_per_stage: int = 2,
            deep_supervision: bool = False,
        ):
            super().__init__()
            
            from nnunet.network_architecture.generic_UNet import Generic_UNet
            import torch.nn as nn
            
            # Create nnU-Net model with 2D convolutions
            self.model = Generic_UNet(
                input_channels=in_channels,
                base_num_features=base_num_features,
                num_classes=num_classes,
                num_pool=num_pool,
                num_conv_per_stage=num_conv_per_stage,
                feat_map_mul_on_downscale=2,
                conv_op=nn.Conv2d,
                norm_op=nn.BatchNorm2d,
                norm_op_kwargs={'eps': 1e-5, 'affine': True},
                dropout_op=nn.Dropout2d,
                dropout_op_kwargs={'p': 0, 'inplace': True},
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
                deep_supervision=deep_supervision,
                dropout_in_localization=False,
                final_nonlin=lambda x: x,  # No final activation, raw logits
                weightInitializer=None,  # Use default He initialization
                pool_op_kernel_sizes=None,  # Auto-configure
                conv_kernel_sizes=None,  # Auto-configure
                upscale_logits=False,
                convolutional_pooling=False,
                convolutional_upsampling=False,
            )
            
            self.deep_supervision = deep_supervision
        
        def forward(self, x):
            output = self.model(x)
            # If deep supervision is enabled, model returns list of outputs
            # We only want the final output for inference
            if self.deep_supervision and isinstance(output, (list, tuple)):
                return output[0]
            return output

    class _FCN(nn.Module):
        """FCN wrapper using PyTorch's torchvision implementation.
        
        Fully Convolutional Network for semantic segmentation.
        Uses torchvision's pretrained FCN models (ResNet50 or ResNet101 backbone).
        
        Args:
            encoder_name: Encoder backbone name ("resnet50" or "resnet101"). Default: "resnet50"
            in_channels: Number of input channels (default: 1 for grayscale CT)
            num_classes: Number of output classes (default: 3)
        """
        def __init__(
            self,
            encoder_name: str = "resnet50",
            in_channels: int = 1,
            num_classes: int = SEG_N_CLASSES,
        ):
            super().__init__()
            
            # Load pretrained FCN model from torchvision
            if encoder_name == "resnet50":
                self.model = models.segmentation.fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
            elif encoder_name == "resnet101":
                self.model = models.segmentation.fcn_resnet101(weights=FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
            else:
                raise ValueError(f"FCN only supports 'resnet50' or 'resnet101' encoders, got '{encoder_name}'")
            
            # Modify input layer for grayscale input
            if in_channels != 3:
                self.model.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            
            # Modify classifier for our number of classes
            self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
            
            # Also modify aux classifier if it exists
            if hasattr(self.model, 'aux_classifier'):
                self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
        def forward(self, x):
            # FCN returns a dict with 'out' and optionally 'aux'
            output = self.model(x)
            return output['out']

    @staticmethod
    def get_model(
        model_name: str, 
        encoder_name: str, 
        encoder_weights: str = "imagenet",
        num_classes: int = SEG_N_CLASSES
    ) -> nn.Module:
        """Factory method to create segmentation models.
        
        Args:
            model_name: Name of the segmentation architecture
            encoder_name: Name of the encoder backbone
            encoder_weights: Pretrained weights ("imagenet" or None)
            num_classes: Number of output classes (default: 2 for binary segmentation)
        
        Returns:
            Configured segmentation model
        
        Note:
            Automatically determines input channels based on config.USE_PIXEL_THRESHOLD_SEPARATION:
            - If False: 1 channel (standard grayscale CT)
            - If True: 4 channels (PTS multi-channel input)
        """
        # Import config here to avoid circular imports
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from config.common import config
        
        # Determine number of input channels based on PTS configuration
        in_channels = 4 if config.USE_PIXEL_THRESHOLD_SEPARATION else 1
        
        model_name_lower = model_name.lower()
        
        if "unet" in model_name_lower and "upernet" not in model_name_lower and "unet++" not in model_name_lower and "swinunetr" not in model_name_lower and "swin_unetr" not in model_name_lower and "nnunet" not in model_name_lower:
            return SegmentationModels2D._UNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
            
        elif "unet++" in model_name_lower:
            return SegmentationModels2D._UNetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
        
        elif "deeplabv3+" in model_name_lower:
            return SegmentationModels2D._DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
        
        elif "fpn" in model_name_lower:
            return SegmentationModels2D._FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
        
        elif "manet" in model_name_lower:
            return SegmentationModels2D._MANet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
        
        elif "linknet" in model_name_lower:
            return SegmentationModels2D._LinkNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
        
        elif "pspnet" in model_name_lower:
            return SegmentationModels2D._PSPNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
        
        elif "upernet" in model_name_lower:
            return SegmentationModels2D._UperNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
        
        elif "dpt" in model_name_lower:
            return SegmentationModels2D._DPT(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
            
        elif "segformer" in model_name_lower:
            return SegmentationModels2D._Segformer(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
        
        elif "swinunetr" in model_name_lower or "swin_unetr" in model_name_lower:
            return SegmentationModels2D._SwinUNETR2D(in_channels=in_channels, num_classes=num_classes)
        
        elif "fcn" in model_name_lower:
            # FCN from torchvision - only uses encoder_name (no encoder_weights)
            return SegmentationModels2D._FCN(
                encoder_name=encoder_name,
                in_channels=in_channels,
                num_classes=num_classes
            )
        
        elif "nnunet" in model_name_lower:
            # nnU-Net - self-configuring architecture
            # encoder_name and encoder_weights are ignored (nnUNet doesn't use pretrained encoders)
            return SegmentationModels2D._nnUNet(
                in_channels=in_channels,
                num_classes=num_classes,
            )
        
        else:
            raise ValueError(
                f"Model '{model_name}' not recognized. "
                f"Available models: unet, unet++, deeplabv3+, fpn, manet, linknet, pspnet, upernet, dpt, segformer, swinunetr, nnunet, fcn"
            )
