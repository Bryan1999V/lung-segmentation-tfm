import torch.nn as nn
from monai.networks.nets import (
    UNet as MonaiUNet,
    VNet,
    SwinUNETR,
    AttentionUnet,
    SegResNet,
    DynUNet,
)
import segmentation_models_pytorch_3d as smp3d

SEG_N_CLASSES_3D = 2  # 0: background, 1: nodule (benign or malignant)


class SegmentationModels3D:
    """Factory class for 3D segmentation models.
    
    This class provides various 3D segmentation architectures optimized for
    binary volumetric nodule segmentation using MONAI library.
    All models are designed to work with 3D CT volumes.
    """

    class _UNet3D(nn.Module):
        """3D U-Net architecture from MONAI.
        
        Standard U-Net adapted for 3D volumetric segmentation.
        
        Args:
            spatial_dims: Number of spatial dimensions (default: 3)
            in_channels: Number of input channels (default: 1 for grayscale CT)
            out_channels: Number of output classes (default: 3)
            channels: Sequence of channel counts for each level
            strides: Sequence of strides for each level
            num_res_units: Number of residual units at each level
        """
        def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = SEG_N_CLASSES_3D,
            channels: tuple = (16, 32, 64, 128, 256),
            strides: tuple = (2, 2, 2, 2),
            num_res_units: int = 2,
        ):
            super().__init__()
            
            self.model = MonaiUNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
                num_res_units=num_res_units,
            )
        
        def forward(self, x):
            return self.model(x)

    class _VNet3D(nn.Module):
        """V-Net architecture from MONAI.
        
        V-Net architecture specifically designed for 3D medical image segmentation.
        Uses residual connections and PReLU activations.
        
        Args:
            spatial_dims: Number of spatial dimensions (default: 3)
            in_channels: Number of input channels (default: 1 for grayscale CT)
            out_channels: Number of output classes (default: 2)
        """
        def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = SEG_N_CLASSES_3D,
        ):
            super().__init__()
            
            self.model = VNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
            )
        
        def forward(self, x):
            return self.model(x)

    class _SwinUNETR3D(nn.Module):
        """Swin UNETR architecture from MONAI.
        
        Swin Transformer-based U-Net for 3D medical image segmentation.
        Combines Swin Transformer encoder with CNN decoder.
        
        Args:
            img_size: Input image size (height, width, depth)
            in_channels: Number of input channels (default: 1 for grayscale CT)
            out_channels: Number of output classes (default: 2)
            feature_size: Feature dimension size
            use_checkpoint: Use gradient checkpointing to save memory
        """
        def __init__(
            self,
            img_size: tuple = (96, 96, 96),
            in_channels: int = 1,
            out_channels: int = SEG_N_CLASSES_3D,
            feature_size: int = 24,
            use_checkpoint: bool = False,
        ):
            super().__init__()
            
            self.model = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                use_checkpoint=use_checkpoint,
            )
        
        def forward(self, x):
            return self.model(x)

    class _AttentionUNet3D(nn.Module):
        """Attention U-Net architecture from MONAI.
        
        U-Net with attention gates for improved segmentation performance.
        Attention mechanisms help focus on relevant features.
        
        Args:
            spatial_dims: Number of spatial dimensions (default: 3)
            in_channels: Number of input channels (default: 1 for grayscale CT)
            out_channels: Number of output classes (default: 2)
            channels: Sequence of channel counts for each level
            strides: Sequence of strides for each level
        """
        def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = SEG_N_CLASSES_3D,
            channels: tuple = (16, 32, 64, 128, 256),
            strides: tuple = (2, 2, 2, 2),
        ):
            super().__init__()
            
            self.model = AttentionUnet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
            )
        
        def forward(self, x):
            return self.model(x)

    class _SegResNet3D(nn.Module):
        """SegResNet architecture from MONAI.
        
        Residual segmentation network with deep supervision.
        Efficient architecture for 3D medical image segmentation.
        
        Args:
            spatial_dims: Number of spatial dimensions (default: 3)
            in_channels: Number of input channels (default: 1 for grayscale CT)
            out_channels: Number of output classes (default: 2)
            init_filters: Number of filters in the first layer
        """
        def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = SEG_N_CLASSES_3D,
            init_filters: int = 8,
        ):
            super().__init__()
            
            self.model = SegResNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                init_filters=init_filters,
            )
        
        def forward(self, x):
            return self.model(x)

    class _DynUNet3D(nn.Module):
        """Dynamic U-Net architecture from MONAI.
        
        Dynamically configurable U-Net that can adapt to different input sizes.
        Used in nnU-Net framework.
        
        Args:
            spatial_dims: Number of spatial dimensions (default: 3)
            in_channels: Number of input channels (default: 1 for grayscale CT)
            out_channels: Number of output classes (default: 2)
            kernel_size: Convolution kernel size for each layer
            strides: Strides for each layer
            upsample_kernel_size: Kernel size for upsampling
        """
        def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = SEG_N_CLASSES_3D,
            kernel_size: list = None,
            strides: list = None,
            upsample_kernel_size: list = None,
        ):
            super().__init__()
            
            # Default configuration for 96x96x96 volumes
            if kernel_size is None:
                kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
            if strides is None:
                strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
            if upsample_kernel_size is None:
                upsample_kernel_size = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
            
            self.model = DynUNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                upsample_kernel_size=upsample_kernel_size,
            )
        
        def forward(self, x):
            return self.model(x)

    # Placeholder implementations for models not directly available in MONAI
    # These would require custom implementations or alternative libraries
    
    class _MANet3D(nn.Module):
        """MA-Net 3D (Multi-scale Attention Network).
        
        Multi-scale attention network from segmentation-models-pytorch-3d.
        If not available, falls back to MONAI's Attention U-Net.
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet50", "resnet101")
            encoder_weights: Pretrained weights ("imagenet", None)
            in_channels: Number of input channels (default: 1)
            classes: Number of output classes (default: 2)
        """
        def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_weights: str = None,  # 3D models typically don't have imagenet weights
            in_channels: int = 1,
            classes: int = SEG_N_CLASSES_3D,
        ):
            super().__init__()
            self.model = smp3d.MAnet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        
        def forward(self, x):
            return self.model(x)

    class _LinkNet3D(nn.Module):
        """LinkNet 3D.
        
        Efficient encoder-decoder architecture from segmentation-models-pytorch-3d.
        If not available, falls back to MONAI's SegResNet.
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet34", "resnet50")
            encoder_weights: Pretrained weights ("imagenet", None)
            in_channels: Number of input channels (default: 1)
            classes: Number of output classes (default: 2)
        """
        def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = None,
            in_channels: int = 1,
            classes: int = SEG_N_CLASSES_3D,
        ):
            super().__init__()
            self.model = smp3d.Linknet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        
        def forward(self, x):
            return self.model(x)

    class _FPN3D(nn.Module):
        """Feature Pyramid Network 3D.
        
        Feature Pyramid Network from segmentation-models-pytorch-3d.
        If not available, falls back to MONAI's U-Net.
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet50", "resnet101")
            encoder_weights: Pretrained weights ("imagenet", None)
            in_channels: Number of input channels (default: 1)
            classes: Number of output classes (default: 2)
        """
        def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_weights: str = None,
            in_channels: int = 1,
            classes: int = SEG_N_CLASSES_3D,
        ):
            super().__init__()
            self.model = smp3d.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        
        def forward(self, x):
            return self.model(x)

    class _PSPNet3D(nn.Module):
        """Pyramid Scene Parsing Network 3D.
        
        PSPNet with spatial pyramid pooling from segmentation-models-pytorch-3d.
        If not available, falls back to MONAI's SegResNet.
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet50", "resnet101")
            encoder_weights: Pretrained weights ("imagenet", None)
            in_channels: Number of input channels (default: 1)
            classes: Number of output classes (default: 2)
        """
        def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_weights: str = None,
            in_channels: int = 1,
            classes: int = SEG_N_CLASSES_3D,
        ):
            super().__init__()
            self.model = smp3d.PSPNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        
        def forward(self, x):
            return self.model(x)

    class _DeepLabV3_3D(nn.Module):
        """DeepLabV3Plus 3D.
        
        DeepLabV3+ with atrous convolutions from segmentation-models-pytorch-3d.
        If not available, falls back to MONAI's SegResNet.
        
        Args:
            encoder_name: Name of encoder (e.g., "resnet50", "resnet101")
            encoder_weights: Pretrained weights ("imagenet", None)
            in_channels: Number of input channels (default: 1)
            classes: Number of output classes (default: 2)
        """
        def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_weights: str = None,
            in_channels: int = 1,
            classes: int = SEG_N_CLASSES_3D,
        ):
            super().__init__()
            self.model = smp3d.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        
        def forward(self, x):
            return self.model(x)

    @staticmethod
    def get_model(
        model_name: str, 
        img_size: tuple = (96, 96, 96),
        channels: tuple = (16, 32, 64, 128, 256),
        strides: tuple = (2, 2, 2, 2),
        encoder_name: str = "resnet50",
        encoder_weights: str = None,
    ) -> nn.Module:
        """Get a 3D segmentation model by name.
        
        Available models from MONAI:
        - unet3d: Standard 3D U-Net
        - vnet3d: V-Net with residual connections
        - swinunetr3d: Swin Transformer-based U-Net
        - attentionunet3d: U-Net with attention gates
        - segresnet3d: Residual segmentation network
        - dynunet3d: Dynamic U-Net (nnU-Net style)
        
        Models from segmentation-models-pytorch-3d (with encoder support):
        - manet3d: Multi-scale Attention Network
        - linknet3d: Efficient encoder-decoder
        - fpn3d: Feature Pyramid Network
        - pspnet3d: Pyramid Scene Parsing Network
        - deeplabv3_3d: DeepLabV3+ with atrous convolutions
        
        Args:
            model_name: Name of the model architecture
            img_size: Input image size (height, width, depth) - used for SwinUNETR
            channels: Channel configuration for encoder-decoder models
            strides: Stride configuration for downsampling
            encoder_name: Encoder backbone (for smp3d models, e.g., "resnet50")
            encoder_weights: Pretrained weights (typically None for 3D)
        
        Returns:
            Initialized 3D segmentation model
        
        Examples:
            >>> model = SegmentationModels3D.get_model("unet3d")
            >>> model = SegmentationModels3D.get_model("vnet3d")
            >>> model = SegmentationModels3D.get_model("swinunetr3d", img_size=(96, 96, 96))
            >>> model = SegmentationModels3D.get_model("manet3d", encoder_name="resnet50")
        """
        model_name_lower = model_name.lower()
        
        # Native MONAI implementations
        if "unet" in model_name_lower:
            return SegmentationModels3D._UNet3D(
                channels=channels,
                strides=strides
            )
        
        elif "vnet" in model_name_lower:
            return SegmentationModels3D._VNet3D()
        
        elif "swinunetr" in model_name_lower:
            return SegmentationModels3D._SwinUNETR3D(img_size=img_size)
        
        elif "attentionunet" in model_name_lower:
            return SegmentationModels3D._AttentionUNet3D(
                channels=channels,
                strides=strides
            )
        
        elif "segresnet" in model_name_lower:
            return SegmentationModels3D._SegResNet3D()
        
        elif "dynunet" in model_name_lower:
            return SegmentationModels3D._DynUNet3D()
        
        # Models from segmentation-models-pytorch-3d
        elif "manet" in model_name_lower:
            return SegmentationModels3D._MANet3D(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=channels,
                classes=SEG_N_CLASSES_3D,
            )
        
        elif "linknet" in model_name_lower:
            return SegmentationModels3D._LinkNet3D(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=channels,
                classes=SEG_N_CLASSES_3D,
            )
        
        elif "fpn" in model_name_lower:
            return SegmentationModels3D._FPN3D(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=channels,
                classes=SEG_N_CLASSES_3D,
            )
        
        elif "pspnet" in model_name_lower:
            return SegmentationModels3D._PSPNet3D(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=channels,
                classes=SEG_N_CLASSES_3D,
            )
        
        elif "deeplabv3" in model_name_lower:
            return SegmentationModels3D._DeepLabV3_3D(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=channels,
                classes=SEG_N_CLASSES_3D,
            )
        
        else:
            raise ValueError(
                f"Model '{model_name}' not recognized. "
                f"Available models: unet3d, vnet3d, swinunetr3d, attentionunet3d, "
                f"segresnet3d, dynunet3d, manet3d, linknet3d, fpn3d, pspnet3d, deeplabv3_3d"
            )
