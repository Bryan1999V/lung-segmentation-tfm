"""
Model Complexity Analysis: Params and GFLOPs
=============================================

Calculates parameters and GFLOPs for all classification, segmentation,
and foundation models used in the project.

Usage:
    python calculate_model_complexity.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import pandas as pd
from ptflops import get_model_complexity_info
from torchinfo import summary
import warnings
warnings.filterwarnings('ignore')

from config.common import config
from nodule_classification.models.models_2d import ClassificationModels2D
from nodule_segmentation.models.models_2d import SegmentationModels2D
from nodule_segmentation.foundation_models.models.medsam2_wrapper import MedSAM2Wrapper
from nodule_segmentation.foundation_models.models.sam2_wrapper import SAM2Wrapper
from nodule_segmentation.foundation_models.models.sam3_wrapper import SAM3Wrapper


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_flops_params(model, input_shape, device='cpu'):
    """Calculate FLOPs and params using ptflops.
    
    Args:
        model: PyTorch model
        input_shape: Tuple of (C, H, W) for input
        device: Device to run calculation on
        
    Returns:
        Tuple of (GFLOPs, params_millions)
    """
    try:
        model = model.to(device)
        model.eval()
        
        # Use ptflops to calculate
        macs, params = get_model_complexity_info(
            model, 
            input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        
        # Convert MACs to FLOPs (1 MAC ≈ 2 FLOPs)
        flops = 2 * macs
        gflops = flops / 1e9
        params_m = params / 1e6
        
        return gflops, params_m
        
    except Exception as e:
        print(f"    ⚠️  Error with ptflops: {e}")
        # Fallback to simple parameter counting
        total_params, _ = count_parameters(model)
        return None, total_params / 1e6


def analyze_classification_models():
    """Analyze all classification models."""
    print("\n" + "="*80)
    print("📊 CLASSIFICATION MODELS")
    print("="*80)
    
    results = []
    
    # Input shape for classification (typically 224x224 RGB)
    input_shape = (3, 224, 224)
    
    models_config = [
        ("ResNet-18", "resnet", "18"),
        ("ResNet-50", "resnet", "50"),
        ("ResNet-152", "resnet", "152"),
        ("EfficientNet-V2-S", "efficientnet", "S"),
        ("EfficientNet-V2-M", "efficientnet", "M"),
        ("EfficientNet-V2-L", "efficientnet", "L"),
        ("ConvNeXt-Tiny", "convnext", "S"),
        ("ConvNeXt-Base", "convnext", "M"),
        ("ConvNeXt-Large", "convnext", "L"),
        ("ViT-B/16", "vit", "S"),
        ("ViT-L/16", "vit", "M"),
        ("ViT-H/14", "vit", "L"),
        ("PVT-v2-B2", "pvt", "S"),
        ("PVT-v2-B4", "pvt", "M"),
        ("PVT-v2-B5", "pvt", "L"),
    ]
    
    # Add DenseNet121 and MobileNetV2 directly from torchvision
    additional_models = [
        ("DenseNet-121", None, None),
        ("MobileNet-V2", None, None),
    ]
    
    for model_name, model_type, weight_size in models_config:
        print(f"\n🔍 Analyzing {model_name}...")
        
        try:
            model = ClassificationModels2D.get_model(model_type, weight_size)
            gflops, params_m = get_flops_params(model, input_shape)
            
            results.append({
                'Model': model_name,
                'Params (M)': f"{params_m:.2f}",
                'GFLOPs': f"{gflops:.2f}" if gflops is not None else "N/A"
            })
            
            print(f"   ✅ Params: {params_m:.2f}M, GFLOPs: {gflops:.2f}" if gflops else f"   ✅ Params: {params_m:.2f}M")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'Model': model_name,
                'Params (M)': "Error",
                'GFLOPs': "Error"
            })
    
    # Analyze additional models (DenseNet, MobileNet) from torchvision directly
    for model_name, _, _ in additional_models:
        print(f"\n🔍 Analyzing {model_name}...")
        
        try:
            import torchvision.models as models
            from torchvision.models import DenseNet121_Weights, MobileNet_V2_Weights
            
            if model_name == "DenseNet-121":
                model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
                # Replace classifier for binary classification
                num_features = model.classifier.in_features
                model.classifier = nn.Linear(num_features, 1)
                
            elif model_name == "MobileNet-V2":
                model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
                # Replace classifier for binary classification
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(nn.Linear(num_features, 1))
            
            gflops, params_m = get_flops_params(model, input_shape)
            
            results.append({
                'Model': model_name,
                'Params (M)': f"{params_m:.2f}",
                'GFLOPs': f"{gflops:.2f}" if gflops is not None else "N/A"
            })
            
            print(f"   ✅ Params: {params_m:.2f}M, GFLOPs: {gflops:.2f}" if gflops else f"   ✅ Params: {params_m:.2f}M")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'Model': model_name,
                'Params (M)': "Error",
                'GFLOPs': "Error"
            })
    
    return pd.DataFrame(results)


def analyze_segmentation_models():
    """Analyze all segmentation models."""
    print("\n" + "="*80)
    print("📊 SEGMENTATION MODELS")
    print("="*80)
    
    results = []
    
    # Input shape for segmentation (64x64 grayscale patches)
    input_shape = (1, 64, 64)
    
    models_config = [
        ("UNet", "unet", "resnet34"),
        ("UNet++", "unet++", "resnet34"),
        ("DeepLabV3+", "deeplabv3+", "resnet34"),
        ("FPN", "fpn", "resnet34"),
        ("MANet", "manet", "resnet34"),
        ("LinkNet", "linknet", "resnet34"),
        ("PSPNet", "pspnet", "resnet34"),
        ("UperNet", "upernet", "resnet34"),
        ("FCN", "fcn", "resnet50"),
        ("Segformer", "segformer", "resnet34"),
    ]
    
    for model_name, model_type, encoder in models_config:
        print(f"\n🔍 Analyzing {model_name}...")
        
        try:
            model = SegmentationModels2D.get_model(model_type, encoder, encoder_weights=None)
            gflops, params_m = get_flops_params(model, input_shape)
            
            results.append({
                'Model': model_name,
                'Encoder': encoder,
                'Params (M)': f"{params_m:.2f}",
                'GFLOPs': f"{gflops:.2f}" if gflops is not None else "N/A"
            })
            
            print(f"   ✅ Params: {params_m:.2f}M, GFLOPs: {gflops:.2f}" if gflops else f"   ✅ Params: {params_m:.2f}M")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'Model': model_name,
                'Encoder': encoder,
                'Params (M)': "Error",
                'GFLOPs': "Error"
            })
    
    # SwinUNETR (requires MONAI)
    print(f"\n🔍 Analyzing SwinUNETR...")
    try:
        model = SegmentationModels2D.get_model("swinunetr", "none", encoder_weights=None)
        gflops, params_m = get_flops_params(model, input_shape)
        
        results.append({
            'Model': 'SwinUNETR',
            'Encoder': 'Swin Transformer',
            'Params (M)': f"{params_m:.2f}",
            'GFLOPs': f"{gflops:.2f}" if gflops is not None else "N/A"
        })
        
        print(f"   ✅ Params: {params_m:.2f}M, GFLOPs: {gflops:.2f}" if gflops else f"   ✅ Params: {params_m:.2f}M")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results.append({
            'Model': 'SwinUNETR',
            'Encoder': 'Swin Transformer',
            'Params (M)': "Error",
            'GFLOPs': "Error"
        })
    
    return pd.DataFrame(results)


def analyze_foundation_models():
    """Analyze foundation models (MedSAM2, SAM2, SAM3)."""
    print("\n" + "="*80)
    print("📊 FOUNDATION MODELS")
    print("="*80)
    
    results = []
    
    # Foundation models use larger input sizes
    foundation_configs = [
        ("MedSAM2", 512),
        ("SAM2", 512),
        ("SAM3", 1008),
    ]
    
    for model_name, img_size in foundation_configs:
        print(f"\n🔍 Analyzing {model_name} ({img_size}x{img_size})...")
        
        try:
            # Create model wrapper
            if model_name == "MedSAM2":
                wrapper = MedSAM2Wrapper(
                    checkpoint_path=config.MEDSAM2_CHECKPOINT,
                    config_path=config.MEDSAM2_CONFIG,
                    device='cpu',
                    image_size=img_size
                )
                model = wrapper.image_predictor.model
                input_shape = (3, img_size, img_size)
                
            elif model_name == "SAM2":
                wrapper = SAM2Wrapper(
                    checkpoint_path=config.SAM2_CHECKPOINT,
                    config_path=config.SAM2_CONFIG,
                    device='cpu',
                    image_size=img_size
                )
                model = wrapper.image_predictor.model
                input_shape = (3, img_size, img_size)
                
            elif model_name == "SAM3":
                wrapper = SAM3Wrapper(
                    checkpoint_path=config.SAM3_CHECKPOINT,
                    config_path=config.SAM3_CONFIG,
                    device='cpu',
                    image_size=img_size
                )
                model = wrapper.model
                input_shape = (3, img_size, img_size)
            
            # Calculate params (FLOPs might be too complex for these models)
            total_params, _ = count_parameters(model)
            params_m = total_params / 1e6
            
            # Try to get FLOPs (might fail for complex models)
            try:
                gflops, params_m_calc = get_flops_params(model, input_shape, device='cpu')
            except:
                gflops = None
                print(f"   ⚠️  Could not calculate GFLOPs (model too complex)")
            
            results.append({
                'Model': model_name,
                'Image Size': f"{img_size}x{img_size}",
                'Params (M)': f"{params_m:.2f}",
                'GFLOPs': f"{gflops:.2f}" if gflops is not None else "N/A (Complex)"
            })
            
            print(f"   ✅ Params: {params_m:.2f}M" + (f", GFLOPs: {gflops:.2f}" if gflops else ""))
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'Model': model_name,
                'Image Size': f"{img_size}x{img_size}",
                'Params (M)': "Error",
                'GFLOPs': "Error"
            })
    
    return pd.DataFrame(results)


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("🔬 MODEL COMPLEXITY ANALYSIS")
    print("="*80)
    print("\nCalculating parameters and GFLOPs for all models...")
    print("Note: This may take several minutes.\n")
    
    # Analyze each category
    df_classification = analyze_classification_models()
    df_segmentation = analyze_segmentation_models()
    df_foundation = analyze_foundation_models()
    
    # Save results
    output_dir = Path(config.WORKDIR) / "results" / "model_complexity"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    
    # Save individual CSVs
    df_classification.to_csv(output_dir / "classification_models_complexity.csv", index=False)
    print(f"✅ Classification: {output_dir / 'classification_models_complexity.csv'}")
    
    df_segmentation.to_csv(output_dir / "segmentation_models_complexity.csv", index=False)
    print(f"✅ Segmentation: {output_dir / 'segmentation_models_complexity.csv'}")
    
    df_foundation.to_csv(output_dir / "foundation_models_complexity.csv", index=False)
    print(f"✅ Foundation: {output_dir / 'foundation_models_complexity.csv'}")
    
    # Print summary tables
    print("\n" + "="*80)
    print("📋 CLASSIFICATION MODELS SUMMARY")
    print("="*80)
    print(df_classification.to_string(index=False))
    
    print("\n" + "="*80)
    print("📋 SEGMENTATION MODELS SUMMARY")
    print("="*80)
    print(df_segmentation.to_string(index=False))
    
    print("\n" + "="*80)
    print("📋 FOUNDATION MODELS SUMMARY")
    print("="*80)
    print(df_foundation.to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📁 Results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
