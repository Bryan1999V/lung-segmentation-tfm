"""
Predicted Masks Export Script: Foundation Models on LUNA16/LUNA25

This script generates predicted masks from different foundation models
(SAM2, SAM3, MedSAM2) for all nodules in selected scans and saves them individually
with the structure: {output_dir}/{scan}/{nodule_id}/{foundation_model}.png

For LUNA16: Also saves ground truth masks as 'ground_truth.png'
For LUNA25: Only saves foundation model predictions (no ground truth available)

Usage:
    python qualitative_prediction.py --dataset LUNA16  # Default
    python qualitative_prediction.py --dataset LUNA25
"""

import os
from os.path import join
import ast
import logging
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import SimpleITK as sitk
import torch
import pandas as pd
import cv2

# Setup paths
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.common import config
from nodule_segmentation.foundation_models import utils
from nodule_segmentation.foundation_models.models.medsam2_wrapper import MedSAM2Wrapper
from nodule_segmentation.foundation_models.models.sam2_wrapper import SAM2Wrapper
from nodule_segmentation.foundation_models.models.sam3_wrapper import SAM3Wrapper
from nodule_segmentation.foundation_models.models.linknet_wrapper import LinkNetWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Set random seeds
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)

torch.set_float32_matmul_precision('high')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Foundation model paths (same for both datasets)
MEDSAM2_CHECKPOINT = config.MEDSAM2_CHECKPOINT
MEDSAM2_CONFIG = config.MEDSAM2_CONFIG
SAM2_CHECKPOINT = config.SAM2_CHECKPOINT
SAM2_CONFIG = config.SAM2_CONFIG
SAM2_IMSIZE = config.SAM2_IMG_SIZE
SAM3_CHECKPOINT = config.SAM3_CHECKPOINT
SAM3_CONFIG = config.SAM3_CONFIG
SAM3_IMSIZE = config.SAM3_IMG_SIZE

# LinkNet checkpoint (set to None to skip LinkNet)
LINKNET_CHECKPOINT = "/workspace/code/lung-segmentation-tfm/results/best_model/segmentation/best_model_ever.pth"  # Update with path to your trained LinkNet checkpoint

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
THRESHOLD = config.THRESHOLD
OUTPUT_BASE_DIR = Path(config.BEST_MODEL_PATH) / ".." / "foundation_models_masks"

# Dataset configurations
DATASET_CONFIG = {
    "LUNA16": {
        "imgs_path": config.LUNA16_CT_DIR,
        "masks_path": config.LUNA16_CT_MASKS_DIR,
        "annotations_csv": config.LUNA16_ANNOTATIONS_CSV_PATH,
        "imsize": config.LUNA16_IMG_SIZE,
        "has_gt": True,
    },
    "LUNA25": {
        "imgs_path": config.LUNA25_CT_DIR,
        "masks_path": None,
        "annotations_csv": config.LUNA25_ANNOTATIONS_CSV_PATH,
        "imsize": config.LUNA25_IMG_SIZE,
        "has_gt": False,
    },
}

# Model wrappers (global state)
medsam2_wrapper = None
sam2_wrapper = None
sam3_wrapper = None
linknet_wrapper = None

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def init_wrappers(imsize: int):
    """Initialize all model wrappers.
    
    Args:
        imsize: Image size for preprocessing
    """
    global medsam2_wrapper, sam2_wrapper, sam3_wrapper, linknet_wrapper

    logger.info("🤖 Initializing MedSAM2...")
    medsam2_wrapper = MedSAM2Wrapper(
        checkpoint_path=MEDSAM2_CHECKPOINT,
        config_path=MEDSAM2_CONFIG,
        device=DEVICE,
        image_size=imsize
    )

    logger.info("🤖 Initializing SAM2...")
    sam2_wrapper = SAM2Wrapper(
        checkpoint_path=SAM2_CHECKPOINT,
        config_path=SAM2_CONFIG,
        device=DEVICE,
        image_size=SAM2_IMSIZE
    )

    logger.info("🤖 Initializing SAM3...")
    sam3_wrapper = SAM3Wrapper(
        checkpoint_path=SAM3_CHECKPOINT,
        config_path=SAM3_CONFIG,
        device=DEVICE,
        image_size=SAM3_IMSIZE
    )
    
    # Initialize LinkNet if checkpoint is provided
    if LINKNET_CHECKPOINT and os.path.exists(LINKNET_CHECKPOINT):
        logger.info("🤖 Initializing LinkNet (64×64 patches)...")
        linknet_wrapper = LinkNetWrapper(
            checkpoint_path=LINKNET_CHECKPOINT,
            device=DEVICE,
            patch_size=64
        )
    else:
        if LINKNET_CHECKPOINT:
            logger.warning(f"⚠️  LinkNet checkpoint not found: {LINKNET_CHECKPOINT}")
        linknet_wrapper = None

    logger.info("✅ All models initialized\n")


def load_and_preprocess_nodule(scan_uid: str, centroid_list, bbox_coords, 
                               imgs_path: str, masks_path: Optional[str], has_gt: bool) -> Optional[Dict]:
    """Load CT image and GT mask (if available) for the specified dataset"""
    try:
        # Load CT scan (try .mhd first, then .mha for LUNA25)
        mhd_path = join(imgs_path, f"{scan_uid}.mhd")
        mha_path = join(imgs_path, f"{scan_uid}.mha")
        
        if os.path.exists(mhd_path):
            img_path = mhd_path
        elif os.path.exists(mha_path):
            img_path = mha_path
        else:
            return None

        sitk_img = sitk.ReadImage(img_path)
        img_3d = sitk.GetArrayFromImage(sitk_img)
        
        # Get image metadata for LinkNet and coordinate conversion
        voxel_origin = np.array(sitk_img.GetOrigin())
        voxel_spacing = np.array(sitk_img.GetSpacing())

        # Load GT mask only if available (LUNA16)
        mask_gt = None
        mask_format = None
        if has_gt and masks_path:
            mask_path = join(masks_path, f"{scan_uid}.npy")
            if not os.path.exists(mask_path):
                return None

            mask_gt_orig = np.load(mask_path)

            # Determine mask format
            if mask_gt_orig.shape == (img_3d.shape[1], img_3d.shape[2], img_3d.shape[0]):
                mask_gt, mask_format = mask_gt_orig, "YXZ"
            elif mask_gt_orig.shape == img_3d.shape:
                mask_gt, mask_format = mask_gt_orig, "ZYX"
            else:
                return None

        # Extract coordinates based on mask format (exactly as in evaluate_foundation_models.py)
        if mask_gt is not None and mask_format == "YXZ":
            centroid_z = centroid_list[2]
            centroid_y = centroid_list[0]
            centroid_x = centroid_list[1]
            bbox_2d = (bbox_coords[0], bbox_coords[2], bbox_coords[1], bbox_coords[3])
            hu_at_centroid = img_3d[centroid_z, centroid_y, centroid_x]
            mask_gt_slice = mask_gt[:, :, centroid_z]  # (Y, X)
        elif mask_gt is not None and mask_format == "ZYX":  # ZYX
            centroid_z = centroid_list[0]
            centroid_y = centroid_list[1]
            centroid_x = centroid_list[2]
            bbox_2d = (bbox_coords[2], bbox_coords[4], bbox_coords[3], bbox_coords[5])
            hu_at_centroid = img_3d[centroid_z, centroid_y, centroid_x]
            mask_gt_slice = mask_gt[centroid_z, :, :]  # (Y, X)
        else:  # No GT mask (LUNA25) - use centroid from annotations
            # LUNA25 centroids are in world coordinates (mm), need to convert to voxel coords
            # centroid_list is [z_world, y_world, x_world]
            # BUT voxel_origin and voxel_spacing from SimpleITK are [X, Y, Z]
            z_world, y_world, x_world = centroid_list
            
            # Correct coordinate conversion accounting for axis order
            centroid_x = (x_world - voxel_origin[0]) / voxel_spacing[0]
            centroid_y = (y_world - voxel_origin[1]) / voxel_spacing[1]
            centroid_z = (z_world - voxel_origin[2]) / voxel_spacing[2]
            
            # Clamp to valid range
            centroid_z = int(np.clip(centroid_z, 0, img_3d.shape[0]-1))
            centroid_y = int(np.clip(centroid_y, 0, img_3d.shape[1]-1))
            centroid_x = int(np.clip(centroid_x, 0, img_3d.shape[2]-1))
            
            # Create bbox around centroid for SAM3 (SAM3 requires bbox, cannot use just points)
            # Use a small margin to match typical nodule sizes in LUNA16 (8-20 pixels)
            # Margin of 8 pixels creates a 16×16 bbox, similar to real nodule bboxes
            bbox_margin = 8
            y_min = max(0, centroid_y - bbox_margin)
            y_max = min(img_3d.shape[1] - 1, centroid_y + bbox_margin)
            x_min = max(0, centroid_x - bbox_margin)
            x_max = min(img_3d.shape[2] - 1, centroid_x + bbox_margin)
            bbox_2d = (y_min, x_min, y_max, x_max)  # Format: (y_min, x_min, y_max, x_max)
            
            hu_at_centroid = img_3d[centroid_z, centroid_y, centroid_x]
            mask_gt_slice = None

        # Determine windowing
        nodule_type, window_level, window_width = utils.determine_nodule_type_and_windowing(hu_at_centroid)

        # Extract single slice
        img_single_slice = img_3d[centroid_z:centroid_z+1]  # (1, H, W)

        # World matrix for coordinate transformations (3x3, not 4x4)
        world_matrix = np.eye(3)

        return {
            'scan_uid': scan_uid,
            'img_single_slice': img_single_slice,
            'mask_gt_slice': mask_gt_slice,
            'centroid_z': centroid_z,
            'centroid_y': centroid_y,
            'centroid_x': centroid_x,
            'bbox_2d': bbox_2d,
            'window_level': window_level,
            'window_width': window_width,
            'nodule_type': nodule_type,
            # For LinkNet: 3D data and coordinate system info
            'ct_3d': img_3d,
            'voxel_origin': voxel_origin,
            'world_matrix': world_matrix,
            'voxel_spacing': voxel_spacing,
        }

    except Exception as e:
        logger.error(f"Error loading nodule data for {scan_uid}: {e}")
        return None


# ============================================================================
# INFERENCE (Same pattern as evaluate_foundation_models.py)
# ============================================================================

@torch.inference_mode()
def get_model_predictions(nodule_data: Dict) -> Dict:
    """Get predictions from all models for a nodule.
    
    Includes foundation models (MedSAM2, SAM2, SAM3) and custom LinkNet (if available).
    LinkNet uses 64×64 patches extracted from the center, while fundational models use 512×512.
    """
    global medsam2_wrapper, sam2_wrapper, sam3_wrapper, linknet_wrapper

    predictions = {}

    img_single_slice = nodule_data['img_single_slice']
    centroid_y = nodule_data['centroid_y']
    centroid_x = nodule_data['centroid_x']
    centroid_z = nodule_data['centroid_z']
    bbox_2d = nodule_data['bbox_2d']
    window_level = nodule_data['window_level']
    window_width = nodule_data['window_width']

    # MedSAM2 (point-based prompt - centroid only, no bbox)
    try:
        img_tensor, video_height, video_width = medsam2_wrapper.preprocess_image(
            img_single_slice, window_level=window_level, window_width=window_width
        )
        pred_mask_3d = medsam2_wrapper.predict(
            img_tensor=img_tensor,
            video_height=video_height,
            video_width=video_width,
            centroid_z=0,
            centroid_y=centroid_y,
            centroid_x=centroid_x,
            bbox=None,
            confidence_threshold=THRESHOLD,
            z_range=None
        )
        predictions['medsam2'] = pred_mask_3d[0]  # Extract slice
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"  ⚠️  MedSAM2 failed: {e}")
        predictions['medsam2'] = None

    # SAM2 (point-based prompt - centroid only, no bbox)
    try:
        img_tensor, video_height, video_width = sam2_wrapper.preprocess_image(
            img_single_slice, window_level=window_level, window_width=window_width
        )
        pred_mask_3d = sam2_wrapper.predict(
            img_tensor=img_tensor,
            video_height=video_height,
            video_width=video_width,
            centroid_z=0,
            centroid_y=centroid_y,
            centroid_x=centroid_x,
            bbox=None,
            confidence_threshold=THRESHOLD,
            z_range=(0, 0)
        )
        predictions['sam2'] = pred_mask_3d[0]  # Extract slice
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"  ⚠️  SAM2 failed: {e}")
        predictions['sam2'] = None

    # SAM3 (requires bbox - point prompts not supported in SAM3)
    try:
        img_array, video_height, video_width = sam3_wrapper.preprocess_image(
            img_single_slice, window_level=window_level, window_width=window_width
        )
        pred_mask_3d = sam3_wrapper.predict(
            img_array=img_array,
            video_height=video_height,
            video_width=video_width,
            centroid_z=0,
            centroid_y=centroid_y,
            centroid_x=centroid_x,
            bbox=bbox_2d,
            confidence_threshold=THRESHOLD,
            z_range=(0, 0)
        )
        predictions['sam3'] = pred_mask_3d[0]  # Extract slice
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"  ⚠️  SAM3 failed: {e}")
        predictions['sam3'] = None

    # LinkNet (64×64 patch-based model)
    if linknet_wrapper is not None:
        try:
            # LinkNet needs the full 3D CT volume for patch extraction
            # Get it from nodule_data if available
            if 'ct_3d' in nodule_data and 'voxel_origin' in nodule_data:
                ct_3d = nodule_data['ct_3d']
                voxel_origin = nodule_data['voxel_origin']
                world_matrix = nodule_data['world_matrix']
                voxel_spacing = nodule_data['voxel_spacing']
                
                pred_mask = linknet_wrapper.predict(
                    ct_3d=ct_3d,
                    centroid_y=centroid_y,
                    centroid_x=centroid_x,
                    centroid_z=centroid_z,
                    bbox=bbox_2d,
                    src_voxel_origin=voxel_origin,
                    src_world_matrix=world_matrix,
                    src_voxel_spacing=voxel_spacing,
                    confidence_threshold=THRESHOLD,
                )
                # LinkNet returns (1, 64, 64), need to resize to 512×512 for consistency
                linknet_mask = pred_mask[0]  # Extract from (1, H, W) -> (H, W)
                
                # Resize from 64×64 to 512×512 using bilinear interpolation
                if linknet_mask.shape != (512, 512):
                    linknet_mask = cv2.resize(linknet_mask.astype(np.float32), (512, 512), 
                                              interpolation=cv2.INTER_LINEAR)
                
                predictions['linknet'] = linknet_mask
            else:
                logger.warning(f"  ⚠️  LinkNet: Missing CT volume data, skipping")
                predictions['linknet'] = None
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"  ⚠️  LinkNet failed: {e}")
            predictions['linknet'] = None
    else:
        predictions['linknet'] = None

    return predictions


# ============================================================================
# MASK SAVING
# ============================================================================

def save_mask_to_file(mask: Optional[np.ndarray], output_path: Path, 
                      color_rgb: tuple = (1.0, 1.0, 1.0)) -> bool:
    """Save a single mask as a colored image file.
    
    Args:
        mask: Segmentation mask (2D binary array or None)
        output_path: Path where to save the mask image
        color_rgb: RGB color tuple (r, g, b) in [0, 1] range
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        if mask is None:
            # Save empty mask (black background)
            empty_mask = np.zeros((512, 512, 3), dtype=np.uint8)
            plt.imsave(output_path, empty_mask)
        else:
            # Create colored mask: white areas for mask > 0, black elsewhere
            h, w = mask.shape
            colored_mask = np.zeros((h, w, 3), dtype=np.float32)
            
            # Normalize mask to [0, 1]
            mask_norm = mask.astype(np.float32)
            if mask_norm.max() > 0:
                mask_norm = mask_norm / mask_norm.max()
            
            # Apply color: where mask > 0, set the color
            mask_bool = mask_norm > 0.5
            for c in range(3):
                colored_mask[mask_bool, c] = color_rgb[c]
            
            # Save as uint8
            colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)
            plt.imsave(output_path, colored_mask_uint8)
        return True
    except Exception as e:
        logger.error(f"Error saving mask to {output_path}: {e}")
        return False


def save_nodule_predictions(dataset_name: str, scan_uid: str, nodule_idx: int, 
                           predictions: Dict, mask_gt: Optional[np.ndarray] = None) -> bool:
    """Save predictions for a single nodule in structure: {dataset}/{scan}/{nodule_id}/{model}.png
    
    Args:
        dataset_name: Dataset name ('LUNA16' or 'LUNA25')
        scan_uid: Scan UID
        nodule_idx: Nodule index within the scan
        predictions: Dictionary with model names as keys and masks as values
        mask_gt: Ground truth mask (optional, for LUNA16)
        
    Returns:
        True if all models saved successfully
    """
    # Create directory structure: {dataset}/{scan}/{nodule_id}/
    scan_dir = OUTPUT_BASE_DIR / dataset_name / scan_uid / f"nodule_{nodule_idx:03d}"
    scan_dir.mkdir(parents=True, exist_ok=True)
    
    # Define colors for each model
    model_colors = {
        'medsam2': (1.0, 0.0, 0.0),  # Red
        'sam2': (0.0, 1.0, 0.0),     # Green
        'sam3': (0.0, 0.0, 1.0),     # Blue
        'linknet': (0.0, 1.0, 1.0),  # Cyan
    }
    
    success = True
    
    # Save ground truth if available (LUNA16)
    if mask_gt is not None:
        output_path = scan_dir / "ground_truth.png"
        if save_mask_to_file(mask_gt, output_path, color_rgb=(1.0, 1.0, 0.0)):  # Yellow
            logger.info(f"    ✅ Saved ground_truth: {output_path}")
        else:
            success = False
    
    # Save model predictions
    for model_name, color_rgb in model_colors.items():
        mask = predictions.get(model_name)
        output_path = scan_dir / f"{model_name}.png"
        
        if not save_mask_to_file(mask, output_path, color_rgb=color_rgb):
            success = False
        else:
            logger.info(f"    ✅ Saved {model_name}: {output_path}")
    
    return success


def save_original_ct_image(dataset_name: str, scan_uid: str, nodule_idx: int, 
                          img_single_slice: np.ndarray) -> bool:
    """Save the original CT image for a specific nodule slice.
    
    Args:
        dataset_name: Dataset name ('LUNA16' or 'LUNA25')
        scan_uid: Scan UID
        nodule_idx: Nodule index within the scan
        img_single_slice: Single CT slice (1, H, W)
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Create nodule directory: {dataset}/{scan}/{nodule_id}/
        nodule_dir = OUTPUT_BASE_DIR / dataset_name / scan_uid / f"nodule_{nodule_idx:03d}"
        nodule_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original image as grayscale
        output_path = nodule_dir / "original.png"
        
        # Normalize to [0, 1] and convert to uint8
        img_slice = img_single_slice[0]  # Extract from (1, H, W) -> (H, W)
        img_norm = img_slice.astype(np.float32)
        img_min = img_norm.min()
        img_max = img_norm.max()
        
        if img_max > img_min:
            img_norm = (img_norm - img_min) / (img_max - img_min)
        
        img_uint8 = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)
        
        # Save using matplotlib (converts to PNG grayscale)
        plt.imsave(output_path, img_uint8, cmap='gray')
        
        logger.info(f"    ✅ Saved original: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving original image for {scan_uid}, nodule {nodule_idx}: {e}")
        return False


# ============================================================================
# VISUALIZATION (Deprecated - kept for reference)
# ============================================================================

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    img_min = img.min()
    img_max = img.max()
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - img_min) / (img_max - img_min)).astype(np.float32)


def apply_mask_overlay(image: np.ndarray, mask: np.ndarray, 
                       cmap_name: str = 'Reds', alpha: float = 0.6) -> np.ndarray:
    """Apply colored mask overlay on image."""
    if mask is None:
        return np.stack([image, image, image], axis=-1)

    image = normalize_image(image)
    image = np.clip(image, 0, 1)

    cmap = plt.get_cmap(cmap_name)
    rgb_image = np.stack([image, image, image], axis=-1)
    
    mask_bool = mask > 0.5
    mask_colored = cmap(mask.astype(float))[:, :, :3]
    
    overlay = rgb_image.copy()
    overlay[mask_bool] = (1 - alpha) * rgb_image[mask_bool] + alpha * mask_colored[mask_bool]

    return overlay


def create_mask_overlay_red(image: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Create colored mask visualization without background.
    
    Args:
        mask: Segmentation mask (2D binary array)
        color_rgb: RGB color tuple (r, g, b) in [0, 1]
    
    Returns:
        RGB image where:
        - Mask values > 0: specified color
        - Mask values <= 0: black [0, 0, 0]
    """
    if mask is None:
        h, w = image.shape
        return np.zeros((h, w, 3), dtype=np.float32)
    
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.float32)
    
    # Set mask to specified color
    mask_bool = mask > 0
    rgb_mask[mask_bool, 0] = 1.0  # Red channel
    
    return rgb_mask


def create_colored_mask_only(mask: np.ndarray, color_rgb: tuple = (1.0, 0.0, 0.0)) -> np.ndarray:
    """Create colored mask visualization without background.
    
    Args:
        mask: Segmentation mask (2D binary array)
        color_rgb: RGB color tuple (r, g, b) in [0, 1]
    
    Returns:
        RGB image where:
        - Mask values > 0: specified color
        - Mask values <= 0: black [0, 0, 0]
    """
    if mask is None:
        return np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.float32)
    
    # Set mask to specified color where mask > 0
    mask_bool = mask > 0
    for c in range(3):
        rgb_mask[mask_bool, c] = color_rgb[c]
    
    return rgb_mask


def create_comparison_figure(samples_list: list) -> None:
    """Create comparison visualization."""
    
    num_samples = len(samples_list)
    num_columns = 5
    
    fig = plt.figure(figsize=(40, 24))
    gs = GridSpec(num_samples + 1, num_columns, figure=fig, hspace=0.1, wspace=0.1)
    
    column_titles = ['Original CT', 'Ground Truth', 'SAM2', 'MedSAM2', 'SAM3']
    colors_dict = {
        'gt': 'Greens',
        'sam2': 'Blues',
        'medsam2': 'Reds',
        'sam3': 'Oranges'
    }
    
    # Add column titles
    for col_idx, title in enumerate(column_titles):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.text(0.5, 0.5, title, transform=ax.transAxes,
               ha='center', va='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax.axis('off')
    
    # Add data rows
    for row_idx, sample_data in enumerate(samples_list, 1):
        logger.info(f"Visualizing sample {row_idx}/{num_samples}...")
        
        img_slice = sample_data['img_slice']
        mask_gt = sample_data['mask_gt']
        predictions = sample_data['predictions']
        scan_uid = sample_data['scan_uid']
        
        img_norm = normalize_image(img_slice)
        
        # Column 0: Original CT
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(img_norm, cmap='gray')
        ax.set_ylabel(f'{scan_uid}', fontsize=10)
        ax.axis('off')
        
        # Column 1: Ground Truth
        ax = fig.add_subplot(gs[row_idx, 1])
        overlay_gt = create_colored_mask_only(mask_gt, color_rgb=(0.0, 1.0, 0.0))  # Green
        ax.imshow(overlay_gt)
        ax.axis('off')
        
        # Column 2: SAM2
        ax = fig.add_subplot(gs[row_idx, 2])
        pred_sam2 = predictions.get('sam2')
        if pred_sam2 is not None:
            overlay = create_colored_mask_only(pred_sam2, color_rgb=(0.0, 0.0, 1.0))  # Blue
            ax.imshow(overlay)
        else:
            ax.imshow(np.zeros_like(img_norm), cmap='gray')
            ax.text(0.5, 0.5, 'FAILED', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='red', fontweight='bold')
        ax.axis('off')
        
        # Column 3: MedSAM2
        ax = fig.add_subplot(gs[row_idx, 3])
        pred_medsam2 = predictions.get('medsam2')
        if pred_medsam2 is not None:
            overlay = create_colored_mask_only(pred_medsam2, color_rgb=(1.0, 0.0, 0.0))  # Red
            ax.imshow(overlay)
        else:
            ax.imshow(np.zeros_like(img_norm), cmap='gray')
            ax.text(0.5, 0.5, 'FAILED', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='red', fontweight='bold')
        ax.axis('off')
        
        # Column 4: SAM3
        ax = fig.add_subplot(gs[row_idx, 4])
        pred_sam3 = predictions.get('sam3')
        if pred_sam3 is not None:
            overlay = create_colored_mask_only(pred_sam3, color_rgb=(1.0, 0.5, 0.0))  # Orange
            ax.imshow(overlay)
        else:
            ax.imshow(np.zeros_like(img_norm), cmap='gray')
            ax.text(0.5, 0.5, 'FAILED', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='red', fontweight='bold')
        ax.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.6, label='Ground Truth'),
        mpatches.Patch(facecolor='blue', alpha=0.6, label='SAM2'),
        mpatches.Patch(facecolor='red', alpha=0.6, label='MedSAM2'),
        mpatches.Patch(facecolor='orange', alpha=0.6, label='SAM3'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              bbox_to_anchor=(0.5, -0.02), fontsize=20, frameon=True)
    
    # Save
    output_path = OUTPUT_BASE_DIR / 'foundation_models_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n📊 Saving visualization to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved!\n")
    
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution: Process LUNA16 and LUNA25 datasets sequentially with random scans."""
    
    logger.info("\n" + "="*80)
    logger.info("🔍 Predicted Masks Export - Foundation Models on LUNA16/LUNA25")
    logger.info("="*80 + "\n")

    # Process each dataset
    for dataset_name in ["LUNA16", "LUNA25"]:
        logger.info("\n" + "="*80)
        logger.info(f"📊 Processing {dataset_name}")
        logger.info("="*80 + "\n")
        
        # Get dataset configuration
        config_dict = DATASET_CONFIG[dataset_name]
        imgs_path = config_dict["imgs_path"]
        masks_path = config_dict["masks_path"]
        annotations_csv = config_dict["annotations_csv"]
        imsize = config_dict["imsize"]
        has_gt = config_dict["has_gt"]
        
        logger.info(f"✅ Configured for dataset: {dataset_name}")
        logger.info(f"   Images: {imgs_path}")
        if has_gt:
            logger.info(f"   GT Masks: {masks_path}")
        logger.info(f"   Annotations: {annotations_csv}\n")
        
        # Initialize models for this dataset
        init_wrappers(imsize=imsize)

        # Load annotations
        logger.info(f"📋 Loading annotations...")
        annotations_df = pd.read_csv(annotations_csv)
        
        # Handle different CSV formats
        if dataset_name == "LUNA16":
            # LUNA16 format: centroid as string, bbox as string, scan column
            annotations_df['centroid'] = annotations_df['centroid'].apply(ast.literal_eval)
            annotations_df['bbox'] = annotations_df['bbox'].apply(utils.parse_bbox_from_string)
            annotations_df['scan'] = annotations_df['scan']  # Already correct column name
        else:  # LUNA25
            # LUNA25 format: CoordX, CoordY, CoordZ as separate columns, SeriesInstanceUID as scan UID
            # Coords are in world space (mm), convert to voxel coordinates later
            annotations_df['centroid'] = annotations_df.apply(
                lambda row: [float(row['CoordZ']), float(row['CoordY']), float(row['CoordX'])],
                axis=1
            )
            # LUNA25 doesn't have bbox, use dummy values
            annotations_df['bbox'] = annotations_df.apply(
                lambda row: (0, 0, 512, 512),
                axis=1
            )
            # Use SeriesInstanceUID as scan UID (rename for consistency)
            annotations_df['scan'] = annotations_df['SeriesInstanceUID']

        # Filter valid scans (check for .mhd or .mha files)
        valid_scans = []
        for s in annotations_df['scan'].unique():
            mhd_exists = os.path.exists(join(imgs_path, f"{s}.mhd"))
            mha_exists = os.path.exists(join(imgs_path, f"{s}.mha"))
            if mhd_exists or mha_exists:
                valid_scans.append(s)
        
        # For LUNA16, also check for GT masks
        if has_gt:
            valid_scans = [
                s for s in valid_scans
                if os.path.exists(join(masks_path, f"{s}.npy"))
            ]
        
        annotations_df = annotations_df[annotations_df['scan'].isin(valid_scans)]

        # Select 3 random scans
        all_scans = list(annotations_df['scan'].unique())
        if len(all_scans) < 3:
            logger.warning(f"⚠️  Only {len(all_scans)} scans available, using all")
            selected_scans = all_scans
        else:
            selected_scans = random.sample(all_scans, 3)
        
        logger.info(f"📌 Randomly selected {len(selected_scans)} scans:\n")
        for i, scan in enumerate(selected_scans, 1):
            logger.info(f"  {i}. {scan}")
        
        scans_annotations = annotations_df[annotations_df['scan'].isin(selected_scans)]
        
        if len(scans_annotations) == 0:
            logger.error(f"❌ No nodules found in selected scans")
            continue
        
        logger.info(f"📊 Found {len(scans_annotations)} nodules across {len(selected_scans)} scans\n")

        # Track statistics
        total_nodules = 0
        processed_nodules = 0
        failed_nodules = 0
        
        # Group by scan
        scans_grouped = scans_annotations.groupby('scan')
        
        for scan_idx, (scan_uid, scan_group) in enumerate(scans_grouped, 1):
            logger.info(f"\n[Scan {scan_idx}/{len(scans_grouped)}] Processing scan: {scan_uid}")
            logger.info(f"  Found {len(scan_group)} nodules")
            
            # Process each nodule in the scan
            for nodule_idx, (_, row) in enumerate(scan_group.iterrows(), 1):
                total_nodules += 1
                
                logger.info(f"  [{nodule_idx}/{len(scan_group)}] Nodule {nodule_idx:03d}...")

                # Load data
                nodule_data = load_and_preprocess_nodule(
                    row['scan'], row['centroid'], row['bbox'],
                    imgs_path, masks_path, has_gt
                )
                if nodule_data is None:
                    logger.warning(f"    ⚠️  Failed to load, skipping")
                    failed_nodules += 1
                    continue

                # Get predictions
                predictions = get_model_predictions(nodule_data)

                # Save original CT image for this nodule's slice
                save_original_ct_image(dataset_name, scan_uid, nodule_idx, nodule_data['img_single_slice'])

                # Save predictions
                mask_gt = nodule_data.get('mask_gt_slice') if has_gt else None
                if save_nodule_predictions(dataset_name, scan_uid, nodule_idx, predictions, mask_gt=mask_gt):
                    processed_nodules += 1
                else:
                    failed_nodules += 1

        logger.info("\n" + "="*80)
        logger.info(f"📈 {dataset_name} Summary")
        logger.info("="*80)
        logger.info(f"Total nodules: {total_nodules}")
        logger.info(f"Successfully processed: {processed_nodules}")
        logger.info(f"Failed: {failed_nodules}")
        logger.info(f"Ground Truth masks: {'✅ Yes' if has_gt else '❌ No'}")
        logger.info(f"Output directory: {OUTPUT_BASE_DIR / dataset_name}")
        logger.info("="*80 + "\n")

    logger.info("="*80)
    logger.info("✅ All datasets processed!")
    logger.info(f"📁 Output base directory: {OUTPUT_BASE_DIR}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
