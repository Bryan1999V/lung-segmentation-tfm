"""
Foundation Models Evaluation on LUNA16
========================================

This script evaluates foundation models on LUNA16 dataset calculating DICE and IoU metrics.

Adapted from: https://github.com/bowang-lab/MedSAM2/blob/main/examples/infer_CT_LUNA25.py

Key features:
- Evaluates each nodule independently using only the central slice
- Uses bbox from CSV annotations
- Calculates quantitative metrics (DICE, IoU) per nodule
- Processes entire dataset or random subset
"""

import os
from os.path import join
import ast
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
import pandas as pd
import random

from config.common import config
from metrics.segmentation import calculate_dice, calculate_iou, calculate_precision, calculate_sensitivity
from nodule_segmentation.foundation_models import utils
from nodule_segmentation.foundation_models.models.medsam2_wrapper import MedSAM2Wrapper
from nodule_segmentation.foundation_models.models.sam2_wrapper import SAM2Wrapper
from nodule_segmentation.foundation_models.models.sam3_wrapper import SAM3Wrapper
from nodule_segmentation.foundation_models.models.medsegx_wrapper import MedSegXWrapper

torch.set_float32_matmul_precision('high')
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
np.random.seed(config.SEED)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths from config
IMGS_PATH = config.LUNA16_CT_DIR
MASKS_PATH = config.LUNA16_CT_MASKS_DIR
ANNOTATIONS_CSV = config.LUNA16_ANNOTATIONS_CSV_PATH

# Configuration
MEDSAM2_CHECKPOINT = config.MEDSAM2_CHECKPOINT
MEDSAM2_CONFIG = config.MEDSAM2_CONFIG
SAM2_CHECKPOINT = config.SAM2_CHECKPOINT
SAM2_CONFIG = config.SAM2_CONFIG
SAM2_IMSIZE = config.SAM2_IMG_SIZE
SAM3_CHECKPOINT = config.SAM3_CHECKPOINT
SAM3_CONFIG = config.SAM3_CONFIG
SAM3_IMSIZE = config.SAM3_IMG_SIZE
MEDSEGX_CHECKPOINT = config.MEDSEGX_CHECKPOINT
MEDSEGX_SAM_CHECKPOINT = config.MEDSEGX_SAM_CHECKPOINT
MEDSEGX_MODEL_TYPE = config.MEDSEGX_MODEL_TYPE
MEDSEGX_IMSIZE = config.MEDSEGX_IMG_SIZE
OUTPUT_CSV = config.FOUNDATION_SEGMENTATION_CSV_RESULTS_PATH
NUM_WORKERS = config.LUNA16_NUM_WORKERS
IMSIZE = config.LUNA16_IMG_SIZE
TEST_MODE = False  # Enable test mode

# Model wrappers (initialized in each worker process)
medsam2_wrapper = None
sam2_wrapper = None
sam3_wrapper = None
medsegx_wrapper = None


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

@torch.inference_mode()
def evaluate_single_nodule(nodule_data):
    """Evaluates all models on a single nodule (single slice)."""
    global medsam2_wrapper, sam2_wrapper, sam3_wrapper, medsegx_wrapper

    # Initialize MedSAM2 wrapper if not exist (in each worker process)
    if medsam2_wrapper is None:
        medsam2_wrapper = MedSAM2Wrapper(
            checkpoint_path=MEDSAM2_CHECKPOINT,
            config_path=MEDSAM2_CONFIG,
            device='cuda',
            image_size=IMSIZE
        )

    # Initialize SAM2 wrapper if not exist
    if sam2_wrapper is None:
        sam2_wrapper = SAM2Wrapper(
            checkpoint_path=SAM2_CHECKPOINT,
            config_path=SAM2_CONFIG,
            device='cuda',
            image_size=SAM2_IMSIZE
        )

    # Initialize SAM3 wrapper if not exist
    if sam3_wrapper is None:
        sam3_wrapper = SAM3Wrapper(
            checkpoint_path=SAM3_CHECKPOINT,
            config_path=SAM3_CONFIG,
            device='cuda',
            image_size=SAM3_IMSIZE
        )

    # Initialize MedSegX wrapper if not exist
    if medsegx_wrapper is None:
        medsegx_wrapper = MedSegXWrapper(
            checkpoint_path=MEDSEGX_CHECKPOINT,
            sam_checkpoint_path=MEDSEGX_SAM_CHECKPOINT,
            device='cuda',
            model_type=MEDSEGX_MODEL_TYPE,
            image_size=MEDSEGX_IMSIZE
        )

    try:
        scan_uid = nodule_data['scan']
        centroid_list = nodule_data['centroid']
        bbox_coords = nodule_data['bbox']

        # Load CT scan
        mhd_path = join(IMGS_PATH, f"{scan_uid}.mhd")
        if not os.path.exists(mhd_path):
            return None

        sitk_img = sitk.ReadImage(mhd_path)
        img_3d = sitk.GetArrayFromImage(sitk_img)

        # Load GT mask
        mask_path = join(MASKS_PATH, f"{scan_uid}.npy")
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

        # Extract coordinates based on mask format
        if mask_format == "YXZ":
            centroid_z = centroid_list[2]
            centroid_y = centroid_list[0]
            centroid_x = centroid_list[1]
            # bbox: [dim1_min, dim1_max, dim2_min, dim2_max, dim3_min, dim3_max]
            # For YXZ: dim1=Y, dim2=X, dim3=Z
            bbox_2d = (bbox_coords[0], bbox_coords[2], bbox_coords[1], bbox_coords[3])  # (y_min, x_min, y_max, x_max)
            hu_at_centroid = img_3d[centroid_z, centroid_y, centroid_x]
            mask_gt_slice = mask_gt[:, :, centroid_z]  # (Y, X)
        else:  # ZYX
            centroid_z = centroid_list[0]
            centroid_y = centroid_list[1]
            centroid_x = centroid_list[2]
            # For ZYX: dim1=Z, dim2=Y, dim3=X
            bbox_2d = (bbox_coords[2], bbox_coords[4], bbox_coords[3], bbox_coords[5])  # (y_min, x_min, y_max, x_max)
            hu_at_centroid = img_3d[centroid_z, centroid_y, centroid_x]
            mask_gt_slice = mask_gt[centroid_z, :, :]  # (Y, X)

        # Determine windowing
        nodule_type, window_level, window_width = utils.determine_nodule_type_and_windowing(hu_at_centroid)

        results = []

        # Evaluate with all four models on this single nodule
        for model_name, wrapper in [
            ('medsam2', medsam2_wrapper),
            ('sam2', sam2_wrapper),
            ('sam3', sam3_wrapper),
            ('medsegx', medsegx_wrapper)
        ]:
            try:
                # Create single-slice "volume" for preprocessing
                img_single_slice = img_3d[centroid_z:centroid_z+1]  # (1, H, W)

                # Preprocess (can fail with CUDA OOM)
                try:
                    img_tensor, video_height, video_width = wrapper.preprocess_image(
                        img_single_slice, window_level=window_level, window_width=window_width
                    )
                except (RuntimeError, MemoryError) as preprocess_error:
                    if 'out of memory' in str(preprocess_error).lower() or 'cuda' in str(preprocess_error).lower():
                        print(f"⚠️  {model_name} CUDA OOM during preprocessing, skipping...")
                        torch.cuda.empty_cache()
                        continue
                    raise  # Re-raise if not a memory error

                # Predict on single slice
                if model_name == 'sam3':
                    # SAM3 expects full volume but we only have one slice
                    img_array = img_tensor  # For SAM3, preprocess returns array not tensor
                    pred_mask_3d = wrapper.predict(
                        img_array=img_array,
                        video_height=video_height,
                        video_width=video_width,
                        centroid_z=0,  # Only one slice, so index is 0
                        centroid_y=centroid_y,
                        centroid_x=centroid_x,
                        bbox=bbox_2d,
                        confidence_threshold=config.THRESHOLD,
                        z_range=(0, 0)  # Process only the single slice
                    )
                elif model_name == 'sam2':
                    # SAM2 with MedSAM2 checkpoint
                    pred_mask_3d = wrapper.predict(
                        img_tensor=img_tensor,
                        video_height=video_height,
                        video_width=video_width,
                        centroid_z=0,  # Only one slice
                        centroid_y=centroid_y,
                        centroid_x=centroid_x,
                        bbox=bbox_2d,
                        confidence_threshold=config.THRESHOLD,
                        z_range=(0, 0)  # Single-slice mode
                    )
                else:
                    # MedSAM2 and MedSegX
                    pred_mask_3d = wrapper.predict(
                        img_tensor=img_tensor,
                        video_height=video_height,
                        video_width=video_width,
                        centroid_z=0,  # Only one slice
                        centroid_y=centroid_y,
                        centroid_x=centroid_x,
                        bbox=bbox_2d,
                        confidence_threshold=config.THRESHOLD,
                        z_range=(0, 0) if model_name == 'medsegx' else None  # MedSegX needs z_range
                    )

                # Extract the predicted slice (index 0 since we only have one slice)
                pred_mask_slice = pred_mask_3d[0]  # (H, W)

                torch.cuda.empty_cache()

                # Verify shapes match
                if pred_mask_slice.shape != mask_gt_slice.shape:
                    print(f"⚠️  {model_name} shape mismatch: pred {pred_mask_slice.shape} vs GT {mask_gt_slice.shape}")
                    continue

                # Calculate metrics on single slice
                dice = calculate_dice(pred_mask_slice, mask_gt_slice)
                iou = calculate_iou(pred_mask_slice, mask_gt_slice)
                precision = calculate_precision(pred_mask_slice, mask_gt_slice)
                sensitivity = calculate_sensitivity(pred_mask_slice, mask_gt_slice)

                results.append({
                    'model': model_name,
                    'dice': dice,
                    'iou': iou,
                    'precision': precision,
                    'sensitivity': sensitivity
                })

            except (RuntimeError, MemoryError) as e:
                if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                    print(f"⚠️  {model_name} CUDA OOM: {str(e)[:100]}")
                else:
                    print(f"⚠️  {model_name} failed: {type(e).__name__}: {str(e)[:100]}")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"⚠️  {model_name} unexpected error: {type(e).__name__}: {str(e)[:100]}")
                torch.cuda.empty_cache()
                continue

        return results

    except Exception as e:
        print(f"⚠️  Failed to process nodule: {type(e).__name__}: {str(e)}")
        torch.cuda.empty_cache()
        return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print(f"\n🚀 Foundation Models - LUNA16 Evaluation (Per-Nodule)")
    print(f"🤖 Models: MedSAM2({IMSIZE}px), SAM2({SAM2_IMSIZE}px), SAM3({SAM3_IMSIZE}px), MedSegX({MEDSEGX_IMSIZE}px)")
    print(f"📊 Output: {OUTPUT_CSV.name}\n")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Load annotations
    annotations_df = pd.read_csv(ANNOTATIONS_CSV)

    # Parse centroid and bbox columns
    annotations_df['centroid'] = annotations_df['centroid'].apply(ast.literal_eval)
    annotations_df['bbox'] = annotations_df['bbox'].apply(utils.parse_bbox_from_string)

    # Filter for scans that have both image and mask
    valid_scans = [
        s for s in annotations_df['scan'].unique()
        if os.path.exists(join(IMGS_PATH, f"{s}.mhd")) and 
           os.path.exists(join(MASKS_PATH, f"{s}.npy"))
    ]
    annotations_df = annotations_df[annotations_df['scan'].isin(valid_scans)]

    if len(annotations_df) == 0:
        print("❌ No valid nodules found")
        exit(1)

    if TEST_MODE:
        random.seed(config.SEED)
        # Sample 20 random nodules for testing
        sample_size = min(20, len(annotations_df))
        annotations_df = annotations_df.sample(n=sample_size, random_state=config.SEED)

    print(f"Processing {len(annotations_df)} nodules...")

    # Prepare nodule data for evaluation
    nodules_to_process = []
    for idx, row in annotations_df.iterrows():
        nodules_to_process.append({
            'scan': row['scan'],
            'centroid': row['centroid'],
            'bbox': row['bbox']
        })

    all_results = []

    if NUM_WORKERS == 1:
        for nodule_data in tqdm(nodules_to_process):
            result = evaluate_single_nodule(nodule_data)
            if result:
                all_results.extend(result)
    else:
        mp.set_start_method('spawn', force=True)
        with mp.Pool(processes=NUM_WORKERS) as pool, tqdm(total=len(nodules_to_process)) as pbar:
            for result in pool.imap_unordered(evaluate_single_nodule, nodules_to_process):
                if result:
                    all_results.extend(result)
                pbar.update()

    if not all_results:
        print("\n⚠️ No results generated")
        raise SystemExit(0)

    df_results = pd.DataFrame(all_results)
    print(f"\n✅ Evaluation completed\n")

    # Create summary with simplified format
    summary_rows = []
    for model_name in df_results['model'].unique():
        mr = df_results[df_results['model'] == model_name]

        dice_mean, dice_std = mr['dice'].mean(), mr['dice'].std()
        iou_mean, iou_std = mr['iou'].mean(), mr['iou'].std()
        precision_mean, precision_std = mr['precision'].mean(), mr['precision'].std()
        sensitivity_mean, sensitivity_std = mr['sensitivity'].mean(), mr['sensitivity'].std()

        print(f"🤖 {model_name.upper()}:")
        print(f"   DICE={dice_mean:.4f}±{dice_std:.4f}, IoU={iou_mean:.4f}±{iou_std:.4f}")
        print(f"   Precision={precision_mean:.4f}±{precision_std:.4f}, Sensitivity={sensitivity_mean:.4f}±{sensitivity_std:.4f}")
        print(f"   Total nodules: {len(mr)}\n")

        # Determine config, image size, and checkpoint
        if model_name == 'medsam2':
            cfg = str(MEDSAM2_CONFIG.name) if MEDSAM2_CONFIG else "MedSAM2-CTLesion"
            size = f"{IMSIZE}x{IMSIZE}"
            ckpt = str(MEDSAM2_CHECKPOINT) if MEDSAM2_CHECKPOINT else "N/A"
        elif model_name == 'sam2':
            cfg = "SAM2-Original-MedSAM2-Checkpoint"
            size = f"{SAM2_IMSIZE}x{SAM2_IMSIZE}"
            ckpt = str(SAM2_CHECKPOINT) if SAM2_CHECKPOINT else "N/A"
        elif model_name == 'sam3':
            cfg = "HuggingFace-Base"
            size = f"{SAM3_IMSIZE}x{SAM3_IMSIZE}"
            ckpt = "HuggingFace-Auto" if SAM3_CHECKPOINT is None else str(SAM3_CHECKPOINT)
        elif model_name == 'medsegx':
            cfg = MEDSEGX_MODEL_TYPE
            size = f"{MEDSEGX_IMSIZE}x{MEDSEGX_IMSIZE}"
            ckpt = str(MEDSEGX_CHECKPOINT) if MEDSEGX_CHECKPOINT else "N/A"
        else:
            cfg = "unknown"
            size = "unknown"
            ckpt = "unknown"

        summary_rows.append({
            'model': model_name,
            'config': cfg,
            'checkpoint': ckpt,
            'image_size': size,
            'dice': f"{dice_mean:.4f} ± {dice_std:.4f}",
            'iou': f"{iou_mean:.4f} ± {iou_std:.4f}",
            'precision': f"{precision_mean:.4f} ± {precision_std:.4f}",
            'sensitivity': f"{sensitivity_mean:.4f} ± {sensitivity_std:.4f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    file_exists = os.path.exists(OUTPUT_CSV)
    df_summary.to_csv(OUTPUT_CSV, mode='a' if file_exists else 'w', header=not file_exists, index=False)
    print(f"\n📊 Results {'appended' if file_exists else 'saved'}: {OUTPUT_CSV.name}\n")

