"""Nodule Classification DataLoader Module.

This module provides optimized data loading functionality for lung nodule classification tasks,
including 2D and 3D patch extraction, augmentation, and mask-based filtering.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import logging

from config.common import config
from utilities import utils


# ============================================================================
# Dataset Classes
# ============================================================================


class CTNoduleDataset(Dataset):
    """Lung Nodule Classification Dataset with Mask Filtering.

    Specialized dataset for nodule classification that applies mask filtering
    to extract only nodule pixels from patches, improving classification performance.
    Supports both 2D and 3D modes with comprehensive augmentation.

    Args:
        data_dir: Path to nodule_blocks data directory
        masks_dir: Path to nodule_blocks_masks directory
        dataset: DataFrame with dataset information (must contain columns:
                 caseid, coordX, coordY, coordZ, label, AnnotationID, radius)
        translations: Whether to apply random translations
        rotations: Tuple with rotation ranges (angle_x, angle_y, angle_z)
        size_px: Size of patch in pixels
        size_mm: Size of patch in mm
        mode: "2D" or "3D"
        use_luna25_data: Whether to use LUNA25 (True) or LUNA16 (False) data
        model_name: Name of the model for channel determination
        filtering: Whether to apply mask filtering (default: False)
    """

    def __init__(
        self,
        data_dir: str,
        masks_dir: str,
        dataset: pd.DataFrame,
        translations: bool = None,
        rotations: tuple = None,
        size_px: int = 64,
        size_mm: int = 50,
        mode: str = "2D",
        use_luna25_data: bool = True,
        model_name: str = "",
        filtering: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.masks_dir = Path(masks_dir)
        self.dataset = dataset
        self.rotations = rotations
        self.translations = translations
        self.size_px = size_px
        self.size_mm = size_mm
        self.mode = mode
        self.use_luna25_data = use_luna25_data
        self.model_name = model_name
        self.filtering = filtering

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single nodule patch with mask filtering.

        Args:
            idx: Index of sample

        Returns:
            Dictionary containing:
                - image: Processed patch tensor
                - label: Binary classification label
                - ID: Annotation ID
                - image_path: Path to source image
        """
        row = self.dataset.iloc[idx]

        label = int(row.label)
        annotation_id = row.AnnotationID

        # Load CT data - use mmap_mode for faster loading
        img_path = self.data_dir / "image" / f"{annotation_id}.npy"
        metadata_path = self.data_dir / "metadata" / f"{annotation_id}.npy"

        ct_data = np.load(img_path, mmap_mode="r")
        metadata = np.load(metadata_path, allow_pickle=True).item()
        voxel_origin = metadata["origin"]
        world_matrix = metadata["transform"]
        voxel_spacing = metadata["spacing"]

        # Load mask data (always load for proper processing)
        mask_path = self.masks_dir / f"{annotation_id}.npy"
        try:
            mask_data = np.load(mask_path, mmap_mode="r")
        except:
            mask_data = None

        # Calculate output shape based on mode
        if self.mode == "2D":
            output_shape = (1, self.size_px, self.size_px)
        else:
            output_shape = (self.size_px, self.size_px, self.size_px)
        voxel_spacing_out = tuple([self.size_mm / self.size_px] * 3)

        # Use center coordinate from PATCH_SIZE (dimensions of preprocessed nodule blocks)
        coord = tuple(np.array(config.PATCH_SIZE) // 2)

        # Determine translation radius
        translations = None
        if self.translations:
            translations = 2.5  # Fixed radius for speed

        # Extract image patch
        patch = utils.extract_patch(
            ct_data=ct_data,
            coord=coord,
            src_voxel_origin=voxel_origin,
            src_world_matrix=world_matrix,
            src_voxel_spacing=voxel_spacing,
            output_shape=output_shape,
            voxel_spacing=voxel_spacing_out,
            rotations=self.rotations,
            translations=translations,
            coord_space_world=False,
            mode=self.mode,
            model_name=self.model_name,
            mask_data=None,  # Don't apply mask during extraction
        )

        # Apply weighted mask filtering if enabled and mask available
        if self.filtering and mask_data is not None:
            # Extract mask patch with same transformations
            mask_patch = utils.extract_patch(
                ct_data=mask_data,
                coord=coord,
                src_voxel_origin=voxel_origin,
                src_world_matrix=world_matrix,
                src_voxel_spacing=voxel_spacing,
                output_shape=output_shape,
                voxel_spacing=voxel_spacing_out,
                rotations=self.rotations,
                translations=translations,
                coord_space_world=False,
                mode=self.mode,
                model_name="mask",  # Special identifier for masks
                mask_data=None,
            )

            # Apply weighted mask: nodule regions = 100%, background = 30%
            if self.mode == "2D":
                if mask_patch.ndim == 3:  # (C, H, W)
                    mask_2d = mask_patch[0]
                elif mask_patch.ndim == 2:  # (H, W)
                    mask_2d = mask_patch
                else:
                    raise ValueError(f"Unexpected mask shape: {mask_patch.shape}")

                # Create weighted mask
                weighted_mask = np.where(mask_2d > 0, 1.0, 0.3)

                # Apply to all channels
                if patch.ndim == 3:  # (C, H, W)
                    for c in range(patch.shape[0]):
                        patch[c] = patch[c] * weighted_mask
                elif patch.ndim == 2:  # (H, W)
                    patch = patch * weighted_mask
            else:  # 3D mode
                if mask_patch.ndim == 4:  # (C, D, H, W)
                    mask_3d = mask_patch[0]
                elif mask_patch.ndim == 3:  # (D, H, W)
                    mask_3d = mask_patch
                else:
                    raise ValueError(f"Unexpected mask shape: {mask_patch.shape}")

                weighted_mask = np.where(mask_3d > 0, 1.0, 0.3)

                if patch.ndim == 4:  # (C, D, H, W)
                    for c in range(patch.shape[0]):
                        patch[c] = patch[c] * weighted_mask
                elif patch.ndim == 3:  # (D, H, W)
                    patch = patch * weighted_mask

        # CRITICAL: Normalize AFTER weighted mask (like in original code)
        patch = patch.astype(np.float32)
        patch = utils.clip_and_scale(patch)

        # Convert to tensors
        image_tensor = torch.from_numpy(patch)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "ID": annotation_id,
        }

    def __repr__(self) -> str:
        return (
            f"CTNoduleDataset(\n"
            f"  samples={len(self)},\n"
            f"  mode={self.mode},\n"
            f"  size_px={self.size_px},\n"
            f"  data_source={'LUNA25' if self.use_luna25_data else 'LUNA16'}\n"
            f")"
        )


# ============================================================================
# DataLoader Factory
# ============================================================================


def get_nodule_classification_data_loader(
    csv_name: str,
    mode: str = "2D",
    workers: int = 0,
    batch_size: int = 64,
    size_px: int = config.SIZE_PX,
    size_mm: int = config.SIZE_MM,
    use_luna25_data: bool = True,
    shuffle: bool = False,
    model_name: str = "",
    rotations: tuple = None,
    translations: bool = None,
    filtering: bool = False,
) -> DataLoader:
    """Create a DataLoader for nodule classification with mask filtering.

    Args:
        csv_name: Name of CSV file (e.g., 'train.csv', 'valid.csv', 'test.csv')
        mode: "2D" or "3D"
        workers: Number of worker processes for data loading
        batch_size: Batch size
        size_px: Pixel size for patches
        size_mm: Physical size of patch in mm
        use_luna25_data: Whether to use LUNA25 (True) or LUNA16 (False) data
        shuffle: Whether to shuffle data (overridden by sampler)
        model_name: Name of model for channel determination
        rotations: Rotation ranges for augmentation
        translations: Whether to apply random translations
        data_dir: Override default data directory
        masks_dir: Override default masks directory
        csv_dir: Override default CSV directory
        filtering: Whether to apply mask filtering (default: False)

    Returns:
        DataLoader configured for nodule classification
    """
    # Set default directories
    data_dir = (
        config.LUNA25_NODULES_DIR
        if use_luna25_data
        else config.LUNA16_NODULES_DIR
    )
    masks_dir = (
        config.LUNA25_NODULES_MASKS_DIR
        if use_luna25_data
        else config.LUNA16_NODULES_MASKS_DIR
    )
    csv_dir = config.LUNA25_CSV_DIR if use_luna25_data else config.LUNA16_CSV_DIR

    csv_path = Path(csv_dir) / csv_name

    # Load dataset
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logging.info(
        f"Label distribution for <{csv_name}>: malign <{df['label'].sum()}> - benign: <{len(df) - df['label'].sum()}>"
    )

    # Apply weighted sampling for LUNA25 training data to handle class imbalance
    sampler = None
    if use_luna25_data and csv_name == "train.csv":
        labels = df["label"].values
        weights = utils.make_weights_for_balanced_classes(labels)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    # Create dataset
    dataset = CTNoduleDataset(
        data_dir=data_dir,
        masks_dir=masks_dir,
        dataset=df,
        translations=translations,
        rotations=rotations,
        size_mm=size_mm,
        size_px=size_px,
        mode=mode,
        use_luna25_data=use_luna25_data,
        model_name=model_name,
        filtering=filtering,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=utils.worker_init_fn,
    )

    logging.info(f"DataLoader created: {len(dataset)} samples, {len(dataloader)} batches")

    return dataloader
