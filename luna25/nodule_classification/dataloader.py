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

from luna25.config.common import config
from luna25.utils.utils import extract_patch, make_weights_for_balanced_classes, worker_init_fn


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

        # Validate required columns
        required_cols = ["caseid", "coordX", "coordY", "coordZ", "label", "AnnotationID"]
        missing_cols = [col for col in required_cols if col not in dataset.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")

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

        caseid = row.caseid
        label = int(row.label)
        annotation_id = row.AnnotationID
        coord = np.array([row.coordX, row.coordY, row.coordZ])

        # Load CT data
        img_path = self.data_dir / f"{caseid}.npy"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        ct_data_dict = np.load(img_path, allow_pickle=True).item()
        ct_data = ct_data_dict["ct_data"]
        voxel_origin = ct_data_dict["origin"]
        world_matrix = ct_data_dict["worldmatrix"]
        voxel_spacing = ct_data_dict["voxel_spacing"]

        # Calculate output shape
        output_shape = (self.size_px, self.size_px, self.size_px)
        voxel_spacing_out = tuple([self.size_mm / self.size_px] * 3)

        # Extract patch
        patch = extract_patch(
            ct_data=ct_data,
            coord=coord,
            src_voxel_origin=voxel_origin,
            src_world_matrix=world_matrix,
            src_voxel_spacing=voxel_spacing,
            output_shape=output_shape,
            voxel_spacing=voxel_spacing_out,
            rotations=self.rotations,
            translations=self.translations,
            coord_space_world=False,
            mode=self.mode,
            model_name=self.model_name,
        )

        # Load and apply mask filtering
        mask_path = self.masks_dir / f"{caseid}.npy"
        if mask_path.exists():
            try:
                mask_data_dict = np.load(mask_path, allow_pickle=True).item()
                mask_data = mask_data_dict.get("mask_data", None)

                if mask_data is not None:
                    # Extract mask patch using same parameters
                    mask_patch = extract_patch(
                        ct_data=mask_data.astype(np.float32),
                        coord=coord,
                        src_voxel_origin=voxel_origin,
                        src_world_matrix=world_matrix,
                        src_voxel_spacing=voxel_spacing,
                        output_shape=output_shape,
                        voxel_spacing=voxel_spacing_out,
                        rotations=None,  # No augmentation for mask
                        translations=None,
                        coord_space_world=False,
                        mode=self.mode,
                        model_name=self.model_name,
                    )

                    # Apply mask filtering (zero out non-nodule regions)
                    if self.mode == "2D":
                        # mask_patch shape: (C, H, W)
                        binary_mask = (mask_patch > 0).astype(np.float32)
                    else:  # 3D
                        # mask_patch shape: (1, D, H, W)
                        binary_mask = (mask_patch > 0).astype(np.float32)

                    patch = patch * binary_mask

            except Exception as e:
                logging.warning(f"Could not load/apply mask for {caseid}: {e}")

        # Convert to tensors
        image_tensor = torch.from_numpy(patch).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "ID": annotation_id,
            "image_path": str(img_path),
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
    data_dir: str = None,
    masks_dir: str = None,
    csv_dir: str = None,
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

    Returns:
        DataLoader configured for nodule classification
    """
    # Set default directories
    if data_dir is None:
        data_dir = (
            "/workspace/data/LUNA25/luna25_nodule_blocks"
            if use_luna25_data
            else "/workspace/data/LUNA16/luna16_nodule_blocks"
        )

    if masks_dir is None:
        masks_dir = (
            "/workspace/data/LUNA25/luna25_nodule_blocks_masks"
            if use_luna25_data
            else "/workspace/data/LUNA16/luna16_nodule_blocks_masks"
        )

    if csv_dir is None:
        csv_dir = "/workspace/data/LUNA25/dataset_csv" if use_luna25_data else "/workspace/data/LUNA16/annotations"

    csv_path = Path(csv_dir) / csv_name

    logging.info(
        f"[Nodule Classification] Loading data from:\n"
        f"  Data: {data_dir}\n"
        f"  Masks: {masks_dir}\n"
        f"  CSV: {csv_path}\n"
        f"  Mode: {mode}, Batch size: {batch_size}"
    )

    # Load dataset
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Apply weighted sampling for LUNA25 training data to handle class imbalance
    sampler = None
    if use_luna25_data and csv_name == "train.csv":
        labels = df["label"].values
        weights = make_weights_for_balanced_classes(labels)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        logging.info(
            f"Applied weighted sampling for class balance (positive: {labels.sum()}, negative: {len(labels) - labels.sum()})"
        )

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
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
    )

    logging.info(f"DataLoader created: {len(dataset)} samples, {len(dataloader)} batches")

    return dataloader
