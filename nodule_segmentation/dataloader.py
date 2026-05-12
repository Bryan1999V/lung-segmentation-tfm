"""Nodule Segmentation DataLoader Module.

This module provides data loading functionality for binary lung nodule segmentation:
    - Class 0: No pulmonary nodule (background)
    - Class 1: Pulmonary nodule (benign or malignant)

The dataset uses binary masks where any nodule (regardless of label) is mapped to class 1.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging

from config.common import config
from utilities import utils
from nodule_segmentation.losses import compute_dist_map_transform


# ============================================================================
# Preprocessing Functions
# ============================================================================


def apply_pixel_threshold_separation(ct_patch: np.ndarray, thresholds: dict = None) -> np.ndarray:
    """Apply Pixel Threshold Separation (PTS) to create multi-channel input.
    
    This technique, inspired by improved V-Net papers, separates CT images into
    multiple channels based on Hounsfield Unit (HU) intensity ranges. Instead of
    trying to predict nodule locations, it provides density information that helps
    the network learn which combinations of densities indicate nodules.
    
    Args:
        ct_patch: CT image patch in HU values, shape (C, H, W) or (C, D, H, W)
                  Expected to be in raw HU range (typically -1000 to 400+)
                  from extract_patch (NOT normalized yet)
        thresholds: Dictionary with threshold values. If None, uses config defaults.
    
    Returns:
        Multi-channel tensor with shape (4, H, W) or (4, D, H, W):
            - Channel 0: Very low density (< -600 HU) - air/lung/ground-glass
            - Channel 1: Low-mid density (-600 to -100 HU) - lung tissue
            - Channel 2: Mid-high density (-100 to 100 HU) - soft tissue/solid nodules
            - Channel 3: Very high density (> 100 HU) - calcifications/contrast
    
    Strategy:
        Nodules have huge HU variability and appear in MULTIPLE channels:
        - Ground-glass nodules: Ch0 + Ch1 (< -100 HU)
        - Part-solid nodules: Ch1 + Ch2 (-100 to 0 HU)
        - Solid nodules: Ch2 (0 to 100 HU)
        - Calcified nodules: Ch3 (> 100 HU)
        The network learns which combination indicates nodules.
    """
    if thresholds is None:
        thresholds = config.PTS_THRESHOLDS
    
    # Extract first channel if multi-channel input (squeeze channel dimension)
    if ct_patch.ndim == 3:  # (C, H, W)
        if ct_patch.shape[0] == 1:
            ct_image_hu = ct_patch[0]  # (H, W)
        else:
            ct_image_hu = ct_patch
    elif ct_patch.ndim == 4:  # (C, D, H, W) for 3D
        if ct_patch.shape[0] == 1:
            ct_image_hu = ct_patch[0]  # (D, H, W)
        else:
            ct_image_hu = ct_patch
    elif ct_patch.ndim == 2:  # (H, W)
        ct_image_hu = ct_patch
    else:
        ct_image_hu = ct_patch
    
    # IMPORTANT: Work directly with HU values (no desnormalization needed)
    # extract_patch returns raw HU values, not normalized
    
    # Create range-based channels (multi-range strategy)
    channels = []
    
    # Channel 0: Very low density - HU < -600 (air/lung parenchyma/ground-glass)
    very_low = (ct_image_hu < thresholds['low_density']).astype(np.float32)
    channels.append(very_low)
    
    # Channel 1: Low-mid density - HU between -600 and -100 (lung + some nodules)
    low_mid = ((ct_image_hu >= thresholds['low_density']) & 
               (ct_image_hu < thresholds['mid_density'])).astype(np.float32)
    channels.append(low_mid)
    
    # Channel 2: Mid-high density - HU between -100 and 100 (soft tissue/solid nodules)
    mid_high = ((ct_image_hu >= thresholds['mid_density']) & 
                (ct_image_hu < thresholds['high_density'])).astype(np.float32)
    channels.append(mid_high)
    
    # Channel 3: Very high density - HU > 100 (calcifications/contrast)
    very_high = (ct_image_hu >= thresholds['high_density']).astype(np.float32)
    channels.append(very_high)
    
    # Stack channels: result is (4, H, W) or (4, D, H, W)
    multi_channel = np.stack(channels, axis=0)
    
    return multi_channel


# ============================================================================
# Dataset Classes
# ============================================================================


class CTNoduleSegmentationDataset(Dataset):
    """Lung Nodule Segmentation Dataset with Binary Labels.
    
    This dataset loads CT nodule patches and their corresponding masks as
    binary masks (0: background, 1: nodule) regardless of nodule type.
    
    Args:
        data_dir: Path to nodule_blocks data directory
        masks_dir: Path to nodule_blocks_masks directory
        dataset: DataFrame with dataset information (must contain columns:
                 caseid, coordX, coordY, coordZ, label, AnnotationID, radius)
                 where label is: 0=benign, 1=malignant (used for reference only)
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

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single nodule patch with binary mask.
        
        Args:
            idx: Index of sample
        
        Returns:
            Dictionary containing:
                - image: Processed patch tensor (C, H, W) for 2D or (C, D, H, W) for 3D
                - mask: Binary segmentation mask (H, W) for 2D or (D, H, W) for 3D
                      0: background, 1: nodule (any type)
                - ID: Annotation ID
        """
        row = self.dataset.iloc[idx]

        label = int(row.label)  # 0: benign, 1: malignant (kept for reference)
        annotation_id = row.AnnotationID

        # Load CT data - use mmap_mode for faster loading
        img_path = self.data_dir / "image" / f"{annotation_id}.npy"
        metadata_path = self.data_dir / "metadata" / f"{annotation_id}.npy"

        ct_data = np.load(img_path, mmap_mode="r")
        metadata = np.load(metadata_path, allow_pickle=True).item()
        voxel_origin = metadata["origin"]
        world_matrix = metadata["transform"]
        voxel_spacing = metadata["spacing"]

        # Load mask data
        mask_path = self.masks_dir / f"{annotation_id}.npy"
        try:
            mask_data = np.load(mask_path, mmap_mode="r")
        except FileNotFoundError:
            logging.warning(f"Mask not found for {annotation_id}, creating empty mask")
            mask_data = np.zeros_like(ct_data)

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
            mask_data=None,
        )

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

        # Process mask: convert from (C, H, W) or (C, D, H, W) to (H, W) or (D, H, W)
        if self.mode == "2D":
            if mask_patch.ndim == 3:  # (C, H, W)
                mask_2d = mask_patch[0]
            elif mask_patch.ndim == 2:  # (H, W)
                mask_2d = mask_patch
            else:
                raise ValueError(f"Unexpected mask shape: {mask_patch.shape}")
            
            # Keep binary mask: 0: background, 1: nodule (any type)
            binary_mask = (mask_2d > 0).astype(np.int64)
            final_mask = binary_mask
            
        else:  # 3D mode
            if mask_patch.ndim == 4:  # (C, D, H, W)
                mask_3d = mask_patch[0]
            elif mask_patch.ndim == 3:  # (D, H, W)
                mask_3d = mask_patch
            else:
                raise ValueError(f"Unexpected mask shape: {mask_patch.shape}")
            
            # Keep binary mask: 0: background, 1: nodule (any type)
            binary_mask = (mask_3d > 0).astype(np.int64)
            final_mask = binary_mask

        # Apply Pixel Threshold Separation if enabled
        if config.USE_PIXEL_THRESHOLD_SEPARATION:
            # PTS expects RAW HU values from extract_patch (not normalized)
            # extract_patch returns values in HU range (typically -1000 to 400+)
            patch = patch.astype(np.float32)
            patch = apply_pixel_threshold_separation(patch, config.PTS_THRESHOLDS)
        else:
            # Standard normalization
            patch = patch.astype(np.float32)
            patch = utils.clip_and_scale(patch)

        # Compute distance maps for boundary loss
        # Use pixel spacing = 1.0 for simplicity (or could use actual spacing)
        num_classes = 2
        dist_map = compute_dist_map_transform(
            seg_mask=final_mask,
            num_classes=num_classes,
            resolution=(1.0, 1.0) if self.mode == "2D" else (1.0, 1.0, 1.0)
        )
        
        # Convert to tensors
        image_tensor = torch.from_numpy(patch)
        mask_tensor = torch.from_numpy(final_mask).long()
        dist_map_tensor = torch.from_numpy(dist_map)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "dist_map": dist_map_tensor,
            "ID": annotation_id,
        }

    def __repr__(self) -> str:
        return (
            f"CTNoduleSegmentationDataset(\n"
            f"  samples={len(self)},\n"
            f"  mode={self.mode},\n"
            f"  size_px={self.size_px},\n"
            f"  data_source={'LUNA25' if self.use_luna25_data else 'LUNA16'}\n"
            f")"
        )


# ============================================================================
# DataLoader Factory
# ============================================================================


def get_nodule_segmentation_data_loader(
    csv_name: str,
    mode: str = "2D",
    workers: int = 0,
    batch_size: int = 16,
    size_px: int = config.SIZE_PX,
    size_mm: int = config.SIZE_MM,
    use_luna25_data: bool = True,
    shuffle: bool = False,
    model_name: str = "",
    rotations: tuple = None,
    translations: bool = None,
    masks_dir: Path = None,
    num_samples: int = None,
) -> DataLoader:
    """Create a DataLoader for nodule segmentation.
    
    Args:
        csv_name: Name of CSV file (e.g., 'train.csv', 'valid.csv', 'test.csv')
        mode: "2D" or "3D"
        workers: Number of worker processes for data loading
        batch_size: Batch size (smaller than classification due to memory requirements)
        size_px: Pixel size for patches
        size_mm: Physical size of patch in mm
        use_luna25_data: Whether to use LUNA25 (True) or LUNA16 (False) data
        shuffle: Whether to shuffle data
        model_name: Name of model for channel determination
        rotations: Rotation ranges for augmentation
        translations: Whether to apply random translations
        masks_dir: Custom masks directory (overrides default)
        num_samples: Number of samples to use (None = use all samples)
    
    Returns:
        DataLoader configured for nodule segmentation
    """
    # Set default directories
    data_dir = (
        config.LUNA25_NODULES_DIR
        if use_luna25_data
        else config.LUNA16_NODULES_DIR
    )
    
    # Use custom masks_dir if provided, otherwise use default
    if masks_dir is None:
        masks_dir = (
            config.LUNA25_NODULES_MASKS_DIR
            if use_luna25_data
            else config.LUNA16_NODULE_MASKS_DIR
        )
    
    csv_dir = config.LUNA25_CSV_DIR if use_luna25_data else config.LUNA16_CSV_DIR

    csv_path = Path(csv_dir) / csv_name

    # Load dataset
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    total_samples = len(df)
    
    # Subset to num_samples if specified (only for training)
    if num_samples is not None and num_samples > 0 and num_samples < len(df):
        df = df.iloc[:num_samples].reset_index(drop=True)
        logging.info(f"Using subset: {num_samples} samples out of {total_samples} total samples")
    
    # Calculate class distribution
    benign_count = (df['label'] == 0).sum()
    malignant_count = (df['label'] == 1).sum()
    
    logging.info(
        f"Dataset <{csv_name}>: {len(df)} samples - "
        f"Benign: {benign_count} ({benign_count/len(df)*100:.1f}%) - "
        f"Malignant: {malignant_count} ({malignant_count/len(df)*100:.1f}%)"
    )

    # Create dataset
    dataset = CTNoduleSegmentationDataset(
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
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=utils.worker_init_fn,
    )

    logging.info(f"DataLoader created: {len(dataset)} samples, {len(dataloader)} batches")

    return dataloader


# ============================================================================
# 3-Class Segmentation Dataset (Background, Benign, Malignant)
# ============================================================================


class CTNoduleSegmentation3ClassDataset(Dataset):
    """Lung Nodule Segmentation Dataset with 3-Class Labels.
    
    This dataset loads CT nodule patches and their corresponding masks as
    3-class masks:
        - Class 0: Background (no nodule)
        - Class 1: Benign nodule
        - Class 2: Malignant nodule
    
    The class is determined by the 'label' field in the dataset CSV.
    
    Args:
        data_dir: Path to nodule_blocks data directory
        masks_dir: Path to nodule_blocks_masks directory
        dataset: DataFrame with dataset information (must contain columns:
                 caseid, coordX, coordY, coordZ, label, AnnotationID, radius)
                 where label is: 0=benign, 1=malignant
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

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single nodule patch with 3-class mask.
        
        Args:
            idx: Index of sample
        
        Returns:
            Dictionary containing:
                - image: Processed patch tensor (C, H, W) for 2D or (C, D, H, W) for 3D
                - mask: 3-class segmentation mask (H, W) for 2D or (D, H, W) for 3D
                      0: background, 1: benign nodule, 2: malignant nodule
                - label: Classification label (0: benign, 1: malignant)
                - dist_map: Distance map for boundary loss (K, H, W) or (K, D, H, W)
                - ID: Annotation ID
        """
        row = self.dataset.iloc[idx]

        label = int(row.label)  # 0: benign, 1: malignant
        annotation_id = row.AnnotationID

        # Load CT data - use mmap_mode for faster loading
        img_path = self.data_dir / "image" / f"{annotation_id}.npy"
        metadata_path = self.data_dir / "metadata" / f"{annotation_id}.npy"

        ct_data = np.load(img_path, mmap_mode="r")
        metadata = np.load(metadata_path, allow_pickle=True).item()
        voxel_origin = metadata["origin"]
        world_matrix = metadata["transform"]
        voxel_spacing = metadata["spacing"]

        # Load mask data
        mask_path = self.masks_dir / f"{annotation_id}.npy"
        try:
            mask_data = np.load(mask_path, mmap_mode="r")
        except FileNotFoundError:
            logging.warning(f"Mask not found for {annotation_id}, creating empty mask")
            mask_data = np.zeros_like(ct_data)

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
            mask_data=None,
        )

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

        # Process mask: convert from (C, H, W) or (C, D, H, W) to (H, W) or (D, H, W)
        # Then create 3-class mask based on label
        if self.mode == "2D":
            if mask_patch.ndim == 3:  # (C, H, W)
                mask_2d = mask_patch[0]
            elif mask_patch.ndim == 2:  # (H, W)
                mask_2d = mask_patch
            else:
                raise ValueError(f"Unexpected mask shape: {mask_patch.shape}")
            
            # Create 3-class mask: 0=background, 1=benign, 2=malignant
            multiclass_mask = np.zeros_like(mask_2d, dtype=np.int64)
            nodule_region = mask_2d > 0
            if label == 0:  # benign
                multiclass_mask[nodule_region] = 1
            elif label == 1:  # malignant
                multiclass_mask[nodule_region] = 2
            final_mask = multiclass_mask
            
        else:  # 3D mode
            if mask_patch.ndim == 4:  # (C, D, H, W)
                mask_3d = mask_patch[0]
            elif mask_patch.ndim == 3:  # (D, H, W)
                mask_3d = mask_patch
            else:
                raise ValueError(f"Unexpected mask shape: {mask_patch.shape}")
            
            # Create 3-class mask: 0=background, 1=benign, 2=malignant
            multiclass_mask = np.zeros_like(mask_3d, dtype=np.int64)
            nodule_region = mask_3d > 0
            if label == 0:  # benign
                multiclass_mask[nodule_region] = 1
            elif label == 1:  # malignant
                multiclass_mask[nodule_region] = 2
            final_mask = multiclass_mask

        # Apply Pixel Threshold Separation if enabled
        if config.USE_PIXEL_THRESHOLD_SEPARATION:
            patch = patch.astype(np.float32)
            patch = apply_pixel_threshold_separation(patch, config.PTS_THRESHOLDS)
        else:
            # Standard normalization
            patch = patch.astype(np.float32)
            patch = utils.clip_and_scale(patch)

        # Compute distance maps for boundary loss
        num_classes = 3  # background, benign, malignant
        dist_map = compute_dist_map_transform(
            seg_mask=final_mask,
            num_classes=num_classes,
            resolution=(1.0, 1.0) if self.mode == "2D" else (1.0, 1.0, 1.0)
        )
        
        # Convert to tensors
        image_tensor = torch.from_numpy(patch)
        mask_tensor = torch.from_numpy(final_mask).long()
        dist_map_tensor = torch.from_numpy(dist_map)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label_tensor,
            "dist_map": dist_map_tensor,
            "ID": annotation_id,
        }

    def __repr__(self) -> str:
        return (
            f"CTNoduleSegmentation3ClassDataset(\n"
            f"  samples={len(self)},\n"
            f"  mode={self.mode},\n"
            f"  size_px={self.size_px},\n"
            f"  classes=3 (bg, benign, malignant),\n"
            f"  data_source={'LUNA25' if self.use_luna25_data else 'LUNA16'}\n"
            f")"
        )


def get_nodule_segmentation_3class_data_loader(
    csv_name: str,
    mode: str = "2D",
    workers: int = 0,
    batch_size: int = 16,
    size_px: int = config.SIZE_PX,
    size_mm: int = config.SIZE_MM,
    use_luna25_data: bool = True,
    shuffle: bool = False,
    model_name: str = "",
    rotations: tuple = None,
    translations: bool = None,
    masks_dir: Path = None,
    num_samples: int = None,
) -> DataLoader:
    """Create a DataLoader for 3-class nodule segmentation.
    
    This creates a dataloader that returns 3-class masks:
        - Class 0: Background
        - Class 1: Benign nodule
        - Class 2: Malignant nodule
    
    Args:
        csv_name: Name of CSV file (e.g., 'train.csv', 'valid.csv', 'test.csv')
        mode: "2D" or "3D"
        workers: Number of worker processes for data loading
        batch_size: Batch size
        size_px: Pixel size for patches
        size_mm: Physical size of patch in mm
        use_luna25_data: Whether to use LUNA25 (True) or LUNA16 (False) data
        shuffle: Whether to shuffle data
        model_name: Name of model for channel determination
        rotations: Rotation ranges for augmentation
        translations: Whether to apply random translations
        masks_dir: Custom masks directory (overrides default)
        num_samples: Number of samples to use (None = use all samples)
    
    Returns:
        DataLoader configured for 3-class nodule segmentation
    """
    # Set default directories
    data_dir = (
        config.LUNA25_NODULES_DIR
        if use_luna25_data
        else config.LUNA16_NODULES_DIR
    )
    
    # Use custom masks_dir if provided, otherwise use default
    if masks_dir is None:
        masks_dir = (
            config.LUNA25_NODULES_MASKS_DIR
            if use_luna25_data
            else config.LUNA16_NODULE_MASKS_DIR
        )
    
    csv_dir = config.LUNA25_CSV_DIR if use_luna25_data else config.LUNA16_CSV_DIR
    csv_path = Path(csv_dir) / csv_name

    # Load dataset
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    total_samples = len(df)
    
    # Subset to num_samples if specified
    if num_samples is not None and num_samples > 0 and num_samples < len(df):
        df = df.iloc[:num_samples].reset_index(drop=True)
        logging.info(f"Using subset: {num_samples} samples out of {total_samples} total samples")
    
    # Calculate class distribution
    benign_count = (df['label'] == 0).sum()
    malignant_count = (df['label'] == 1).sum()
    
    logging.info(
        f"3-Class Dataset <{csv_name}>: {len(df)} samples - "
        f"Benign: {benign_count} ({benign_count/len(df)*100:.1f}%) - "
        f"Malignant: {malignant_count} ({malignant_count/len(df)*100:.1f}%)"
    )

    # Create dataset
    dataset = CTNoduleSegmentation3ClassDataset(
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
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=utils.worker_init_fn,
    )

    logging.info(f"3-Class DataLoader created: {len(dataset)} samples, {len(dataloader)} batches")

    return dataloader
