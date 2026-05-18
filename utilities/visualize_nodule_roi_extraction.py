"""
Visualization of Nodule ROI Extraction Process

This module generates academic-quality figures demonstrating the complete
pipeline for extracting a Region of Interest (ROI) centered on a pulmonary
nodule from a CT volume. This process is fundamental in CAD systems for
nodule segmentation and classification.

The visualization shows three stages:
1. Full axial CT slice with nodule location marked
2. Cropped region around the nodule
3. Final resized and normalized ROI used as model input

Author: TFM System
Date: 2026
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage


# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CT Normalization Functions
# ============================================================================

def normalize_hu_range(
    image_data: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0
) -> np.ndarray:
    """
    Normalize HU values to [0, 1] range by clipping to a specified window.
    
    This is the standard normalization used in CAD systems for nodule
    classification and segmentation.
    
    Args:
        image_data: Input array in Hounsfield Units
        hu_min: Minimum HU value (default: -1000 for air)
        hu_max: Maximum HU value (default: 400 for soft tissue)
    
    Returns:
        Normalized array in [0, 1]
    
    Notes:
        Common HU ranges:
        - Lung window: [-1000, 400]
        - Soft tissue window: [-160, 240]
        - Bone window: [-500, 1500]
    """
    # Clip to HU range
    clipped = np.clip(image_data, hu_min, hu_max)
    
    # Normalize to [0, 1]
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    
    return normalized.astype(np.float32)


def normalize_zero_mean_unit_variance(image_data: np.ndarray) -> np.ndarray:
    """
    Normalize image to zero mean and unit variance (Z-score normalization).
    
    Args:
        image_data: Input array
    
    Returns:
        Normalized array
    """
    mean = np.mean(image_data)
    std = np.std(image_data)
    
    if std > 0:
        normalized = (image_data - mean) / std
    else:
        normalized = image_data - mean
    
    return normalized.astype(np.float32)


# ============================================================================
# File I/O Functions
# ============================================================================

def find_luna25_volumes(data_dir: Path) -> List[Path]:
    """
    Find all LUNA25 CT volume files (MHA, DICOM, or Nifti).
    
    Args:
        data_dir: Path to LUNA25 root directory
    
    Returns:
        List of volume file paths
    """
    if data_dir.name == "luna25_images":
        luna25_images = data_dir
    else:
        luna25_images = data_dir / "luna25_images"
    
    if not luna25_images.exists():
        raise FileNotFoundError(f"LUNA25 images directory not found: {luna25_images}")
    
    # Look for MHA/NIfTI files
    mha_files = list(luna25_images.glob("*.mha"))
    nii_files = list(luna25_images.glob("*.nii.gz")) + list(luna25_images.glob("*.nii"))
    
    volumes = mha_files + nii_files
    
    # If no files found, look for DICOM directories
    if not volumes:
        all_items = list(luna25_images.glob("*"))
        volumes = [v for v in all_items if v.is_dir()]
    
    logger.info(f"Found {len(volumes)} LUNA25 volumes")
    
    return volumes


def load_ct_volume(volume_path: Path) -> Tuple[np.ndarray, sitk.Image]:
    """
    Load 3D CT volume from MHA, NIfTI, or DICOM directory.
    
    Args:
        volume_path: Path to volume file or DICOM directory
    
    Returns:
        Tuple of (3D numpy array (Z, Y, X) in HU, SimpleITK Image object)
    """
    volume_path = Path(volume_path)
    
    if volume_path.is_file():
        logger.info(f"Loading volume: {volume_path.name}")
        image_sitk = sitk.ReadImage(str(volume_path))
    elif volume_path.is_dir():
        logger.info(f"Loading DICOM series: {volume_path.name}")
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(volume_path))
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {volume_path}")
        
        reader.SetFileNames(dicom_files)
        image_sitk = reader.Execute()
    else:
        raise FileNotFoundError(f"Volume path does not exist: {volume_path}")
    
    # Convert to numpy (Z, Y, X)
    image_array = sitk.GetArrayFromImage(image_sitk)
    image_array = image_array.astype(np.float32)
    
    logger.info(f"Volume shape: {image_array.shape}")
    logger.info(f"HU range: [{image_array.min():.1f}, {image_array.max():.1f}]")
    logger.info(f"Spacing: {image_sitk.GetSpacing()}")
    logger.info(f"Origin: {image_sitk.GetOrigin()}")
    
    return image_array, image_sitk


def load_nodule_annotations(csv_path: Path) -> pd.DataFrame:
    """
    Load LUNA25 nodule annotations from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DataFrame with nodule annotations
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} nodule annotations")
    
    return df


def get_series_nodules(df: pd.DataFrame, series_uid: str) -> List[Dict]:
    """
    Get all nodule annotations for a specific SeriesInstanceUID.
    
    Args:
        df: DataFrame with nodule annotations
        series_uid: SeriesInstanceUID to filter by
    
    Returns:
        List of nodule dictionaries
    """
    series_nodules = df[df['SeriesInstanceUID'] == series_uid]
    
    if len(series_nodules) == 0:
        return []
    
    nodules = []
    for _, row in series_nodules.iterrows():
        nodules.append({
            'coord_x': row['CoordX'],
            'coord_y': row['CoordY'],
            'coord_z': row['CoordZ'],
            'lesion_id': row['LesionID'],
            'label': row['label']
        })
    
    return nodules


# ============================================================================
# Coordinate Transformation Functions
# ============================================================================

def world_to_voxel(
    world_coord: Tuple[float, float, float],
    image_sitk: sitk.Image
) -> Tuple[int, int, int]:
    """
    Convert world coordinates (mm) to voxel indices.
    
    Args:
        world_coord: (x, y, z) in mm (world coordinates)
        image_sitk: SimpleITK image object
    
    Returns:
        (z_idx, y_idx, x_idx) voxel indices for numpy array indexing
    """
    origin = image_sitk.GetOrigin()  # (x, y, z) in mm
    spacing = image_sitk.GetSpacing()  # (x, y, z) in mm/voxel
    
    x_world, y_world, z_world = world_coord
    
    x_idx = int(round((x_world - origin[0]) / spacing[0]))
    y_idx = int(round((y_world - origin[1]) / spacing[1]))
    z_idx = int(round((z_world - origin[2]) / spacing[2]))
    
    # Return in numpy indexing order (Z, Y, X)
    return z_idx, y_idx, x_idx


def voxel_to_world(
    voxel_coord: Tuple[int, int, int],
    image_sitk: sitk.Image
) -> Tuple[float, float, float]:
    """
    Convert voxel indices to world coordinates (mm).
    
    Args:
        voxel_coord: (z_idx, y_idx, x_idx) voxel indices
        image_sitk: SimpleITK image object
    
    Returns:
        (x, y, z) in mm (world coordinates)
    """
    origin = image_sitk.GetOrigin()
    spacing = image_sitk.GetSpacing()
    
    z_idx, y_idx, x_idx = voxel_coord
    
    x_world = origin[0] + x_idx * spacing[0]
    y_world = origin[1] + y_idx * spacing[1]
    z_world = origin[2] + z_idx * spacing[2]
    
    return x_world, y_world, z_world


# ============================================================================
# ROI Extraction Functions
# ============================================================================

def extract_roi_around_nodule(
    volume_3d: np.ndarray,
    center_voxel: Tuple[int, int, int],
    roi_size_voxels: Tuple[int, int, int]
) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
    """
    Extract a 3D ROI centered on a nodule from a CT volume.
    
    Args:
        volume_3d: Full CT volume (Z, Y, X)
        center_voxel: (z, y, x) center coordinate in voxel space
        roi_size_voxels: (depth, height, width) of ROI in voxels
    
    Returns:
        Tuple of (ROI array, bounding box coordinates (z_min, z_max, y_min, y_max, x_min, x_max))
    """
    z_center, y_center, x_center = center_voxel
    depth, height, width = roi_size_voxels
    
    # Calculate half sizes
    half_depth = depth // 2
    half_height = height // 2
    half_width = width // 2
    
    # Calculate bounds
    z_min = max(0, z_center - half_depth)
    z_max = min(volume_3d.shape[0], z_center + half_depth)
    y_min = max(0, y_center - half_height)
    y_max = min(volume_3d.shape[1], y_center + half_height)
    x_min = max(0, x_center - half_width)
    x_max = min(volume_3d.shape[2], x_center + half_width)
    
    # Extract ROI
    roi = volume_3d[z_min:z_max, y_min:y_max, x_min:x_max]
    
    logger.info(f"Extracted ROI with bounds: z=[{z_min}, {z_max}], y=[{y_min}, {y_max}], x=[{x_min}, {x_max}]")
    logger.info(f"ROI shape: {roi.shape}")
    
    return roi, (z_min, z_max, y_min, y_max, x_min, x_max)


def resize_roi(
    roi: np.ndarray,
    target_size: Tuple[int, int, int],
    order: int = 3
) -> np.ndarray:
    """
    Resize ROI to target size using interpolation.
    
    Args:
        roi: Input ROI array
        target_size: (depth, height, width) target size
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
    
    Returns:
        Resized ROI
    """
    # Calculate zoom factors
    zoom_factors = [
        target_size[0] / roi.shape[0],
        target_size[1] / roi.shape[1],
        target_size[2] / roi.shape[2]
    ]
    
    # Resize using scipy
    resized = ndimage.zoom(roi, zoom_factors, order=order)
    
    logger.info(f"Resized ROI from {roi.shape} to {resized.shape}")
    
    return resized


def extract_2d_roi_from_slice(
    slice_2d: np.ndarray,
    center_voxel: Tuple[int, int],
    roi_size: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract a 2D ROI from an axial slice.
    
    Args:
        slice_2d: 2D axial slice (Y, X)
        center_voxel: (y, x) center coordinate
        roi_size: (height, width) of ROI
    
    Returns:
        Tuple of (ROI array, bounding box (y_min, y_max, x_min, x_max))
    """
    y_center, x_center = center_voxel
    height, width = roi_size
    
    half_height = height // 2
    half_width = width // 2
    
    y_min = max(0, y_center - half_height)
    y_max = min(slice_2d.shape[0], y_center + half_height)
    x_min = max(0, x_center - half_width)
    x_max = min(slice_2d.shape[1], x_center + half_width)
    
    roi = slice_2d[y_min:y_max, x_min:x_max]
    
    return roi, (y_min, y_max, x_min, x_max)


# ============================================================================
# Visualization Functions
# ============================================================================

def draw_bbox_on_axis(
    ax: plt.Axes,
    bbox: Tuple[int, int, int, int],
    color: str = 'red',
    linewidth: float = 2.0,
    linestyle: str = '--'
):
    """
    Draw a bounding box on a matplotlib axis.
    
    Args:
        ax: Matplotlib axis
        bbox: (y_min, y_max, x_min, x_max) bounding box
        color: Box color
        linewidth: Line width
        linestyle: Line style
    """
    y_min, y_max, x_min, x_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    
    rect = patches.Rectangle(
        (x_min, y_min), width, height,
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none',
        linestyle=linestyle
    )
    ax.add_patch(rect)


def draw_circle_on_axis(
    ax: plt.Axes,
    center: Tuple[int, int],
    radius: float,
    color: str = 'red',
    linewidth: float = 2.0
):
    """
    Draw a circle marker on a matplotlib axis.
    
    Args:
        ax: Matplotlib axis
        center: (y, x) center coordinate
        radius: Circle radius
        color: Circle color
        linewidth: Line width
    """
    y_center, x_center = center
    
    circle = patches.Circle(
        (x_center, y_center), radius,
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none'
    )
    ax.add_patch(circle)


def create_roi_extraction_figure(
    full_slice: np.ndarray,
    nodule_center: Tuple[int, int],
    cropped_roi: np.ndarray,
    final_roi: np.ndarray,
    bbox_2d: Tuple[int, int, int, int],
    output_path: Optional[Path] = None,
    roi_size_target: Optional[Tuple[int, int]] = None,
    dpi: int = 300
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create academic-quality figure showing ROI extraction process.
    
    Args:
        full_slice: Full axial CT slice
        nodule_center: (y, x) nodule center in full slice
        cropped_roi: Cropped region around nodule
        final_roi: Final resized and normalized ROI
        bbox_2d: (y_min, y_max, x_min, x_max) bounding box in full slice
        output_path: Path to save figure (without extension)
        roi_size_target: Target ROI size for display
        dpi: Output DPI resolution
    
    Returns:
        Tuple of (figure, axes array)
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)
    
    # =====================================================================
    # Subplot (a): Full CT slice with nodule marked
    # =====================================================================
    ax_full = axes[0]
    
    im_full = ax_full.imshow(full_slice, cmap='gray', origin='lower')
    
    # Draw bounding box
    draw_bbox_on_axis(ax_full, bbox_2d, color='red', linewidth=2.5, linestyle='--')
    
    # Draw circle at nodule center
    draw_circle_on_axis(ax_full, nodule_center, radius=10, color='red', linewidth=2.0)
    
    # Remove axes
    ax_full.set_xticks([])
    ax_full.set_yticks([])
    ax_full.spines['top'].set_visible(False)
    ax_full.spines['right'].set_visible(False)
    ax_full.spines['bottom'].set_visible(False)
    ax_full.spines['left'].set_visible(False)
    
    # Title
    ax_full.set_title(
        "(a) CT original con nódulo marcado",
        fontsize=14,
        fontweight='bold',
        pad=10
    )
    
    # Add colorbar
    cbar_full = plt.colorbar(im_full, ax=ax_full, fraction=0.046, pad=0.04)
    cbar_full.set_label('HU', fontsize=11)
    
    # =====================================================================
    # Subplot (b): Cropped region
    # =====================================================================
    ax_crop = axes[1]
    
    im_crop = ax_crop.imshow(cropped_roi, cmap='gray', origin='lower')
    
    # Remove axes
    ax_crop.set_xticks([])
    ax_crop.set_yticks([])
    ax_crop.spines['top'].set_visible(False)
    ax_crop.spines['right'].set_visible(False)
    ax_crop.spines['bottom'].set_visible(False)
    ax_crop.spines['left'].set_visible(False)
    
    # Title
    if roi_size_target:
        title_crop = f"(b) Región recortada ({roi_size_target[0]}×{roi_size_target[1]} px)"
    else:
        title_crop = f"(b) Región recortada ({cropped_roi.shape[1]}×{cropped_roi.shape[0]} px)"
    
    ax_crop.set_title(
        title_crop,
        fontsize=14,
        fontweight='bold',
        pad=10
    )
    
    # Add colorbar
    cbar_crop = plt.colorbar(im_crop, ax=ax_crop, fraction=0.046, pad=0.04)
    cbar_crop.set_label('HU', fontsize=11)
    
    # =====================================================================
    # Subplot (c): Final ROI
    # =====================================================================
    ax_final = axes[2]
    
    im_final = ax_final.imshow(final_roi, cmap='gray', origin='lower', vmin=0, vmax=1)
    
    # Remove axes
    ax_final.set_xticks([])
    ax_final.set_yticks([])
    ax_final.spines['top'].set_visible(False)
    ax_final.spines['right'].set_visible(False)
    ax_final.spines['bottom'].set_visible(False)
    ax_final.spines['left'].set_visible(False)
    
    # Title
    if roi_size_target:
        title_final = f"(c) ROI final normalizada ({roi_size_target[0]}×{roi_size_target[1]} px)"
    else:
        title_final = f"(c) ROI final normalizada ({final_roi.shape[1]}×{final_roi.shape[0]} px)"
    
    ax_final.set_title(
        title_final,
        fontsize=14,
        fontweight='bold',
        pad=10
    )
    
    # Add colorbar
    cbar_final = plt.colorbar(im_final, ax=ax_final, fraction=0.046, pad=0.04)
    cbar_final.set_label('Valor normalizado [0,1]', fontsize=11)
    
    # =====================================================================
    # Figure Layout
    # =====================================================================
    plt.tight_layout()
    
    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as PNG
        save_path = output_path.with_suffix('.png')
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            format='png'
        )
        logger.info(f"Figure saved: {save_path}")
    
    return fig, axes


# ============================================================================
# Main Execution Function
# ============================================================================

def main(
    data_dir: Path = Path("/workspace/data/LUNA25"),
    output_dir: Path = Path("/workspace/results/figures"),
    csv_path: Optional[Path] = None,
    roi_size_mm: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    target_roi_size: Tuple[int, int, int] = (64, 64, 64),
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    seed: Optional[int] = None
):
    """
    Main function: Generate ROI extraction visualization.
    
    Args:
        data_dir: Path to LUNA25 dataset root
        output_dir: Path to save output figures
        csv_path: Path to CSV with nodule annotations
        roi_size_mm: ROI size in mm (depth, height, width)
        target_roi_size: Target ROI size after resizing (depth, height, width)
        hu_min: Minimum HU for normalization
        hu_max: Maximum HU for normalization
        seed: Random seed for reproducibility
    """
    # Set random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")
    
    logger.info("=" * 70)
    logger.info("NODULE ROI EXTRACTION VISUALIZATION")
    logger.info("=" * 70)
    
    # Load nodule annotations
    if csv_path is None:
        csv_path = data_dir / "dataset_csv" / "LUNA25_Public_Training_Development_Data.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    nodule_df = load_nodule_annotations(csv_path)
    
    # Find volumes
    volumes = find_luna25_volumes(data_dir)
    
    if not volumes:
        raise ValueError("No LUNA25 volumes found")
    
    # Filter volumes with nodules
    series_with_nodules = nodule_df['SeriesInstanceUID'].unique()
    volumes_with_nodules = [
        v for v in volumes
        if any(uid in v.stem for uid in series_with_nodules)
    ]
    
    if not volumes_with_nodules:
        raise ValueError("No volumes found with nodule annotations")
    
    logger.info(f"Found {len(volumes_with_nodules)} volumes with nodules")
    
    # Select random volume
    selected_volume = random.choice(volumes_with_nodules)
    series_uid = selected_volume.stem
    
    logger.info(f"Selected volume: {selected_volume.name}")
    
    # Load CT volume
    volume_3d, image_sitk = load_ct_volume(selected_volume)
    
    # Get nodules for this volume
    nodules = get_series_nodules(nodule_df, series_uid)
    
    if not nodules:
        raise ValueError(f"No nodules found for series {series_uid}")
    
    logger.info(f"Found {len(nodules)} nodule(s) in this volume")
    
    # Select first nodule
    nodule = nodules[0]
    logger.info(f"Using nodule LesionID={nodule['lesion_id']}, label={nodule['label']}")
    logger.info(f"World coordinates: ({nodule['coord_x']:.2f}, {nodule['coord_y']:.2f}, {nodule['coord_z']:.2f}) mm")
    
    # Convert world coordinates to voxel
    world_coord = (nodule['coord_x'], nodule['coord_y'], nodule['coord_z'])
    z_center, y_center, x_center = world_to_voxel(world_coord, image_sitk)
    
    logger.info(f"Voxel coordinates: (z={z_center}, y={y_center}, x={x_center})")
    
    # Get spacing to convert mm to voxels
    spacing = image_sitk.GetSpacing()  # (x, y, z)
    
    # Calculate ROI size in voxels
    roi_size_voxels = (
        int(roi_size_mm[0] / spacing[2]),  # depth (z)
        int(roi_size_mm[1] / spacing[1]),  # height (y)
        int(roi_size_mm[2] / spacing[0])   # width (x)
    )
    
    logger.info(f"ROI size: {roi_size_mm} mm → {roi_size_voxels} voxels")
    
    # =====================================================================
    # Step 1: Extract 3D ROI
    # =====================================================================
    roi_3d, bbox_3d = extract_roi_around_nodule(
        volume_3d,
        (z_center, y_center, x_center),
        roi_size_voxels
    )
    
    # =====================================================================
    # Step 2: Get axial slice containing nodule
    # =====================================================================
    full_slice = volume_3d[z_center, :, :]
    
    # Calculate 2D bounding box for visualization
    half_height = roi_size_voxels[1] // 2
    half_width = roi_size_voxels[2] // 2
    
    y_min = max(0, y_center - half_height)
    y_max = min(full_slice.shape[0], y_center + half_height)
    x_min = max(0, x_center - half_width)
    x_max = min(full_slice.shape[1], x_center + half_width)
    
    bbox_2d = (y_min, y_max, x_min, x_max)
    
    # Extract 2D cropped region
    cropped_slice = full_slice[y_min:y_max, x_min:x_max]
    
    # =====================================================================
    # Step 3: Resize and normalize ROI
    # =====================================================================
    
    # Resize to target size
    resized_roi_3d = resize_roi(roi_3d, target_roi_size, order=3)
    
    # Normalize HU values
    normalized_roi_3d = normalize_hu_range(resized_roi_3d, hu_min, hu_max)
    
    # Get central slice of final ROI
    final_roi_slice = normalized_roi_3d[target_roi_size[0] // 2, :, :]
    
    logger.info(f"Final ROI shape: {normalized_roi_3d.shape}")
    logger.info(f"Final ROI value range: [{normalized_roi_3d.min():.3f}, {normalized_roi_3d.max():.3f}]")
    
    # =====================================================================
    # Step 4: Create visualization
    # =====================================================================
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"nodule_roi_extraction_{series_uid}"
    
    fig, axes = create_roi_extraction_figure(
        full_slice,
        (y_center, x_center),
        cropped_slice,
        final_roi_slice,
        bbox_2d,
        output_path=output_path,
        roi_size_target=(target_roi_size[2], target_roi_size[1]),
        dpi=300
    )
    
    logger.info("=" * 70)
    logger.info("ROI EXTRACTION VISUALIZATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    
    return fig, axes, {
        'volume_name': selected_volume.name,
        'nodule': nodule,
        'voxel_coords': (z_center, y_center, x_center),
        'roi_3d': normalized_roi_3d,
        'roi_shape': normalized_roi_3d.shape
    }


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate academic figures of nodule ROI extraction process"
    )
    parser.add_argument(
        '--data_dir',
        type=Path,
        default="/workspace/data/LUNA25",
        help="Path to LUNA25 dataset"
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default="/workspace/results/figures",
        help="Output directory for figures"
    )
    parser.add_argument(
        '--csv_path',
        type=Path,
        default=None,
        help="Path to CSV with nodule annotations"
    )
    parser.add_argument(
        '--roi_size_mm',
        type=float,
        nargs=3,
        default=[50.0, 50.0, 50.0],
        help="ROI size in mm (depth height width)"
    )
    parser.add_argument(
        '--target_roi_size',
        type=int,
        nargs=3,
        default=[64, 64, 64],
        help="Target ROI size in pixels (depth height width)"
    )
    parser.add_argument(
        '--hu_min',
        type=float,
        default=-1000.0,
        help="Minimum HU for normalization (default: -1000)"
    )
    parser.add_argument(
        '--hu_max',
        type=float,
        default=400.0,
        help="Maximum HU for normalization (default: 400)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Run main function
    fig, axes, results = main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        roi_size_mm=tuple(args.roi_size_mm),
        target_roi_size=tuple(args.target_roi_size),
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        seed=args.seed
    )
    
    plt.show()
