"""
Visualization of CT Lung Windowing Effect

This module generates academic-quality figures demonstrating the effect of 
applying lung windowing to CT scans. The script integrates with the CAD system's
preprocessing pipeline to ensure consistency with actual processing.

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


# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CT Windowing Functions (Aligned with CAD System)
# ============================================================================

def apply_ct_windowing(
    image_data: np.ndarray,
    window_level: float = -750,
    window_width: float = 1500
) -> np.ndarray:
    """
    Apply CT windowing to normalize HU values to [0, 255].
    
    This function replicates the preprocessing used in the CAD system's
    foundation models (SAM2, SAM3, MedSAM2).
    
    Args:
        image_data: 3D CT volume in Hounsfield Units (HU)
        window_level: Window center (default 40 HU for solid nodules)
        window_width: Window width (default 400 HU for solid nodules)
    
    Returns:
        Normalized image array in [0, 255] as uint8
    
    Notes:
        Windowing formula:
            lower_bound = window_level - window_width / 2 = 40 - 200 = -160
            upper_bound = window_level + window_width / 2 = 40 + 200 = 240
            
        This window captures:
        - Ground-glass components (around -300 to -500 HU)
        - Soft tissue (0 to 100 HU)
        - Part of lung tissue
    """
    # Calculate bounds
    lower_bound = window_level - window_width / 2
    upper_bound = window_level + window_width / 2
    
    logger.info(f"Applying lung windowing: level={window_level}, width={window_width}")
    logger.info(f"HU range: [{lower_bound:.1f}, {upper_bound:.1f}]")
    
    # Clip to window bounds
    windowed = np.clip(image_data, lower_bound, upper_bound)
    
    # Normalize to [0, 255]
    windowed_min = np.min(windowed)
    windowed_max = np.max(windowed)
    
    if windowed_max > windowed_min:
        normalized = (windowed - windowed_min) / (windowed_max - windowed_min) * 255.0
    else:
        normalized = np.zeros_like(windowed)
    
    return normalized.astype(np.uint8)


def apply_clip_and_scale(
    image_data: np.ndarray,
    clip_min: float = -1024,
    clip_max: float = 3072
) -> np.ndarray:
    """
    Alternative normalization: clip to range then scale to [0, 1].
    
    Args:
        image_data: Input array
        clip_min: Minimum clipping value (default HU min)
        clip_max: Maximum clipping value (default HU max)
    
    Returns:
        Normalized array in [0, 1]
    """
    clipped = np.clip(image_data, clip_min, clip_max)
    normalized = (clipped - clip_min) / (clip_max - clip_min)
    return normalized.astype(np.float32)


# ============================================================================
# File I/O Functions
# ============================================================================

def find_luna25_volumes(data_dir: Path) -> list:
    """
    Find all LUNA25 CT volume files (MHA, DICOM, or Nifti).
    
    Args:
        data_dir: Path to LUNA25 root directory
    
    Returns:
        List of volume file paths (either .mha files or DICOM directories)
    """
    # Check if data_dir already points to luna25_images or if we need to append it
    if data_dir.name == "luna25_images":
        luna25_images = data_dir
    else:
        luna25_images = data_dir / "luna25_images"
    
    logger.info(f"Looking for volumes in: {luna25_images}")
    
    if not luna25_images.exists():
        raise FileNotFoundError(f"LUNA25 images directory not found: {luna25_images}")
    
    # First, look for MHA/NIfTI files
    mha_files = list(luna25_images.glob("*.mha"))
    nii_files = list(luna25_images.glob("*.nii.gz")) + list(luna25_images.glob("*.nii"))
    
    volumes = mha_files + nii_files
    
    # If no MHA/NIfTI files found, look for DICOM directories
    if not volumes:
        all_items = list(luna25_images.glob("*"))
        volumes = [v for v in all_items if v.is_dir()]
    
    logger.info(f"Found {len(volumes)} LUNA25 volumes")
    
    if volumes:
        logger.info(f"First volume example: {volumes[0].name}")
    
    return volumes


def load_ct_volume(volume_path: Path) -> Tuple[np.ndarray, sitk.Image]:
    """
    Load 3D CT volume from MHA, NIfTI, or DICOM directory.
    
    Args:
        volume_path: Path to volume file (.mha, .nii, .nii.gz) or DICOM directory
    
    Returns:
        Tuple of (3D numpy array (Z, Y, X) in Hounsfield Units, SimpleITK Image object)
    """
    volume_path = Path(volume_path)
    
    # Check if it's a file (MHA, NIfTI) or directory (DICOM)
    if volume_path.is_file():
        # Load MHA or NIfTI file
        logger.info(f"Loading volume from file: {volume_path.name}")
        image_sitk = sitk.ReadImage(str(volume_path))
    
    elif volume_path.is_dir():
        # Load DICOM series
        logger.info(f"Loading DICOM series from: {volume_path.name}")
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(volume_path))
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {volume_path}")
        
        logger.info(f"Found {len(dicom_files)} DICOM slices")
        reader.SetFileNames(dicom_files)
        image_sitk = reader.Execute()
    
    else:
        raise FileNotFoundError(f"Volume path does not exist: {volume_path}")
    
    # Convert to numpy (Z, Y, X)
    image_array = sitk.GetArrayFromImage(image_sitk)
    
    # Ensure float32 for proper HU values
    image_array = image_array.astype(np.float32)
    
    logger.info(f"Volume shape: {image_array.shape}, dtype: {image_array.dtype}")
    logger.info(f"HU range: [{image_array.min():.1f}, {image_array.max():.1f}]")
    
    return image_array, image_sitk


def extract_central_axial_slice(volume_3d: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Extract the central axial (Z) slice from 3D volume.
    
    Args:
        volume_3d: 3D volume (Z, Y, X)
    
    Returns:
        Tuple of (2D slice, slice index)
    """
    center_z = volume_3d.shape[0] // 2
    axial_slice = volume_3d[center_z, :, :]
    
    logger.info(f"Extracted central axial slice at z={center_z}")
    
    return axial_slice, center_z


# ============================================================================
# Nodule Annotation Functions
# ============================================================================

def load_nodule_annotations(csv_path: Path) -> pd.DataFrame:
    """
    Load LUNA25 nodule annotations from CSV file.
    
    Args:
        csv_path: Path to CSV file with nodule annotations
    
    Returns:
        DataFrame with nodule annotations
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} nodule annotations from CSV")
    
    return df


def get_series_nodules(df: pd.DataFrame, series_uid: str) -> List[Dict]:
    """
    Get all nodule annotations for a specific SeriesInstanceUID.
    
    Args:
        df: DataFrame with nodule annotations
        series_uid: SeriesInstanceUID to filter by
    
    Returns:
        List of nodule dictionaries with coordinates
    """
    # Filter by SeriesInstanceUID
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
    
    logger.info(f"Found {len(nodules)} nodule(s) in series {series_uid[:20]}...")
    
    return nodules


def world_to_voxel(world_coord: Tuple[float, float, float], 
                   image_sitk: sitk.Image) -> Tuple[int, int, int]:
    """
    Convert world coordinates (mm) to voxel indices.
    
    Args:
        world_coord: (x, y, z) in mm (world coordinates)
        image_sitk: SimpleITK image object with origin and spacing info
    
    Returns:
        (z_idx, y_idx, x_idx) voxel indices for numpy array indexing
    """
    # Get image metadata
    origin = image_sitk.GetOrigin()  # (x, y, z) in mm
    spacing = image_sitk.GetSpacing()  # (x, y, z) in mm/voxel
    
    # Convert world to continuous index
    x_world, y_world, z_world = world_coord
    
    x_idx = (x_world - origin[0]) / spacing[0]
    y_idx = (y_world - origin[1]) / spacing[1]
    z_idx = (z_world - origin[2]) / spacing[2]
    
    # Round to nearest integer
    x_idx = int(round(x_idx))
    y_idx = int(round(y_idx))
    z_idx = int(round(z_idx))
    
    # Return in numpy indexing order (Z, Y, X)
    return z_idx, y_idx, x_idx


def create_nodule_bbox(nodule: Dict, 
                       image_sitk: sitk.Image,
                       slice_z: int,
                       bbox_size_mm: float = 30.0,
                       z_tolerance: int = 5) -> Optional[Tuple[int, int, int, int]]:
    """
    Create bounding box for nodule if it's visible in the given slice.
    
    Args:
        nodule: Dictionary with nodule world coordinates
        image_sitk: SimpleITK image object
        slice_z: Z index of the slice to check
        bbox_size_mm: Size of bounding box in mm (default 30mm = ~6cm diameter)
        z_tolerance: Number of slices tolerance for nodule visibility
    
    Returns:
        (y_min, x_min, y_max, x_max) bounding box or None if nodule not in slice
    """
    # Convert nodule world coordinates to voxel
    world_coord = (nodule['coord_x'], nodule['coord_y'], nodule['coord_z'])
    z_idx, y_idx, x_idx = world_to_voxel(world_coord, image_sitk)
    
    # Check if nodule is in this slice (with tolerance)
    if abs(z_idx - slice_z) > z_tolerance:
        return None
    
    logger.info(f"Nodule found at voxel: (z={z_idx}, y={y_idx}, x={x_idx})")
    logger.info(f"Nodule in target slice {slice_z} (distance: {abs(z_idx - slice_z)} slices)")
    
    # Get spacing to convert mm to voxels
    spacing = image_sitk.GetSpacing()  # (x, y, z)
    
    # Calculate bbox size in voxels
    half_size_x = int(bbox_size_mm / spacing[0] / 2)
    half_size_y = int(bbox_size_mm / spacing[1] / 2)
    
    # Create bounding box (in 2D slice coordinates)
    x_min = max(0, x_idx - half_size_x)
    x_max = x_idx + half_size_x
    y_min = max(0, y_idx - half_size_y)
    y_max = y_idx + half_size_y
    
    # Clip to image bounds
    image_size = image_sitk.GetSize()  # (x, y, z)
    x_max = min(x_max, image_size[0] - 1)
    y_max = min(y_max, image_size[1] - 1)
    
    logger.info(f"Bounding box (2D): x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    return (y_min, x_min, y_max, x_max)


# ============================================================================
# Visualization Functions
# ============================================================================

def create_windowing_comparison_figure(
    original_slice: np.ndarray,
    windowed_slice: np.ndarray,
    window_level: float = 40,
    window_width: float = 400,
    output_path: Optional[Path] = None,
    nodule_bbox: Optional[Tuple[int, int, int, int]] = None,
    dpi: int = 300
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create an academic-quality figure comparing original and windowed CT slices.
    
    Args:
        original_slice: Original CT slice in HU
        windowed_slice: Windowed CT slice (0-255)
        window_level: Window level used (for reference)
        window_width: Window width used (for reference)
        output_path: Path to save figure (if None, won't save)
        nodule_bbox: Optional nodule bounding box (y_min, x_min, y_max, x_max)
        dpi: Output DPI resolution
    
    Returns:
        Tuple of (figure, axes array)
    """
    # Create figure with high DPI for academic publication
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=dpi)
    
    # =====================================================================
    # Subplot (a): Original CT Image
    # =====================================================================
    ax_orig = axes[0]
    
    # Display with grayscale colormap
    im_orig = ax_orig.imshow(original_slice, cmap='gray', origin='lower')
    
    # Draw nodule bbox if provided
    if nodule_bbox is not None:
        y_min, x_min, y_max, x_max = nodule_bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2.5, edgecolor='red', facecolor='none',
            linestyle='--', alpha=0.8
        )
        ax_orig.add_patch(rect)
    
    # Remove axes
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])
    ax_orig.spines['top'].set_visible(False)
    ax_orig.spines['right'].set_visible(False)
    ax_orig.spines['bottom'].set_visible(False)
    ax_orig.spines['left'].set_visible(False)
    
    # Title (paper-style, small font)
    ax_orig.set_title(
        "(a) Imagen CT original",
        fontsize=14, fontweight='bold', pad=10
    )
    
    # Add colorbar
    cbar_orig = plt.colorbar(im_orig, ax=ax_orig, fraction=0.046, pad=0.04)
    cbar_orig.set_label('HU', fontsize=11)
    
    # =====================================================================
    # Subplot (b): Windowed CT Image
    # =====================================================================
    ax_wind = axes[1]
    
    im_wind = ax_wind.imshow(windowed_slice, cmap='gray', origin='lower', vmin=0, vmax=255)
    
    # Draw nodule bbox if provided
    if nodule_bbox is not None:
        y_min, x_min, y_max, x_max = nodule_bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2.5, edgecolor='red', facecolor='none',
            linestyle='--', alpha=0.8
        )
        ax_wind.add_patch(rect)
    
    # Remove axes
    ax_wind.set_xticks([])
    ax_wind.set_yticks([])
    ax_wind.spines['top'].set_visible(False)
    ax_wind.spines['right'].set_visible(False)
    ax_wind.spines['bottom'].set_visible(False)
    ax_wind.spines['left'].set_visible(False)
    
    # Title (paper-style)
    ax_wind.set_title(
        f"(b) Ventana pulmonar (L={window_level}, W={window_width})",
        fontsize=14, fontweight='bold', pad=10
    )
    
    # Add colorbar
    cbar_wind = plt.colorbar(im_wind, ax=ax_wind, fraction=0.046, pad=0.04)
    cbar_wind.set_label('Valor normalizado', fontsize=11)
    
    # =====================================================================
    # Figure Layout
    # =====================================================================
    plt.tight_layout()
    
    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            format='png'
        )
        logger.info(f"Figure saved to: {output_path}")
    
    return fig, axes


# ============================================================================
# Main Execution Function
# ============================================================================

def main(
    data_dir: Path = Path("/workspace/data/LUNA25"),
    output_dir: Path = Path("/workspace/results/figures"),
    csv_path: Optional[Path] = None,
    window_level: float = 40,
    window_width: float = 400,
    seed: Optional[int] = None,
    nodule_bbox: Optional[Tuple[int, int, int, int]] = None,
    bbox_size_mm: float = 30.0
):
    """
    Main function: Load LUNA25 volume, apply windowing, generate figure.
    
    Args:
        data_dir: Path to LUNA25 dataset root
        output_dir: Path to save output figures
        csv_path: Path to CSV with nodule annotations (if None, uses default)
        window_level: Window level for CT windowing
        window_width: Window width for CT windowing
        seed: Random seed for reproducibility
        nodule_bbox: Optional manual nodule bounding box (overrides auto-detection)
        bbox_size_mm: Size of bounding box in mm (default 30mm)
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")
    
    logger.info("=" * 70)
    logger.info("CT LUNG WINDOWING VISUALIZATION")
    logger.info("=" * 70)
    
    # Load nodule annotations if CSV provided
    nodule_df = None
    if csv_path is None:
        csv_path = data_dir / "dataset_csv" / "LUNA25_Public_Training_Development_Data.csv"
    
    if csv_path.exists():
        nodule_df = load_nodule_annotations(csv_path)
    else:
        logger.warning(f"CSV file not found: {csv_path}. Nodules won't be annotated.")
    
    # Find volumes
    volumes = find_luna25_volumes(data_dir)
    
    if not volumes:
        raise ValueError("No LUNA25 volumes found")
    
    # Select random volume (prefer volumes with nodules if CSV available)
    if nodule_df is not None:
        # Get list of series UIDs with nodules
        series_with_nodules = nodule_df['SeriesInstanceUID'].unique()
        
        # Filter volumes to those with nodules
        volumes_with_nodules = [
            v for v in volumes 
            if any(uid in v.stem for uid in series_with_nodules)
        ]
        
        if volumes_with_nodules:
            logger.info(f"Found {len(volumes_with_nodules)} volumes with nodule annotations")
            selected_volume = random.choice(volumes_with_nodules)
        else:
            logger.warning("No volumes found with nodule annotations, selecting random volume")
            selected_volume = random.choice(volumes)
    else:
        selected_volume = random.choice(volumes)
    
    logger.info(f"Selected volume: {selected_volume.name}")
    
    # Extract SeriesInstanceUID from filename (remove .mha extension)
    series_uid = selected_volume.stem
    
    # Load CT volume
    volume_3d, image_sitk = load_ct_volume(selected_volume)
    
    # Extract central axial slice
    original_slice, slice_idx = extract_central_axial_slice(volume_3d)
    
    # Find nodules in this volume and create bbox if present
    if nodule_df is not None and nodule_bbox is None:
        nodules = get_series_nodules(nodule_df, series_uid)
        
        if nodules:
            # Try to find a nodule visible in the central slice first
            for nodule in nodules:
                bbox = create_nodule_bbox(
                    nodule, 
                    image_sitk, 
                    slice_idx,
                    bbox_size_mm=bbox_size_mm
                )
                if bbox is not None:
                    nodule_bbox = bbox
                    logger.info(f"Using nodule LesionID={nodule['lesion_id']} for annotation")
                    break
            
            # If no nodule in central slice, use the first nodule's slice
            if nodule_bbox is None:
                logger.info("No nodules visible in central slice, using nodule's slice instead")
                first_nodule = nodules[0]
                world_coord = (first_nodule['coord_x'], first_nodule['coord_y'], first_nodule['coord_z'])
                z_idx, y_idx, x_idx = world_to_voxel(world_coord, image_sitk)
                
                # Update slice to nodule's position
                if 0 <= z_idx < volume_3d.shape[0]:
                    slice_idx = z_idx
                    original_slice = volume_3d[slice_idx, :, :]
                    logger.info(f"Switched to slice z={slice_idx} containing nodule")
                    
                    # Create bbox for this nodule
                    nodule_bbox = create_nodule_bbox(
                        first_nodule,
                        image_sitk,
                        slice_idx,
                        bbox_size_mm=bbox_size_mm,
                        z_tolerance=0  # Exact match since we're on the nodule's slice
                    )
                    if nodule_bbox:
                        logger.info(f"Using nodule LesionID={first_nodule['lesion_id']} for annotation")
        else:
            logger.info("No nodules found in this volume")
    
    # Apply windowing
    windowed_slice = apply_ct_windowing(
        original_slice,
        window_level=window_level,
        window_width=window_width
    )
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figure
    output_path = output_dir / f"ct_windowing_comparison_{selected_volume.name}.png"
    
    fig, axes = create_windowing_comparison_figure(
        original_slice,
        windowed_slice,
        window_level=window_level,
        window_width=window_width,
        output_path=output_path,
        nodule_bbox=nodule_bbox,
        dpi=300
    )
    
    logger.info("=" * 70)
    logger.info("FIGURE GENERATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    
    return fig, axes, {
        'original_slice': original_slice,
        'windowed_slice': windowed_slice,
        'volume_name': selected_volume.name,
        'slice_index': slice_idx
    }


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate academic figures of CT lung windowing effect"
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
        help="Path to CSV with nodule annotations (optional)"
    )
    parser.add_argument(
        '--bbox_size_mm',
        type=float,
        default=30.0,
        help="Bounding box size in mm (default: 30mm)"
    )
    parser.add_argument(
        '--window_level',
        type=float,
        default=-750,
        help="Window level (default: -750 for lung window)"
    )
    parser.add_argument(
        '--window_width',
        type=float,
        default=1500,
        help="Window width (default: 1500 for lung window)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Run main function
    fig, axes, results = main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        window_level=args.window_level,
        window_width=args.window_width,
        seed=args.seed,
        bbox_size_mm=args.bbox_size_mm
    )
    
    plt.show()
