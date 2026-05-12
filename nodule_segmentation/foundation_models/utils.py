"""
Utility functions for LUNA16 foundation models evaluation.

This module contains shared functions that can be reused across different
segmentation models (MedSAM2, SAM3, etc.).
"""

import numpy as np
from scipy import ndimage


def preprocess(image_data, window_level=40, window_width=400):
    """
    Preprocesses CT image with windowing for lung.

    Args:
        image_data: 3D array with HU values
        window_level: Window center (default 40 HU for solid nodules)
        window_width: Window width (default 400 HU for solid nodules)

    Returns:
        Array normalized to [0, 255]
    
    Note:
        Adaptive windowing recommendations:
        - Ground-glass nodules (HU < -300): level=-600, width=1600
        - Solid nodules (-300 <= HU <= 100): level=40, width=400
        - Calcified nodules (HU > 100): level=200, width=1000
    """
    lower_bound = window_level - window_width / 2
    upper_bound = window_level + window_width / 2
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - np.min(image_data_pre))
        / (np.max(image_data_pre) - np.min(image_data_pre))
        * 255.0
    )
    image_data_pre = image_data_pre.astype(np.uint8)
    return image_data_pre


def resize_grayscale_to_rgb_and_resize(gray_array, image_size):
    """
    Resizes 3D grayscale array to RGB with specific size.

    Args:
        gray_array: Array (D, H, W) in grayscale
        image_size: Target size (e.g. 512)

    Returns:
        Array (D, 3, image_size, image_size)
    """
    resized_array = np.zeros(
        (gray_array.shape[0], 3, image_size, image_size), dtype=np.uint8
    )

    for i in range(gray_array.shape[0]):
        # Convert to RGB (3 channels)
        img_rgb = np.stack([gray_array[i]] * 3, axis=-1)  # (H, W, 3)

        # Resize using scipy
        zoom_factors = (image_size / gray_array.shape[1], 
                       image_size / gray_array.shape[2], 
                       1)
        img_resized = ndimage.zoom(img_rgb, zoom_factors, order=1)

        # Transpose to (3, H, W)
        img_array = np.transpose(img_resized, (2, 0, 1))
        resized_array[i] = img_array

    return resized_array


def parse_bbox_from_string(bbox_str):
    """
    Parses bbox string from CSV format to coordinates.
    
    Args:
        bbox_str: String like "(slice(359, 385, None), slice(158, 177, None), slice(36, 45, None))"
    
    Returns:
        tuple: (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    import re
    pattern = r'slice\((\d+), (\d+), None\)'
    matches = re.findall(pattern, bbox_str)
    
    z_min, z_max = int(matches[0][0]), int(matches[0][1])
    y_min, y_max = int(matches[1][0]), int(matches[1][1])
    x_min, x_max = int(matches[2][0]), int(matches[2][1])
    
    return (z_min, z_max, y_min, y_max, x_min, x_max)


def extract_nodules_info(mask_gt):
    """
    Extracts information of individual nodules from GT mask.

    Args:
        mask_gt: Binary 3D mask (Z, Y, X) with all nodules

    Returns:
        list: List of dictionaries with info for each nodule:
              - nodule_id: Nodule ID
              - bbox: (z_min, z_max, y_min, y_max, x_min, x_max)
              - centroid: (z, y, x) centroid coordinates
              - mask: Binary mask of this nodule only
    """
    # Label connected components (each nodule = component)
    labeled_mask, num_nodules = ndimage.label(mask_gt)

    nodules_info = []

    for nodule_id in range(1, num_nodules + 1):
        # Extract mask of this specific nodule
        nodule_mask = (labeled_mask == nodule_id)

        # Find bbox
        coords = np.where(nodule_mask)
        if len(coords[0]) == 0:
            continue

        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()

        # Calculate centroid
        z_cent = int(np.mean(coords[0]))
        y_cent = int(np.mean(coords[1]))
        x_cent = int(np.mean(coords[2]))

        nodules_info.append({
            'nodule_id': nodule_id,
            'bbox': (z_min, z_max, y_min, y_max, x_min, x_max),
            'centroid': (z_cent, y_cent, x_cent),
            'mask': nodule_mask,
            'num_voxels': nodule_mask.sum()
        })

    return nodules_info


def determine_nodule_type_and_windowing(hu_value):
    """
    Determines nodule type and optimal windowing parameters based on HU value.
    
    Args:
        hu_value: HU value at nodule centroid
    
    Returns:
        tuple: (nodule_type, window_level, window_width)
    """
    if hu_value < -300:
        # Ground-glass nodule (low density)
        return "ground-glass", -600, 1600
    elif hu_value > 100:
        # Calcified nodule (high density)
        return "calcified", 200, 1000
    else:
        # Solid/semi-solid nodule
        return "solid", 40, 400
