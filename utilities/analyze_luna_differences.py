#!/usr/bin/env python
"""Analyze differences between LUNA16 and LUNA25 datasets."""

import sys
sys.path.insert(0, '/workspace/code/lung-segmentation-tfm')

import numpy as np
import pandas as pd
import ast
import SimpleITK as sitk
from os.path import join
import os

from config.common import config
from nodule_segmentation.foundation_models import utils

print("\n" + "="*90)
print("🔬 LUNA16 vs LUNA25 Dataset Analysis")
print("="*90 + "\n")

# ============================================================================
# LUNA16 ANALYSIS
# ============================================================================
print("█ LUNA16 ANALYSIS")
print("─" * 90)

df16 = pd.read_csv(config.LUNA16_ANNOTATIONS_CSV_PATH)
df16['centroid'] = df16['centroid'].apply(ast.literal_eval)
df16['bbox'] = df16['bbox'].apply(utils.parse_bbox_from_string)

scan16 = df16['scan'].iloc[0]
centroid16 = df16['centroid'].iloc[0]  # Already in voxel coords

print(f"\n1. Data Format:")
print(f"   Scan UID: {scan16}")
print(f"   Centroid format: Voxel coordinates [y, x, z]")
print(f"   Sample centroid: {centroid16}")
print(f"   Sample centroid types: {[type(x).__name__ for x in centroid16]}")

# Load LUNA16 image
mhd_path16 = join(config.LUNA16_CT_DIR, f"{scan16}.mhd")
sitk_img16 = sitk.ReadImage(mhd_path16)
img_3d_16 = sitk.GetArrayFromImage(sitk_img16)

print(f"\n2. Image Properties:")
print(f"   Shape (Z, Y, X): {img_3d_16.shape}")
print(f"   Data type: {img_3d_16.dtype}")
print(f"   HU range: [{img_3d_16.min():.1f}, {img_3d_16.max():.1f}]")
print(f"   Origin: {np.array(sitk_img16.GetOrigin())}")
print(f"   Spacing (X, Y, Z): {np.array(sitk_img16.GetSpacing())}")
print(f"   Direction: {sitk_img16.GetDirection()}")

# Get voxel at centroid
y16, x16, z16 = centroid16
print(f"\n3. Centroid Investigation:")
print(f"   Centroid indices (y, x, z): ({y16}, {x16}, {z16})")
print(f"   Accessing as img_3d[z, y, x]: img_3d[{z16}, {y16}, {x16}]")
print(f"   HU value at centroid: {img_3d_16[z16, y16, x16]:.1f}")

# ============================================================================
# LUNA25 ANALYSIS
# ============================================================================
print("\n" + "─" * 90)
print("█ LUNA25 ANALYSIS")
print("─" * 90)

df25 = pd.read_csv(config.LUNA25_ANNOTATIONS_CSV_PATH)
scan25 = df25['SeriesInstanceUID'].iloc[0]
coords25_world = [df25['CoordZ'].iloc[0], df25['CoordY'].iloc[0], df25['CoordX'].iloc[0]]

print(f"\n1. Data Format:")
print(f"   Scan UID: {scan25}")
print(f"   Coordinate format: World coordinates [Z, Y, X] in mm")
print(f"   Sample coords: {coords25_world}")
print(f"   Sample coords types: {[type(x).__name__ for x in coords25_world]}")

# Load LUNA25 image
mha_path25 = join(config.LUNA25_CT_DIR, f"{scan25}.mha")
sitk_img25 = sitk.ReadImage(mha_path25)
img_3d_25 = sitk.GetArrayFromImage(sitk_img25)

print(f"\n2. Image Properties:")
print(f"   Shape (Z, Y, X): {img_3d_25.shape}")
print(f"   Data type: {img_3d_25.dtype}")
print(f"   HU range: [{img_3d_25.min():.1f}, {img_3d_25.max():.1f}]")
print(f"   Origin: {np.array(sitk_img25.GetOrigin())}")
print(f"   Spacing (X, Y, Z): {np.array(sitk_img25.GetSpacing())}")
print(f"   Direction: {sitk_img25.GetDirection()}")

# Convert world coords to voxel coords
voxel_origin25 = np.array(sitk_img25.GetOrigin())
voxel_spacing25 = np.array(sitk_img25.GetSpacing())

centroid_world = np.array(coords25_world)
centroid_voxel = (centroid_world - voxel_origin25) / voxel_spacing25

print(f"\n3. Coordinate Conversion:")
print(f"   World coords: {coords25_world}")
print(f"   Voxel origin: {voxel_origin25}")
print(f"   Voxel spacing: {voxel_spacing25}")
print(f"   Calculated voxel coords (before clamp): {centroid_voxel}")
print(f"   Calculated voxel coords (after clamp): {np.clip(centroid_voxel, 0, [img_3d_25.shape[i]-1 for i in range(3)])}")

z25_v, y25_v, x25_v = np.clip(centroid_voxel, 0, [img_3d_25.shape[i]-1 for i in range(3)])
z25_v, y25_v, x25_v = int(z25_v), int(y25_v), int(x25_v)

print(f"   Accessing as img_3d[z, y, x]: img_3d[{z25_v}, {y25_v}, {x25_v}]")
print(f"   HU value at centroid: {img_3d_25[z25_v, y25_v, x25_v]:.1f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "─" * 90)
print("█ COMPARISON & POTENTIAL ISSUES")
print("─" * 90)

print(f"\n1. Image Size:")
print(f"   LUNA16: {img_3d_16.shape} vs LUNA25: {img_3d_25.shape}")
if img_3d_16.shape != img_3d_25.shape:
    print(f"   ⚠️  DIFFERENT SHAPES!")

print(f"\n2. Voxel Spacing:")
print(f"   LUNA16 spacing: {np.array(sitk_img16.GetSpacing())}")
print(f"   LUNA25 spacing: {voxel_spacing25}")
if not np.allclose(np.array(sitk_img16.GetSpacing()), voxel_spacing25):
    print(f"   ⚠️  DIFFERENT SPACING!")

print(f"\n3. HU Range:")
print(f"   LUNA16: [{img_3d_16.min():.1f}, {img_3d_16.max():.1f}]")
print(f"   LUNA25: [{img_3d_25.min():.1f}, {img_3d_25.max():.1f}]")

print(f"\n4. Coordinate System:")
print(f"   LUNA16 uses: Voxel coordinates directly from annotations")
print(f"   LUNA25 uses: World coordinates (mm) - need conversion")

print(f"\n5. File Format:")
print(f"   LUNA16: .mhd format")
print(f"   LUNA25: .mha format")

# ============================================================================
# DETAILED WINDOWING ANALYSIS
# ============================================================================
print("\n" + "─" * 90)
print("█ WINDOWING ANALYSIS")
print("─" * 90)

hu16 = img_3d_16[z16, y16, x16]
hu25 = img_3d_25[z25_v, y25_v, x25_v]

print(f"\nLUNA16 Nodule HU value: {hu16:.1f}")
nodule_type16, wl16, ww16 = utils.determine_nodule_type_and_windowing(hu16)
print(f"  Type: {nodule_type16}, Window Level: {wl16}, Window Width: {ww16}")

print(f"\nLUNA25 Nodule HU value: {hu25:.1f}")
nodule_type25, wl25, ww25 = utils.determine_nodule_type_and_windowing(hu25)
print(f"  Type: {nodule_type25}, Window Level: {wl25}, Window Width: {ww25}")

# ============================================================================
# SLICING VERIFICATION
# ============================================================================
print("\n" + "─" * 90)
print("█ SLICE EXTRACTION VERIFICATION")
print("─" * 90)

print(f"\nLUNA16:")
print(f"  Extracting slice at z={z16} from shape {img_3d_16.shape}")
slice16 = img_3d_16[z16:z16+1]
print(f"  Result shape: {slice16.shape}")
if z16 >= img_3d_16.shape[0]:
    print(f"  ⚠️  Z INDEX OUT OF BOUNDS! {z16} >= {img_3d_16.shape[0]}")

print(f"\nLUNA25:")
print(f"  Extracting slice at z={z25_v} from shape {img_3d_25.shape}")
slice25 = img_3d_25[z25_v:z25_v+1]
print(f"  Result shape: {slice25.shape}")
if z25_v >= img_3d_25.shape[0]:
    print(f"  ⚠️  Z INDEX OUT OF BOUNDS! {z25_v} >= {img_3d_25.shape[0]}")

print("\n" + "="*90)
print("✅ Analysis complete!")
print("="*90 + "\n")
