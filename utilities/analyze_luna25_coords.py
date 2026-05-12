#!/usr/bin/env python
"""Deep dive into LUNA25 coordinate system issue."""

import sys
sys.path.insert(0, '/workspace/code/lung-segmentation-tfm')

import numpy as np
import pandas as pd
import SimpleITK as sitk
from os.path import join
from scipy.ndimage import center_of_mass

from config.common import config

print("\n" + "="*90)
print("🔬 LUNA25 Coordinate System Deep Analysis")
print("="*90 + "\n")

# Load LUNA25
df25 = pd.read_csv(config.LUNA25_ANNOTATIONS_CSV_PATH)
scan25 = df25['SeriesInstanceUID'].iloc[0]

# Try multiple scans to verify pattern
print("Testing 5 random LUNA25 scans to understand coordinate mapping:\n")

for test_idx in range(5):
    print(f"─ Test {test_idx + 1}: {df25.iloc[test_idx]['SeriesInstanceUID']}")
    
    scan_uid = df25.iloc[test_idx]['SeriesInstanceUID']
    coord_z_world = float(df25.iloc[test_idx]['CoordZ'])
    coord_y_world = float(df25.iloc[test_idx]['CoordY'])
    coord_x_world = float(df25.iloc[test_idx]['CoordX'])
    
    mha_path = join(config.LUNA25_CT_DIR, f"{scan_uid}.mha")
    
    try:
        sitk_img = sitk.ReadImage(mha_path)
        img_3d = sitk.GetArrayFromImage(sitk_img)
        
        origin = np.array(sitk_img.GetOrigin())  # [X, Y, Z]
        spacing = np.array(sitk_img.GetSpacing())  # [X, Y, Z]
        
        # LUNA25 coords are [Z, Y, X] in world space
        # But SimpleITK GetOrigin/GetSpacing return [X, Y, Z]
        
        # Correct conversion (accounting for axis order):
        # World X = origin[0] + voxel_x * spacing[0]
        # World Y = origin[1] + voxel_y * spacing[1]
        # World Z = origin[2] + voxel_z * spacing[2]
        
        # Reverse: voxel_coord = (world_coord - origin) / spacing
        
        voxel_x = (coord_x_world - origin[0]) / spacing[0]
        voxel_y = (coord_y_world - origin[1]) / spacing[1]
        voxel_z = (coord_z_world - origin[2]) / spacing[2]
        
        print(f"  World coords [Z, Y, X]: [{coord_z_world:.2f}, {coord_y_world:.2f}, {coord_x_world:.2f}]")
        print(f"  Origin [X, Y, Z]: {origin}")
        print(f"  Spacing [X, Y, Z]: {spacing}")
        print(f"  Voxel coords [X, Y, Z]: [{voxel_x:.2f}, {voxel_y:.2f}, {voxel_z:.2f}]")
        print(f"  Image shape [Z, Y, X]: {img_3d.shape}")
        
        # Clamp to valid range
        voxel_z_clamped = int(np.clip(voxel_z, 0, img_3d.shape[0]-1))
        voxel_y_clamped = int(np.clip(voxel_y, 0, img_3d.shape[1]-1))
        voxel_x_clamped = int(np.clip(voxel_x, 0, img_3d.shape[2]-1))
        
        print(f"  Clamped voxel indices [Z, Y, X]: [{voxel_z_clamped}, {voxel_y_clamped}, {voxel_x_clamped}]")
        
        # Check HU value
        hu = img_3d[voxel_z_clamped, voxel_y_clamped, voxel_x_clamped]
        print(f"  HU value at centroid: {hu:.1f}")
        
        # Also check raw indices conversion (IMAGE SPACE vs WORLD SPACE)
        print(f"  ✓ Conversion successful\n")
        
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

print("\n" + "="*90)
print("Manual verification with first scan slice inspection:")
print("="*90 + "\n")

# Load first scan
scan_uid = df25.iloc[0]['SeriesInstanceUID']
mha_path = join(config.LUNA25_CT_DIR, f"{scan_uid}.mha")
sitk_img = sitk.ReadImage(mha_path)
img_3d = sitk.GetArrayFromImage(sitk_img)

print(f"Scan: {scan_uid}")
print(f"Shape: {img_3d.shape}")

# Sample some slices to see what we're getting
for z in [0, img_3d.shape[0]//4, img_3d.shape[0]//2, 3*img_3d.shape[0]//4, -1]:
    slice_data = img_3d[z]
    non_air = np.sum(slice_data > -600)
    print(f"  Slice Z={z:3d}: {non_air:6d} voxels > -600 HU (likely nodule candidates)")

print("\n" + "="*90)
print("✅ Analysis complete!")
print("="*90 + "\n")
