"""nnU-Net Wrapper for Foundation Models Comparison.

This module provides a wrapper around the trained nnU-Net model to make it compatible
with the foundation models interface. It extracts 64×64 patches (as used in training)
and generates segmentation predictions.
"""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import SimpleITK as sitk

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from config.common import config
from nodule_segmentation.models.models_2d import SegmentationModels2D
from utilities import utils


class NNUNetWrapper:
    """Wrapper for nnU-Net model to segment nodules using 64×64 patches.
    
    This class handles:
    - Loading the trained nnU-Net model
    - Extracting patches from full CT volumes
    - Running inference
    - Returning predictions in compatible format
    
    Args:
        checkpoint_path: Path to nnU-Net model checkpoint
        device: Device to run inference on ("cuda:0", "cpu", etc)
        patch_size: Size of patches to extract (default: 64)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        patch_size: int = 64,
    ):
        """Initialize nnU-Net wrapper.
        
        Args:
            checkpoint_path: Path to saved nnU-Net checkpoint
            device: Device for inference
            patch_size: Patch size in pixels
        """
        self.device = device
        self.patch_size = patch_size
        
        # Initialize nnU-Net model
        self.model = SegmentationModels2D.get_model(
            model_name="nnUNet",
            encoder_name="",  # nnU-Net doesn't use pretrained encoders
            encoder_weights=None
        )
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"✅ nnU-Net loaded from {checkpoint_path}")
    
    def preprocess_patch(
        self,
        patch: np.ndarray,
    ) -> torch.Tensor:
        """Preprocess patch for model input.
        
        Args:
            patch: CT patch, shape (H, W) or (C, H, W) with HU values
        
        Returns:
            Preprocessed tensor ready for model input
        """
        # Ensure 2D
        if patch.ndim == 3:
            patch = patch[0]  # Take first channel if multi-channel
        
        # Clip HU values
        patch = np.clip(patch, -1000, 400)
        
        # Normalize to [0, 1]
        patch = (patch + 1000) / 1400.0
        patch = np.clip(patch, 0, 1)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        patch = patch[np.newaxis, ...]
        
        # Convert to tensor and add batch dimension
        patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)  # (1, 1, H, W)
        
        return patch_tensor.to(self.device)
    
    def extract_patch_from_volume(
        self,
        ct_data: np.ndarray,
        centroid: tuple,
        src_voxel_origin: np.ndarray,
        src_world_matrix: np.ndarray,
        src_voxel_spacing: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Extract 64×64 patch from CT volume centered at nodule.
        
        Args:
            ct_data: 3D CT volume
            centroid: Nodule centroid coordinates (z, y, x) in voxel space
            src_voxel_origin: Voxel space origin
            src_world_matrix: World to voxel transformation matrix
            src_voxel_spacing: Voxel spacing
        
        Returns:
            Extracted patch (64, 64) or None if extraction fails
        """
        try:
            # Extract 64×64 patch using same logic as training
            output_shape = (1, self.patch_size, self.patch_size)
            voxel_spacing = (50.0 / self.patch_size, 50.0 / self.patch_size, 50.0 / self.patch_size)
            
            # Convert centroid tuple to numpy array if needed
            coord = np.array(centroid)
            
            patch = utils.extract_patch(
                ct_data=ct_data,
                coord=coord,
                src_voxel_origin=src_voxel_origin,
                src_world_matrix=src_world_matrix,
                src_voxel_spacing=src_voxel_spacing,
                output_shape=output_shape,
                voxel_spacing=voxel_spacing,
                rotations=None,  # No augmentation for inference
                translations=None,
                coord_space_world=False,
                mode="2D",
                model_name="nnunet-2d",
            )
            
            # patch shape: (1, 64, 64) from extract_patch for 2D
            return patch
            
        except Exception as e:
            print(f"❌ Error extracting patch: {e}")
            return None
    
    @torch.inference_mode()
    def predict(
        self,
        ct_3d: np.ndarray,
        centroid_y: int,
        centroid_x: int,
        centroid_z: int,
        bbox: tuple,
        src_voxel_origin: np.ndarray,
        src_world_matrix: np.ndarray,
        src_voxel_spacing: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> np.ndarray:
        """Run inference on a nodule in 3D CT volume.
        
        Args:
            ct_3d: 3D CT volume
            centroid_y, centroid_x, centroid_z: Nodule centroid coordinates
            bbox: Bounding box (not used by nnU-Net, for compatibility)
            src_voxel_origin: Voxel space origin
            src_world_matrix: World to voxel transformation matrix
            src_voxel_spacing: Voxel spacing
            confidence_threshold: Threshold for binary prediction
        
        Returns:
            Segmentation mask (1, H, W) with values [0, 1]
        """
        # Determine which z to use (handle different coordinate orderings)
        centroid = (int(centroid_z), int(centroid_y), int(centroid_x))
        
        # Extract patch
        patch = self.extract_patch_from_volume(
            ct_3d,
            centroid,
            src_voxel_origin,
            src_world_matrix,
            src_voxel_spacing,
        )
        
        if patch is None:
            # Return empty prediction
            return np.zeros((1, self.patch_size, self.patch_size), dtype=np.float32)
        
        # Preprocess patch
        patch_tensor = self.preprocess_patch(patch)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(patch_tensor)  # (1, 2, 64, 64) - binary segmentation
        
        # Get probability for class 1 (nodule)
        probs = torch.softmax(logits, dim=1)  # (1, 2, 64, 64)
        nodule_prob = probs[:, 1, :, :].cpu().numpy()  # (1, 64, 64)
        
        # Apply threshold
        pred_mask = (nodule_prob > confidence_threshold).astype(np.float32)
        
        return pred_mask
