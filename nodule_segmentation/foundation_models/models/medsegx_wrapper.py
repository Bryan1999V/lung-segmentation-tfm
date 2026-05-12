"""
MedSegX Model Wrapper for LUNA16 Evaluation
============================================

Encapsulates MedSegX model for nodule segmentation.
Provides a clean interface for preprocessing and inference.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, '/workspace/code/MedSegX')
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from model.medsegx import MedSegX

from nodule_segmentation.foundation_models import utils


class MedSegXWrapper:
    """
    Wrapper class for MedSegX model.

    This class encapsulates the MedSegX model and provides
    methods for preprocessing CT images and predicting nodule masks.
    """

    def __init__(self, checkpoint_path, sam_checkpoint_path, device='cuda', 
                 model_type='vit_b', image_size=256, 
                 bottleneck_dim=16, embedding_dim=16, expert_num=4):
        """
        Initialize MedSegX wrapper.

        Args:
            checkpoint_path: Path to MedSegX model checkpoint
            sam_checkpoint_path: Path to SAM base checkpoint
            device: Device to run model on ('cuda' or 'cpu')
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            image_size: Target image size for model (default: 256)
            bottleneck_dim: Adapter bottleneck dimension (default: 16)
            embedding_dim: Modal and organ embedding dimension (default: 16)
            expert_num: Number of MoE experts (default: 4)
        """
        self.checkpoint_path = checkpoint_path
        self.sam_checkpoint_path = sam_checkpoint_path
        self.device = device
        self.model_type = model_type
        self.image_size = image_size
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.expert_num = expert_num
        self.model = None

        # SAM normalization constants (pixel values in [0, 255])
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the MedSegX model."""
        # Build SAM base model
        sam_model = sam_model_registry[self.model_type](
            image_size=self.image_size,
            keep_resolution=True,
            checkpoint=self.sam_checkpoint_path
        )

        # Build MedSegX model with adapters
        self.model = MedSegX(
            sam_model,
            bottleneck_dim=self.bottleneck_dim,
            embedding_dim=self.embedding_dim,
            expert_num=self.expert_num
        ).to(self.device)

        # Load MedSegX checkpoint
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_parameters(checkpoint["model"])

        self.model.eval()

        # Move normalization constants to device
        if self.device == 'cuda':
            self.pixel_mean = self.pixel_mean.cuda()
            self.pixel_std = self.pixel_std.cuda()

        # Transform for resizing
        self.box_transform = ResizeLongestSide(self.image_size)

    def preprocess_image(self, img_3d, window_level=40, window_width=400):
        """
        Preprocess 3D CT image for MedSegX inference.

        Args:
            img_3d: 3D CT scan in HU values (Z, Y, X)
            window_level: Window center for CT windowing
            window_width: Window width for CT windowing

        Returns:
            tuple: (preprocessed_tensor, video_height, video_width)
                - preprocessed_tensor: Normalized tensor (D, 3, H, W) on device
                - video_height: Original height before resizing
                - video_width: Original width before resizing
        """
        # 1. Apply CT windowing
        img_preprocessed = utils.preprocess(img_3d, window_level=window_level, window_width=window_width)

        # 2. Get original dimensions
        video_height = img_preprocessed.shape[1]
        video_width = img_preprocessed.shape[2]

        # 3. Resize to target size and convert to RGB
        if video_height != self.image_size or video_width != self.image_size:
            img_resized = utils.resize_grayscale_to_rgb_and_resize(img_preprocessed, self.image_size)
        else:
            # Convert grayscale to RGB
            img_resized = np.repeat(img_preprocessed[:, np.newaxis, :, :], 3, axis=1)

        # 4. Keep in [0, 255] range (MedSegX expects this)
        # img_resized is already in [0, 255] from preprocessing

        # 5. Move to device
        try:
            img_tensor = torch.from_numpy(img_resized).float().to(self.device)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                raise MemoryError(f"CUDA out of memory during preprocessing")
            raise

        return img_tensor, video_height, video_width

    @torch.inference_mode()
    def predict(self, img_tensor, video_height, video_width,
                centroid_z, centroid_y, centroid_x, bbox=None,
                confidence_threshold=0.5, z_range=None):
        """
        Predict nodule segmentation mask using MedSegX.

        Args:
            img_tensor: Preprocessed image tensor (D, 3, H, W) on device in [0, 255]
            video_height: Original video height
            video_width: Original video width
            centroid_z: Z coordinate of nodule centroid
            centroid_y: Y coordinate of nodule centroid
            centroid_x: X coordinate of nodule centroid
            bbox: Bounding box (y_min, x_min, y_max, x_max) for 2D slice
            confidence_threshold: Threshold for mask confidence (default: 0.5)
            z_range: Optional (z_min, z_max) tuple to limit slice processing

        Returns:
            np.ndarray: Predicted 3D binary mask (D, H, W)
        """
        # Initialize output mask
        num_slices = img_tensor.shape[0]
        pred_mask_3d = np.zeros((num_slices, video_height, video_width), dtype=bool)

        # Scale bbox to image_size coordinates
        scale_y = self.image_size / video_height
        scale_x = self.image_size / video_width

        # Bbox prompt: [x_min, y_min, x_max, y_max] in scaled coordinates
        box = torch.tensor([[
            bbox[1] * scale_x,  # x_min
            bbox[0] * scale_y,  # y_min
            bbox[3] * scale_x,  # x_max
            bbox[2] * scale_y   # y_max
        ]], dtype=torch.float32, device=self.device)

        # MedSegX metadata (CT lung nodule task)
        modal = torch.tensor([2], device=self.device)  # CT modality
        organ_1 = torch.tensor([1], device=self.device)  # body
        organ_2 = torch.tensor([3], device=self.device)  # chest
        organ_3 = torch.tensor([1], device=self.device)  # left lung (approximation)
        organ_4 = torch.tensor([0], device=self.device)  # task 0 (generic)
        organ = (organ_1, organ_2, organ_3, organ_4)

        # Define slice range: use z_range if provided, otherwise fallback to ±10 slices
        if z_range is not None:
            start_slice, end_slice = z_range
            start_slice = max(0, start_slice)
            end_slice = min(num_slices, end_slice + 1)
        else:
            # Fallback to ±10 slices around centroid
            start_slice = max(0, centroid_z - 10)
            end_slice = min(num_slices, centroid_z + 11)

        for slice_idx in range(start_slice, end_slice):
            # Prepare data dict for this slice
            data = {
                'img': img_tensor[slice_idx:slice_idx+1],  # (1, 3, H, W)
                'box': box,  # (1, 4)
                'modal': modal,  # (1,)
                'organ': organ  # tuple of tensors
            }

            try:
                # Forward pass - MedSegX generates 3 candidate masks
                # Use simple forward call, then select best mask using heuristics
                mask_predictions = self.model(data)  # (1, 3, H, W)

                # Resize to original resolution if needed
                if mask_predictions.shape[-1] != video_height:
                    mask_predictions = F.interpolate(
                        mask_predictions,
                        size=(video_height, video_width),
                        mode='bilinear',
                        antialias=True
                    )

                # Apply sigmoid
                mask_prob = torch.sigmoid(mask_predictions)  # (1, 3, H, W)

                # SAM3 produces 3 masks with different granularities
                # Without GT, we use a heuristic: select mask that best fits the bbox
                # Strategy: Choose mask with highest probability within bbox region

                # Get bbox coordinates (already in scaled coordinates for image_size)
                bbox_2d = box[0].cpu().numpy()  # (4,) [x_min, y_min, x_max, y_max]

                # Convert to pixel coordinates in the resized mask space
                # bbox is in image_size coordinates, mask_prob is in (video_height, video_width)
                scale_x = mask_prob.shape[3] / self.image_size
                scale_y = mask_prob.shape[2] / self.image_size

                x_min = int(bbox_2d[0] * scale_x)
                y_min = int(bbox_2d[1] * scale_y)
                x_max = int(bbox_2d[2] * scale_x)
                y_max = int(bbox_2d[3] * scale_y)

                # Clamp to image bounds
                x_min = max(0, min(x_min, mask_prob.shape[3] - 1))
                x_max = max(0, min(x_max, mask_prob.shape[3] - 1))
                y_min = max(0, min(y_min, mask_prob.shape[2] - 1))
                y_max = max(0, min(y_max, mask_prob.shape[2] - 1))

                # Calculate mean probability within bbox for each mask
                best_score = -1
                best_idx = 0
                for idx in range(3):
                    mask_prob_roi = mask_prob[0, idx, y_min:y_max+1, x_min:x_max+1]
                    mean_prob = mask_prob_roi.mean().item()

                    if mean_prob > best_score:
                        best_score = mean_prob
                        best_idx = idx

                # Use best mask
                mask = (mask_prob[0, best_idx] > confidence_threshold).cpu().numpy()
                pred_mask_3d[slice_idx] = mask

            except Exception as e:
                print(f"Error processing slice {slice_idx}: {e}")
                continue

        # Post-process: keep only largest connected component near centroid
        # For single slice, use 2D processing; otherwise use 3D
        if pred_mask_3d.any():
            # Check if processing single slice
            num_active_slices = np.sum(pred_mask_3d.any(axis=(1, 2)))

            if num_active_slices == 1:
                # Single slice - use 2D connected components (more efficient)
                slice_idx = centroid_z
                if pred_mask_3d[slice_idx].any():
                    labeled_2d, num_features = ndimage.label(pred_mask_3d[slice_idx])
                    if num_features > 0:
                        # Keep only component at centroid
                        component_at_centroid = labeled_2d[centroid_y, centroid_x]
                        if component_at_centroid > 0:
                            pred_mask_3d[slice_idx] = (labeled_2d == component_at_centroid)
                        else:
                            # Keep largest component
                            sizes = ndimage.sum(pred_mask_3d[slice_idx], labeled_2d, range(1, num_features + 1))
                            max_label = sizes.argmax() + 1
                            pred_mask_3d[slice_idx] = (labeled_2d == max_label)
            else:
                # Multiple slices - use 3D connected components
                labeled, num_features = ndimage.label(pred_mask_3d)
                if num_features > 0:
                    # Keep only the component that contains the centroid
                    component_at_centroid = labeled[centroid_z, centroid_y, centroid_x]
                    if component_at_centroid > 0:
                        pred_mask_3d = (labeled == component_at_centroid)
                    else:
                        # Keep largest component
                        sizes = ndimage.sum(pred_mask_3d, labeled, range(1, num_features + 1))
                        max_label = sizes.argmax() + 1
                        pred_mask_3d = (labeled == max_label)

        return pred_mask_3d

    def __repr__(self):
        return (f"MedSegXWrapper(checkpoint={self.checkpoint_path}, "
                f"sam_checkpoint={self.sam_checkpoint_path}, "
                f"device={self.device}, model_type={self.model_type}, "
                f"image_size={self.image_size})")
