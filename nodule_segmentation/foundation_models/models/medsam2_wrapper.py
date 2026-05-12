"""
MedSAM2 Model Wrapper for LUNA16 Evaluation
============================================

Encapsulates MedSAM2-CTLesion model for nodule segmentation.
Provides a clean interface for preprocessing and inference.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, '/workspace/code/MedSAM2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from nodule_segmentation.foundation_models import utils


class MedSAM2Wrapper:
    """
    Wrapper class for MedSAM2-CTLesion model.

    This class encapsulates the MedSAM2 video predictor and provides
    methods for preprocessing CT images and predicting nodule masks.
    """

    def __init__(self, checkpoint_path, config_path, device='cuda', image_size=512):
        """
        Initialize MedSAM2 wrapper.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to model configuration file
            device: Device to run model on ('cuda' or 'cpu')
            image_size: Target image size for resizing (default: 512)
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.image_size = image_size
        self.image_predictor = None

        # ImageNet normalization constants
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

        # Initialize predictors
        self._initialize_predictors()

    def _initialize_predictors(self):
        """Initialize image predictor for single-slice inference."""
        sam2_model = build_sam2(
            str(self.config_path),
            str(self.checkpoint_path),
            device=self.device
        )
        self.image_predictor = SAM2ImagePredictor(sam2_model)

        # Move normalization constants to device
        if 'cuda' in self.device:
            self.img_mean = self.img_mean.to(self.device)
            self.img_std = self.img_std.to(self.device)
        elif self.device == 'cpu':
            self.img_mean = self.img_mean.cpu()
            self.img_std = self.img_std.cpu()

    def preprocess_image(self, img_3d, window_level=40, window_width=400):
        """
        Preprocess 3D CT image for MedSAM2 inference.

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
            img_resized = img_preprocessed[:, None].repeat(3, axis=1)

        # 4. Normalize to [0, 1]
        img_resized = img_resized / 255.0

        # 5. Move to device
        try:
            img_tensor = torch.from_numpy(img_resized).to(self.device)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                raise MemoryError(f"CUDA out of memory during preprocessing")
            raise

        # 6. Apply ImageNet normalization
        img_tensor -= self.img_mean
        img_tensor /= self.img_std

        return img_tensor, video_height, video_width

    @torch.inference_mode()
    def predict(self, img_tensor, video_height, video_width,
                centroid_z, centroid_y, centroid_x, bbox=None,
                confidence_threshold=0.5, z_range=None):
        """
        Predict nodule segmentation mask using MedSAM2 (single-slice only).

        Args:
            img_tensor: Preprocessed image tensor (1, 3, H, W) on device
            video_height: Original video height
            video_width: Original video width
            centroid_z: Z coordinate (unused, always 0 for single-slice)
            centroid_y: Y coordinate of nodule centroid
            centroid_x: X coordinate of nodule centroid
            bbox: Optional bounding box (y_min, x_min, y_max, x_max)
            confidence_threshold: Threshold for mask confidence (default: 0.5)
            z_range: Unused parameter (kept for API compatibility)

        Returns:
            np.ndarray: Predicted 3D binary mask (1, H, W)
        """
        return self._predict_single_slice(
            img_tensor, video_height, video_width,
            centroid_y, centroid_x, bbox, confidence_threshold
        )

    @torch.inference_mode()
    def _predict_single_slice(self, img_tensor, video_height, video_width,
                               centroid_y, centroid_x, bbox=None,
                               confidence_threshold=0.5):
        """
        Predict on a single slice using image predictor (optimized).

        Args:
            img_tensor: Preprocessed image tensor (1, 3, H, W) on device
            video_height: Original height
            video_width: Original width
            centroid_y: Y coordinate
            centroid_x: X coordinate
            bbox: Optional bbox (y_min, x_min, y_max, x_max)
            confidence_threshold: Threshold for mask

        Returns:
            np.ndarray: Predicted mask (1, H, W)
        """
        # Extract single image (3, H, W)
        img_2d = img_tensor[0]  # (3, H, W)

        # Denormalize to [0, 255] for SAM2ImagePredictor
        img_2d = img_2d * self.img_std + self.img_mean
        img_2d = img_2d * 255.0
        img_2d = img_2d.clamp(0, 255)

        # Convert to numpy (H, W, 3) and uint8
        img_np = img_2d.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # Set image
        self.image_predictor.set_image(img_np)

        # Prepare prompt
        if bbox is not None:
            # Bbox: [x_min, y_min, x_max, y_max]
            box = np.array([bbox[1], bbox[0], bbox[3], bbox[2]], dtype=np.float32)
            masks, scores, logits = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False
            )
        else:
            # Point prompt
            point_coords = np.array([[centroid_x, centroid_y]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
            masks, scores, logits = self.image_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=False
            )

        # Get mask (1, H, W)
        pred_mask = masks[0]  # (H, W)

        # Resize to original dimensions if needed
        if pred_mask.shape != (video_height, video_width):
            from scipy.ndimage import zoom
            zoom_factors = (video_height / pred_mask.shape[0], video_width / pred_mask.shape[1])
            pred_mask = zoom(pred_mask.astype(float), zoom_factors, order=0) > 0.5

        # Return as 3D (1, H, W)
        return pred_mask[None, :, :].astype(bool)

    def __repr__(self):
        return (f"MedSAM2Wrapper(checkpoint={self.checkpoint_path}, "
                f"config={self.config_path}, device={self.device}, "
                f"image_size={self.image_size})")
