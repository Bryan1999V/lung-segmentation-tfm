"""
SAM2 Model Wrapper for LUNA16 Evaluation (Original SAM2 + MedSAM2 Checkpoint)
==============================================================================

Encapsulates SAM2 video predictor using:
- SAM2 original module from Meta (/workspace/code/sam2)
- MedSAM2 checkpoint fine-tuned on CT lesions
- MedSAM2 configuration (512px, optimized for medical imaging)

Provides a clean interface for preprocessing and inference on 3D CT volumes.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, '/workspace/code/sam2')
from sam2.sam2_image_predictor import SAM2ImagePredictor

from nodule_segmentation.foundation_models import utils


class SAM2Wrapper:
    """
    Wrapper class for SAM2 original module with MedSAM2 checkpoint.

    This class encapsulates the SAM2 video predictor and provides
    methods for preprocessing CT images and predicting nodule masks.

    Architecture:
    - Uses SAM2 original codebase from Meta
    - Loads MedSAM2_CTLesion.pt checkpoint (fine-tuned on CT lesions)
    - Uses 512px configuration optimized for medical imaging
    - Supports single-slice prediction via SAM2ImagePredictor
    """

    def __init__(self, checkpoint_path, config_path, device='cuda', image_size=512):
        """
        Initialize SAM2 wrapper with MedSAM2 checkpoint.

        Args:
            checkpoint_path: Path to MedSAM2 checkpoint (default: MedSAM2_CTLesion.pt)
            config_path: Path to model configuration file (default: sam2.1_hiera_t512.yaml)
            device: Device to run model on ('cuda' or 'cpu')
            image_size: Target image size for resizing (default: 512)
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.image_size = image_size
        self.image_predictor = None  # For single-slice inference

        # ImageNet normalization constants
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

        # Initialize predictor
        self._initialize_predictor()

    def _initialize_predictor(self):
        """Initialize the SAM2 image predictor for single-slice inference."""
        # Build SAM2 model using direct config loading (not Hydra)
        import torch
        from omegaconf import OmegaConf
        from hydra.utils import instantiate

        # Load config directly from file
        cfg = OmegaConf.load(self.config_path)

        # Instantiate model
        model = instantiate(cfg.model, _recursive_=True)

        # Load checkpoint
        if self.checkpoint_path is not None:
            sd = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)["model"]
            missing_keys, unexpected_keys = model.load_state_dict(sd)
            if missing_keys:
                print(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {unexpected_keys}")

        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        # Create image predictor for efficient single-slice processing
        self.image_predictor = SAM2ImagePredictor(model)

        # Fix _bb_feat_sizes for 512px images (SAM2 has hardcoded 1024px values)
        # This is required for proper feature extraction at different resolutions
        hires_size = model.image_size // 4  # 512 // 4 = 128
        self.image_predictor._bb_feat_sizes = [[hires_size // (2**k)]*2 for k in range(3)]
        # Result: [(128, 128), (64, 64), (32, 32)] for 512px

        # Move normalization constants to device
        if 'cuda' in self.device:
            self.img_mean = self.img_mean.to(self.device)
            self.img_std = self.img_std.to(self.device)
        elif self.device == 'cpu':
            self.img_mean = self.img_mean.cpu()
            self.img_std = self.img_std.cpu()

    def preprocess_image(self, img_3d, window_level=40, window_width=400):
        """
        Preprocess 3D CT image for SAM2 inference.
        
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
                confidence_threshold=0.0, z_range=None):
        """
        Predict nodule segmentation mask using SAM2 (single-slice only).

        Args:
            img_tensor: Preprocessed image tensor (1, 3, H, W) on device
            video_height: Original video height
            video_width: Original video width
            centroid_z: Z coordinate (unused, always 0 for single-slice)
            centroid_y: Y coordinate of nodule centroid
            centroid_x: X coordinate of nodule centroid
            bbox: Optional bounding box (y_min, x_min, y_max, x_max)
            confidence_threshold: Threshold for mask confidence (default: 0.0)
            z_range: Unused parameter (kept for API compatibility)

        Returns:
            np.ndarray: Predicted 3D binary mask (1, H, W)
        """
        # Extract single slice (D=1, C=3, H, W) -> (C, H, W)
        img_slice = img_tensor[0]  # (3, H, W) - already normalized with ImageNet stats
        # Denormalize back to [0, 1] range
        img_slice = img_slice * self.img_std + self.img_mean

        # Convert to [0, 255] range (expected by SAM2ImagePredictor)
        img_slice = img_slice * 255.0

        # Convert to HWC format for SAM2ImagePredictor
        img_slice_hwc = img_slice.permute(1, 2, 0)  # (H, W, 3)

        # Convert to numpy and ensure uint8 (expected format)
        img_array = img_slice_hwc.cpu().numpy().astype(np.uint8)

        # Set image in predictor (will apply transforms internally)
        self.image_predictor.set_image(img_array)

        # Prepare prompt
        if bbox is not None:
            # Bbox prompt: [x_min, y_min, x_max, y_max]
            # Scale bbox to image size
            scale_y = self.image_size / video_height
            scale_x = self.image_size / video_width

            bbox_scaled = np.array([
                bbox[1] * scale_x,  # x_min
                bbox[0] * scale_y,  # y_min
                bbox[3] * scale_x,  # x_max
                bbox[2] * scale_y   # y_max
            ], dtype=np.float32)

            masks, scores, logits = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox_scaled[None, :],  # (1, 4)
                multimask_output=False
            )
        else:
            # Point prompt at centroid
            point_coords = np.array([[centroid_x, centroid_y]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)

            # Scale point to image size
            scale_y = self.image_size / video_height
            scale_x = self.image_size / video_width
            point_coords[:, 0] *= scale_x
            point_coords[:, 1] *= scale_y

            masks, scores, logits = self.image_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=False
            )

        # Get mask (already in correct size from predictor)
        pred_mask = masks[0]  # (H, W) at image_size

        # Resize back to original size if needed
        if pred_mask.shape != (video_height, video_width):
            from scipy.ndimage import zoom
            zoom_factors = (video_height / pred_mask.shape[0], video_width / pred_mask.shape[1])
            pred_mask = zoom(pred_mask.astype(float), zoom_factors, order=1) > 0.5

        # Return as 3D array with single slice
        return np.array([pred_mask], dtype=bool)

    def __repr__(self):
        return (f"SAM2Wrapper(checkpoint={self.checkpoint_path.name if hasattr(self.checkpoint_path, 'name') else self.checkpoint_path}, "
                f"config={self.config_path.name if hasattr(self.config_path, 'name') else self.config_path}, "
                f"device={self.device}, image_size={self.image_size})")
