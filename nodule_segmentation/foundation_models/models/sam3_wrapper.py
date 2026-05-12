"""
SAM3 Model Wrapper for LUNA16 Evaluation
=========================================

Encapsulates SAM3 Image Predictor model for nodule segmentation.
Uses slice-by-slice processing with Sam3Processor API.
"""

import sys
import os
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/workspace/code/sam3')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox

from nodule_segmentation.foundation_models import utils


class SAM3Wrapper:
    """
    Wrapper class for SAM3 image predictor model.

    Uses slice-by-slice processing with Sam3Processor API.
    Processes CT slices around the nodule centroid for segmentation.
    """

    def __init__(self, checkpoint_path=None, config_path=None, device='cuda', image_size=1008):
        """
        Initialize SAM3 wrapper.

        Args:
            checkpoint_path: Path to model checkpoint (optional, downloads from HF if None)
            config_path: Path to BPE vocabulary file (default: uses SAM3 assets)
            device: Device to run model on ('cuda' or 'cpu')
            image_size: Target image size for resizing (default: 1008 for SAM3)
        """
        self.checkpoint_path = checkpoint_path
        self.bpe_path = config_path  # Use config_path as BPE vocab path
        self.device = device
        self.image_size = image_size
        self.model = None
        self.processor = None

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SAM3 image predictor model."""
        # Set default BPE path if not provided
        if self.bpe_path is None:
            import sam3
            sam3_pkg_dir = os.path.dirname(sam3.__file__)
            self.bpe_path = f"{sam3_pkg_dir}/assets/bpe_simple_vocab_16e6.txt.gz"

        # Build SAM3 image model (downloads checkpoint from HF if not provided)
        self.model = build_sam3_image_model(
            bpe_path=self.bpe_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device
        )

        # Ensure model is in float32 and on correct device
        self.model = self.model.to(self.device).float()
        self.model.eval()

        # Create processor with HIGH confidence threshold to reduce over-segmentation
        # When evaluating full volume, we need to be very conservative
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)

    def preprocess_image(self, img_3d, window_level=40, window_width=400):
        """
        Preprocess 3D CT image for SAM3 inference.

        NOTE: This method returns the full volume but SAM3 processes slice-by-slice.

        Args:
            img_3d: 3D CT scan in HU values (Z, Y, X)
            window_level: Window center for CT windowing
            window_width: Window width for CT windowing

        Returns:
            tuple: (preprocessed_array, video_height, video_width)
                - preprocessed_array: CT windowed array (Z, Y, X) in [0, 255]
                - video_height: Original height
                - video_width: Original width
        """
        # Apply CT windowing to [0, 255]
        img_preprocessed = utils.preprocess(img_3d, window_level=window_level, window_width=window_width)

        # Get original dimensions
        video_height = img_preprocessed.shape[1]
        video_width = img_preprocessed.shape[2]

        return img_preprocessed, video_height, video_width

    @torch.inference_mode()
    def predict(self, img_array, video_height, video_width,
                centroid_z, centroid_y, centroid_x, bbox=None,
                confidence_threshold=0.5, z_range=None):
        """
        Predict nodule segmentation mask using SAM3 image predictor.

        Processes slices within the nodule's z-range to avoid false positives.
        Uses bbox prompts only (text prompts don't work on medical CT scans).

        Args:
            img_array: Preprocessed CT array (Z, Y, X) in [0, 255]
            video_height: Original height
            video_width: Original width
            centroid_z: Z coordinate of nodule centroid
            centroid_y: Y coordinate of nodule centroid
            centroid_x: X coordinate of nodule centroid
            bbox: Bounding box (y_min, x_min, y_max, x_max) for 2D slice
            confidence_threshold: Threshold for mask confidence (default: 0.5)
            z_range: Optional (z_min, z_max) tuple to limit slice processing

        Returns:
            np.ndarray: Predicted 3D binary mask (Z, H, W)
        """
        # Initialize output mask
        pred_mask_3d = np.zeros((img_array.shape[0], video_height, video_width), dtype=bool)

        # Define slice range
        if z_range is not None:
            start_z, end_z = z_range
            start_z = max(0, start_z)
            end_z = min(img_array.shape[0], end_z + 1)
        else:
            # Fallback to ±10 slices around centroid
            slice_range = 10
            start_z = max(0, centroid_z - slice_range)
            end_z = min(img_array.shape[0], centroid_z + slice_range + 1)

        # Convert bbox to (x, y, w, h) format for SAM3
        y_min, x_min, y_max, x_max = bbox
        bbox_xywh = [x_min, y_min, x_max - x_min, y_max - y_min]

        try:
            # Process each slice around centroid
            for z in range(start_z, end_z):
                # Get 2D slice
                slice_2d = img_array[z]  # (H, W) in [0, 255]

                # Convert to PIL Image (RGB)
                slice_rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=-1).astype(np.uint8)
                pil_image = Image.fromarray(slice_rgb)

                # Set image in processor (creates new inference_state)
                inference_state = self.processor.set_image(pil_image)

                # Reset prompts for this new state
                self.processor.reset_all_prompts(inference_state)

                # Add bbox prompt (normalized to [0, 1] coordinates)
                box_input_xywh = torch.tensor(bbox_xywh).view(-1, 4)
                box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
                norm_box_cxcywh = normalize_bbox(
                    box_input_cxcywh,
                    pil_image.width,
                    pil_image.height
                ).flatten().tolist()

                # Add geometric prompt (box)
                inference_state = self.processor.add_geometric_prompt(
                    state=inference_state,
                    box=norm_box_cxcywh,
                    label=True  # Positive prompt
                )

                # Get prediction masks from inference_state
                if 'masks' in inference_state and len(inference_state['masks']) > 0:
                    # Get first mask
                    mask = inference_state['masks'][0]  # Can be (H, W) or (1, H, W) or (B, H, W)

                    # Convert to numpy if tensor
                    if torch.is_tensor(mask):
                        mask = mask.cpu().numpy()

                    # Squeeze to remove batch/channel dimensions
                    while mask.ndim > 2:
                        mask = mask.squeeze(0)

                    # Resize to original dimensions if needed
                    if mask.shape[0] != video_height or mask.shape[1] != video_width:
                        from scipy.ndimage import zoom
                        zoom_factors = (video_height / mask.shape[0], video_width / mask.shape[1])
                        mask = zoom(mask.astype(float), zoom_factors, order=0) > 0.5

                    pred_mask_3d[z] = mask.astype(bool)

        except Exception as e:
            print(f"Error during SAM3 prediction: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Post-process: Keep only the largest connected component near centroid
        # For single slice (z_range=(0,0)), use 2D processing; otherwise use 3D
        if pred_mask_3d.any():
            from scipy import ndimage

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
                    # Find component at centroid
                    component_at_centroid = labeled[centroid_z, centroid_y, centroid_x]

                    if component_at_centroid > 0:
                        # Keep only the component containing the centroid
                        pred_mask_3d = (labeled == component_at_centroid)
                    else:
                        # Keep largest component
                        sizes = ndimage.sum(pred_mask_3d, labeled, range(1, num_features + 1))
                        max_label = sizes.argmax() + 1
                        pred_mask_3d = (labeled == max_label)

        return pred_mask_3d

    def __repr__(self):
        return (f"SAM3Wrapper(checkpoint={self.checkpoint_path}, "
                f"device={self.device}, image_size={self.image_size})")
