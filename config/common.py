from pathlib import Path


class Configuration(object):
    def __init__(self) -> None:

        # Working directory
        self.WORKDIR = Path("/workspace/code/lung-segmentation-tfm")

        # LUNA25 dataset directories
        self.LUNA25_CT_DIR = Path("/workspace/data/LUNA25/luna25_images")
        self.LUNA25_NODULES_DIR = Path("/workspace/data/LUNA25/luna25_nodule_blocks")
        self.LUNA25_NODULES_MASKS_DIR = Path("/workspace/data/LUNA25/luna25_nodule_blocks_masks")
        self.LUNA25_CSV_DIR = Path("/workspace/data/LUNA25/dataset_csv")
        self.LUNA25_ANNOTATIONS_CSV_PATH = Path("/workspace/data/LUNA25/dataset_csv/LUNA25_Public_Training_Development_Data.csv")
        self.LUNA25_IMG_SIZE = 512

        # LUNA16 dataset directories
        self.LUNA16_ANNOTATIONS_CSV_PATH = Path("/workspace/data/LUNA16/annotations/luna16_annotations.csv")
        self.LUNA16_CT_DIR = Path("/workspace/data/LUNA16/luna16_images")
        self.LUNA16_NODULES_DIR = Path("/workspace/data/LUNA16/luna16_nodule_blocks")
        self.LUNA16_CT_MASKS_DIR = Path("/workspace/data/LUNA16/luna16_nodule_masks")
        self.LUNA16_NODULE_MASKS_DIR = Path("/workspace/data/LUNA16/luna16_nodule_blocks_masks")  # MedSAM2 masks
        self.LUNA16_NODULE_MASKS_GT_DIR = Path("/workspace/data/LUNA16/luna16_nodule_blocks_masks_gt")  # GT masks
        self.LUNA16_CSV_DIR = Path("/workspace/data/LUNA16/annotations")
        self.LUNA16_IMG_SIZE = 512
        self.LUNA16_NUM_WORKERS = 2

        # Training parameters
        self.SEED = 42
        self.SIZE_MM = 50
        self.SIZE_PX = 32
        self.PATCH_SIZE = [64, 128, 128]  # Dimensions of preprocessed nodule volume blocks
        self.ROTATION = ((-20, 20), (-20, 20), (-20, 20))
        self.TRANSLATION = True
        self.PATIENCE = 15
        self.THRESHOLD = 0.5

        # Pixel Threshold Separation (PTS) - Inspired by improved V-Net paper
        # Enables multi-channel input based on HU intensity thresholds
        self.USE_PIXEL_THRESHOLD_SEPARATION = False  # Enable/disable PTS preprocessing (default: False)
        self.PTS_THRESHOLDS = {
            'low_density': -600,    # Ground-glass nodules + lung (-∞ to -600)
            'mid_density': -100,    # Transition region (-600 to -100)
            'high_density': 100,    # Soft tissue/solid nodules (-100 to 100)
            'very_high': 300,       # Calcifications/contrast (> 100)
        }
        # Strategy: Multi-range approach instead of trying to predict nodules
        # Nodules have huge HU variability (-1024 to +2096), they appear in MULTIPLE channels:
        # - Ground-glass nodules: appear in Ch0 + Ch1
        # - Part-solid nodules: appear in Ch1 + Ch2
        # - Solid nodules: appear in Ch2 + Ch3
        # - Calcified nodules: appear in Ch3
        # The network learns which COMBINATION of channels indicates nodules
        # Channel interpretation:
        #   0: Very low density (< -600 HU) = air + lung + ground-glass nodules
        #   1: Low-mid density (-600 to -100 HU) = lung tissue + some ground-glass
        #   2: Mid-high density (-100 to 100 HU) = soft tissue (solid nodules)
        #   3: Very high density (> 100 HU) = calcifications + contrast
        # When PTS is enabled:
        # - Channel 0: Low density region (< air threshold)
        # - Channel 1: Nodule candidate region (soft_tissue to nodule_max)
        # - Channel 2: High density region (> nodule_max)
        # - Channel 3: Original normalized image (optional)

        # Experiment paths
        self.CLASSIFICATION_EXPERIMENT_PATH = self.WORKDIR / "config" / "classification_experiments.yaml"
        self.SEGMENTATION_EXPERIMENT_PATH = self.WORKDIR / "config" / "segmentation_experiments.yaml"

        # Best models directory
        self.BEST_MODEL_PATH = self.WORKDIR / "results" / "best_model"

        # CSV results path
        self.CLASSIFICATION_CSV_RESULTS_PATH = self.WORKDIR / "results" / "classification_results.csv"
        self.SEGMENTATION_CSV_RESULTS_PATH = self.WORKDIR / "results" / "segmentation_results.csv"
        self.FOUNDATION_SEGMENTATION_CSV_RESULTS_PATH = self.WORKDIR / "results" / "foundation_models_nodule_segmentation_results.csv"

        # MedSAM2 configuration
        self.MEDSAM2_CHECKPOINT = Path("/workspace/code/MedSAM2/checkpoints/MedSAM2_latest.pt")
        self.MEDSAM2_CONFIG = Path("configs/sam2.1_hiera_t512")  # Hydra config name (without .yaml, relative to sam2/configs/)

        # SAM2 configuration (using sam2.1_hiera_tiny.pt from MedSAM2 repo)
        self.SAM2_CHECKPOINT = Path("/workspace/code/MedSAM2/checkpoints/sam2.1_hiera_tiny.pt")
        self.SAM2_CONFIG = Path("/workspace/code/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml")
        self.SAM2_IMG_SIZE = 512

        # SAM3 configuration
        self.SAM3_CHECKPOINT = None  # Downloads from HuggingFace automatically
        self.SAM3_CONFIG = None  # Not used
        self.SAM3_IMG_SIZE = 1008

        # MedSegX configuration
        self.MEDSEGX_CHECKPOINT = Path("/workspace/code/MedSegX/playground/MedSegX/medsegx_vit_b.pth")
        self.MEDSEGX_SAM_CHECKPOINT = Path("/workspace/code/MedSegX/playground/SAM/sam_vit_b_01ec64.pth")
        self.MEDSEGX_MODEL_TYPE = "vit_b"
        self.MEDSEGX_IMG_SIZE = 256

config = Configuration()