from pathlib import Path


class Configuration(object):
    def __init__(self) -> None:

        # Working directory
        self.WORKDIR = Path("/workspace/code/lung-segmentation-tfm")
        
        # Best models directory
        self.RESULTS = self.WORKDIR / "results" / "best_models"

        # LUNA25 dataset directories
        self.LUNA25_NODULES_DIR = Path("/workspace/data/LUNA25/luna25_nodule_blocks")
        self.LUNA25_NODULES_MASKS_DIR = Path("/workspace/data/LUNA25/luna25_nodule_blocks_masks")
        self.LUNA25_CSV_TRAIN_DIR = Path("/workspace/data/LUNA25/dataset_csv/train.csv")
        self.LUNA25_CSV_VALID_DIR = Path("/workspace/data/LUNA25/dataset_csv/valid.csv")
        self.LUNA25_CSV_TEST_DIR = Path("/workspace/data/LUNA25/dataset_csv/test.csv")

        # LUNA16 dataset directories
        self.LUNA16_CT_DIR = Path("/workspace/data/LUNA16/luna16_images")
        self.LUNA16_CT_MASKS_DIR = Path("/workspace/data/LUNA16/luna16_masks")
        self.LUNA16_NODULES_DIR = Path("/workspace/data/LUNA16/luna16_nodule_blocks")
        self.LUNA16_NODULES_MASKS_DIR = Path("/workspace/data/LUNA16/luna16_nodule_blocks_masks")
        self.LUNA16_CSV_TRAIN_DIR = Path("/workspace/data/LUNA16/annotations/train.csv")
        self.LUNA16_CSV_VALID_DIR = Path("/workspace/data/LUNA16/annotations/valid.csv")
        self.LUNA16_CSV_TEST_DIR = Path("/workspace/data/LUNA16/annotations/test.csv")

        # Training parameters
        self.SEED = 42
        self.SIZE_MM = 50
        self.SIZE_PX = 32
        self.ROTATION = ((-20, 20), (-20, 20), (-20, 20))
        self.TRANSLATION = True
        self.PATIENCE = 30
        

config = Configuration()