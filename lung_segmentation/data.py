"""."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pydicom
import torch
from pandas import DataFrame
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

DEFAULT_RANDOM_SEED = 42

MAX_PIXEL_VALUE = 255.
PIXEL_THRESHOLD = 200


class NumberMaskError(Exception):
    """Exception when the number of masks does not match with the number of data."""

    def __init__(self, message: str) -> None:
        """
        Raise the exception with the given message.

        :param message: message to display in the traceback.
        """
        super().__init__(message)


class ChestCTDatasetCsv(Dataset):
    """
    Create a dataset for chest computed tomography data files based on the CSV's information.

    Each pixel in the mask data is represented by a different color, and the labels are as follows:

    - 0: background
    - 1: trachea
    - 2: heart
    - 3: lung
    """

    def __init__(self, images_path: str, masks_path: str,  df: DataFrame) -> None:
        """
        Get the data files, and masks if it is given, based on the given paths.

        :param images_path: path where the data files are located.
        :param masks_path: path where the masks files are located (for training and validation process).
        :param df: dataframe containing the data files and masks from a csv file.
        """
        self._df = df.reset_index(drop=True)

        if not Path(images_path).exists():
            msg = f"<{images_path}> directory does not exist!"
            raise NotADirectoryError(msg)
        self._images_path = images_path

        if not Path(masks_path).exists():
            msg = f"<{masks_path}> directory does not exist!"
            raise NotADirectoryError(msg)
        self._masks_path = masks_path

    def __len__(self) -> None:
        """Get the total number of files found on the given path."""
        return len(self._df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the data and corresponding mask for the given index.

        :param index: index of the data item to retrieve.
        :return: tuple containing the data tensor and the mask tensor.
        """
        lung_channel = 2
        img_path = Path(self._images_path) / self._df.loc[index, "ImageId"]
        mask_path = Path(self._masks_path) / self._df.loc[index, "MaskId"]

        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.float32) / MAX_PIXEL_VALUE
        image = torch.tensor(image).permute(2, 0, 1)

        mask = Image.open(mask_path).convert("RGB")
        mask = np.array(mask, dtype=np.int64)
        mask = mask[:, :, lung_channel] / MAX_PIXEL_VALUE
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)

        return image, mask

class CtLungIldDataset(Dataset):
    """Create a dataset for chest computed tomography images and masks data stored in dicom files."""

    def __init__(self, database_path: str) -> None:
        """
        Initialize the dataset with the images and masks path from the given database path.

        :param database_path: path where the dicom files are located. The structure of the path should be:
            <database_path>/
            ├── <patient_id_1>/
            │   ├── lung_mask/
            │   │   ├── lung_mask_<mask_id_1>.dcm
            │   │   ├── lung_mask_<mask_id_2>.dcm
            │   ├── CT-<image_id_1>.dcm
            │   ├── CT-<image_id_2>.dcm
            ...
        """
        self._images_path = list(Path(database_path).rglob("**/*CT*.dcm"))
        self._masks_path = list(Path(database_path).rglob("**/lung_mask/lung_mask*.dcm"))

        if len(self._images_path) != len(self._masks_path):
            msg = (
                f"Number of images <{len(self._images_path)}> does not match with the number of masks "
                f"<{len(self._masks_path)}>!"
            )
            raise NumberMaskError(msg)

    def __len__(self) -> None:
        """Get the total number of files found on the given path."""
        return len(self._images_path)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the data and corresponding mask for the given index.

        :param index: index of the data item to retrieve.
        :return: tuple containing the data tensor and the mask tensor.
        """
        image_path = self._images_path[index]
        mask_path = self._masks_path[index]

        print(f"Image path: {image_path}, Mask path: {mask_path}")

        image = pydicom.dcmread(image_path).pixel_array
        mask = pydicom.dcmread(mask_path).pixel_array

        print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        print(f"Image unique values: {np.unique(image)}, Mask unique values: {np.unique(mask)}")

        return image, mask

def split_dataset(
    dataset: Dataset,
    val_split_size: float,
    test_split_size: float,
    seed: int = DEFAULT_RANDOM_SEED
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Split the complete dataset into training, validation, and test datasets.

    The training dataset will contain the remaining data after splitting the validation and test datasets.

    :param dataset: dataset with every images and masks.
    :param val_split_size: size of the validation split.
    :param test_split_size: size of the test split.
    :return: DataLoader object.
    """
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_split_size, random_state=seed)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=val_split_size, random_state=seed)
    return train_dataset, val_dataset, test_dataset
