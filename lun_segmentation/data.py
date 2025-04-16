"""."""

from __future__ import annotations

import math
from pathlib import Path

import nrrd
import pydicom
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_TRAIN_SPLIT_SIZE = 0.8
DEFAULT_TRAIN_MODE = True
DEFAULT_OPACITY_VALUE = 0.5

FULL_DATASET_SIZE = 1.0
TWO_DIM_SHAPE = 2
MAX_PIXEL_NORMALIZED_VALUE = 1.0
MAX_NUM_COLUMN = 5


class NumberMaskError(Exception):
    """Exception when the number of masks does not match with the number of data."""

    def __init__(self, message: str) -> None:
        """
        Raise the exception with the given message.

        :param message: message to display in the traceback.
        """
        super().__init__(message)


class CTLungDataset(Dataset):
    """
    Create a dataset for chest computed tomography data files based on a given path.

    For training dataset, the mask
    """

    def __init__(self, data_path: str, masks_path: str | None = None) -> None:
        """
        Get the data files, and masks if it is given, based on the given paths.

        :param data_path: path where the data files are located.
        :param masks_path: path where the masks files are located (for training and validation process).
        """
        super().__init__()

        if not Path(data_path).exists():
            msg = f"<{data_path}> directory does not exist!"
            raise NotADirectoryError(msg)
        self._data_files = sorted(Path(data_path).rglob("*"), key=lambda p: p.stem)

        self._masks_files = None
        if masks_path:
            if not Path(masks_path).exists():
                msg = f"<{masks_path}> directory does not exist!"
                raise NotADirectoryError(msg)

            self._masks_files = sorted(Path(masks_path).rglob("*"), key=lambda p: p.stem)
            if len(self._masks_files) != len(self._data_files):
                msg = (
                    f"The number of masks <{len(self._masks_files)}> does not match with the number of data "
                    f"<{len(self._data_files)}>!"
                )
                raise NumberMaskError(msg)

            assert all(  # noqa: S101
                self._data_files[i].stem
                == f"{self._masks_files[i].stem.split('mask_')[0]}{self._masks_files[i].stem.split('mask_')[1]}"
                for i in range(len(self._masks_files))
            ), f"The masks files <{self._masks_files}> do not match with the data files <{self._data_files}>!"

    def __len__(self) -> None:
        """Get the total number of files found on the given path."""
        return len(self._data_files)

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the data and corresponding mask (if available) for the given index.

        :param index: index of the data item to retrieve.
        :return: tuple containing the data tensor and the mask tensor (if mask path is provided).
        """
        data = self._get_tensor_data_from_file(self._data_files[index])
        if self._masks_files:
            mask = self._get_tensor_data_from_file(self._masks_files[index])
            return data, mask

        return data

    def _get_tensor_data_from_file(self, filepath: Path) -> torch.Tensor:
        """
        Retrieve the data from the given file and convert it to a tensor.

        :param filepath: path to the file containing the data.
        :return: data as a tensor object.
        """
        conversion_file = {
            ".nrrd": lambda f: self._nrrd_to_tensor(f),
            ".dcm": lambda f: self._dcm_to_tensor(f),
            ".jpg": lambda f: self._jpg_png_to_tensor(f),
            ".png": lambda f: self._jpg_png_to_tensor(f),
        }
        return conversion_file[filepath.suffix](filepath)

    def _nrrd_to_tensor(self, filepath: Path) -> torch.Tensor:
        """
        Retrieve the data from the given NRRD file and convert it to a tensor.

        :param filepath: path to the NRRD file containing the data.
        :return: data as a tensor object.
        """
        data, _ = nrrd.read(filepath.as_posix())
        data_tensor = torch.tensor(data, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)

        if data_tensor.max() > MAX_PIXEL_NORMALIZED_VALUE:
            data_tensor /= data_tensor.max().item()

        return data_tensor

    def _dcm_to_tensor(self, filepath: Path) -> torch.Tensor:
        """
        Retrieve the data from the given DICOM file and convert it to a tensor.

        :param filepath: path to the DICOM file containing the data.
        :return: data as a tensor object.
        """
        data = pydicom.dcmread(filepath.as_posix())
        data_tensor = torch.tensor(data.pixel_array, dtype=torch.float32, device=DEVICE)

        if data_tensor.max() > MAX_PIXEL_NORMALIZED_VALUE:
            data_tensor /= data_tensor.max().item()

        if data_tensor.ndim == TWO_DIM_SHAPE:
            data_tensor = data_tensor.unsqueeze(dim=0)

        return data_tensor

    def _jpg_png_to_tensor(self, filepath: Path) -> torch.Tensor:
        """
        Retrieve the data from the given JPG/PNG file and convert it to a tensor.

        :param filepath: path to the JPG/PNG file containing the data.
        :return: data as a tensor object.
        """
        return transforms.ToTensor()(Image.open(filepath))


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    train_mode: bool = DEFAULT_TRAIN_MODE,
) -> DataLoader:
    """
    Create a DataLoader based on the given dataset and batch size with shuffled samples by default.

    :param dataset: dataset to create the DataLoader from.
    :param batch_size: number of samples in each batch.
    :param num_workers: number of subprocesses to use for data loading.
    :param train_mode: whether the DataLoader is for training or validation. Default is True (training mode).
    :return: DataLoader object.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_mode,
        num_workers=num_workers,
        drop_last=train_mode,
        pin_memory=True,
    )


def plot_images_and_masks_overlapped(dataloader: DataLoader, mask_opacity: int = DEFAULT_OPACITY_VALUE) -> None:
    """
    Plot some images and masks overlapped for each batch of the given DataLoader.

    :param dataloader: DataLoader containing the images and masks to plot.
    :param mask_opacity: opacity value for the mask. Default value is 0.5.
    """
    imgs_data, masks_data = next(iter(dataloader))
    plt.figure(figsize=(10, 10))
    n_columns = min(MAX_NUM_COLUMN, dataloader.batch_size)
    n_rows = math.ceil(dataloader.batch_size / n_columns)

    for i in range(dataloader.batch_size):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.imshow(imgs_data[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
        plt.imshow(masks_data[i].permute(1, 2, 0).cpu().numpy(), cmap="gray", alpha=mask_opacity)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
