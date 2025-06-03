"""."""

from __future__ import annotations

import re
from pathlib import Path

import albumentations
import numpy as np
import pydicom
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

DEFAULT_RANDOM_SEED = 42

MAX_N_IMAGES_TO_VISUALIZE = 10
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

    def __init__(self, database_path: str, transform: albumentations.Compose = None, roi_mask: bool = False) -> None:
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
        self._roi_mask = roi_mask
        self._transform = transform
        self._images_path = []
        self._masks_path = []

        mask_prefix = "roi_mask_" if roi_mask else "lung_mask_"

        for patient_path in Path(database_path).glob("*"):
            images_path = sorted([p for p in patient_path.rglob("*.dcm") if mask_prefix not in p.name])
            masks_path = sorted(patient_path.rglob(f"{mask_prefix}*.dcm"), key=self._mask_sort_key)

            self._images_path.extend(images_path)
            self._masks_path.extend(masks_path)

        assert len(self._images_path) == len(self._masks_path), (
            f"Number of images <{len(self._images_path)}> does not match with the number of masks <{len(self._masks_path)}>!"
        )

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

        print(f"Processing image: {'/'.join(image_path.parts[-4:])}, mask: {'/'.join(mask_path.parts[-4:])}")

        image = pydicom.dcmread(image_path).pixel_array
        if not self._roi_mask:
            min_hu = np.min(image)
            max_hu = 400
            image = (np.clip(image, min_hu, max_hu) - min_hu) / (max_hu - min_hu)

        mask = pydicom.dcmread(mask_path).pixel_array
        if not self._roi_mask:
            max_mask_value = np.max(mask)
            mask = mask / int(np.max(mask)) if max_mask_value != 0 else mask

        if self._transform:
            augmented = self._transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        # print(f"Image unique values: {np.unique(image)}, Mask unique values: {np.unique(mask)}")

        image = image.float()
        mask = mask.unsqueeze(0)
        return image, mask

    @staticmethod
    def _mask_sort_key(path: Path) -> str:
        """
        Sort key for the mask files based on the number in the filename.

        :param path: path to the mask file.
        :return: sorted path with the number padded to 4 digits.
        """
        m = re.search(r"(.+lung_mask_\d+_)(\d+)(\.dcm)", path.as_posix()) if not "roi_mask_" in path.name else re.search(r"(.+roi_mask_\d+_)(\d+)(\.dcm)", path.as_posix())
        if m:
            prefix, num, suffix = m.groups()
            return f"{prefix}{int(num):04d}{suffix}"
        return path.as_posix()


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


def visualize_dataset_images(dataset: Dataset, n_images: int = MAX_N_IMAGES_TO_VISUALIZE) -> None:
    """
    Visualize the number of images and masks from the given dataset.

    :param dataset: dataset containing the images and masks.
    :param n_images: number of images to visualize.
    """
    n_rows = 2

    plt.figure()
    for i in range(n_images):
        random_idx = np.random.randint(0, len(dataset))
        img, mask = dataset[random_idx]
        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.permute(1, 2, 0).numpy()

        plt.subplot(n_rows, n_images, i + 1)
        plt.imshow(img_np, cmap="gray")
        plt.title("Image")
        plt.axis("off")

        plt.subplot(n_rows, n_images, i + 1 + n_images)
        plt.imshow(mask_np, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_roi_pixel_distribution() -> None:
    """Visualize the pixel distribution of the ROI masks in the dataset."""
    classes_values = {
        1: "Healthy",
        2: "Emphysema",
        3: "Ground\nGlass",
        4: "Fibrosis",
        5: "Micronodule",
        6: "Consolidation",
        7: "Bronchial\nWall\nThickening",
        8: "Reticulation",
        9: "Macronodules",
        10: "Cysts",
        11: "Peripheral\nmicronodules",
        12: "Bronchiectasis",
        13: "Air\nTrapping",
        14: "Early\nFibrosis",
        15: "Increased\nattenuation",
        16: "Tuberculosis",
        17: "Pcp",
    }
    pixel_value_per_class = {
        "Healthy": 0,
        "Emphysema": 0,
        "Ground\nGlass": 0,
        "Fibrosis": 0,
        "Micronodule": 0,
        "Consolidation": 0,
        "Bronchial\nWall\nThickening": 0,
        "Reticulation": 0,
        "Macronodules": 0,
        "Cysts": 0,
        "Peripheral\nmicronodules": 0,
        "Bronchiectasis": 0,
        "Air\nTrapping": 0,
        "Early\nFibrosis": 0,
        "Increased\nattenuation": 0,
        "Tuberculosis": 0,
        "Pcp": 0,
    }

    transform = albumentations.Compose([albumentations.ToTensorV2(),])
    dataset = CtLungIldDataset("/workspace/data/ILD_DB/ILD_DB_volumeROIs", transform=transform, roi_mask=True)

    for _, data in enumerate(dataset):
        _, mask = data
        mask_np = mask.squeeze().numpy()
        unique_values, counts = np.unique(mask_np, return_counts=True)
        for value, count in zip(unique_values, counts):
            if value in classes_values:
                pixel_value_per_class[classes_values[value]] += int(count)

    for class_name, pixel_count in pixel_value_per_class.items():
        class_name = class_name.replace("\n", " ")
        print(f"{class_name}: {pixel_count} pixels")

    plt.figure()
    plt.bar(pixel_value_per_class.keys(), pixel_value_per_class.values(), color='skyblue')
    plt.title("Classes per Pixel Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Pixels")
    plt.show()


def visualize_roi_mask_distribution() -> None:
    """Visualize the distribution of the ROI masks in the dataset."""
    classes_values = {
        1: "Healthy",
        2: "Emphysema",
        3: "Ground\nGlass",
        4: "Fibrosis",
        5: "Micronodule",
        6: "Consolidation",
        7: "Bronchial\nWall\nThickening",
        8: "Reticulation",
        9: "Macronodules",
        10: "Cysts",
        11: "Peripheral\nmicronodules",
        12: "Bronchiectasis",
        13: "Air\nTrapping",
        14: "Early\nFibrosis",
        15: "Increased\nattenuation",
        16: "Tuberculosis",
        17: "Pcp",
    }
    pixel_value_per_class = {
        "Healthy": 0,
        "Emphysema": 0,
        "Ground\nGlass": 0,
        "Fibrosis": 0,
        "Micronodule": 0,
        "Consolidation": 0,
        "Bronchial\nWall\nThickening": 0,
        "Reticulation": 0,
        "Macronodules": 0,
        "Cysts": 0,
        "Peripheral\nmicronodules": 0,
        "Bronchiectasis": 0,
        "Air\nTrapping": 0,
        "Early\nFibrosis": 0,
        "Increased\nattenuation": 0,
        "Tuberculosis": 0,
        "Pcp": 0,
    }

    transform = albumentations.Compose([albumentations.ToTensorV2(),])
    dataset = CtLungIldDataset("/workspace/data/ILD_DB/ILD_DB_volumeROIs", transform=transform, roi_mask=True)

    for _, data in enumerate(dataset):
        _, mask = data
        mask_np = mask.squeeze().numpy()
        unique_values = np.unique(mask_np)
        for value in unique_values:
            if value in classes_values:
                pixel_value_per_class[classes_values[value]] += 1

    for class_name, mask_count in pixel_value_per_class.items():
        class_name = class_name.replace("\n", " ")
        print(f"{class_name}: {mask_count} masks")

    plt.figure()
    plt.bar(pixel_value_per_class.keys(), pixel_value_per_class.values(), color='skyblue')
    plt.title("Classes per Mask Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Masks")
    plt.show()


def visualize_roi_mask() -> None:
    """Visualize the ROI masks in the dataset."""
    classes_values = {
        1: "Healthy",
        2: "Emphysema",
        3: "Ground\nGlass",
        4: "Fibrosis",
        5: "Micronodule",
        6: "Consolidation",
        7: "Bronchial\nWall\nThickening",
        8: "Reticulation",
        9: "Macronodules",
        10: "Cysts",
        11: "Peripheral\nmicronodules",
        12: "Bronchiectasis",
        13: "Air\nTrapping",
        14: "Early\nFibrosis",
        15: "Increased\nattenuation",
        16: "Tuberculosis",
        17: "Pcp",
    }
    transform = albumentations.Compose([albumentations.ToTensorV2(),])
    dataset = CtLungIldDataset("/workspace/data/ILD_DB/ILD_DB_volumeROIs", transform=transform, roi_mask=True)

    n_images = 5
    random_idx = np.random.choice(len(dataset), n_images, replace=False)
    print(f"Randomly selected indices: {random_idx}")
    plt.figure(figsize=(15, 5))

    for i, idx in enumerate(random_idx):
        img, mask = dataset[idx]
        img_np = img.permute(1, 2, 0).squeeze().numpy()
        mask_np = mask.squeeze().numpy()

        mask_unique = np.unique(mask_np)
        class_names = [classes_values.get(int(value), "No Class") for value in mask_unique if value != 0]
        print(f"Mask unique values: {mask_unique}")

        mask_np = mask_np / np.max(mask_np) if np.max(mask_np) != 0 else mask_np
        mask_np = (mask_np * MAX_PIXEL_VALUE).astype(np.uint8)

        plt.subplot(2, n_images, i + 1)
        plt.imshow(img_np, cmap="gray")
        plt.title(f"Image with {', '.join(class_names) if class_names else 'none'} classes")
        plt.axis("off")

        plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(mask_np, cmap="gray")
        plt.title(f"Mask with {', '.join(class_names) if class_names else 'none'} classes")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_classes_distribution_in_test_set() -> None:
    """Visualize the distribution of the classes in the test set."""
    num_classes = {
        "healthy": 0,
        "emphysema": 0,
        "ground_glass": 0,
        "fibrosis": 0,
        "micronodules": 0,
        "consolidation": 0,
        "bronchial_wall_thickening": 0,
        "reticulation": 0,
        "macronodules": 0,
        "cysts": 0,
        "peripheral_micronodules": 0,
        "bronchiectasis": 0,
        "air_trapping": 0,
        "early_fibrosis": 0,
        "increased_attenuation": 0,
        "tuberculosis": 0,
        "pcp": 0,
    }

    for p in Path("/workspace/data/ILD_DB/ILD_DB_talismanTestSuite").glob("*.tif"):
        m = re.match(r"(.+)_patch\d+_patient.*\.tif", p.name)
        class_name = m.group(1)
        num_classes[class_name] += 1

    for class_name, count in num_classes.items():
        if count == 0:
            continue
        print(f"{class_name}: {count} images")

    num_classes = {k: v for k, v in num_classes.items() if v > 0}

    plt.figure()
    plt.bar(num_classes.keys(), num_classes.values(), color='skyblue')
    plt.title("Classes Distribution in Test Set")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.show()


def visualize_random_tif_images() -> None:
    """Visualize random tif images from the dataset."""
    images_path = list(Path("/workspace/data/ILD_DB/ILD_DB_talismanTestSuite").glob("*.tif"))
    n_images = 10
    random_idx = np.random.choice(len(images_path), n_images, replace=False)
    plt.figure()

    for i, idx in enumerate(random_idx):
        img_path = images_path[idx]
        img = Image.open(img_path)

        print(f"Image path: {img_path.name}")
        print(f"Image size: {img.size}")

        plt.subplot(1, n_images, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(img_path.name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
