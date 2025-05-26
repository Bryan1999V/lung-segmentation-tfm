"""Test the lung segmentation model."""

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from lung_segmentation.train import evaluate
from unet import model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = 0.5
SEED = 42


def test_model(
    unet: model.UNet,
    test_loader: DataLoader,
    criterion: object,
    samples_to_visualize: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """
    Test the lung segmentation model.

    :param unet: The U-Net model to test.
    :param test_loader: DataLoader for the test dataset.
    :param criterion: Loss function used during training.
    :param samples_to_visualize: Number of samples to visualize.
    """
    unet = unet.to(DEVICE)
    samples_num = np.random.choice(len(test_loader.dataset), samples_to_visualize, replace=False)

    test_loss, test_dice = evaluate(unet, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} - Test Dice: {test_dice:.4f}")
    print("-" * 30)

    plt.figure()
    num_columns = 3
    for plt_i, idx in enumerate(samples_num):
        img, true_mask = test_loader.dataset[idx]
        img = img.to(DEVICE)
        true_mask = true_mask.to(DEVICE)

        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        true_mask_np = true_mask.permute(1, 2, 0).cpu().numpy()
        true_mask_np = np.clip(true_mask_np, 0, 1)

        unet.eval()
        with torch.no_grad():
            pred = unet(img.unsqueeze(0))
            pred = torch.sigmoid(pred)
            pred = (pred > threshold).float().squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred = np.clip(pred, 0, 1)

        plt.subplot(samples_to_visualize, num_columns, num_columns * plt_i + 1)
        plt.imshow(img_np, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(samples_to_visualize, num_columns, num_columns * plt_i + 2)
        plt.imshow(true_mask_np, cmap="gray")
        plt.title("True mask")
        plt.axis("off")

        plt.subplot(samples_to_visualize, num_columns, num_columns * plt_i + 3)
        plt.imshow(pred, cmap="gray")
        plt.title("Pred mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
