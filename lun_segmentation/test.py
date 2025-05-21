"""Test the lung segmentation model."""

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from lun_segmentation.train import evaluate
from unet import model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def mask_to_rgb(mask):
    """
    Convierte una máscara [H, W] con valores 0–2 a una imagen RGB [H, W, 3]
    """
    color_map = {
        0: [0, 0, 0],   # Fondo
        1: [255, 0, 0],   # Pulmón - rojo
        2: [0, 255, 0],   # Corazón - verde
        3: [0, 0, 255]    # Tráquea - azul
    }

    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for cls, color in color_map.items():
        rgb_mask[mask == cls] = color

    return rgb_mask

def test_model(
    model: model.UNet,
    test_loader: DataLoader,
    criterion: object,
    samples_to_visualize: int,
) -> None:
    """
    Test the lung segmentation model.

    :param model: The U-Net model to test.
    :param test_loader: DataLoader for the test dataset.
    :param criterion: Loss function used during training.
    :param samples_to_visualize: Number of samples to visualize.
    """
    model = model.to(DEVICE)
    rng = np.random.default_rng(SEED)
    samples_num = rng.integers(0, len(test_loader.dataset), samples_to_visualize)

    test_loss, test_dice = evaluate(model, test_loader, criterion)
    test_dice_mean = np.mean(test_dice[1:])
    print(f"Test Loss: {test_loss:.4f} - Validation Dice: {test_dice_mean:.4f}")
    print(f"Validation Dice Coefficients - Lung: {test_dice[3]:.4f}, Heart: {test_dice[2]:.4f}, Trachea: {test_dice[1]:.4f}")
    print(f"-" * 30)

    plt.figure()
    num_columns = 3
    for plt_i, idx in enumerate(samples_num):
        img, true_mask = test_loader.dataset[idx]
        img = img.to(DEVICE)
        true_mask = true_mask.to(DEVICE)

        img_np = img.squeeze(0).cpu().numpy()
        true_mask_np = true_mask.cpu().numpy()

        model.eval()
        with torch.no_grad():
            pred = model(img.unsqueeze(0))
            pred = pred.squeeze(0)
            pred_np = torch.argmax(pred, dim=0).cpu().numpy()

        true_mask_rgb = mask_to_rgb(true_mask_np)
        pred_rgb = mask_to_rgb(pred_np)

        plt.subplot(samples_to_visualize, num_columns, num_columns * plt_i + 1)
        plt.imshow(img_np, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(samples_to_visualize, num_columns, num_columns * plt_i + 2)
        plt.imshow(true_mask_rgb)
        plt.title("True mask")
        plt.axis("off")

        plt.subplot(samples_to_visualize, num_columns, num_columns * plt_i + 3)
        plt.imshow(pred_rgb)
        plt.title("Pred mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
