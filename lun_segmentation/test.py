"""Test the lung segmentation model."""

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional

from lun_segmentation import data, metrics
from lun_segmentation.train import evaluate
from unet import model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

TEST_BATCH_SIZE = 4


def test_model(
    model: model.UNet,
    dataset: data.CTLungDataset,
    samples_to_visualize: int,
    prediction_threshold: float = 0.5,
) -> None:
    """
    Test the lung segmentation model.

    :param model: The U-Net model to test.
    :param dataset: The dataset containing the test images.
    """
    model = model.to(DEVICE)
    rng = np.random.default_rng(SEED)
    samples_num = rng.integers(0, len(dataset), samples_to_visualize)

    test_loader = data.get_dataloader(dataset, batch_size=TEST_BATCH_SIZE, num_workers=2, train_mode=False)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

    plt.figure()
    num_columns = 3
    for plt_i, idx in enumerate(samples_num):
        img, true_mask = dataset[idx]

        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        true_mask_np = true_mask.permute(1, 2, 0).cpu().numpy()
        true_mask_np = np.clip(true_mask_np, 0, 1)

        model.eval()
        with torch.no_grad():
            output_mask = model(img.unsqueeze(0).to(DEVICE))
            output_mask = functional.sigmoid(output_mask)
            output_mask = (output_mask > prediction_threshold).float()[0].permute(1, 2, 0).cpu().numpy()
            output_mask = np.clip(output_mask, 0, 1)

        plt.subplot(samples_to_visualize, num_columns, num_columns * plt_i + 1)
        plt.imshow(img_np, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(samples_to_visualize, num_columns, num_columns * plt_i + 2)
        plt.imshow(true_mask_np, cmap="gray")
        plt.title("True mask")
        plt.axis("off")

        plt.subplot(samples_to_visualize, num_columns, num_columns * plt_i + 3)
        plt.imshow(output_mask, cmap="gray")
        plt.title("Pred mask")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("/workspace/lung-segmentation-tfm/resources/imgs/test_results.png")
    plt.close()
