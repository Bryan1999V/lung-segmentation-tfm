"""Implementation of the training and validation loop for lung segmentation."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lung_segmentation import metrics
from unet import model

USE_DICE_FOR_ACCURACY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42


def train(  # noqa: PLR0913
    unet: model.UNet,
    n_epochs: int,
    criterion: object,
    optimizer: object,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> dict[str, np.ndarray]:
    """
    Train the unet using the given optimizer and number of epochs.

    :param unet: unet to train.
    :param n_epochs: number of epochs to train the unet.
    :param criterion: loss function to use for training.
    :param optimizer: optimizer to use for training.
    :param train_loader: DataLoader containing the training data.
    :param val_loader: DataLoader containing the validation data.
    :return: dictionary containing the training and validation loss and accuracy (dice) values for each epoch.
    """
    best_val_loss = float("inf")
    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
    }

    unet = unet.to(DEVICE)
    for epoch in range(n_epochs):
        total_train_loss = 0.
        total_train_dice = 0.
        count = 1

        unet.train()
        with tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch") as pbar:
            for images, masks in train_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = unet(images)
                loss = criterion(outputs, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                dice, _ = metrics.calculate_metrics(outputs, masks)
                total_train_dice += dice

                pbar.update(images.size(0))
                pbar.set_postfix({"Train Loss": total_train_loss / count, "Train Dice": total_train_dice / count})
                count += 1

        val_loss, val_dice = evaluate(unet, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f} - Validation Dice: {val_dice:.4f}")
        print("-" * 30)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                unet.state_dict(),
                "/root/models/best_unet_model_bryan.pth",
            )

        history["loss"].append(total_train_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["accuracy"].append(total_train_dice / len(train_loader))
        history["val_accuracy"].append(val_dice)

    return history


def evaluate(
    unet: model.UNet,
    val_loader: DataLoader,
    criterion: object,
) -> tuple[float, float]:
    """
    Evaluate the unet using the given validation DataLoader.

    :param unet: unet model to evaluate.
    :param val_loader: DataLoader containing the validation data.
    :param criterion: loss function to use for evaluation.
    :return: tuple containing the validation cost, accuracy, Dice coefficient and IoU.
    """
    unet.eval()
    total_loss = 0.
    total_dice = 0.

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = unet(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            dice, _ = metrics.calculate_metrics(outputs, masks)
            total_dice += dice

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)

    return avg_loss, avg_dice
