"""Implementation of the training and validation loop for lung segmentation."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lun_segmentation import metrics
from unet import model

USE_DICE_FOR_ACCURACY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42


def train(  # noqa: PLR0913
    model: model.UNet,
    n_epochs: int,
    criterion: object,
    optimizer: object,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> dict[str, np.ndarray]:
    """
    Train the model using the given optimizer and number of epochs.

    :param model: model to train.
    :param n_epochs: number of epochs to train the model.
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

    model = model.to(DEVICE)
    for epoch in range(n_epochs):
        train_loss = 0.0
        total_dice_lung = 0.0
        total_dice_heart = 0.0
        total_dice_trachea = 0.0
        total_dice_mean = 0.0

        count = 1

        model.train()
        with tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch") as pbar:
            for images, masks in train_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                _, dice_trachea, dice_heart, dice_lung = metrics.multiclass_dice_coefficient(outputs, masks)
                total_dice_trachea += dice_trachea
                total_dice_heart += dice_heart
                total_dice_lung += dice_lung
                total_dice_mean += (dice_trachea + dice_heart + dice_lung) / 3
                count += 1

                pbar.update(images.size(0))
                pbar.set_postfix(
                    {
                        "Train Loss": train_loss / count,
                        "Dice Trachea": total_dice_trachea / count,
                        "Dice Heart": total_dice_heart / count,
                        "Dice Lung": total_dice_lung / count,
                        "Dice Mean": total_dice_mean / count,
                    },
                )

        val_loss, val_dice = evaluate(model, val_loader, criterion)
        val_dice_mean = sum(val_dice[1:]) / len(val_dice[1:])

        print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_dice_mean:.4f}")
        print(f"Validation Dice Coefficients - Lung: {val_dice[3]:.4f}, Heart: {val_dice[2]:.4f}, Trachea: {val_dice[1]:.4f}")
        print(f"-" * 30)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                "/root/models/best_unet_model_bryan.pth",
            )

        history["loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["accuracy"].append(total_dice_mean / len(train_loader))
        history["val_accuracy"].append(val_dice_mean)

    return history


def evaluate(
    model: model.UNet,
    val_loader: DataLoader,
    criterion: object,
) -> tuple[float, float]:
    """
    Evaluate the model using the given validation DataLoader.

    :param model: model to evaluate.
    :param val_loader: DataLoader containing the validation data.
    :param criterion: loss function to use for evaluation.
    :return: tuple containing the validation cost, accuracy, Dice coefficient and IoU.
    """
    model.eval()
    total_loss = 0.0
    total_dice = [0.0, 0.0, 0.0, 0.0]
    count = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)

            dice_scores = metrics.multiclass_dice_coefficient(outputs, masks)
            total_loss += loss.item()
            for i in range(outputs.shape[1]):
                total_dice[i] += dice_scores[i]
            count += 1

    avg_loss = total_loss / count
    avg_dice = [d / count for d in total_dice]

    return avg_loss, avg_dice
