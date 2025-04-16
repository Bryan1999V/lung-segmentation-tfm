"""Implementation of the training and validation loop for lung segmentation."""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lun_segmentation import data, metrics
from unet import model

USE_DICE_FOR_ACCURACY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42


def train(  # noqa: PLR0913
    model: model.UNet,
    optimizer: object,
    n_epochs: int,
    batch_size: int,
    dataset: data.CTLungDataset,
    val_split_size: float,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Train the model using the given optimizer and number of epochs.

    :param model: model to train.
    :param optimizer: optimizer to use for training.
    :param n_epochs: number of epochs to train the model.
    :param batch_size: number of samples in each batch.
    :param dataset: dataset to use for training.
    :param val_split_size: size of the validation dataset. The value should be between 0.0 and 1.0. The rest will be the
        size for the training dataset.
    :param threshold: threshold to binarize the predictions.
    :return: dictionary containing the training and validation loss and accuracy (dice) values for each epoch.
    """
    best_val_loss = float("inf")
    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
    }

    train_set, val_set = train_test_split(
        dataset,
        test_size=val_split_size,  # Proportion of data for validation
        random_state=SEED,  # Seed for reproducibility
    )
    train_loader = data.get_dataloader(train_set, batch_size=batch_size, num_workers=2, train_mode=True)
    val_loader = data.get_dataloader(val_set, batch_size=batch_size, num_workers=2, train_mode=False)
    loss_module = metrics.BCEDiceLoss()

    model = model.to(DEVICE)
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_val_accuracy = 0.0
        epoch_val_loss = 0.0

        model.train()
        with tqdm(total=len(train_set), desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch") as pbar:
            for nb, (images, masks) in enumerate(train_loader, start=1):
                x = images.to(DEVICE, dtype=torch.float32, memory_format=torch.channels_last)
                y_true = masks.to(DEVICE, dtype=torch.float32)

                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_module(y_pred, y_true)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)

                dice, iou = metrics.calculate_metrics(y_pred, y_true, threshold)
                epoch_accuracy += (dice if USE_DICE_FOR_ACCURACY else iou) * x.size(0)

                val_loss, val_accuracy = evaluate(model, val_loader)
                epoch_val_loss += val_loss
                epoch_val_accuracy += val_accuracy

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        model.state_dict(),
                        "/workspace/lung-segmentation-tfm/resources/models/my_best_model.pth",
                    )

                pbar.update(x.size(0))
                pbar.set_postfix(
                    {
                        "T-Loss": epoch_loss / (nb * x.size(0)),
                        "T-Accuracy": epoch_accuracy / (nb * x.size(0)),
                        "V-Loss": epoch_val_loss / nb,
                        "V-Accuracy": epoch_val_accuracy / nb,
                    },
                )
                model.train()

        history["loss"].append(epoch_loss / len(train_loader.dataset))
        history["val_loss"].append(epoch_val_loss / len(train_loader))
        history["accuracy"].append(epoch_accuracy / len(train_loader.dataset))
        history["val_accuracy"].append(epoch_val_accuracy / len(train_loader))

        print(
            f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {history['loss'][-1]:.4f} - "
            f"Train accuracy: {history['accuracy'][-1]:.4f} - Val Loss: {history['val_loss'][-1]:.4f} - "
            f"Val accuracy: {history['val_accuracy'][-1]:.4f}",
        )

    return history


def evaluate(
    model: model.UNet,
    val_loader: data.DataLoader,
    prediction_threshold: float = 0.5,
) -> tuple[float, float]:
    """
    Evaluate the model using the given validation DataLoader.

    :param model: model to evaluate.
    :param val_loader: DataLoader containing the validation data.
    :return: tuple containing the validation cost, accuracy, Dice coefficient and IoU.
    """
    model.eval()
    val_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    loss_module = metrics.BCEDiceLoss()

    with torch.no_grad():
        for images, masks in val_loader:
            x = images.to(DEVICE, dtype=torch.float32, memory_format=torch.channels_last)
            y_true = masks.to(DEVICE, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_module(y_pred, y_true)
            val_loss += loss.item() * x.size(0)

            dice, iou = metrics.calculate_metrics(y_pred, y_true, prediction_threshold)
            total_dice += dice * x.size(0)
            total_iou += iou * x.size(0)

    val_accuracy = (
        total_dice / len(val_loader.dataset) if USE_DICE_FOR_ACCURACY else total_iou / len(val_loader.dataset)
    )
    return val_loss / len(val_loader.dataset), val_accuracy
