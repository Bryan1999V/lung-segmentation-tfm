"""Implementation of the training and validation loop for lung segmentation."""

import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from lun_segmentation import data
from unet import model

DEFAULT_EPSILON = 1e-6

DICE_NUMERATOR_FACTOR = 2.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(  # noqa: PLR0913
    model: model.UNet,
    optimizer: object,
    n_epochs: int,
    batch_size: int,
    dataset: data.CTLungDataset,
    train_split_size: float,
) -> None:
    """
    Train the model using the given optimizer and number of epochs.

    :param model: model to train.
    :param optimizer: optimizer to use for training.
    :param n_epochs: number of epochs to train the model.
    :param batch_size: number of samples in each batch.
    :param dataset: dataset to use for training.
    :param train_split_size: size of the training dataset. The value should be between 0.0 and 1.0. The rest will be the
        size for the validation dataset.
    """
    train_set, val_set = data.train_val_split(dataset, train_split_size)
    train_loader = data.get_dataloader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = data.get_dataloader(val_set, batch_size=batch_size, shuffle=False)
    loss_module = BCEWithLogitsLoss()

    model = model.to(DEVICE)
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        train_dice_coeff = 0.0
        train_iou_coeff = 0.0

        with tqdm(total=len(train_set), desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch") as pbar:
            for batch in train_loader:
                images = batch[0].to(DEVICE, dtype=torch.float32, memory_format=torch.channels_last)
                masks = batch[1].to(DEVICE, dtype=torch.float32)

                pred = model(images)
                loss = loss_module(pred.squeeze(1), masks.squeeze(1))
                train_dice_coeff = dice_coefficient(masks, pred, DEFAULT_EPSILON)
                train_iou_coeff = iou_coefficient(masks, pred, DEFAULT_EPSILON)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                epoch_loss += loss.item()

                val_loss, val_dice_coeff, val_iou_coeff = evaluate(model, val_loader)
                pbar.set_postfix(
                    {
                        "T-Loss": loss.item(),
                        "T-Dice": train_dice_coeff,
                        "T-IOU": train_iou_coeff,
                        "V-Loss": val_loss,
                        "V-Dice": val_dice_coeff,
                        "V-IOU": val_iou_coeff,
                    },
                )

        print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {epoch_loss / len(train_loader):.4f}")


def evaluate(
    model: model.UNet,
    val_loader: data.DataLoader,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[float, float, float]:
    """
    Evaluate the model using the given validation DataLoader.

    :param model: model to evaluate.
    :param val_loader: DataLoader containing the validation data.
    :return: tuple containing the validation cost, accuracy, Dice coefficient and IoU.
    """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(DEVICE, dtype=torch.float32)
            masks = batch[1].to(DEVICE, dtype=torch.float32)

            pred = model(images)
            loss = BCEWithLogitsLoss()(pred, masks)

            val_loss = loss.item()
            dice_coeff = dice_coefficient(masks, pred, epsilon)
            iou_coeff = iou_coefficient(masks, pred, epsilon)

    model.train()
    return val_loss, dice_coeff, iou_coeff


def dice_coefficient(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float) -> float:
    """
    Calculate the Dice coefficient between the true and predicted masks.

    :param y_true: true mask.
    :param y_pred: predicted mask.
    :param epsilon: small value to avoid division by zero.
    :return: Dice coefficient.
    """
    intersection = (y_true * y_pred).sum()
    return ((2.0 * intersection + epsilon) / (y_true.sum() + y_pred.sum() + epsilon)).mean().item()


def iou_coefficient(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float) -> float:
    """
    Calculate the Intersection over Union (IoU) coefficient between the true and predicted masks.

    :param y_true: true mask.
    :param y_pred: predicted mask.
    :param epsilon: small value to avoid division by zero.
    :return: IoU coefficient.
    """
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return ((intersection + epsilon) / (union + epsilon)).item()
