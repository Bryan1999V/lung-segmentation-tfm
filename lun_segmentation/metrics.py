"""Different metrics for evaluating segmentation models."""

import torch
from torch import nn

DICE_NUMERATOR_FACTOR = 2.0
DEFAULT_EPSILON = 1e-6


class DiceLoss(nn.Module):
    """Dice Loss implementation for measuring the overlap between predictions and targets."""

    def __init__(self) -> None:
        """Initialize DiceLoss with a small epsilon to avoid division by zero."""
        super().__init__()
        self._eps = DEFAULT_EPSILON

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Forward pass for Dice Loss computation.

        :param outputs: model predictions before activation.
        :param targets: ground truth masks.
        :return: computed Dice Loss.
        """
        probabilities = torch.sigmoid(outputs)
        probabilities = probabilities.view(probabilities.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probabilities * targets).sum(dim=1)
        union = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (DICE_NUMERATOR_FACTOR * intersection + self._eps) / (union + self._eps)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combination of Binary Cross-Entropy (BCE) Loss and Dice Loss for segmentation tasks."""

    def __init__(self) -> None:
        """Initialize BCEDiceLoss with BCE and Dice components."""
        super().__init__()
        self._bce = nn.BCEWithLogitsLoss()  # BCE loss for pixel-wise binary classification
        self._dice = DiceLoss()  # Dice loss for overlap measurement

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Forward pass for BCEDiceLoss.

        :param outputs: model predictions before activation.
        :param targets: ground truth masks.
        :return: computed BCEDiceLoss.
        """
        bce_loss = self._bce(outputs, targets)
        dice_loss = self._dice(outputs, targets)
        return bce_loss + dice_loss


def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    """
    Calculate Dice Coefficient and Intersection over Union (IoU) for model predictions.

    :param outputs: model predictions before activation.
    :param targets: ground truth masks.
    :param threshold: threshold to binarize predictions.
    :return: tuple containing Dice Coefficient and IoU.
    """
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (DICE_NUMERATOR_FACTOR * intersection) / (union + DEFAULT_EPSILON)

    intersection_iou = (preds * targets).sum(dim=1)
    union_iou = (preds.sum(dim=1) + targets.sum(dim=1)) - intersection_iou
    iou = (intersection_iou) / (union_iou + DEFAULT_EPSILON)

    return dice.mean().item(), iou.mean().item()


def dice_coefficient(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the Dice coefficient between the true and predicted masks.

    :param y_true: true mask.
    :param y_pred: predicted mask.
    :return: Dice coefficient.
    """
    intersection = (y_true * y_pred).sum()
    denominator = y_true.sum() + y_pred.sum() + DEFAULT_EPSILON
    dice = (DICE_NUMERATOR_FACTOR * intersection) / denominator
    return dice.item()


def iou_coefficient(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the Intersection over Union (IoU) coefficient between the true and predicted masks.

    :param y_true: true mask.
    :param y_pred: predicted mask.
    :return: IoU coefficient.
    """
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection + DEFAULT_EPSILON
    iou = intersection / union
    return iou.item()

def multiclass_dice_coefficient(preds, targets, epsilon=DEFAULT_EPSILON) -> list[float]:
    """
    Calculate the Dice coefficient for each class in a multi-class segmentation task.

    :param preds: predicted masks (logits).
    :param targets: true masks.
    :param epsilon: small value to avoid division by zero.
    :return: list of Dice coefficients for each class.
    """
    n_classes = preds.shape[1]
    preds = torch.argmax(preds, dim=1)  # [B, H, W]
    dice_scores = []

    for cls in range(n_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersection) / (union + epsilon)
        dice_scores.append(dice.item())

    return dice_scores