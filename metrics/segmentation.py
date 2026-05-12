import numpy as np


def calculate_dice(pred, gt):
    """
    Calculates DICE coefficient between prediction and ground truth.

    DICE = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred: Predicted mask (boolean or binary)
        gt: Ground truth mask (boolean or binary)

    Returns:
        float: DICE score [0, 1]
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()

    if pred_bool.sum() + gt_bool.sum() == 0:
        return 1.0  # Both empty = perfect

    dice = 2.0 * intersection / (pred_bool.sum() + gt_bool.sum())
    return dice


def calculate_iou(pred, gt):
    """
    Calculates Intersection over Union (IoU) between prediction and ground truth.

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        pred: Predicted mask (boolean or binary)
        gt: Ground truth mask (boolean or binary)

    Returns:
        float: IoU score [0, 1]
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        return 1.0  # Both empty = perfect

    iou = intersection / union
    return iou


def calculate_precision(pred, gt):
    """
    Calculates Precision between prediction and ground truth.

    Precision = TP / (TP + FP)
    Where TP = True Positives, FP = False Positives

    Args:
        pred: Predicted mask (boolean or binary)
        gt: Ground truth mask (boolean or binary)

    Returns:
        float: Precision score [0, 1]
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, ~gt_bool).sum()

    if tp + fp == 0:
        return 1.0 if tp == 0 else 0.0  # No positive predictions = undefined, return 1.0 or 0.0

    precision = tp / (tp + fp)
    return precision


def calculate_sensitivity(pred, gt):
    """
    Calculates Sensitivity (Recall) between prediction and ground truth.

    Sensitivity = TP / (TP + FN)
    Where TP = True Positives, FN = False Negatives

    Args:
        pred: Predicted mask (boolean or binary)
        gt: Ground truth mask (boolean or binary)

    Returns:
        float: Sensitivity score [0, 1]
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    tp = np.logical_and(pred_bool, gt_bool).sum()
    fn = np.logical_and(~pred_bool, gt_bool).sum()

    if tp + fn == 0:
        return 1.0 if tp == 0 else 0.0  # No positive ground truth = undefined, return 1.0 or 0.0

    sensitivity = tp / (tp + fn)
    return sensitivity
