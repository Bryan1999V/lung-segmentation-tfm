"""Nodule Segmentation and Classification Multi-Task Module.

This module provides multi-task learning functionality for simultaneous:
    - Nodule Segmentation: Binary mask prediction (background vs nodule)
    - Malignancy Classification: Binary classification (benign vs malignant)

The multi-task approach leverages shared features from a common encoder
to improve both tasks through joint optimization.
"""

from nodule_multitasking.dataloader import (
    CTNoduleSegmentationAndClassificationDataset,
    get_nodule_segmentation_and_classification_data_loader,
)

__all__ = [
    "CTNoduleSegmentationAndClassificationDataset",
    "get_nodule_segmentation_and_classification_data_loader",
]
