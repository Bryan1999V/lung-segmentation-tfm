"""
Boundary Loss for Highly Unbalanced Segmentation.

Official implementation from: https://github.com/LIVIAETS/boundary-loss
Paper: "Boundary loss for highly unbalanced segmentation"
Kervadec et al., 2019 (MIDL) / 2021 (Medical Image Analysis)
https://arxiv.org/abs/1812.07032

MIT License - Copyright (c) 2023 Hoel Kervadec

Adapted for lung nodule segmentation with standard PyTorch interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance
from typing import Optional, List, Tuple, Set, cast
from torch import Tensor, einsum


# ============================================================================
# Utility Functions (from official repository)
# ============================================================================

def uniq(a: Tensor) -> Set:
    """Get unique values from tensor."""
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub) -> bool:
    """Check if tensor values are subset of given values."""
    return uniq(a).issubset(sub)


def simplex(t: Tensor, axis=1) -> bool:
    """Check if tensor is a simplex (values sum to 1)."""
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    """Check if tensor is one-hot encoded."""
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    """
    Convert class indices to one-hot encoding.
    
    Args:
        seg: Tensor of shape (B, H, W) with class indices
        K: Number of classes
    
    Returns:
        One-hot tensor of shape (B, K, H, W)
    """
    assert sset(seg, list(range(K))), (uniq(seg), K)
    
    b, *img_shape = seg.shape
    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(
        1, seg[:, None, ...], 1
    )
    
    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)
    
    return res


def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, ...] = None, dtype=None) -> np.ndarray:
    """
    Compute signed distance transform from one-hot encoded segmentation.
    
    Official implementation from LIVIAETS/boundary-loss repository.
    
    Args:
        seg: One-hot encoded segmentation of shape (K, H, W) or (K, D, H, W)
             where K is the number of classes
        resolution: Pixel/voxel spacing for each axis (e.g., (1.0, 1.0) for 2D)
        dtype: Data type for output
    
    Returns:
        Distance map of same shape as seg, with signed distances:
        - Negative inside the object
        - Positive outside the object
        - Zero at the boundary
    """
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)
    
    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(bool)
        
        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel
    
    return res


# ============================================================================
# SurfaceLoss (Boundary Loss) - Official Implementation
# ============================================================================

class SurfaceLoss(nn.Module):
    """
    Surface/Boundary Loss for highly unbalanced segmentation.
    
    Official implementation from: https://github.com/LIVIAETS/boundary-loss
    
    The loss computes a weighted sum over softmax probabilities, where
    the weights are pre-computed signed distance maps from ground truth.
    
    Args:
        idc: List of class indices to supervise with boundary loss.
             For example: [1, 2] to only supervise nodule classes
    
    Usage:
        >>> # For binary nodule segmentation (background, nodule)
        >>> loss_fn = SurfaceLoss(idc=[1])  # Only supervise nodules
        >>> 
        >>> # probs: (B, C, H, W) probabilities after softmax
        >>> # dist_maps: (B, C, H, W) pre-computed distance maps
        >>> loss = loss_fn(probs, dist_maps)
    """
    
    def __init__(self, idc: List[int], **kwargs):
        super(SurfaceLoss, self).__init__()
        self.idc: List[int] = idc
        print(f"Initialized {self.__class__.__name__} with idc={idc}")
    
    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        """
        Compute surface/boundary loss.
        
        Args:
            probs: Softmax probabilities of shape (B, C, H, W) or (B, C, D, H, W)
            dist_maps: Pre-computed distance maps of shape (B, C, H, W) or (B, C, D, H, W)
        
        Returns:
            Scalar loss value (can be negative)
        """
        assert simplex(probs)
        assert not one_hot(dist_maps)
        
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)
        
        # Detect if 2D or 3D based on number of dimensions
        if pc.ndim == 4:  # 2D: (B, C, H, W)
            multipled = einsum("bkwh,bkwh->bkwh", pc, dc)
        elif pc.ndim == 5:  # 3D: (B, C, D, H, W)
            multipled = einsum("bkdhw,bkdhw->bkdhw", pc, dc)
        else:
            raise ValueError(f"Unexpected number of dimensions: {pc.ndim}. Expected 4 (2D) or 5 (3D).")
        
        loss = multipled.mean()
        
        return loss


# Alias for compatibility
BoundaryLoss = SurfaceLoss


# ============================================================================
# Helper Functions for Integration
# ============================================================================

def compute_distance_maps_batch(
    targets: Tensor,
    num_classes: int,
    resolution: Tuple[float, ...] = (1.0, 1.0),
) -> Tensor:
    """
    Compute distance maps for a batch of segmentation masks.
    
    Args:
        targets: Ground truth masks of shape (B, H, W) with class indices
        num_classes: Number of classes
        resolution: Pixel/voxel spacing
    
    Returns:
        Distance maps of shape (B, C, H, W)
    """
    batch_size = targets.shape[0]
    device = targets.device
    
    # Convert to one-hot
    one_hot_targets = class2one_hot(targets, num_classes)
    
    # Compute distance maps
    dist_maps = np.zeros_like(one_hot_targets.cpu().numpy(), dtype=np.float32)
    
    for b in range(batch_size):
        dist_maps[b] = one_hot2dist(
            one_hot_targets[b].cpu().numpy(),
            resolution=resolution
        )
    
    return torch.from_numpy(dist_maps).to(device)


def compute_dist_map_transform(
    seg_mask: np.ndarray,
    num_classes: int,
    resolution: Tuple[float, ...] = (1.0, 1.0),
) -> np.ndarray:
    """
    Transform for computing distance maps in the dataloader.
    
    This should be called in the __getitem__ method of your dataset.
    
    Args:
        seg_mask: Segmentation mask of shape (H, W) with class indices
        num_classes: Number of classes
        resolution: Pixel/voxel spacing
    
    Returns:
        Distance maps of shape (C, H, W)
    """
    # Convert to one-hot
    one_hot_mask = np.zeros((num_classes, *seg_mask.shape), dtype=np.int32)
    for c in range(num_classes):
        one_hot_mask[c] = (seg_mask == c).astype(np.int32)
    
    # Compute distance transform
    dist_map = one_hot2dist(one_hot_mask, resolution=resolution, dtype=np.float32)
    
    return dist_map


# ============================================================================
# Combined Loss for Easy Integration
# ============================================================================

class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy + Boundary Loss.
    
    This combines:
    1. Cross-Entropy: Region-based loss for overall segmentation
    2. Boundary Loss: Boundary-focused loss for sharp edges
    
    Args:
        num_classes: Number of classes
        ce_weight: Class weights for cross-entropy loss
        boundary_idc: Classes to supervise with boundary loss
        boundary_weight: Weight for boundary loss (α in the paper)
        resolution: Pixel spacing for distance computation
        
    Usage:
        >>> loss_fn = CombinedLoss(
        ...     num_classes=2,
        ...     ce_weight=torch.tensor([0.1, 1.0]),
        ...     boundary_idc=[1],
        ...     boundary_weight=1.0,
        ... )
        >>> 
        >>> outputs = model(images)  # (B, C, H, W)
        >>> dist_maps = batch["dist_map"]  # Pre-computed
        >>> loss = loss_fn(outputs, targets, dist_maps)
    """
    
    def __init__(
        self,
        num_classes: int,
        ce_weight: Optional[Tensor] = None,
        boundary_idc: Optional[List[int]] = None,
        boundary_weight: float = 1.0,
        resolution: Tuple[float, ...] = (1.0, 1.0),
    ):
        super(CombinedLoss, self).__init__()
        
        self.num_classes = num_classes
        self.boundary_weight = boundary_weight
        self.resolution = resolution
        
        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
        
        # Boundary loss
        if boundary_idc is None:
            # Default: supervise all classes except background
            boundary_idc = list(range(1, num_classes))
        self.boundary_loss = SurfaceLoss(idc=boundary_idc)
        
    def forward(
        self,
        outputs: Tensor,
        targets: Tensor,
        dist_maps: Optional[Tensor] = None,
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            outputs: Model logits of shape (B, C, H, W)
            targets: Ground truth masks of shape (B, H, W)
            dist_maps: Pre-computed distance maps of shape (B, C, H, W)
                      If None, will be computed on-the-fly (slower)
        
        Returns:
            Dictionary with individual and total losses
        """
        # Cross-entropy loss
        ce_loss_value = self.ce_loss(outputs, targets)
        
        # Get softmax probabilities for boundary loss
        probs = F.softmax(outputs, dim=1)
        
        # Compute distance maps if not provided
        if dist_maps is None:
            dist_maps = compute_distance_maps_batch(
                targets=targets,
                num_classes=self.num_classes,
                resolution=self.resolution,
            )
        
        # Boundary loss
        boundary_loss_value = self.boundary_loss(probs, dist_maps)
        
        # Combined loss
        total_loss = ce_loss_value + self.boundary_weight * boundary_loss_value
        
        return {
            'total': total_loss,
            'ce': ce_loss_value.item(),
            'boundary': boundary_loss_value.item(),
        }
    
    def set_boundary_weight(self, weight: float):
        """Update the boundary loss weight (for scheduling)."""
        self.boundary_weight = weight


# ============================================================================
# Dice + Boundary Loss (Alternative)
# ============================================================================

class GeneralizedDice(nn.Module):
    """Generalized Dice Loss from official repository."""
    
    def __init__(self, idc: List[int], **kwargs):
        super(GeneralizedDice, self).__init__()
        self.idc: List[int] = idc
        print(f"Initialized {self.__class__.__name__} with idc={idc}")
    
    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)
        
        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)
        
        w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
        union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))
        
        divided: Tensor = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)
        
        loss = divided.mean()
        
        return loss


class DiceBoundaryLoss(nn.Module):
    """Combined Dice + Boundary Loss."""
    
    def __init__(
        self,
        num_classes: int,
        boundary_idc: Optional[List[int]] = None,
        dice_idc: Optional[List[int]] = None,
        boundary_weight: float = 1.0,
        resolution: Tuple[float, ...] = (1.0, 1.0),
    ):
        super(DiceBoundaryLoss, self).__init__()
        
        self.num_classes = num_classes
        self.boundary_weight = boundary_weight
        self.resolution = resolution
        
        if dice_idc is None:
            dice_idc = list(range(num_classes))
        if boundary_idc is None:
            boundary_idc = list(range(1, num_classes))
        
        self.dice_loss = GeneralizedDice(idc=dice_idc)
        self.boundary_loss = SurfaceLoss(idc=boundary_idc)
    
    def forward(
        self,
        outputs: Tensor,
        targets: Tensor,
        dist_maps: Optional[Tensor] = None,
    ) -> dict:
        """Compute Dice + Boundary loss."""
        # Get probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Convert targets to one-hot for Dice
        targets_one_hot = class2one_hot(targets, self.num_classes).float()
        
        # Dice loss
        dice_loss_value = self.dice_loss(probs, targets_one_hot)
        
        # Compute distance maps if not provided
        if dist_maps is None:
            dist_maps = compute_distance_maps_batch(
                targets=targets,
                num_classes=self.num_classes,
                resolution=self.resolution,
            )
        
        # Boundary loss
        boundary_loss_value = self.boundary_loss(probs, dist_maps)
        
        # Combined loss
        total_loss = dice_loss_value + self.boundary_weight * boundary_loss_value
        
        return {
            'total': total_loss,
            'dice': dice_loss_value.item(),
            'boundary': boundary_loss_value.item(),
        }
    
    def set_boundary_weight(self, weight: float):
        """Update boundary loss weight."""
        self.boundary_weight = weight
