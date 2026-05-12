"""Multi-Task Loss for Nodule Segmentation and Classification.

This module implements combined loss functions for simultaneous:
    - Nodule Segmentation (with Boundary Loss)
    - Malignancy Classification (with Focal Loss for class imbalance)

Based on multi-task learning literature for medical image analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from torch import Tensor

from nodule_segmentation.losses import CombinedLoss, SurfaceLoss


# ============================================================================
# Focal Loss for Classification (Handles Class Imbalance)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    From "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    https://arxiv.org/abs/1708.02002
    
    Focal loss applies a modulating term to the cross entropy loss to focus
    learning on hard negative examples. Particularly useful for imbalanced
    datasets like benign/malignant nodule classification.
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
               or a Tensor of weights for each class
        gamma: Exponent of the modulating factor (1 - p_t)^gamma
        reduction: Specifies reduction to apply: 'none' | 'mean' | 'sum'
        
    Shape:
        - Input: (N, C) where N = batch size, C = number of classes
        - Target: (N,) where each value is 0 ≤ targets[i] ≤ C-1
        - Output: scalar if reduction is 'mean' or 'sum', otherwise (N,)
    """
    
    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: Predictions (logits) of shape (N, C)
            targets: Ground truth class indices of shape (N,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# Multi-Task Loss
# ============================================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task nodule segmentation and classification.
    
    This loss combines:
    1. Segmentation Loss: Cross-Entropy + Boundary Loss for precise nodule delineation
    2. Classification Loss: Focal Loss for malignancy prediction
    
    The losses are weighted to balance the two tasks during training.
    
    Args:
        num_seg_classes: Number of segmentation classes (default: 2)
        num_cls_classes: Number of classification classes (default: 2)
        seg_weight: Weight for segmentation loss (α)
        cls_weight: Weight for classification loss (β)
        seg_ce_weight: Class weights for segmentation cross-entropy
        boundary_idc: Classes to supervise with boundary loss
        boundary_weight: Weight for boundary loss component
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        use_focal: Whether to use focal loss (True) or standard CE (False)
        
    Usage:
        >>> loss_fn = MultiTaskLoss(
        ...     seg_weight=1.0,
        ...     cls_weight=0.5,
        ...     use_focal=True,
        ... )
        >>> 
        >>> seg_output, cls_output = model(images)
        >>> loss_dict = loss_fn(
        ...     seg_output, cls_output,
        ...     seg_targets, cls_targets,
        ...     dist_maps
        ... )
        >>> loss = loss_dict['total']
    """
    
    def __init__(
        self,
        num_seg_classes: int = 2,
        num_cls_classes: int = 2,
        seg_weight: float = 1.0,
        cls_weight: float = 0.5,
        seg_ce_weight: Optional[Tensor] = None,
        boundary_idc: Optional[list] = None,
        boundary_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_focal: bool = True,
    ):
        super(MultiTaskLoss, self).__init__()
        
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.use_focal = use_focal
        
        # Segmentation loss (Cross-Entropy + Boundary)
        self.seg_loss = CombinedLoss(
            num_classes=num_seg_classes,
            ce_weight=seg_ce_weight,
            boundary_idc=boundary_idc,
            boundary_weight=boundary_weight,
        )
        
        # Classification loss
        if use_focal:
            self.cls_loss = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction='mean',
            )
        else:
            self.cls_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        seg_output: Tensor,
        cls_output: Tensor,
        seg_target: Tensor,
        cls_target: Tensor,
        dist_maps: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            seg_output: Segmentation logits of shape (B, C_seg, H, W)
            cls_output: Classification logits of shape (B, C_cls)
            seg_target: Segmentation ground truth of shape (B, H, W)
            cls_target: Classification ground truth of shape (B,)
            dist_maps: Pre-computed distance maps of shape (B, C_seg, H, W)
        
        Returns:
            Dictionary containing:
                - total: Combined weighted loss
                - seg_total: Total segmentation loss
                - seg_ce: Segmentation cross-entropy loss
                - seg_boundary: Segmentation boundary loss
                - cls: Classification loss
        """
        # Compute segmentation loss
        seg_loss_dict = self.seg_loss(seg_output, seg_target, dist_maps)
        seg_total_loss = seg_loss_dict['total']
        
        # Compute classification loss
        cls_loss_value = self.cls_loss(cls_output, cls_target)
        
        # Combined multi-task loss
        total_loss = (
            self.seg_weight * seg_total_loss +
            self.cls_weight * cls_loss_value
        )
        
        return {
            'total': total_loss,
            'seg_total': seg_total_loss.item() if isinstance(seg_total_loss, Tensor) else seg_total_loss,
            'seg_ce': seg_loss_dict['ce'],
            'seg_boundary': seg_loss_dict['boundary'],
            'cls': cls_loss_value.item() if isinstance(cls_loss_value, Tensor) else cls_loss_value,
        }
    
    def set_task_weights(self, seg_weight: float, cls_weight: float):
        """Update task weights (for dynamic weight adjustment)."""
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
    
    def set_boundary_weight(self, weight: float):
        """Update boundary loss weight."""
        self.seg_loss.set_boundary_weight(weight)


# ============================================================================
# Dynamic Weight Averaging (DWA) - Optional Advanced Feature
# ============================================================================

class DynamicWeightAveraging:
    """
    Dynamic Weight Averaging for multi-task learning.
    
    From "End-to-End Multi-Task Learning with Attention" (Liu et al., 2019)
    https://arxiv.org/abs/1803.10704
    
    Automatically adjusts task weights based on the rate of change of losses.
    Tasks that are learning slowly get higher weights.
    
    Args:
        num_tasks: Number of tasks
        temperature: Temperature parameter T (default: 2.0)
        
    Usage:
        >>> dwa = DynamicWeightAveraging(num_tasks=2)
        >>> # After each epoch
        >>> task_losses = [seg_loss, cls_loss]
        >>> new_weights = dwa.get_weights(task_losses)
        >>> loss_fn.set_task_weights(new_weights[0], new_weights[1])
    """
    
    def __init__(self, num_tasks: int = 2, temperature: float = 2.0):
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.previous_losses = None
        
    def get_weights(self, current_losses: list) -> list:
        """
        Compute task weights based on loss dynamics.
        
        Args:
            current_losses: List of current task losses
            
        Returns:
            List of task weights
        """
        if self.previous_losses is None:
            # First epoch: use equal weights
            self.previous_losses = current_losses
            return [1.0] * self.num_tasks
        
        # Compute rate of change
        loss_ratios = [
            current_losses[i] / self.previous_losses[i]
            for i in range(self.num_tasks)
        ]
        
        # Apply temperature
        weights = [
            self.num_tasks * torch.exp(ratio / self.temperature) 
            for ratio in loss_ratios
        ]
        
        # Normalize
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Update previous losses
        self.previous_losses = current_losses
        
        return weights


# ============================================================================
# Uncertainty Weighting (Alternative to DWA)
# ============================================================================

class UncertaintyWeighting(nn.Module):
    """
    Learnable task uncertainty weighting.
    
    From "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018)
    https://arxiv.org/abs/1705.07115
    
    Learns task weights as model parameters during training.
    
    Args:
        num_tasks: Number of tasks (default: 2)
        
    Usage:
        >>> uncertainty_weight = UncertaintyWeighting(num_tasks=2)
        >>> # In training loop
        >>> weighted_loss = uncertainty_weight(seg_loss, cls_loss)
        >>> # Don't forget to add uncertainty_weight.parameters() to optimizer!
    """
    
    def __init__(self, num_tasks: int = 2):
        super(UncertaintyWeighting, self).__init__()
        # Log variance parameters (learned during training)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, *task_losses) -> Tensor:
        """
        Compute uncertainty-weighted multi-task loss.
        
        Args:
            *task_losses: Variable number of task losses
            
        Returns:
            Weighted combined loss
        """
        total_loss = 0
        for i, loss in enumerate(task_losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss
    
    def get_current_weights(self) -> list:
        """Get current task weights."""
        with torch.no_grad():
            weights = [torch.exp(-log_var).item() for log_var in self.log_vars]
        return weights


# ============================================================================
# Convenience Function
# ============================================================================

def get_multitask_loss(
    seg_weight: float = 1.0,
    cls_weight: float = 0.5,
    use_focal: bool = True,
    use_uncertainty: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create multi-task loss.
    
    Args:
        seg_weight: Weight for segmentation task
        cls_weight: Weight for classification task
        use_focal: Whether to use Focal Loss for classification
        use_uncertainty: Whether to use learnable uncertainty weighting
        **kwargs: Additional arguments for MultiTaskLoss
        
    Returns:
        Multi-task loss module
    """
    if use_uncertainty:
        # Return wrapper with uncertainty weighting
        base_loss = MultiTaskLoss(
            seg_weight=1.0,  # Will be learned
            cls_weight=1.0,  # Will be learned
            use_focal=use_focal,
            **kwargs
        )
        uncertainty = UncertaintyWeighting(num_tasks=2)
        
        # Create wrapper
        class UncertaintyMultiTaskLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_loss = base_loss
                self.uncertainty = uncertainty
                
            def forward(self, seg_output, cls_output, seg_target, cls_target, dist_maps=None):
                loss_dict = self.base_loss(seg_output, cls_output, seg_target, cls_target, dist_maps)
                seg_loss = loss_dict['seg_total']
                cls_loss = loss_dict['cls']
                
                # Apply uncertainty weighting
                total_loss = self.uncertainty(seg_loss, cls_loss)
                loss_dict['total'] = total_loss
                loss_dict['uncertainty_weights'] = self.uncertainty.get_current_weights()
                
                return loss_dict
        
        return UncertaintyMultiTaskLoss()
    else:
        return MultiTaskLoss(
            seg_weight=seg_weight,
            cls_weight=cls_weight,
            use_focal=use_focal,
            **kwargs
        )
