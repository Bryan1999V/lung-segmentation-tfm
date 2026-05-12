"""
Training script for multi-task nodule segmentation and classification models.

This script trains, validates, and tests multi-task models that simultaneously:
    1. Segment lung nodules (binary segmentation)
    2. Classify nodule malignancy (benign vs malignant)

The multi-task approach leverages shared features to improve both tasks.
"""

import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.common import config
from nodule_multitasking.models.models_2d import MultiTaskModels2D
from nodule_multitasking.models.models_3d import MultiTaskModels3D
from nodule_multitasking.dataloader import get_nodule_segmentation_and_classification_data_loader
from nodule_multitasking.losses import MultiTaskLoss, get_multitask_loss
from metrics.segmentation import calculate_dice, calculate_iou
from metrics.classification import calculate_specificity


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Enable CUDNN benchmarking for performance
torch.backends.cudnn.benchmark = True

USE_LUNA25_DATASET = True


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Dictionary with classification metrics
    """
    # Ensure we have both classes for metrics calculation
    unique_labels = np.unique(y_true)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    
    # Calculate specificity
    try:
        metrics['specificity'] = calculate_specificity(y_true, y_pred)
    except:
        metrics['specificity'] = 0.0
    
    # Calculate AUC only if we have both classes
    if len(unique_labels) > 1:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            metrics['auc_roc'] = 0.0
    else:
        metrics['auc_roc'] = 0.0
    
    return metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train model for one epoch.
    
    Args:
        model: Multi-task model to train
        dataloader: Training data loader
        criterion: Multi-task loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Dictionary with average losses
    """
    model.train()
    running_losses = {
        'total': 0.0,
        'seg_total': 0.0,
        'seg_ce': 0.0,
        'seg_boundary': 0.0,
        'cls': 0.0,
    }

    with tqdm(dataloader, desc=f"Epoch {epoch} [TRAIN]", leave=False) as pbar:
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            seg_masks = batch["mask"].to(device)
            cls_labels = batch["label"].to(device)
            dist_maps = batch["dist_map"].to(device)

            # Forward pass
            optimizer.zero_grad()
            seg_output, cls_output = model(images)
            
            # Compute multi-task loss
            loss_dict = criterion(seg_output, cls_output, seg_masks, cls_labels, dist_maps)
            loss = loss_dict['total']

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            running_losses['total'] += loss.item()
            running_losses['seg_total'] += loss_dict['seg_total']
            running_losses['seg_ce'] += loss_dict['seg_ce']
            running_losses['seg_boundary'] += loss_dict['seg_boundary']
            running_losses['cls'] += loss_dict['cls']
            
            # Update progress bar
            avg_loss = running_losses['total'] / (batch_idx + 1)
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "seg": f"{running_losses['seg_total']/(batch_idx+1):.4f}",
                "cls": f"{running_losses['cls']/(batch_idx+1):.4f}",
            })

    # Calculate averages
    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in running_losses.items()}


def validate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: str = "VALID",
) -> dict:
    """Validate or test multi-task model.
    
    Args:
        model: Multi-task model to validate
        dataloader: Validation/test data loader
        criterion: Multi-task loss function
        device: Device to run on
        phase: Phase name for logging ("VALID" or "TEST")
    
    Returns:
        Dictionary with metrics for both tasks
    """
    model.eval()
    running_losses = {
        'total': 0.0,
        'seg_total': 0.0,
        'cls': 0.0,
    }
    
    # Segmentation metrics
    all_seg_dice_scores = []
    all_seg_iou_scores = []
    
    # Classification metrics
    all_cls_labels = []
    all_cls_predictions = []
    all_cls_probabilities = []

    with torch.no_grad():
        with tqdm(dataloader, desc=f"[{phase}]", leave=False) as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch["image"].to(device)
                seg_masks = batch["mask"].to(device)
                cls_labels = batch["label"].to(device)
                dist_maps = batch["dist_map"].to(device)

                # Forward pass
                seg_output, cls_output = model(images)
                
                # Compute loss
                loss_dict = criterion(seg_output, cls_output, seg_masks, cls_labels, dist_maps)
                
                # Accumulate losses
                running_losses['total'] += loss_dict['total'].item() if isinstance(loss_dict['total'], torch.Tensor) else loss_dict['total']
                running_losses['seg_total'] += loss_dict['seg_total']
                running_losses['cls'] += loss_dict['cls']

                # Segmentation predictions
                seg_predictions = torch.argmax(seg_output, dim=1)
                
                # Classification predictions
                cls_probabilities = torch.softmax(cls_output, dim=1)
                cls_pred = torch.argmax(cls_probabilities, dim=1)

                # Calculate segmentation metrics per sample
                for i in range(seg_predictions.shape[0]):
                    pred_sample = seg_predictions[i].cpu().numpy()
                    mask_sample = seg_masks[i].cpu().numpy()
                    
                    # Nodule metrics (class 1)
                    pred_nodule = (pred_sample == 1).astype(np.uint8)
                    mask_nodule = (mask_sample == 1).astype(np.uint8)
                    
                    dice = calculate_dice(pred_nodule, mask_nodule)
                    iou = calculate_iou(pred_nodule, mask_nodule)
                    
                    all_seg_dice_scores.append(dice)
                    all_seg_iou_scores.append(iou)

                # Accumulate classification results
                all_cls_labels.extend(cls_labels.cpu().numpy())
                all_cls_predictions.extend(cls_pred.cpu().numpy())
                all_cls_probabilities.extend(cls_probabilities.cpu().numpy())

                # Update progress bar
                avg_loss = running_losses['total'] / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Calculate final metrics
    num_batches = len(dataloader)
    
    # Segmentation metrics
    seg_dice_mean = np.mean(all_seg_dice_scores)
    seg_dice_std = np.std(all_seg_dice_scores)
    seg_iou_mean = np.mean(all_seg_iou_scores)
    seg_iou_std = np.std(all_seg_iou_scores)
    
    # Classification metrics
    all_cls_labels = np.array(all_cls_labels)
    all_cls_predictions = np.array(all_cls_predictions)
    all_cls_probabilities = np.array(all_cls_probabilities)
    
    cls_metrics = calculate_classification_metrics(
        y_true=all_cls_labels,
        y_pred=all_cls_predictions,
        y_pred_proba=all_cls_probabilities,
    )
    
    metrics = {
        # Loss metrics
        "loss_total": running_losses['total'] / num_batches,
        "loss_seg": running_losses['seg_total'] / num_batches,
        "loss_cls": running_losses['cls'] / num_batches,
        
        # Segmentation metrics
        "seg_dice": seg_dice_mean,
        "seg_dice_std": seg_dice_std,
        "seg_iou": seg_iou_mean,
        "seg_iou_std": seg_iou_std,
        
        # Classification metrics
        "cls_accuracy": cls_metrics['accuracy'],
        "cls_precision": cls_metrics['precision'],
        "cls_recall": cls_metrics['recall'],
        "cls_f1": cls_metrics['f1'],
        "cls_auc_roc": cls_metrics['auc_roc'],
        "cls_specificity": cls_metrics['specificity'],
    }

    return metrics


def train_multitask_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
    experiment_name: str,
    save_dir: Path,
    primary_metric: str = "combined",  # "seg_dice", "cls_f1", or "combined"
) -> dict:
    """Train multi-task model with early stopping.
    
    Args:
        model: Multi-task model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Multi-task loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        experiment_name: Name of experiment
        save_dir: Directory to save best model
        primary_metric: Metric to use for model selection
                       ("seg_dice", "cls_f1", or "combined")
    
    Returns:
        Dictionary with best validation metrics
    """
    best_score = -1.0
    best_metrics = {}
    epochs_without_improvement = 0

    logging.info(f"Starting multi-task training for {experiment_name}")
    logging.info(f"Primary metric for model selection: {primary_metric}")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        # Validate
        val_metrics = validate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            phase="VALID",
        )

        # Calculate combined metric for model selection
        if primary_metric == "seg_dice":
            current_score = val_metrics['seg_dice']
        elif primary_metric == "cls_f1":
            current_score = val_metrics['cls_f1']
        else:  # combined
            # Weighted combination: 70% segmentation + 30% classification
            current_score = 0.7 * val_metrics['seg_dice'] + 0.3 * val_metrics['cls_f1']

        # Log metrics
        logging.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_losses['total']:.4f} - "
            f"Val Loss: {val_metrics['loss_total']:.4f}"
        )
        logging.info(
            f"  Segmentation - Dice: {val_metrics['seg_dice']:.4f} ± {val_metrics['seg_dice_std']:.4f}, "
            f"IoU: {val_metrics['seg_iou']:.4f} ± {val_metrics['seg_iou_std']:.4f}"
        )
        logging.info(
            f"  Classification - Acc: {val_metrics['cls_accuracy']:.4f}, "
            f"F1: {val_metrics['cls_f1']:.4f}, "
            f"AUC: {val_metrics['cls_auc_roc']:.4f}"
        )
        logging.info(f"  Combined Score: {current_score:.4f}")

        # Check for improvement
        if current_score > best_score:
            best_score = current_score
            best_metrics = val_metrics.copy()
            best_metrics['combined_score'] = current_score
            epochs_without_improvement = 0

            # Save best model
            save_path = save_dir / "best_multitask_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': best_metrics,
            }, save_path)
            logging.info(f"✓ New best model saved (Score: {best_score:.4f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            logging.info(f"Early stopping triggered after {epoch} epochs")
            break

    return best_metrics


def run_multitask_experiment(
    experiment_name: str,
    experiment_config: dict,
    device: torch.device,
    masks_dir: Path = None,
    num_train_samples: int = None,
    original_experiment_name: str = None,
) -> dict:
    """Run a single multi-task training experiment.
    
    Args:
        experiment_name: Name of experiment
        experiment_config: Configuration dictionary
        device: Device to train on
        masks_dir: Custom masks directory path
        num_train_samples: Number of training samples to use
        original_experiment_name: Original experiment name from YAML (e.g., "unet_2d")
    
    Returns:
        Dictionary with results
    """
    mode = experiment_config.get("Mode", "2D")
    encoder_name = experiment_config.get("Encoder_name", "resnet34")
    encoder_weights = experiment_config.get("Encoder_weights", "imagenet")
    
    # Extract model architecture from original experiment name
    # E.g., "unet_2d" → "unet", "unet++_2d" → "unet++", "nnunet_2d" → "nnunet"
    if original_experiment_name:
        model_arch = original_experiment_name.rsplit('_', 1)[0]  # Remove "_2d" or "_3d" suffix
    else:
        # Fallback if not provided (shouldn't happen)
        model_arch = "unet"
    batch_size = experiment_config.get("Batch", 16)
    num_epochs = experiment_config.get("Epochs", 100)
    learning_rate = experiment_config.get("LR", 1e-4)
    pixel_size = experiment_config.get("Pixel_size", 64)
    workers = experiment_config.get("Workers", 4)
    patience = experiment_config.get("Patience", 15)
    
    # Multi-task specific parameters
    seg_weight = experiment_config.get("Seg_weight", 1.0)
    cls_weight = experiment_config.get("Cls_weight", 0.5)
    use_focal = experiment_config.get("Use_focal_loss", True)
    primary_metric = experiment_config.get("Primary_metric", "combined")

    # Create experiment directory
    exp_dir = config.BEST_MODEL_PATH / "multitask"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n{'='*80}")
    logging.info(f"Multi-Task Experiment: {experiment_name}")
    logging.info(f"Mode: {mode}, Encoder: {encoder_name}, Weights: {encoder_weights}")
    logging.info(f"Batch: {batch_size}, LR: {learning_rate}, Epochs: {num_epochs}")
    logging.info(f"Task Weights - Seg: {seg_weight}, Cls: {cls_weight}")
    logging.info(f"Use Focal Loss: {use_focal}, Primary Metric: {primary_metric}")
    logging.info(f"{'='*80}\n")

    # Create dataloaders
    train_loader = get_nodule_segmentation_and_classification_data_loader(
        csv_name="train.csv",
        mode=mode,
        workers=workers,
        batch_size=batch_size,
        size_px=pixel_size,
        use_luna25_data=USE_LUNA25_DATASET,  # Use LUNA25 for labels
        shuffle=True,
        model_name=experiment_name,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        masks_dir=masks_dir,
        num_samples=num_train_samples,
    )

    val_loader = get_nodule_segmentation_and_classification_data_loader(
        csv_name="valid.csv",
        mode=mode,
        workers=workers,
        batch_size=batch_size,
        size_px=pixel_size,
        use_luna25_data=USE_LUNA25_DATASET,
        shuffle=False,
        model_name=experiment_name,
        rotations=None,
        translations=None,
        masks_dir=masks_dir,
    )

    test_loader = get_nodule_segmentation_and_classification_data_loader(
        csv_name="test.csv",
        mode=mode,
        workers=workers,
        batch_size=batch_size,
        size_px=pixel_size,
        use_luna25_data=USE_LUNA25_DATASET,
        shuffle=False,
        model_name=experiment_name,
        rotations=None,
        translations=None,
        masks_dir=masks_dir,
    )

    # Initialize model
    if mode == "2D":
        model = MultiTaskModels2D.get_model(
            model_name=model_arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
        )
    else:
        model = MultiTaskModels3D.get_model(model_name="unet3d")
    
    model.to(device)

    # Multi-task loss function
    criterion = get_multitask_loss(
        seg_weight=seg_weight,
        cls_weight=cls_weight,
        use_focal=use_focal,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    best_metrics = train_multitask_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        experiment_name=experiment_name,
        save_dir=exp_dir,
        primary_metric=primary_metric,
    )

    # Test best model
    logging.info("\nTesting best model...")
    best_model_path = exp_dir / "best_multitask_model.pth"
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        phase="TEST",
    )

    # Log test results
    logging.info("\n" + "="*80)
    logging.info("TEST RESULTS")
    logging.info("="*80)
    logging.info(
        f"Segmentation - Dice: {test_metrics['seg_dice']:.4f} ± {test_metrics['seg_dice_std']:.4f}, "
        f"IoU: {test_metrics['seg_iou']:.4f} ± {test_metrics['seg_iou_std']:.4f}"
    )
    logging.info(
        f"Classification - Acc: {test_metrics['cls_accuracy']:.4f}, "
        f"Precision: {test_metrics['cls_precision']:.4f}, "
        f"Recall: {test_metrics['cls_recall']:.4f}, "
        f"F1: {test_metrics['cls_f1']:.4f}, "
        f"AUC: {test_metrics['cls_auc_roc']:.4f}, "
        f"Spec: {test_metrics['cls_specificity']:.4f}"
    )
    logging.info("="*80 + "\n")

    # Prepare results in CSV format (similar to classification module)
    results = {
        # Experiment configuration
        "Name": experiment_name,
        "Mode": mode,
        "Encoder_name": encoder_name,
        "Encoder_weights": encoder_weights,
        "Epochs": num_epochs,
        "Batch": batch_size,
        "LR": learning_rate,
        "Pixel_size": pixel_size,
        "Workers": workers,
        "Patience": patience,
        "Seg_weight": seg_weight,
        "Cls_weight": cls_weight,
        "Use_focal": use_focal,
        "Primary_metric": primary_metric,
        "LUNA25": "Y" if USE_LUNA25_DATASET else "N",
        
        # Validation metrics - Segmentation
        "V-Dice": round(best_metrics['seg_dice'], 4),
        "V-IoU": round(best_metrics['seg_iou'], 4),
        
        # Validation metrics - Classification
        "V-AUC": round(best_metrics['cls_auc_roc'], 4),
        "V-Acc": round(best_metrics['cls_accuracy'], 4),
        "V-Prec": round(best_metrics['cls_precision'], 4),
        "V-Sens": round(best_metrics['cls_recall'], 4),
        "V-F1": round(best_metrics['cls_f1'], 4),
        "V-Spec": round(best_metrics['cls_specificity'], 4),
        
        # Test metrics - Segmentation
        "T-Dice": round(test_metrics['seg_dice'], 4),
        "T-IoU": round(test_metrics['seg_iou'], 4),
        
        # Test metrics - Classification
        "T-AUC": round(test_metrics['cls_auc_roc'], 4),
        "T-Acc": round(test_metrics['cls_accuracy'], 4),
        "T-Prec": round(test_metrics['cls_precision'], 4),
        "T-Sens": round(test_metrics['cls_recall'], 4),
        "T-F1": round(test_metrics['cls_f1'], 4),
        "T-Spec": round(test_metrics['cls_specificity'], 4),
    }

    return results


def main():
    """Main training loop for multi-task experiments."""
    # Set random seeds for reproducibility
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load experiments configuration from segmentation YAML
    config_file = config.SEGMENTATION_EXPERIMENT_PATH
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        experiments_config = yaml.safe_load(f)

    logging.info(f"Loaded {len(experiments_config)} experiments from {config_file}")

    # Initialize results CSV
    csv_file = config.BEST_MODEL_PATH / "multitask" / "multitask_experiments.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    if csv_file.exists():
        results_df = pd.read_csv(csv_file)
        logging.info(f"Loaded existing results from {csv_file}")
    else:
        results_df = pd.DataFrame()
        logging.info(f"Creating new results file: {csv_file}")

    # Multi-task specific parameters (applied to all experiments)
    multitask_params = {
        "Seg_weight": 3.0,
        "Cls_weight": 0.1,
        "Use_focal_loss": True,
        "Primary_metric": "combined",
        "Patience": config.PATIENCE if hasattr(config, 'PATIENCE') else 15,
    }

    # Masks configurations to test
    masks_configurations = [("GT", config.LUNA16_NODULE_MASKS_GT_DIR)] if not USE_LUNA25_DATASET else [("MEDSAM2", config.LUNA25_NODULES_MASKS_DIR)]

    # Run experiments
    for experiment_name, experiment_config in experiments_config.items():
        if experiment_config is None:
            logging.warning(f"Skipping {experiment_name} (no configuration)")
            continue

        logging.info(f"\n{'='*80}")
        logging.info(f"= STARTING MULTITASK EXPERIMENT: {experiment_name}")
        logging.info(f"{'='*80}\n")

        # Merge experiment config with multitask params
        full_config = {**experiment_config, **multitask_params}

        # Run with different mask types
        for masks_type, masks_dir in masks_configurations:
            try:
                # Create unique experiment name with mask type
                full_experiment_name = f"multitask_{experiment_name}_{masks_type.lower()}"

                logging.info(f"\n{'#'*80}")
                logging.info(f"# Running {full_experiment_name}")
                logging.info(f"# Masks directory: {masks_dir}")
                logging.info(f"{'#'*80}\n")

                # Run experiment
                results = run_multitask_experiment(
                    experiment_name=full_experiment_name,
                    experiment_config=full_config,
                    device=device,
                    masks_dir=masks_dir,
                    original_experiment_name=experiment_name,
                )

                # Add mask type to results
                results["Masks_type"] = masks_type

                # Append to results
                results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)

                # Save after each experiment
                results_df.to_csv(csv_file, index=False)
                logging.info(f"✓ Results saved to {csv_file}")

                # Clean up GPU memory
                torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"✗ Error running {full_experiment_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    # Print final summary
    logging.info(f"\n{'='*80}")
    logging.info("ALL MULTITASK EXPERIMENTS COMPLETED")
    logging.info(f"{'='*80}")
    logging.info(f"Total experiments: {len(results_df)}")
    logging.info(f"Results saved to: {csv_file}")
    
    if len(results_df) > 0:
        # Print top models by validation metrics
        logging.info("\nTop 5 models by Validation Dice:")
        top_dice = results_df.nlargest(5, 'V-Dice')[['Name', 'V-Dice', 'V-IoU', 'V-AUC', 'V-F1']]
        print(top_dice.to_string(index=False))
        
        logging.info("\nTop 5 models by Validation AUC:")
        top_auc = results_df.nlargest(5, 'V-AUC')[['Name', 'V-AUC', 'V-Acc', 'V-F1', 'V-Dice']]
        print(top_auc.to_string(index=False))
    
    logging.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
