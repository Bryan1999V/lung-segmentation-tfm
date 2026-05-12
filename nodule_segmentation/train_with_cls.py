"""
Training script for 3-class nodule segmentation with classification metrics.

This script trains, validates, and tests 3-class segmentation models for
lung nodule segmentation:
    - Class 0: Background (no nodule)
    - Class 1: Benign nodule
    - Class 2: Malignant nodule

Additionally, it extracts binary classification metrics (benign vs malignant)
from the segmentation masks using majority voting.
"""

import logging
import random
import shutil
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
from nodule_segmentation.models.models_2d import SegmentationModels2D
from nodule_segmentation.models.models_3d import SegmentationModels3D
from metrics.segmentation import calculate_dice, calculate_iou
from metrics.classification import extract_classification_from_mask
from nodule_segmentation.dataloader import get_nodule_segmentation_3class_data_loader
from nodule_segmentation.losses import CombinedLoss


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Enable CUDNN benchmarking for performance
torch.backends.cudnn.benchmark = True


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
    from metrics.classification import calculate_specificity
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
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Dictionary with average losses
    """
    model.train()
    running_loss = 0.0

    with tqdm(dataloader, desc=f"Epoch {epoch} [TRAIN]", leave=False) as pbar:
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            dist_maps = batch["dist_map"].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss_dict = criterion(outputs, masks, dist_maps)
            loss = loss_dict['total']

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return {"loss": running_loss / len(dataloader)}


def validate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: str = "VALID",
) -> dict:
    """Validate or test model with both segmentation and classification metrics.
    
    Args:
        model: Model to validate
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to run on
        phase: Phase name for logging ("VALID" or "TEST")
    
    Returns:
        Dictionary with segmentation and classification metrics
    """
    model.eval()
    running_loss = 0.0
    
    # Segmentation metrics
    all_dice_scores = []  # Overall nodule (class 1 + 2)
    all_iou_scores = []
    all_dice_benign = []  # Class 1
    all_dice_malignant = []  # Class 2
    
    # Classification metrics
    all_cls_labels = []
    all_cls_predictions = []
    all_cls_probabilities = []

    with torch.no_grad():
        with tqdm(dataloader, desc=f"[{phase}]", leave=False) as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                dist_maps = batch["dist_map"].to(device)

                # Forward pass
                outputs = model(images)
                loss_dict = criterion(outputs, masks, dist_maps)
                loss = loss_dict['total']

                # Get predictions (argmax over classes)
                predictions = torch.argmax(outputs, dim=1)  # (B, H, W)

                # Calculate segmentation metrics per sample (nodule vs background)
                for i in range(predictions.shape[0]):
                    pred_sample = predictions[i].cpu().numpy()
                    mask_sample = masks[i].cpu().numpy()
                    
                    # Overall nodule metrics (any nodule: class 1 or 2)
                    pred_nodule = (pred_sample > 0).astype(np.uint8)
                    mask_nodule = (mask_sample > 0).astype(np.uint8)
                    
                    dice = calculate_dice(pred_nodule, mask_nodule)
                    iou = calculate_iou(pred_nodule, mask_nodule)
                    
                    all_dice_scores.append(dice)
                    all_iou_scores.append(iou)
                    
                    # Benign nodule metrics (class 1)
                    pred_benign = (pred_sample == 1).astype(np.uint8)
                    mask_benign = (mask_sample == 1).astype(np.uint8)
                    dice_benign = calculate_dice(pred_benign, mask_benign)
                    all_dice_benign.append(dice_benign)
                    
                    # Malignant nodule metrics (class 2)
                    pred_malignant = (pred_sample == 2).astype(np.uint8)
                    mask_malignant = (mask_sample == 2).astype(np.uint8)
                    dice_malignant = calculate_dice(pred_malignant, mask_malignant)
                    all_dice_malignant.append(dice_malignant)

                # Extract classification from multi-class masks
                pred_labels, true_labels, pred_probas = extract_classification_from_mask(
                    pred_masks=predictions.cpu().numpy(),
                    gt_masks=masks.cpu().numpy(),
                    method="majority_vote"
                )
                
                # Accumulate classification results
                all_cls_labels.extend(true_labels)
                all_cls_predictions.extend(pred_labels)
                all_cls_probabilities.extend(pred_probas)

                # Accumulate loss
                running_loss += loss.item()

                # Update progress bar
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Calculate segmentation metrics
    dice_mean = np.mean(all_dice_scores)
    dice_std = np.std(all_dice_scores)
    iou_mean = np.mean(all_iou_scores)
    iou_std = np.std(all_iou_scores)
    
    # Calculate per-class Dice metrics
    dice_benign_mean = np.mean(all_dice_benign)
    dice_benign_std = np.std(all_dice_benign)
    dice_malignant_mean = np.mean(all_dice_malignant)
    dice_malignant_std = np.std(all_dice_malignant)
    
    # Calculate classification metrics
    all_cls_labels = np.array(all_cls_labels)
    all_cls_predictions = np.array(all_cls_predictions)
    all_cls_probabilities = np.array(all_cls_probabilities)
    
    cls_metrics = calculate_classification_metrics(
        y_true=all_cls_labels,
        y_pred=all_cls_predictions,
        y_pred_proba=all_cls_probabilities,
    )
    
    metrics = {
        # Loss
        "loss": running_loss / len(dataloader),
        
        # Segmentation metrics - Overall nodule
        "nodule_dice": dice_mean,
        "nodule_dice_std": dice_std,
        "nodule_iou": iou_mean,
        "nodule_iou_std": iou_std,
        
        # Segmentation metrics - Per class
        "benign_dice": dice_benign_mean,
        "benign_dice_std": dice_benign_std,
        "malignant_dice": dice_malignant_mean,
        "malignant_dice_std": dice_malignant_std,
        
        # Classification metrics
        "cls_accuracy": cls_metrics['accuracy'],
        "cls_precision": cls_metrics['precision'],
        "cls_recall": cls_metrics['recall'],
        "cls_f1": cls_metrics['f1'],
        "cls_auc_roc": cls_metrics['auc_roc'],
        "cls_specificity": cls_metrics['specificity'],
    }

    return metrics


def train_model(
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
) -> dict:
    """Train model with early stopping based on validation Dice.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        experiment_name: Name of experiment
        save_dir: Directory to save best model
    
    Returns:
        Dictionary with best validation metrics
    """
    best_val_dice = -1.0
    best_val_metrics = {}
    epochs_without_improvement = 0

    logging.info(f"Starting training for {experiment_name}")

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

        # Check for improvement
        current_val_dice = val_metrics['nodule_dice']

        if current_val_dice > best_val_dice:
            best_val_dice = current_val_dice
            best_val_metrics = val_metrics.copy()
            epochs_without_improvement = 0

            # Save best model
            model_save_path = save_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': current_val_dice,
            }, model_save_path)

            logging.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Loss: {train_losses['loss']:.4f} - "
                f"Val Dice: {val_metrics['nodule_dice']:.4f} - "
                f"Val AUC: {val_metrics['cls_auc_roc']:.4f} - "
                f"⭐ NEW BEST! Saved to {model_save_path}"
            )
        else:
            epochs_without_improvement += 1
            logging.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Loss: {train_losses['loss']:.4f} - "
                f"Val Dice: {val_metrics['nodule_dice']:.4f} - "
                f"Val AUC: {val_metrics['cls_auc_roc']:.4f} - "
                f"No improvement ({epochs_without_improvement}/{patience})"
            )

        # Early stopping
        if epochs_without_improvement >= patience:
            logging.info(f"Early stopping triggered after {epoch} epochs")
            break

    logging.info(
        f"Training completed. Best Val Dice: {best_val_dice:.4f} "
        f"(AUC: {best_val_metrics['cls_auc_roc']:.4f})"
    )

    return best_val_metrics


def run_experiment(
    experiment_name: str,
    experiment_config: dict,
    device: torch.device,
    masks_type: str = "GT",
    masks_dir: Path = None,
) -> tuple:
    """Run a single experiment with 3-class segmentation.
    
    Args:
        experiment_name: Name of experiment
        experiment_config: Configuration dictionary
        device: Device to train on
        masks_type: Type of masks ("GT" or "MedSAM2")
        masks_dir: Custom masks directory path
    
    Returns:
        Tuple of (results_dict, test_dice, test_iou, model_path)
    """
    mode = experiment_config.get("Mode", "2D")
    encoder_name = experiment_config.get("Encoder_name", "resnet34")
    encoder_weights = experiment_config.get("Encoder_weights", "imagenet")
    
    # Extract model architecture from experiment name
    model_arch = experiment_name.rsplit('_', 1)[0]  # Remove "_2d" or "_3d" suffix
    
    batch_size = experiment_config.get("Batch", 16)
    num_epochs = experiment_config.get("Epochs", 100)
    learning_rate = experiment_config.get("LR", 1e-4)
    pixel_size = experiment_config.get("Pixel_size", 64)
    workers = experiment_config.get("Workers", 4)
    patience = experiment_config.get("Patience", 15)

    # Create experiment directory
    exp_dir = config.BEST_MODEL_PATH / "segmentation_3class" / f"{experiment_name}_{masks_type.lower()}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n{'='*80}")
    logging.info(f"Experiment: {experiment_name} ({masks_type} masks)")
    logging.info(f"3-Class Segmentation: 0=bg, 1=benign, 2=malignant")
    logging.info(f"Mode: {mode}, Encoder: {encoder_name}, Weights: {encoder_weights}")
    logging.info(f"Batch: {batch_size}, LR: {learning_rate}, Epochs: {num_epochs}")
    logging.info(f"{'='*80}\n")

    # Create dataloaders (3-class)
    train_loader = get_nodule_segmentation_3class_data_loader(
        csv_name="train.csv",
        mode=mode,
        workers=workers,
        batch_size=batch_size,
        size_px=pixel_size,
        use_luna25_data=False,  # Use LUNA16
        shuffle=True,
        model_name=experiment_name,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        masks_dir=masks_dir,
    )

    val_loader = get_nodule_segmentation_3class_data_loader(
        csv_name="valid.csv",
        mode=mode,
        workers=workers,
        batch_size=batch_size,
        size_px=pixel_size,
        use_luna25_data=False,
        shuffle=False,
        model_name=experiment_name,
        rotations=None,
        translations=None,
        masks_dir=masks_dir,
    )

    test_loader = get_nodule_segmentation_3class_data_loader(
        csv_name="test.csv",
        mode=mode,
        workers=workers,
        batch_size=batch_size,
        size_px=pixel_size,
        use_luna25_data=False,
        shuffle=False,
        model_name=experiment_name,
        rotations=None,
        translations=None,
        masks_dir=masks_dir,
    )

    # Initialize model with 3 classes
    num_classes = 3
    if mode == "2D":
        model = SegmentationModels2D.get_model(
            model_name=model_arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            num_classes=num_classes,
        )
    else:
        model = SegmentationModels3D.get_model(
            model_name="unet3d",
            num_classes=num_classes,
        )
    
    model.to(device)

    # Loss and optimizer
    criterion = CombinedLoss(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    best_val_metrics = train_model(
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
    )

    # Test best model
    logging.info("\nTesting best model...")
    best_model_path = exp_dir / "best_model.pth"
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        phase="TEST",
    )

    # Log results
    logging.info("\nVALIDATION RESULTS (Best Epoch)")
    logging.info("="*80)
    logging.info(
        f"Segmentation - Overall Dice: {best_val_metrics['nodule_dice']:.4f} ± {best_val_metrics['nodule_dice_std']:.4f}, "
        f"IoU: {best_val_metrics['nodule_iou']:.4f} ± {best_val_metrics['nodule_iou_std']:.4f}"
    )
    logging.info(
        f"Segmentation - Benign Dice: {best_val_metrics['benign_dice']:.4f} ± {best_val_metrics['benign_dice_std']:.4f}, "
        f"Malignant Dice: {best_val_metrics['malignant_dice']:.4f} ± {best_val_metrics['malignant_dice_std']:.4f}"
    )
    logging.info(
        f"Classification - Acc: {best_val_metrics['cls_accuracy']:.4f}, "
        f"Precision: {best_val_metrics['cls_precision']:.4f}, "
        f"Recall: {best_val_metrics['cls_recall']:.4f}, "
        f"F1: {best_val_metrics['cls_f1']:.4f}, "
        f"AUC: {best_val_metrics['cls_auc_roc']:.4f}"
    )
    logging.info("="*80 + "\n")

    logging.info("TEST RESULTS")
    logging.info("="*80)
    logging.info(
        f"Segmentation - Overall Dice: {test_metrics['nodule_dice']:.4f} ± {test_metrics['nodule_dice_std']:.4f}, "
        f"IoU: {test_metrics['nodule_iou']:.4f} ± {test_metrics['nodule_iou_std']:.4f}"
    )
    logging.info(
        f"Segmentation - Benign Dice: {test_metrics['benign_dice']:.4f} ± {test_metrics['benign_dice_std']:.4f}, "
        f"Malignant Dice: {test_metrics['malignant_dice']:.4f} ± {test_metrics['malignant_dice_std']:.4f}"
    )
    logging.info(
        f"Classification - Acc: {test_metrics['cls_accuracy']:.4f}, "
        f"Precision: {test_metrics['cls_precision']:.4f}, "
        f"Recall: {test_metrics['cls_recall']:.4f}, "
        f"F1: {test_metrics['cls_f1']:.4f}, "
        f"AUC: {test_metrics['cls_auc_roc']:.4f}"
    )
    logging.info("="*80 + "\n")

    # Prepare results for CSV
    results = {
        # Experiment configuration
        "Name": f"{experiment_name}_{masks_type.lower()}",
        "Mode": mode,
        "Encoder_name": encoder_name,
        "Encoder_weights": encoder_weights,
        "Epochs": num_epochs,
        "Batch": batch_size,
        "LR": learning_rate,
        "Pixel_size": pixel_size,
        "Workers": workers,
        "Patience": patience,
        "Masks_type": masks_type,
        "Num_classes": 3,
        
        # Validation metrics - Segmentation (overall nodule)
        "V-Dice": round(best_val_metrics['nodule_dice'], 4),
        "V-IoU": round(best_val_metrics['nodule_iou'], 4),
        
        # Validation metrics - Segmentation per class
        "V-Dice-Benign": round(best_val_metrics['benign_dice'], 4),
        "V-Dice-Malignant": round(best_val_metrics['malignant_dice'], 4),
        
        # Validation metrics - Classification
        "V-AUC": round(best_val_metrics['cls_auc_roc'], 4),
        "V-Acc": round(best_val_metrics['cls_accuracy'], 4),
        "V-Prec": round(best_val_metrics['cls_precision'], 4),
        "V-Sens": round(best_val_metrics['cls_recall'], 4),
        "V-F1": round(best_val_metrics['cls_f1'], 4),
        "V-Spec": round(best_val_metrics['cls_specificity'], 4),
        
        # Test metrics - Segmentation (overall nodule)
        "T-Dice": round(test_metrics['nodule_dice'], 4),
        "T-IoU": round(test_metrics['nodule_iou'], 4),
        
        # Test metrics - Segmentation per class
        "T-Dice-Benign": round(test_metrics['benign_dice'], 4),
        "T-Dice-Malignant": round(test_metrics['malignant_dice'], 4),
        
        # Test metrics - Classification
        "T-AUC": round(test_metrics['cls_auc_roc'], 4),
        "T-Acc": round(test_metrics['cls_accuracy'], 4),
        "T-Prec": round(test_metrics['cls_precision'], 4),
        "T-Sens": round(test_metrics['cls_recall'], 4),
        "T-F1": round(test_metrics['cls_f1'], 4),
        "T-Spec": round(test_metrics['cls_specificity'], 4),
    }

    return results, test_metrics['nodule_dice'], test_metrics['nodule_iou'], best_model_path


def main():
    """Main training loop for 3-class segmentation with classification metrics."""
    # Set random seeds for reproducibility
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load experiments configuration
    config_file = config.SEGMENTATION_EXPERIMENT_PATH
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        experiments_config = yaml.safe_load(f)

    # Initialize results CSV
    csv_file = config.BEST_MODEL_PATH / "segmentation_3class" / "results_3class.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    if csv_file.exists():
        results_df = pd.read_csv(csv_file)
        logging.info(f"Loaded existing results from {csv_file}")
    else:
        results_df = pd.DataFrame()
        logging.info(f"Creating new results file: {csv_file}")

    # Track best model across all experiments
    best_test_dice = -1.0
    best_experiment_info = None
    best_model_dir = config.BEST_MODEL_PATH / "segmentation_3class"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = best_model_dir / "best_model_3class_ever.pth"

    # Run experiments with GT masks
    masks_configurations = [
        ("GT", config.LUNA16_NODULE_MASKS_GT_DIR),
    ]
    
    for experiment_name, experiment_config in experiments_config.items():
        if experiment_config is None:
            logging.warning(f"Skipping {experiment_name} (no configuration)")
            continue

        logging.info(f"\n{'='*80}")
        logging.info(f"= STARTING EXPERIMENT: {experiment_name}")
        logging.info(f"{'='*80}\n")
        
        for masks_type, masks_dir in masks_configurations:
            try:
                logging.info(f"\n{'#'*80}")
                logging.info(f"# Running {experiment_name} with {masks_type} masks")
                logging.info(f"# Masks directory: {masks_dir}")
                logging.info(f"{'#'*80}\n")

                results_experiment = run_experiment(
                    experiment_name=experiment_name,
                    experiment_config=experiment_config,
                    device=device,
                    masks_type=masks_type,
                    masks_dir=masks_dir,
                )

                # Unpack results
                results, test_dice, test_iou, model_path = results_experiment

                # Check if this is the best model
                if test_dice > best_test_dice:
                    best_test_dice = test_dice
                    best_experiment_info = {
                        "name": experiment_name,
                        "masks_type": masks_type,
                        "test_dice": test_dice,
                        "test_iou": test_iou,
                        "test_auc": results["T-AUC"],
                        "test_f1": results["T-F1"],
                    }
                    
                    # Copy the best model
                    shutil.copy2(model_path, best_model_path)
                    logging.info(f"🏆 New best model overall! Saved to {best_model_path}")
                    logging.info(f"   Experiment: {experiment_name} ({masks_type})")
                    logging.info(f"   Test Dice: {test_dice:.4f}, AUC: {results['T-AUC']:.4f}")

                # Save results
                results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
                results_df.to_csv(csv_file, index=False)
                logging.info(f"✓ Results saved to {csv_file}")

                # Clean up GPU memory
                torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"✗ Error running {experiment_name} with {masks_type}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    logging.info(f"\n{'='*80}")
    logging.info(f"All experiments completed!")
    logging.info(f"Results saved to: {csv_file}")
    
    # Show best model overall
    if best_experiment_info:
        logging.info(f"\n{'='*80}")
        logging.info(f"🏆 BEST 3-CLASS MODEL OVERALL 🏆")
        logging.info(f"Experiment: {best_experiment_info['name']} ({best_experiment_info['masks_type']})")
        logging.info(f"Best Model Saved: {best_model_path}")
        logging.info(f"  Test Dice:  {best_experiment_info['test_dice']:.4f}")
        logging.info(f"  Test IoU:   {best_experiment_info['test_iou']:.4f}")
        logging.info(f"  Test AUC:   {best_experiment_info['test_auc']:.4f}")
        logging.info(f"  Test F1:    {best_experiment_info['test_f1']:.4f}")
    
    logging.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
