"""
Training script for nodule segmentation models.

This script trains, validates, and tests binary segmentation models for
lung nodule segmentation with 2 classes:
    - Class 0: No pulmonary nodule (background)
    - Class 1: Pulmonary nodule (benign or malignant)
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.common import config
from nodule_segmentation.models.models_2d import SegmentationModels2D
from nodule_segmentation.models.models_3d import SegmentationModels3D
from metrics.segmentation import calculate_dice, calculate_iou
from nodule_segmentation.dataloader import get_nodule_segmentation_data_loader
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


def calculate_segmentation_metrics(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 2) -> dict:
    """Calculate segmentation metrics for binary predictions.
    
    Args:
        predictions: Predicted class labels (H, W) or (B, H, W)
        targets: Ground truth class labels (H, W) or (B, H, W)
        num_classes: Number of classes (default: 2 for binary: 0=background, 1=nodule)
    
    Returns:
        Dictionary with metrics per class and mean metrics
    """
    metrics = {
        'dice': {},
        'iou': {},
        'mean_dice': 0.0,
        'mean_iou': 0.0,
    }
    
    # Calculate metrics for each class
    dice_scores = []
    iou_scores = []
    
    for class_id in range(num_classes):
        pred_binary = (predictions == class_id).astype(np.uint8)
        target_binary = (targets == class_id).astype(np.uint8)
        
        dice = calculate_dice(pred_binary, target_binary)
        iou = calculate_iou(pred_binary, target_binary)
        
        metrics['dice'][f'class_{class_id}'] = dice
        metrics['iou'][f'class_{class_id}'] = iou
        
        dice_scores.append(dice)
        iou_scores.append(iou)
    
    # Calculate mean metrics (mean of both classes)
    metrics['mean_dice'] = np.mean(dice_scores)
    metrics['mean_iou'] = np.mean(iou_scores)
    
    # For binary segmentation, nodule metrics are class 1 metrics
    metrics['nodule_dice'] = metrics['dice']['class_1']
    metrics['nodule_iou'] = metrics['iou']['class_1']
    
    # Background metrics are class 0 metrics
    metrics['background_dice'] = metrics['dice']['class_0']
    metrics['background_iou'] = metrics['iou']['class_0']
    
    return metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Average training loss
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

            # Update metrics
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return running_loss / len(dataloader)


def validate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: str = "VALID",
    num_classes: int = 2,
) -> dict:
    """Validate or test model.
    
    Args:
        model: Model to validate
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to run on
        phase: Phase name for logging ("VALID" or "TEST")
        num_classes: Number of classes
    
    Returns:
        Dictionary with metrics including mean and std
    """
    model.eval()
    running_loss = 0.0
    all_dice_scores = []
    all_iou_scores = []

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

                # Get predictions
                predictions = torch.argmax(outputs, dim=1)

                # Calculate metrics per sample in batch
                for i in range(predictions.shape[0]):
                    pred_sample = predictions[i].cpu().numpy()
                    mask_sample = masks[i].cpu().numpy()
                    
                    # Calculate nodule metrics (class 1)
                    pred_nodule = (pred_sample == 1).astype(np.uint8)
                    mask_nodule = (mask_sample == 1).astype(np.uint8)
                    
                    dice = calculate_dice(pred_nodule, mask_nodule)
                    iou = calculate_iou(pred_nodule, mask_nodule)
                    
                    all_dice_scores.append(dice)
                    all_iou_scores.append(iou)

                # Accumulate loss
                running_loss += loss.item()

                # Update progress bar
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Calculate mean and std
    dice_mean = np.mean(all_dice_scores)
    dice_std = np.std(all_dice_scores)
    iou_mean = np.mean(all_iou_scores)
    iou_std = np.std(all_iou_scores)
    
    metrics = {
        "loss": running_loss / len(dataloader),
        "nodule_dice": dice_mean,
        "nodule_dice_std": dice_std,
        "nodule_iou": iou_mean,
        "nodule_iou_std": iou_std,
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
    """Train model with early stopping.
    
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
    best_dice = -1.0
    best_metrics = {}
    epochs_without_improvement = 0

    logging.info(f"Starting training for {experiment_name}")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(
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

        # Log metrics
        logging.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_metrics['loss']:.4f} - "
            f"Val Dice: {val_metrics['nodule_dice']:.4f} ± {val_metrics['nodule_dice_std']:.4f} - "
            f"Val IoU: {val_metrics['nodule_iou']:.4f} ± {val_metrics['nodule_iou_std']:.4f}"
        )

        # Check for improvement (use nodule_dice as primary metric)
        if val_metrics['nodule_dice'] > best_dice:
            best_dice = val_metrics['nodule_dice']
            best_metrics = val_metrics.copy()
            epochs_without_improvement = 0

            # Save best model
            save_path = save_dir / "best_current_model.pth"
            torch.save(model.state_dict(), save_path)
            logging.info(f"✓ New best model saved (Dice: {best_dice:.4f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            logging.info(f"Early stopping triggered after {epoch} epochs")
            break

    return best_metrics


def run_experiment(
    experiment_name: str,
    experiment_config: dict,
    device: torch.device,
    masks_type: str = "GT",
    masks_dir: Path = None,
    num_train_samples: int = None,
) -> tuple:
    """Run a single training experiment.
    
    Args:
        experiment_name: Name of experiment
        experiment_config: Configuration dictionary
        device: Device to train on
        masks_type: Type of masks being used ("GT" or "MedSAM2")
        masks_dir: Custom masks directory path
        num_train_samples: Number of training samples to use (None = use all)
    
    Returns:
        Tuple of (results_dict, test_dice, test_iou, model_path)
        where results_dict contains CSV-friendly metrics
    """
    mode = experiment_config["Mode"]
    encoder_name = experiment_config["Encoder_name"]
    encoder_weights = experiment_config["Encoder_weights"]
    batch_size = experiment_config["Batch"]
    num_epochs = experiment_config["Epochs"]
    learning_rate = experiment_config["LR"]
    pixel_size = experiment_config["Pixel_size"]
    workers = experiment_config["Workers"]

    # Create experiment directory
    exp_dir = config.BEST_MODEL_PATH / "segmentation"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n{'='*80}")
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"Masks Type: {masks_type}")
    logging.info(f"Mode: {mode}, Encoder: {encoder_name}, Weights: {encoder_weights}")
    logging.info(f"Batch: {batch_size}, LR: {learning_rate}, Epochs: {num_epochs}")
    logging.info(f"{'='*80}\n")

    # Create dataloaders
    train_loader = get_nodule_segmentation_data_loader(
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
        num_samples=num_train_samples,
    )

    val_loader = get_nodule_segmentation_data_loader(
        csv_name="valid.csv",
        mode=mode,
        workers=workers,
        batch_size=batch_size,
        size_px=pixel_size,
        use_luna25_data=False,  # Use LUNA16
        shuffle=False,
        model_name=experiment_name,
        rotations=None,
        translations=None,
        masks_dir=masks_dir,
    )

    test_loader = get_nodule_segmentation_data_loader(
        csv_name="test.csv",
        mode=mode,
        workers=workers,
        batch_size=batch_size,
        size_px=pixel_size,
        use_luna25_data=False,  # Use LUNA16
        shuffle=False,
        model_name=experiment_name,
        rotations=None,
        translations=None,
        masks_dir=masks_dir,
    )

    # Initialize model
    model = SegmentationModels2D.get_model(experiment_name, encoder_name, encoder_weights) if mode == "2D" else SegmentationModels3D.get_model(experiment_name, encoder_name=encoder_name, encoder_weights=encoder_weights)
    model.to(device)

    # Loss function: Combined Cross-Entropy + Boundary Loss
    # Background is abundant (~95%), nodules are rare (~5%)
    class_weights = torch.tensor([0.1, 1.0]).to(device)  # Lower weight for background
    
    # Set resolution based on mode
    resolution = (1.0, 1.0, 1.0) if mode == "3D" else (1.0, 1.0)
    
    criterion = CombinedLoss(
        num_classes=2,
        ce_weight=class_weights,
        boundary_idc=[1],  # Only supervise nodule boundaries
        boundary_weight=1.0,
        resolution=resolution,
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    # Train model
    best_val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        patience=config.PATIENCE,
        experiment_name=experiment_name,
        save_dir=exp_dir,
    )

    # Load best model for testing
    model.load_state_dict(
        torch.load(exp_dir / "best_current_model.pth", map_location=device)
    )

    # Test model
    test_metrics = validate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        phase="TEST",
    )

    logging.info(f"\n{'='*80}")
    logging.info(f"Results for {experiment_name} ({masks_type}):")
    logging.info(f"  Validation - Dice: {best_val_metrics['nodule_dice']:.4f} ± {best_val_metrics['nodule_dice_std']:.4f}, IoU: {best_val_metrics['nodule_iou']:.4f} ± {best_val_metrics['nodule_iou_std']:.4f}")
    logging.info(f"  Test       - Dice: {test_metrics['nodule_dice']:.4f} ± {test_metrics['nodule_dice_std']:.4f}, IoU: {test_metrics['nodule_iou']:.4f} ± {test_metrics['nodule_iou_std']:.4f}")
    logging.info(f"{'='*80}\n")

    # Clean up
    del model
    torch.cuda.empty_cache()

    # Prepare results (only nodule class metrics with mean ± std)
    results = {
        "Name": experiment_name,
        "Masks_Type": masks_type,
        "Mode": mode,
        "Encoder_name": encoder_name,
        "Encoder_weights": encoder_weights,
        "Epochs": num_epochs,
        "Batch": batch_size,
        "LR": learning_rate,
        "Pixel_size": pixel_size,
        "Workers": workers,
        "Num_Train_Samples": num_train_samples if num_train_samples else "full",
        "V-Dice": f"{best_val_metrics['nodule_dice']:.4f} ± {best_val_metrics['nodule_dice_std']:.4f}",
        "V-IoU": f"{best_val_metrics['nodule_iou']:.4f} ± {best_val_metrics['nodule_iou_std']:.4f}",
        "T-Dice": f"{test_metrics['nodule_dice']:.4f} ± {test_metrics['nodule_dice_std']:.4f}",
        "T-IoU": f"{test_metrics['nodule_iou']:.4f} ± {test_metrics['nodule_iou_std']:.4f}",
    }

    # Return results dict for CSV and additional metadata for best model tracking
    return results, test_metrics['nodule_dice'], test_metrics['nodule_iou'], str(exp_dir / "best_current_model.pth")


def main():
    """Main training loop."""
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
    csv_file = config.SEGMENTATION_CSV_RESULTS_PATH
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
    best_model_dir = config.BEST_MODEL_PATH / "segmentation"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = best_model_dir / "best_model_ever.pth"

    # Run experiments: for each model, run with MedSAM2 then GT
    masks_configurations = [
        ("GT", config.LUNA16_NODULE_MASKS_GT_DIR),
        # ("MedSAM2", config.LUNA16_NODULE_MASKS_DIR),
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

                # Unpack results and metrics for best model tracking
                results, test_dice, test_iou, model_path = results_experiment

                # Check if this is the best model across all experiments
                if test_dice > best_test_dice:
                    best_test_dice = test_dice
                    best_experiment_info = {
                        "name": experiment_name,
                        "masks_type": masks_type,
                        "test_dice": test_dice,
                        "test_iou": test_iou,
                        "val_dice": results["V-Dice"],
                        "val_iou": results["V-IoU"],
                        "test_dice_std": results["T-Dice"],  # Contains std info
                        "test_iou_std": results["T-IoU"],    # Contains std info
                    }
                    
                    # Copy the best model from current experiment to best_model_ever.pth
                    shutil.copy2(model_path, best_model_path)
                    logging.info(f"🏆 New best model overall! Saved to {best_model_path}")
                    logging.info(f"   Experiment: {experiment_name} ({masks_type})")
                    logging.info(f"   Test Dice: {test_dice:.4f}")

                # Save results
                results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
                results_df.to_csv(csv_file, index=False)
                logging.info(f"✓ Results saved to {csv_file}")

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
        logging.info(f"🏆 BEST MODEL OVERALL 🏆")
        logging.info(f"Experiment: {best_experiment_info['name']} ({best_experiment_info['masks_type']})")
        logging.info(f"Best Model Saved: {best_model_path}")
        logging.info(f"  Test Dice:  {best_experiment_info['test_dice_std']}")
        logging.info(f"  Test IoU:   {best_experiment_info['test_iou_std']}")
        logging.info(f"  Val Dice:   {best_experiment_info['val_dice']}")
        logging.info(f"  Val IoU:    {best_experiment_info['val_iou']}")
    
    logging.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
