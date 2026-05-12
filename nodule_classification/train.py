"""
Training script for nodule classification models.

This script trains, validates, and tests nodule classification models
with and without mask filtering, following an incremental training strategy.
"""

import logging
import random
import sys
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
from nodule_classification.models.models_2d import ClassificationModels2D
from metrics.classification import calculate_all_metrics
from nodule_classification.dataloader import get_nodule_classification_data_loader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Enable CUDNN benchmarking for performance
torch.backends.cudnn.benchmark = True


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
            labels = batch["label"].to(device).float().unsqueeze(1)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

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
) -> dict:
    """Validate or test model.

    Args:
        model: Model to validate
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to run on
        phase: Phase name for logging ("VALID" or "TEST")

    Returns:
        Dictionary with metrics
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        with tqdm(dataloader, desc=f"[{phase}]", leave=False) as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch["image"].to(device)
                labels = batch["label"].to(device).float().unsqueeze(1)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Get predictions
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()

                # Accumulate results
                running_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                # Update progress bar
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Convert to numpy arrays
    all_labels = np.array(all_labels).flatten()
    all_predictions = np.array(all_predictions).flatten()
    all_probabilities = np.array(all_probabilities).flatten()

    # Calculate metrics
    metrics = calculate_all_metrics(
        y_true=all_labels,
        y_pred=all_predictions,
        y_pred_proba=all_probabilities,
    )

    metrics["loss"] = running_loss / len(dataloader)

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
    best_auc = 0.0
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
            f"Val AUC: {val_metrics['auc']:.4f} - "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        # Check for improvement
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics.copy()
            epochs_without_improvement = 0

            # Save best model
            save_path = save_dir / "best_metric_model.pth"
            torch.save(model.state_dict(), save_path)
            logging.info(f"✓ New best model saved (AUC: {best_auc:.4f})")
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
    filtering: bool = False,
) -> dict:
    """Run a single training experiment.

    Args:
        experiment_name: Name of experiment
        experiment_config: Configuration dictionary
        device: Device to train on
        filtering: Whether to apply mask filtering

    Returns:
        Dictionary with all metrics (train, validation, test)
    """
    mode = experiment_config["Mode"]
    weight_size = str(experiment_config["Weight"])  # Convert to string
    batch_size = experiment_config["Batch"]
    num_epochs = experiment_config["Epochs"]
    learning_rate = experiment_config["LR"]
    pixel_size = experiment_config["Pixel_size"]
    workers = experiment_config["Workers"]

    # Create experiment directory
    exp_dir = config.BEST_MODEL_PATH
    exp_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n{'='*80}")
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"Mode: {mode}, Weight: {weight_size}, Filtering: {filtering}")
    logging.info(f"Batch: {batch_size}, LR: {learning_rate}, Epochs: {num_epochs}")
    logging.info(f"{'='*80}\n")

    # Create dataloaders
    train_loader = get_nodule_classification_data_loader(
        csv_name="train.csv",
        mode=mode,
        workers=workers,
        batch_size=batch_size,
        size_px=pixel_size,
        use_luna25_data=True,
        shuffle=False,  # WeightedRandomSampler will be used
        model_name=experiment_name,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        filtering=filtering,
    )

    val_loader = get_nodule_classification_data_loader(
        csv_name="valid.csv",
        mode=mode,
        workers=workers,
        batch_size=batch_size,
        size_px=pixel_size,
        use_luna25_data=True,
        shuffle=False,
        model_name=experiment_name,
        rotations=None,
        translations=None,
        filtering=filtering,
    )

    test_loader = get_nodule_classification_data_loader(
        csv_name="test.csv",
        mode=mode,
        workers=workers,
        batch_size=batch_size,
        size_px=pixel_size,
        use_luna25_data=True,
        shuffle=False,
        model_name=experiment_name,
        rotations=None,
        translations=None,
        filtering=filtering,
    )

    # Initialize model
    model = ClassificationModels2D.get_model(experiment_name, weight_size).to(device)

    # Loss function with class imbalance handling (LUNA25: 91% negative, 9% positive)
    pos_weight = torch.tensor([0.91 / 0.09]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
        torch.load(exp_dir / "best_metric_model.pth", map_location=device)
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
    logging.info(f"Results for {experiment_name} {'filtered' if filtering else 'unfiltered'}:")
    logging.info(f"  Validation - AUC: {best_val_metrics['auc']:.4f}, Acc: {best_val_metrics['accuracy']:.4f}")
    logging.info(f"  Test       - AUC: {test_metrics['auc']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    logging.info(f"{'='*80}\n")

    # Clean up
    del model
    torch.cuda.empty_cache()

    # Prepare results
    results = {
        "Name": experiment_name,
        "Mode": mode,
        "Weight": weight_size,
        "Filtering": filtering,
        "Epochs": num_epochs,
        "Batch": batch_size,
        "LR": learning_rate,
        "Pixel_size": pixel_size,
        "Workers": workers,
        "V-AUC": round(best_val_metrics['auc'], 4),
        "V-Acc": round(best_val_metrics['accuracy'], 4),
        "V-Prec": round(best_val_metrics['precision'], 4),
        "V-Sens": round(best_val_metrics['sensitivity'], 4),
        "V-F1": round(best_val_metrics['f1_score'], 4),
        "V-Spec": round(best_val_metrics.get('specificity', 0.0), 4),
        "T-AUC": round(test_metrics['auc'], 4),
        "T-Acc": round(test_metrics['accuracy'], 4),
        "T-Prec": round(test_metrics['precision'], 4),
        "T-Sens": round(test_metrics['sensitivity'], 4),
        "T-F1": round(test_metrics['f1_score'], 4),
        "T-Spec": round(test_metrics.get('specificity', 0.0), 4),
    }

    return results


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
    config_file = config.CLASSIFICATION_EXPERIMENT_PATH
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        experiments_config = yaml.safe_load(f)

    # Initialize results CSV
    csv_file = config.CLASSIFICATION_CSV_RESULTS_PATH
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    if csv_file.exists():
        results_df = pd.read_csv(csv_file)
        logging.info(f"Loaded existing results from {csv_file}")
    else:
        results_df = pd.DataFrame()
        logging.info(f"Creating new results file: {csv_file}")

    # Run experiments
    for experiment_name, experiment_config in experiments_config.items():
        if experiment_config is None:
            logging.warning(f"Skipping {experiment_name} (no configuration)")
            continue

        try:
            # Run without filtering
            logging.info(f"\n{'#'*80}")
            logging.info(f"# Running {experiment_name} WITHOUT mask filtering")
            logging.info(f"{'#'*80}\n")

            results_unfiltered = run_experiment(
                experiment_name=experiment_name,
                experiment_config=experiment_config,
                device=device,
                filtering=False,
            )

            # Save results
            results_df = pd.concat([results_df, pd.DataFrame([results_unfiltered])], ignore_index=True)
            results_df.to_csv(csv_file, index=False)
            logging.info(f"✓ Results saved to {csv_file}")

            # Run with filtering
            logging.info(f"\n{'#'*80}")
            logging.info(f"# Running {experiment_name} WITH mask filtering")
            logging.info(f"{'#'*80}\n")

            results_filtered = run_experiment(
                experiment_name=experiment_name,
                experiment_config=experiment_config,
                device=device,
                filtering=True,
            )

            # Save results
            results_df = pd.concat([results_df, pd.DataFrame([results_filtered])], ignore_index=True)
            results_df.to_csv(csv_file, index=False)
            logging.info(f"✓ Results saved to {csv_file}")

        except Exception as e:
            logging.error(f"✗ Error running {experiment_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    logging.info(f"\n{'='*80}")
    logging.info(f"All experiments completed!")
    logging.info(f"Results saved to: {csv_file}")
    logging.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
