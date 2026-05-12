"""
Dataset Scaling Experiment Script

This script runs training experiments with different dataset sizes to analyze
the performance impact of training data quantity.

For each sample size:
- Trains with X samples from training set
- Validates with complete validation set
- Tests with complete test set
- Runs for each model and mask type sequentially

Sample sizes: [60, 120, 240, 480, 600, full_dataset]
"""

import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.common import config
from nodule_segmentation.train import run_experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def main():
    """Main script for dataset scaling experiments."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Sample sizes to test, None = full dataset
    # sample_sizes = [3, 9, 27, 81, 243, None]
    sample_sizes = [2, 6, 18, 54, 162, None]

    # Load experiments configuration
    config_file = config.SEGMENTATION_EXPERIMENT_PATH
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        experiments_config = yaml.safe_load(f)

    # Initialize results CSV
    csv_file = config.SEGMENTATION_CSV_RESULTS_PATH.parent / "segmentation_scaling_results.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    if csv_file.exists():
        results_df = pd.read_csv(csv_file)
        logging.info(f"Loaded existing results from {csv_file}")
    else:
        results_df = pd.DataFrame()
        logging.info(f"Creating new results file: {csv_file}")

    # Masks configurations: (type_name, directory_path)
    masks_configurations = [
        ("GT", config.LUNA16_NODULE_MASKS_GT_DIR),
        # ("MedSAM2", config.LUNA16_NODULE_MASKS_DIR),
    ]

    # ========================================================================
    # Main Loop: For each sample size
    # ========================================================================
    for sample_idx, sample_size in enumerate(sample_sizes, 1):
        size_label = f"{sample_size}_samples" if sample_size else "full_dataset"
        
        logging.info(f"\n{'='*80}")
        logging.info(f"SCALING EXPERIMENT {sample_idx}/{len(sample_sizes)}: {size_label}")
        logging.info(f"{'='*80}\n")

        # ====================================================================
        # For each experiment
        # ====================================================================
        for exp_idx, (experiment_name, experiment_config) in enumerate(experiments_config.items(), 1):
            if experiment_config is None:
                logging.warning(f"Skipping {experiment_name} (no configuration)")
                continue

            logging.info(f"\n{'-'*80}")
            logging.info(f"Experiment {exp_idx}: {experiment_name} [{size_label}]")
            logging.info(f"{'-'*80}\n")

            # ================================================================
            # For each masks type
            # ================================================================
            for masks_type, masks_dir in masks_configurations:
                try:
                    logging.info(f"\n  → Running {experiment_name}")
                    logging.info(f"    Size: {size_label}")
                    logging.info(f"    Masks: {masks_type}\n")

                    results, _, _, _ = run_experiment(
                        experiment_name=experiment_name,
                        experiment_config=experiment_config,
                        device=device,
                        masks_type=masks_type,
                        masks_dir=masks_dir,
                        num_train_samples=sample_size,
                    )

                    # Save results
                    results_df = pd.concat(
                        [results_df, pd.DataFrame([results])],
                        ignore_index=True
                    )
                    results_df.to_csv(csv_file, index=False)
                    logging.info(f"✓ Results saved to {csv_file}\n")

                except Exception as e:
                    logging.error(f"✗ Error running {experiment_name} with {masks_type}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

    logging.info(f"\n{'='*80}")
    logging.info(f"✓ All dataset scaling experiments completed!")
    logging.info(f"Results saved to: {csv_file}")
    logging.info(f"Total experiments: {len(results_df)}")
    logging.info(f"{'='*80}\n")

    # Print summary
    logging.info("\nSummary of Results by Sample Size:")
    logging.info("-" * 80)
    
    for sample_size in sample_sizes:
        size_label = f"{sample_size}_samples" if sample_size else "full_dataset"
        subset_results = results_df[results_df['Num_Train_Samples'] == (sample_size if sample_size else "full")]
        
        if len(subset_results) > 0:
            logging.info(f"\n{size_label} ({len(subset_results)} entries):")
            logging.info(f"  Models tested: {subset_results['Name'].nunique()}")
            logging.info(f"  Mask types: {subset_results['Masks_Type'].unique().tolist()}")


if __name__ == "__main__":
    main()
