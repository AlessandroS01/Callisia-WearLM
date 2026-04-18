"""Shared utilities for model training and testing blocks.

Provides common functions for data loading, directory management, and result saving
across different model blocks.
"""
import csv
import os

import numpy as np

from src.models.evaluation_utils import calculate_metrics


def setup_run_directory(base_dir: str = "history/block_1/") -> str:
    """
    Creates and returns a run-specific directory with incremental numbering.

    This function:
    1. Checks existing run directories in base_dir
    2. Creates a new run directory with the next sequential number
    3. Returns the path to the new run directory

    Params:
        base_dir: Base directory where run folders are stored

    Returns:
        str: Path to the new run directory (e.g., "history/block_1/run_001")
    """
    os.makedirs(base_dir, exist_ok=True)

    # Find the highest run number
    existing_runs = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            if item.startswith("run_") and os.path.isdir(os.path.join(base_dir, item)):
                try:
                    run_num = int(item.split("_")[1])
                    existing_runs.append(run_num)
                except (ValueError, IndexError):
                    pass

    # Create next run directory
    next_run_num = max(existing_runs) + 1 if existing_runs else 1
    run_dir = os.path.join(base_dir, f"run_{next_run_num:03d}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"✓ Created run directory: {run_dir}")
    return run_dir

def save_test_results(
    predictions,
    targets,
    run_dir: str
):
    """
    Saves test predictions and ground truth to a CSV file.

    Params:
        predictions: Array of model predictions
        targets: Array of ground truth labels
        run_dir: Directory where results will be saved

    Returns:
        dict: Dictionary with computed metrics
    """
    if len(predictions) != len(targets):
        print("✗ Predictions and targets have different lengths")
        return {}

    try:
        predictions, targets = np.array(predictions), np.array(targets)

        # Calculate metrics using shared utility
        metrics = calculate_metrics(predictions, targets)

        # Save results
        output_path = os.path.join(run_dir, "test_results.csv")
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile,
                fieldnames=['sample', 'predicted', 'actual', 'absolute_error',
                            'percentage_error'])
            writer.writeheader()
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                abs_error = abs(pred - target)
                writer.writerow({
                    'sample': i + 1,
                    'predicted': round(pred, 4),
                    'actual': round(target, 4),
                    'absolute_error': round(abs_error, 4),
                    'percentage_error':
                        round(abs_error / target * 100 if target != 0 else 0, 2)
                })

        print(f"✓ Test results saved to: {output_path}")
        return {
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2'],
            'mape': metrics['mape'],
            'num_samples': len(predictions)
        }

    except (OSError, ValueError) as e:
        print(f"✗ Error saving test results: {e}")
        return {}
