"""Shared utilities for model evaluation and metrics calculation.

Provides common functions for calculating and displaying test/validation metrics
across different model blocks.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Calculate evaluation metrics comparing predictions to targets.

    Params:
        predictions: Array of model predictions
        targets: Array of ground truth labels

    Returns:
        dict: Dictionary containing MAE, RMSE, R², MAPE metrics
    """
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)

    # Calculate MAPE with handling for zero values
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


def print_metrics_summary(metrics: dict, num_samples: int, run_dir: str = None):
    """
    Print a summary of metrics in a compact format.

    Params:
        metrics: Dictionary with 'mae', 'rmse', 'r2', 'mape' keys
        num_samples: Total number of samples evaluated
        run_dir: Optional path to the results directory
    """
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  MAE: {metrics['mae']:.4f} bpm")
    print(f"  RMSE: {metrics['rmse']:.4f} bpm")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Samples: {num_samples}")
    if run_dir:
        print(f"  Results saved to: {run_dir}")
    print("="*70 + "\n")


def display_sample_predictions(predictions: np.ndarray, targets: np.ndarray, num_samples: int = 10):
    """
    Display sample predictions vs actual values in a formatted table.

    Params:
        predictions: Array of model predictions
        targets: Array of ground truth labels
        num_samples: Number of sample predictions to display (default: 10)
    """
    print(f"Sample Predictions (first {num_samples}):")
    print(f"{'Index':<8} {'Predicted (bpm)':<20} {'Actual (bpm)':<20} {'Error (bpm)':<15}")
    print("-" * 63)

    for i in range(min(num_samples, len(predictions))):
        error = predictions[i] - targets[i]
        print(f"{i+1:<8} {predictions[i]:<20.2f} {targets[i]:<20.2f} {error:<15.2f}")
    print()


def _plot_predictions_vs_actual(ax, targets, predictions):
    """Plot predictions vs actual values with perfect prediction line."""
    ax.scatter(targets, predictions, alpha=0.6, s=50, color='#2E86AB',
               edgecolors='black', linewidth=0.5)
    min_val, max_val = min(targets.min(), predictions.min()), \
                       max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--',
            linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual HR (bpm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted HR (bpm)', fontsize=11, fontweight='bold')
    ax.set_title('Predictions vs Actual', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_residuals(ax, targets, predictions):
    """Plot residuals plot."""
    residuals = predictions - targets
    ax.scatter(targets, residuals, alpha=0.6, s=50, color='#A23B72',
               edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Actual HR (bpm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residuals (Predicted - Actual)', fontsize=11, fontweight='bold')
    ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)


def _plot_error_distribution(ax, targets, predictions, mae):
    """Plot error distribution histogram."""
    errors = np.abs(predictions - targets)
    ax.hist(errors, bins=30, color='#F18F01', edgecolor='black', alpha=0.7)
    ax.axvline(mae, color='red', linestyle='--', linewidth=2, label=f"Mean: {mae:.2f}")
    ax.set_xlabel('Absolute Error (bpm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Absolute Errors', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')


def _plot_predictions_over_samples(ax, targets, predictions):
    """Plot predictions vs actual over samples."""
    sample_indices = np.arange(len(predictions))
    step = max(1, len(predictions) // 100)
    ax.plot(sample_indices[::step], targets[::step], marker='o',
            label='Actual', linewidth=2, markersize=4, color='#2E86AB')
    ax.plot(sample_indices[::step], predictions[::step], marker='s',
            label='Predicted', linewidth=2, markersize=4, color='#A23B72', alpha=0.7)
    ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('HR (bpm)', fontsize=11, fontweight='bold')
    ax.set_title('Predicted vs Actual Over Samples', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_test_results(
    predictions,
    targets,
    metrics: dict,
    output_path: str = "test_analysis.png"
):
    """
    Creates comprehensive test analysis plots.

    Params:
        predictions: Array of model predictions
        targets: Array of ground truth labels
        metrics: Dictionary with computed metrics
        output_path: Path to save the plot image

    Creates a 2x2 subplot showing:
    1. Predictions vs Actual (scatter plot with perfect prediction line)
    2. Residuals plot
    3. Distribution of errors
    4. Predicted vs Actual (line plot)
    """
    try:
        predictions, targets = np.array(predictions), np.array(targets)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Test Set Analysis - HR Estimation Model',
                    fontsize=16, fontweight='bold', y=1.00)

        _plot_predictions_vs_actual(axes[0, 0], targets, predictions)
        _plot_residuals(axes[0, 1], targets, predictions)
        _plot_error_distribution(axes[1, 0], targets, predictions, metrics['mae'])
        _plot_predictions_over_samples(axes[1, 1], targets, predictions)

        # Add metrics text box
        metrics_text = (f"MAE: {metrics['mae']:.4f} bpm\n"
                       f"RMSE: {metrics['rmse']:.4f} bpm\n"
                       f"R²: {metrics['r2']:.4f}\n"
                       f"MAPE: {metrics['mape']:.2f}%\n"
                       f"Samples: {len(predictions)}")
        fig.text(0.98, 0.02, metrics_text, fontsize=10, family='monospace',
                bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.8},
                ha='right', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Test analysis plot saved to: {output_path}")
        plt.close()

    except (OSError, ValueError) as e:
        print(f"✗ Error creating test plots: {e}")
