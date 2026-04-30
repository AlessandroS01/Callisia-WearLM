"""
Evaluation Artifacts Manager - Comprehensive evaluation and visualization utilities.

This module provides all utilities for:
- Calculating evaluation metrics (MAE, RMSE, R², MAPE)
- Saving training metrics to CSV
- Plotting training history and test results
- Managing all evaluation artifacts for both split and LOSO training

Separation of concerns: Keeps evaluation logic independent of training logic.
"""

import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class EvaluationArtifacts:
    """Manager for all evaluation, metrics, and visualization operations."""

    # ==================== METRICS CALCULATION ====================

    @staticmethod
    def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
        """
        Calculate evaluation metrics comparing predictions to targets.

        Args:
            predictions: Array of model predictions
            targets: Array of ground truth labels

        Returns:
            dict: Dictionary containing MAE, RMSE, R², MAPE, and std_error metrics

        Example:
            metrics = EvaluationArtifacts.calculate_metrics(pred_array, target_array)
            print(f"MAE: {metrics['mae']:.4f} bpm")
        """
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)

        # Calculate MAPE with handling for zero values
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((targets - predictions) / targets)) * 100
            mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate standard deviation of absolute errors
        absolute_errors = np.abs(predictions - targets)
        std_error = np.std(absolute_errors)

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'std_error': std_error
        }

    # ==================== METRICS DISPLAY ====================

    @staticmethod
    def print_metrics_summary(metrics: dict, num_samples: int, run_dir: str = None) -> None:
        """
        Print a summary of metrics in a compact format.

        Args:
            metrics: Dictionary with 'mae', 'rmse', 'r2', 'mape', 'std_error' keys
            num_samples: Total number of samples evaluated
            run_dir: Optional path to the results directory
        """
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"  MAE: {metrics['mae']:.4f} bpm")
        print(f"  Std Dev Error: {metrics['std_error']:.4f} bpm")
        print(f"  RMSE: {metrics['rmse']:.4f} bpm")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  Samples: {num_samples}")
        if run_dir:
            print(f"  Results saved to: {run_dir}")
        print("="*70 + "\n")

    @staticmethod
    def display_sample_predictions(predictions: np.ndarray, targets: np.ndarray,
                                   num_samples: int = 10) -> None:
        """
        Display sample predictions vs actual values in a formatted table.

        Args:
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

    # ==================== CSV SAVING ====================

    @staticmethod
    def save_training_metrics(epochs_data: List[Dict], output_path: str) -> None:
        """
        Save training metrics (epoch losses) to a CSV file.

        Args:
            epochs_data: List of dicts with keys: epoch, train_loss, val_loss, best_model
            output_path: Path to save the CSV file

        Example:
            epochs_data = [
                {'epoch': 1, 'train_loss': 0.5432, 'val_loss': 0.5123, 'best_model': True},
                {'epoch': 2, 'train_loss': 0.4921, 'val_loss': 0.4756, 'best_model': False},
            ]
            EvaluationArtifacts.save_training_metrics(epochs_data, "metrics.csv")
        """
        if not epochs_data:
            print("⚠ No data to save")
            return

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['epoch', 'train_loss', 'val_loss', 'best_model']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(epochs_data)

            print(f"✓ Training metrics saved to: {output_path}")
        except OSError as e:
            print(f"✗ Error saving metrics: {e}")

    @staticmethod
    def save_test_results(predictions: List, targets: List, output_path: str) -> None:
        """
        Save detailed test results to a CSV file.

        Args:
            predictions: List of model predictions
            targets: List of ground truth labels
            output_path: Path to save the CSV file

        Creates CSV with columns: sample, predicted, actual, absolute_error, percentage_error
        """
        if not predictions or not targets:
            print("⚠ No data to save")
            return

        try:
            predictions_arr = np.array(predictions)
            targets_arr = np.array(targets)

            absolute_errors = np.abs(predictions_arr - targets_arr)
            percentage_errors = (absolute_errors / np.abs(targets_arr)) * 100
            # Handle division by zero
            percentage_errors = np.nan_to_num(percentage_errors, nan=0.0, posinf=0.0, neginf=0.0)

            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['sample', 'predicted', 'actual', 'absolute_error', 'percentage_error']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for sample_idx, (pred, target, abs_err, pct_err) in enumerate(
                    zip(predictions_arr, targets_arr, absolute_errors, percentage_errors), 1
                ):
                    writer.writerow({
                        'sample': sample_idx,
                        'predicted': round(float(pred), 4),
                        'actual': round(float(target), 4),
                        'absolute_error': round(float(abs_err), 4),
                        'percentage_error': round(float(pct_err), 2)
                    })

            print(f"✓ Test results saved to: {output_path}")
        except OSError as e:
            print(f"✗ Error saving test results: {e}")

    # ==================== PLOTTING ====================

    @staticmethod
    def plot_training_history(metrics_path: str, output_path: str) -> None:
        """
        Plot training and validation loss over epochs from a CSV file.

        Creates a plot showing:
        - Training loss over epochs
        - Validation loss over epochs
        - Marks the best model epoch

        Args:
            metrics_path: Path to CSV file with training metrics
            output_path: Path to save the plot image
        """
        try:
            # Read and parse CSV file
            epochs, train_losses, val_losses, best_epochs = [], [], [], []
            with open(metrics_path, 'r', encoding='utf-8') as csvfile:
                for row in csv.DictReader(csvfile):
                    epochs.append(int(row['epoch']))
                    train_losses.append(float(row['train_loss']))
                    val_losses.append(float(row['val_loss']))
                    if row['best_model'].lower() == 'true':
                        best_epochs.append(int(row['epoch']))

            if not epochs:
                print("⚠ No data found in metrics file")
                return

            # Create plot
            _, ax = plt.subplots(figsize=(12, 6))
            ax.plot(epochs, train_losses, marker='o', label='Training Loss',
                    linewidth=2, markersize=6, color='#2E86AB')
            ax.plot(epochs, val_losses, marker='s', label='Validation Loss',
                    linewidth=2, markersize=6, color='#A23B72')

            # Mark best model if exists
            if best_epochs:
                best_epoch = best_epochs[-1]
                ax.plot(best_epoch, val_losses[best_epoch - 1], marker='*',
                       markersize=20, color='#F18F01', label='Best Model', zorder=5)

            # Configure plot
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax.set_title('Training History - HR Estimation Model',
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xticks(epochs)

            # Add value labels
            for i, (epoch, train_loss, val_loss) in enumerate(
                zip(epochs, train_losses, val_losses)
            ):
                if i % max(1, len(epochs) // 5) == 0:
                    ax.text(epoch, train_loss, f'{train_loss:.3f}',
                           fontsize=8, ha='center', va='bottom', color='#2E86AB')
                    ax.text(epoch, val_loss, f'{val_loss:.3f}',
                           fontsize=8, ha='center', va='bottom', color='#A23B72')

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training history plot saved to: {output_path}")
            plt.close()

        except FileNotFoundError:
            print(f"✗ Metrics file not found: {metrics_path}")
        except (IOError, ValueError) as e:
            print(f"✗ Error creating plot: {e}")

    @staticmethod
    def _plot_predictions_vs_actual(ax, targets, predictions) -> None:
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

    @staticmethod
    def _plot_residuals(ax, targets, predictions) -> None:
        """Plot residuals plot."""
        residuals = predictions - targets
        ax.scatter(targets, residuals, alpha=0.6, s=50, color='#A23B72',
                   edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Actual HR (bpm)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residuals (Predicted - Actual)', fontsize=11, fontweight='bold')
        ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_error_distribution(ax, targets, predictions, mae, std_error=None) -> None:
        """Plot error distribution histogram."""
        errors = np.abs(predictions - targets)
        ax.hist(errors, bins=30, color='#F18F01', edgecolor='black', alpha=0.7)
        ax.axvline(mae, color='red', linestyle='--', linewidth=2, label=f"Mean: {mae:.2f}")
        if std_error is not None:
            ax.axvline(mae + std_error, color='orange', linestyle=':',
                       linewidth=2, label=f"±Std: {std_error:.2f}")
            ax.axvline(mae - std_error, color='orange', linestyle=':', linewidth=2)
        ax.set_xlabel('Absolute Error (bpm)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Absolute Errors', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    @staticmethod
    def _plot_predictions_over_samples(ax, targets, predictions) -> None:
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

    @staticmethod
    def plot_test_results(predictions, targets, metrics: dict,
                         output_path: str = "test_analysis.png") -> None:
        """
        Creates comprehensive test analysis plots.

        Args:
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

            EvaluationArtifacts._plot_predictions_vs_actual(
                axes[0, 0], targets, predictions)
            EvaluationArtifacts._plot_residuals(
                axes[0, 1], targets, predictions)
            EvaluationArtifacts._plot_error_distribution(
                axes[1, 0], targets, predictions, metrics['mae'], metrics.get('std_error'))
            EvaluationArtifacts._plot_predictions_over_samples(
                axes[1, 1], targets, predictions)

            # Add metrics text box
            std_error_text = (f"\nStd Dev Error: {metrics.get('std_error', 0):.4f}"
                              f" bpm") if 'std_error' in metrics else ""
            metrics_text = (f"MAE: {metrics['mae']:.4f} bpm{std_error_text}\n"
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

    # ==================== ARTIFACT MANAGEMENT ====================

    @staticmethod
    def save_fold_artifacts(fold_dir: str, epochs_data: List[Dict],
                           predictions: List, targets: List, test_metrics: Dict) -> None:
        """
        Save all artifacts for a single training fold (split or LOSO).

        Saves to the given fold_dir:
        - training_metrics.csv: Loss values per epoch
        - training_history.png: Training/validation loss curves
        - test_results.csv: Detailed predictions and errors
        - test_analysis.png: Predictions vs ground truth plot with metrics

        Args:
            fold_dir: Directory to save artifacts
            epochs_data: List of epoch metrics dicts
            predictions: Model predictions on test set
            targets: Ground truth test values
            test_metrics: Test performance metrics dict
        """
        # Save training metrics and plot
        metrics_csv = os.path.join(fold_dir, "training_metrics.csv")
        EvaluationArtifacts.save_training_metrics(epochs_data, metrics_csv)

        EvaluationArtifacts.plot_training_history(
            metrics_csv,
            os.path.join(fold_dir, "training_history.png")
        )

        # Save and plot test results
        if test_metrics:
            # Save test results CSV
            EvaluationArtifacts.save_test_results(
                predictions, targets,
                os.path.join(fold_dir, "test_results.csv")
            )

            EvaluationArtifacts.plot_test_results(
                predictions, targets, test_metrics,
                os.path.join(fold_dir, "test_analysis.png")
            )
            print("\n✓ Fold artifacts saved:")
            print("  - training_metrics.csv")
            print("  - training_history.png")
            print("  - test_results.csv")
            print("  - test_analysis.png")
