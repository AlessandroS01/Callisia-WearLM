"""
Module for training the initial 1D CNN for HR estimation from BVP and ACC data.
"""

import os
import csv
import json
from datetime import datetime
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.processors.processor import DaliaProcessor
from src.data.dataset.dalia_dataset import DaliaHRDataset
from src.models.hr_cnn import MultimodalHRNet


def setup_run_directory(base_dir: str = "history/block_1") -> str:
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


def save_run_config(run_dir: str, config: dict, learning_rate: float, num_epochs: int,
                    dataset_config: dict):
    """
    Saves training configuration to a JSON file in the run directory.

    Params:
        run_dir: Path to the run directory
        config: Configuration dictionary
        learning_rate: Learning rate used for training
        num_epochs: Number of epochs
        dataset_config: Dictionary containing training/validation/test patient splits
    """
    config_data = {
        'timestamp': datetime.now().isoformat(),
        'learning_rate': learning_rate,
        'batch_size': config.get('batch_size'),
        'num_epochs': num_epochs,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_params': {
            'mode': 'min',
            'factor': 0.5,
            'patience': 3,
            'min_lr': 1e-7,
            'lr_reduction_factor': 0.5
        },
        'loss_function': 'MSELoss',
        'dataset': dataset_config
    }

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)

    print(f"✓ Configuration saved to: {config_path}")


def load_config(config_path: str = "../../../config.yaml") -> dict:
    """
    Loads the configuration from a YAML file.

    Params:
        config_path: path to the config.yaml file

    Returns:
        dict: Configuration dictionary containing learning_rate, batch_size, and num_epochs
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return config


def retrieve_patient_data(patient: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts input features and labels for the specific patient by using
    the DaliaProcessor object.

    Params:
        patient: patient ID string, e.g. "S1", "S2", ..., "S15"

    Returns:
        tuple: A tuple containing:

            - x (ndarray): The input features for the model, typically
              a 3D array of shape (num_samples, num_channels, sequence_length)
              containing [BVP, ACCx, ACCy, ACCz]

            - y (ndarray): The target labels for the model, typically
              a 1D array of shape (num_samples,) containing the heart rate values.
    """
    patient_path = f"../../../data/processed/dalia/{patient}"
    processor = DaliaProcessor(patient_path)
    x, y = processor.process()

    return x, y


def prepare_dataset(patients: list, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares a dataset by combining data from multiple patients.

    Params:
        patients: list of patient IDs
        dataset_name: name of the dataset (for logging)

    Returns:
        tuple: Combined (x, y) arrays for the dataset
    """
    x_list, y_list = [], []

    for patient in patients:
        x, y = retrieve_patient_data(patient)
        x_list.append(x)
        y_list.append(y)
        print(f"{patient} has x shape: {x.shape} and y shape: {y.shape}")

    x_combined = np.concatenate(x_list, axis=0)
    y_combined = np.concatenate(y_list, axis=0)

    print(f"Combined {dataset_name} data with x shape: {x_combined.shape} and "
          f"y shape: {y_combined.shape}\n")

    return x_combined, y_combined


def setup_training():
    """
    Sets up the training pipeline by loading configuration, preparing datasets,
    and creating DataLoaders for training, validation, and testing.

    This function:
    1. Loads hyperparameters from config.yaml (learning_rate, batch_size, num_epochs)
    2. Retrieves and combines data from all training, validation, and test patients
    3. Creates PyTorch Dataset objects for each split with proper tensor formatting
    4. Creates DataLoaders with appropriate batch sizes and shuffling settings
    5. Validates the data pipeline by retrieving and displaying a sample batch

    Returns:
        tuple: A tuple containing:

            - train_loader (DataLoader): DataLoader for training data with shuffle

            - valid_loader (DataLoader): DataLoader for validation data

            - test_loader (DataLoader): DataLoader for test data

            - learning_rate (float): Learning rate from configuration

            - num_epochs (int): Number of training epochs from configuration

    """
    # Load configuration
    config = load_config()
    batch_size = config["batch_size"]

    print("Configuration loaded:")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of epochs: {config['num_epochs']}\n")

    patient_splits = get_split_patients()

    x_train, y_train = prepare_dataset(
        patient_splits['training_patients'], "training"
    )
    x_valid, y_valid = prepare_dataset(
        patient_splits['validation_patients'], "validation"
    )
    x_test, y_test = prepare_dataset(
        patient_splits['test_patients'], "test"
    )

    # Create PyTorch datasets
    train_dataset = DaliaHRDataset(x_train, y_train)
    valid_dataset = DaliaHRDataset(x_valid, y_valid)
    test_dataset = DaliaHRDataset(x_test, y_test)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}\n")

    # Create DataLoaders with batch_size from config
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}\n")

    return (
        train_loader,
        valid_loader,
        test_loader,
        config["learning_rate"],
        config["num_epochs"]
    )

def train_epoch(
    model,
    train_loader,
    optimizer,
    loss_function,
    device
):
    """
    Executes a single training epoch.

    Params:
        model: The neural network model to train
        train_loader: DataLoader with training data
        optimizer: Optimizer for parameter updates
        loss_function: Loss function to compute training loss
        device: Device to run tensors on (CPU or GPU)

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = loss_function(predictions.squeeze(), y_batch.squeeze())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"- Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / num_batches
    return avg_loss


def validate(
    model,
    valid_loader,
    loss_function,
    device
):
    """
    Evaluates the model on the validation set.

    Params:
        model: The neural network model to validate
        valid_loader: DataLoader with validation data
        loss_function: Loss function to compute validation loss
        device: Device to run tensors on (CPU or GPU)

    Returns:
        float: Average validation loss
    """
    model.eval()
    epoch_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(valid_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)
            loss = loss_function(predictions.squeeze(), y_batch.squeeze())

            epoch_loss += loss.item()
            num_batches += 1

        print(f"  Validation completed - Processed {num_batches} batches")

    avg_loss = epoch_loss / num_batches
    return avg_loss


def test(
    model,
    test_loader,
    loss_function,
    device
):
    """
    Evaluates the model on the test set and collects predictions.

    Params:
        model: The neural network model to test
        test_loader: DataLoader with test data
        loss_function: Loss function to compute test loss
        device: Device to run tensors on (CPU or GPU)

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): Average test loss
            - all_predictions (list): Model predictions on test set
            - all_targets (list): Ground truth labels from test set
    """
    model.eval()
    test_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []

    print("  Processing test batches...")

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)
            loss = loss_function(predictions.squeeze(), y_batch.squeeze())

            test_loss += loss.item()
            num_batches += 1

            all_predictions.extend(predictions.squeeze().cpu().numpy().tolist())
            all_targets.extend(y_batch.squeeze().cpu().numpy().tolist())

            if (batch_idx + 1) % 5 == 0:
                print(f"    Batch [{batch_idx + 1}/{len(test_loader)}] "
                      f"- Loss: {loss.item():.4f}")

    avg_loss = test_loss / num_batches
    print(f"  Test completed - Processed {num_batches} batches "
          f"with {len(all_predictions)} total samples")

    return avg_loss, all_predictions, all_targets


def save_training_metrics(
    epochs_data: list,
    output_path: str = "training_metrics.csv"
):
    """
    Saves training metrics to a CSV file for later analysis.

    Params:
        epochs_data: List of dictionaries containing epoch metrics
                    Each dict should have keys: epoch, train_loss, val_loss, best_model
        output_path: Path to save the CSV file

    Example of epochs_data:
        [
            {'epoch': 1, 'train_loss': 0.5432, 'val_loss': 0.5123, 'best_model': True},
            {'epoch': 2, 'train_loss': 0.4921, 'val_loss': 0.4756, 'best_model': False},
        ]
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


def plot_training_history(
    metrics_path: str = "training_metrics.csv",
    output_path: str = "training_history.png"
):
    """
    Plots training and validation loss over epochs from a CSV file.

    Params:
        metrics_path: Path to the CSV file with training metrics
        output_path: Path to save the plot image

    Creates a plot showing:
    - Training loss over epochs
    - Validation loss over epochs
    - Marks the best model epoch
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
        ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
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


def save_test_results(
    predictions,
    targets,
    output_path: str = "test_results.csv"
):
    """
    Saves test predictions and ground truth to a CSV file.

    Params:
        predictions: List of model predictions
        targets: List of ground truth labels
        output_path: Path to save the CSV file

    Returns:
        dict: Dictionary with computed metrics
    """
    if len(predictions) != len(targets):
        print("✗ Predictions and targets have different lengths")
        return {}

    try:
        predictions, targets = np.array(predictions), np.array(targets)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        # MAPE with handling for zero values
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((targets - predictions) / targets)) * 100
            mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)

        # Save results
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
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'num_samples': len(predictions)
        }


    except (OSError, ValueError) as e:
        print(f"✗ Error saving test results: {e}")
        return {}


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
        predictions: List of model predictions
        targets: List of ground truth labels
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
                       f"Samples: {metrics['num_samples']}")
        fig.text(0.98, 0.02, metrics_text, fontsize=10, family='monospace',
                bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.8},
                ha='right', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Test analysis plot saved to: {output_path}")
        plt.close()

    except (OSError, ValueError) as e:
        print(f"✗ Error creating test plots: {e}")


def train():
    """
    Main training function that orchestrates the complete training pipeline.

    This function:
    1. Sets up the training infrastructure (data loaders, model, optimizer, loss)
    2. Executes training loop with train, validate, and test epochs
    3. Saves the best model based on validation loss
    4. Prints comprehensive training summary

    The pipeline includes:
    - Training phase: Forward pass, backward propagation, parameter updates
    - Validation phase: Model evaluation without gradient computation
    - Testing phase: Final evaluation on held-out test set
    """
    print("="*70)
    print("INITIALIZING TRAINING PIPELINE")
    print("="*70 + "\n")

    # Setup training infrastructure
    (train_loader,
     valid_loader,
     test_loader,
     learning_rate,
     num_epochs) = setup_training()

    # Determine device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}\n")

    # Initialize model, optimizer, and loss function
    print("Initializing model, optimizer, and loss function...")
    model = MultimodalHRNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    print(f"✓ Model: {model.__class__.__name__}")
    print(f"✓ Optimizer: Adam (lr={learning_rate})")
    print("✓ Loss Function: MSELoss\n")

    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
    print("✓ Learning Rate Scheduler: ReduceLROnPlateau")
    print("  - Factor: 0.5, Patience: 3 epochs\n")

    # Create run directory for this training session
    run_dir = setup_run_directory("history/block_1")

    # Save configuration for this run
    save_run_config(
        run_dir,
        load_config(),
        learning_rate,
        num_epochs,
        get_split_patients(),
    )

    # Create model directory for this run
    model_run_dir = os.path.join("../../../models/block_1", f"{os.path.basename(run_dir)}")
    os.makedirs(model_run_dir, exist_ok=True)
    print(f"✓ Model directory created: {model_run_dir}\n")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    no_improve_count = 0
    epochs_data = []  # Store data for CSV

    # Training loop
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")

    for epoch in range(num_epochs):
        print(f"{'─'*70}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'─'*70}")

        print("  [1/2] Training phase...")
        avg_train_loss = train_epoch(
            model, train_loader, optimizer, loss_function, device
        )
        train_losses.append(avg_train_loss)

        print("\n  [2/2] Validation phase...")
        avg_val_loss = validate(
            model, valid_loader, loss_function, device
        )
        val_losses.append(avg_val_loss)

        # Calculate improvement
        improvement = "↓" if avg_val_loss < best_val_loss else "↑"
        loss_diff = abs(avg_val_loss - best_val_loss)

        print("\n  Results:")
        print(f"    Train Loss: {avg_train_loss:.4f}")
        print(f"    Val Loss:   {avg_val_loss:.4f} "
              f"    Improvement: {improvement} ({loss_diff:.4f})")

        # Save best model based on validation loss
        is_best = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(model_run_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            no_improve_count = 0
            is_best = True
            print(f"    ✓ Best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            no_improve_count += 1
            print(f"    • No improvement ({no_improve_count} epochs)")

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Store epoch data for CSV
        epochs_data.append({
            'epoch': epoch + 1,
            'train_loss': round(avg_train_loss, 6),
            'val_loss': round(avg_val_loss, 6),
            'best_model': is_best
        })

        print()

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nStarting Testing Phase...\n")

    # Testing phase
    print("["*35)
    print("  Testing model on held-out test set...")
    avg_test_loss, predictions, targets = test(
        model, test_loader, loss_function, device
    )
    print(f"\n  ✓ Test Loss: {avg_test_loss:.4f}\n")

    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"  Total Epochs:              {num_epochs}")
    print(f"  Final Train Loss:          {train_losses[-1]:.4f}")
    print(f"  Final Val Loss:            {val_losses[-1]:.4f}")
    print(f"  Best Val Loss:             {best_val_loss:.4f}")
    print(f"  Test Loss:                 {avg_test_loss:.4f}")
    print(f"  Test Samples:              {len(predictions)}")
    print(f"  Model saved to:            {model_run_dir}/best_model.pth")
    print("="*70)

    # Save training metrics to CSV
    print("\n" + "="*70)
    print("SAVING TRAINING ARTIFACTS")
    print("="*70)
    save_training_metrics(
        epochs_data,
        os.path.join(run_dir, "training_metrics.csv")
    )

    # Generate and save plot
    plot_training_history(
        os.path.join(run_dir, "training_metrics.csv"),
        os.path.join(run_dir, "training_history.png")
    )

    # Save test results and generate analysis
    print("\n" + "="*70)
    print("SAVING TEST RESULTS & ANALYSIS")
    print("="*70)
    test_metrics = save_test_results(
        predictions,
        targets,
        os.path.join(run_dir, "test_results.csv")
    )

    # Generate test analysis plots
    if test_metrics:
        plot_test_results(
            predictions,
            targets,
            test_metrics,
            os.path.join(run_dir, "test_analysis.png")
        )

        print("\n" + "─"*70)
        print("TEST PERFORMANCE METRICS")
        print("─"*70)
        print(f"  Mean Absolute Error (MAE):       {test_metrics['mae']:.4f} bpm")
        print(f"  Root Mean Squared Error (RMSE):  {test_metrics['rmse']:.4f} bpm")
        print(f"  R² Score:                        {test_metrics['r2']:.4f}")
        print(f"  Mean Absolute Percentage Error:  {test_metrics['mape']:.2f}%")
        print(f"  Total Test Samples:              {test_metrics['num_samples']}")
        print("─"*70)

    print("="*70 + "\n")

def get_split_patients():
    """
    Returns the predefined patient splits for training, validation, and testing.

    Returns:
        dict: Dictionary containing:

            - training_patients (list): List of patient IDs for training

            - validation_patients (list): List of patient IDs for validation

            - test_patients (list): List of patient IDs for testing
    """
    return {
        'training_patients':
            ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"],
        'validation_patients':
            ["S11", "S12"],
        'test_patients':
            ["S13", "S14", "S15"]
    }

if __name__ == "__main__":
    train()
