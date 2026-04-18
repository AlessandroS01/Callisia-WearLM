"""
Module for training the initial 1D CNN for HR estimation from BVP and ACC data.
"""

import csv
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.dataset.hr_dataset import HRDataset
from src.models.block_utils import setup_run_directory, save_test_results
from src.models.evaluation_utils import plot_test_results, print_metrics_summary
from src.models.hr_cnn import MultimodalHRNet
from src.models.training.block_1_data_loader import Block1TrainingDataLoader


def save_run_config(run_dir: str, config: dict, learning_rate: float, num_epochs: int,
                    dataset_config: dict, optimizer_config: dict = None,
                    loss_config: dict = None):
    """
    Saves training configuration to a JSON file in the run directory.

    Params:
        run_dir: Path to the run directory
        config: Configuration dictionary from config.yaml
        learning_rate: Learning rate used for training
        num_epochs: Number of epochs
        dataset_config: Dictionary containing training/validation/test patient splits
        optimizer_config: Dictionary with optimizer name and parameters
        loss_config: Dictionary with loss function name and parameters

    Note:
        If optimizer_config or loss_config are not provided, defaults are used.
        These should come from get_optimizer_config() and get_loss_config().
    """
    # Use defaults if not provided
    if optimizer_config is None:
        optimizer_config = get_optimizer_config()
    if loss_config is None:
        loss_config = get_loss_config()

    config_data = {
        'timestamp': datetime.now().isoformat(),
        'learning_rate': learning_rate,
        'batch_size': config.get('batch_size'),
        'num_epochs': num_epochs,
        'optimizer': optimizer_config.get('name', 'Adam'),
        'optimizer_params': {
            'learning_rate': learning_rate,
            **optimizer_config.get('params', {})
        },
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_params': {
            'mode': 'min',
            'factor': 0.5,
            'patience': 3,
            'min_lr': 1e-7,
            'lr_reduction_factor': 0.5
        },
        'loss_function': loss_config.get('name', 'MSELoss'),
        'loss_function_params': loss_config.get('params', {}),
        'dataset': dataset_config
    }

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)

    print(f"✓ Configuration saved to: {config_path}")


def load_training_config(config_path: str = "../../../config.yaml") -> dict:
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


def get_optimizer_config() -> dict:
    """
    Returns the optimizer configuration.

    Single source of truth for optimizer settings.
    Update here to propagate changes everywhere.

    Returns:
        dict: Optimizer configuration with name and parameters
    """
    return {
        'name': 'Adam',
        'params': {
            'weight_decay': 1e-4
        }
    }


def get_loss_config() -> dict:
    """
    Returns the loss function configuration.

    Single source of truth for loss function settings.
    Update here to propagate changes everywhere.

    Returns:
        dict: Loss function configuration with name and parameters
    """
    return {
        'name': 'HuberLoss',
        'params': {
            'delta': 5.0
        }
    }



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
    config = load_training_config()
    batch_size = config["batch_size"]

    print("Configuration loaded:")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of epochs: {config['num_epochs']}\n")

    patient_splits = get_split_patients()
    loader = Block1TrainingDataLoader()

    x_train, y_train = loader.prepare_dataset(
        patient_splits['training_patients'], "training"
    )
    x_valid, y_valid = loader.prepare_dataset(
        patient_splits['validation_patients'], "validation"
    )
    x_test, y_test = loader.prepare_dataset(
        patient_splits['test_patients'], "test"
    )

    # Create PyTorch datasets
    train_dataset = HRDataset(x_train, y_train)
    valid_dataset = HRDataset(x_valid, y_valid)
    test_dataset = HRDataset(x_test, y_test)

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
        ax.set_ylabel('Loss (Huber)', fontsize=12, fontweight='bold')
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



def _initialize_training_components(learning_rate, run_dir):
    """Initialize model, optimizer, scheduler, and directories."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}\n")

    # Get configurations from single source of truth
    optimizer_cfg = get_optimizer_config()
    loss_cfg = get_loss_config()

    print("Initializing model, optimizer, and loss function...")
    model = MultimodalHRNet().to(device)

    # Create optimizer using configuration
    if optimizer_cfg['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                    **optimizer_cfg['params'])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_cfg['name']}")

    # Create loss function using configuration
    if loss_cfg['name'] == 'HuberLoss':
        loss_function = torch.nn.HuberLoss(**loss_cfg['params'])
    elif loss_cfg['name'] == 'SmoothL1Loss':
        loss_function = torch.nn.SmoothL1Loss(**loss_cfg['params'])
    elif loss_cfg['name'] == 'MSELoss':
        loss_function = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_cfg['name']}")

    print(f"✓ Model: {model.__class__.__name__}")
    print(f"✓ Optimizer: {optimizer_cfg['name']} (lr={learning_rate}, "
          f"weight_decay={optimizer_cfg['params'].get('weight_decay', 0)})")
    print(f"✓ Loss Function: {loss_cfg['name']}"
          f"({', '.join(f'{k}={v}' for k, v in loss_cfg['params'].items())})\n")

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=1, min_lr=1e-7)
    print("✓ Learning Rate Scheduler: ReduceLROnPlateau\n")

    model_run_dir = os.path.join("../../../models/block_1/4rd_version",
                                f"{os.path.basename(run_dir)}")
    os.makedirs(model_run_dir, exist_ok=True)
    print(f"✓ Model directory created: {model_run_dir}\n")

    return device, model, optimizer, loss_function, scheduler, model_run_dir


def _run_training_loop(model, train_loader, valid_loader, optimizer, loss_function,
                       device, scheduler, num_epochs, model_run_dir):
    """Execute training and validation loop."""
    best_val_loss, epochs_data = float('inf'), []
    train_losses, val_losses = [], []

    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")

    for epoch in range(num_epochs):
        print(f"{'─'*70}\nEpoch {epoch + 1}/{num_epochs}\n{'─'*70}")

        print("  [1/2] Training phase...")
        avg_train_loss = train_epoch(model, train_loader, optimizer,
                                    loss_function, device)
        train_losses.append(avg_train_loss)

        print("\n  [2/2] Validation phase...")
        avg_val_loss = validate(model, valid_loader, loss_function, device)
        val_losses.append(avg_val_loss)

        improvement, loss_diff = ("↓" if avg_val_loss < best_val_loss else "↑"), \
                                 abs(avg_val_loss - best_val_loss)
        print(f"\n  Results:\n    Train Loss: {avg_train_loss:.4f}")
        print(f"    Val Loss:   {avg_val_loss:.4f} ({improvement} {loss_diff:.4f})")

        is_best = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_run_dir, "best_model.pth"))
            is_best = True
            print(f"    ✓ Best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            print("    • No improvement")

        scheduler.step(avg_val_loss)
        epochs_data.append({'epoch': epoch + 1, 'train_loss': round(avg_train_loss, 6),
                           'val_loss': round(avg_val_loss, 6), 'best_model': is_best})
        print()

    return train_losses, val_losses, best_val_loss, epochs_data


def _save_training_artifacts(run_dir, epochs_data, predictions, targets, test_metrics):
    """Save all training artifacts and metrics."""
    print("\n" + "="*70)
    print("SAVING TRAINING ARTIFACTS")
    print("="*70)
    save_training_metrics(epochs_data, os.path.join(run_dir, "training_metrics.csv"))
    plot_training_history(os.path.join(run_dir, "training_metrics.csv"),
                         os.path.join(run_dir, "training_history.png"))

    print("\n" + "="*70)
    print("SAVING TEST RESULTS & ANALYSIS")
    print("="*70)
    test_metrics = save_test_results(predictions, targets, run_dir)

    if test_metrics:
        plot_test_results(predictions, targets, test_metrics,
                         os.path.join(run_dir, "test_analysis.png"))
        print("\n" + "─"*70)
        print("TEST PERFORMANCE METRICS")
        print("─"*70)
        print_metrics_summary(test_metrics, test_metrics['num_samples'])
        print("─"*70)


def train():
    """
    Main training function that orchestrates the complete training pipeline.

    Orchestrates model initialization, training loop, and artifact saving.

    Configuration is automatically retrieved from:
    - get_optimizer_config(): Optimizer settings (single source of truth)
    - get_loss_config(): Loss function settings (single source of truth)
    - get_split_patients(): Dataset patient splits (single source of truth)
    """
    print("="*70)
    print("INITIALIZING TRAINING PIPELINE")
    print("="*70 + "\n")

    train_loader, valid_loader, test_loader, learning_rate, num_epochs = setup_training()

    run_dir = setup_run_directory("history/block_1/4th_version")

    # Get configurations from single sources of truth
    optimizer_config = get_optimizer_config()
    loss_config = get_loss_config()

    # Save configuration with actual optimizer and loss settings
    save_run_config(run_dir, load_training_config(), learning_rate, num_epochs,
                    get_split_patients(), optimizer_config, loss_config)

    device, model, optimizer, loss_function, scheduler, model_run_dir = \
        _initialize_training_components(learning_rate, run_dir)

    train_losses, val_losses, best_val_loss, epochs_data = \
        _run_training_loop(model, train_loader, valid_loader, optimizer,
                          loss_function, device, scheduler, num_epochs, model_run_dir)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nStarting Testing Phase...\n")

    print("["*35)
    print("  Testing model on held-out test set...")
    avg_test_loss, predictions, targets = test(model, test_loader,
                                               loss_function, device)
    print(f"\n  ✓ Test Loss: {avg_test_loss:.4f}\n")

    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"  Total Epochs: {num_epochs}")
    print(f"  Final Train Loss: {train_losses[-1]:.4f}")
    print(f"  Final Val Loss: {val_losses[-1]:.4f}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Samples: {len(predictions)}")
    print(f"  Model: {model_run_dir}/best_model.pth")
    print("="*70)

    _save_training_artifacts(run_dir, epochs_data, predictions, targets, None)
    print("="*70 + "\n")

def get_split_patients():
    """
    Returns the predefined patient splits for training, validation, and testing.

    Returns:
        dict: Dictionary containing patient splits from the data loader
    """
    loader = Block1TrainingDataLoader()
    return loader.get_patients()

if __name__ == "__main__":
    train()
