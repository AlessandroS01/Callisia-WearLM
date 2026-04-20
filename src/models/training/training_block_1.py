"""
Block 1 Training - ENTRY POINT + CONFIG MANAGEMENT

RESPONSIBILITIES:
- Entry points: train_split(), train_loso()
- Config management: [load_training_config(), get_optimizer_config(),
                        get_loss_config(), save_run_config()]

ALL TRAINING LOGIC is in training_strategy.py

DEPENDENCY FLOW:
training_block_1.py (Entry point + Config)
  ↓ imports TrainingStrategy
training_strategy.py (Orchestrator + Training logic)
"""

import json
import os
from datetime import datetime

import yaml
from torch.utils.data import DataLoader

from src.data.dataset.hr_dataset import HRDataset
from src.models.block_utils import setup_run_directory
from src.models.training.block_1_data_loader import Block1TrainingDataLoader
from src.models.training.training_strategy import TrainingStrategy


def save_run_config(run_dir: str, config: dict, learning_rate: float, num_epochs: int,
                    dataset_config: dict, optimizer_config: dict = None,
                    loss_config: dict = None, training_method: str = 'split'):
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
        training_method: Training method used ('split' or 'loso')

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
        'training_method': training_method,
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
    Returns the optimizer configuration from config.yaml.

    Single source of truth for optimizer settings.
    Update config.yaml to change optimizer settings everywhere.

    Returns:
        dict: Optimizer configuration with name and parameters
    """
    config = load_training_config()
    return {
        'name': config.get('optimizer', 'Adam'),
        'params': {
            'weight_decay': config.get('optimizer_weight_decay', 1e-4)
        }
    }

def get_loss_config() -> dict:
    """
    Returns the loss function configuration from config.yaml.

    Single source of truth for loss function settings.
    Update config.yaml to change loss function settings everywhere.

    Returns:
        dict: Loss function configuration with name and parameters
    """
    config = load_training_config()
    loss_name = config.get('loss_function', 'HuberLoss')

    # Build parameters based on loss function type
    if loss_name == 'HuberLoss':
        params = {'delta': config.get('loss_delta', 5.0)}
    elif loss_name == 'SmoothL1Loss':
        params = {'beta': config.get('loss_beta', 0.5)}
    else:
        params = {}

    return {
        'name': loss_name,
        'params': params
    }

def train(method: str = 'split') -> None:
    """
    Main training entry point.

    Handles common setup:
    1. Load configuration
    2. Setup training (get dataloaders)
    3. Retrieve patient splits / subjects
    4. Create run directory
    5. Save run configuration
    6. Pass all data to TrainingStrategy for orchestration

    Args:
        method (str): 'split' for fixed split training or 'loso' for LOSO cross-validation
    """

    # ==================== 1. LOAD CONFIGURATION ====================
    config = load_training_config()
    version = config.get('version', '5th_version')
    learning_rate = config.get('learning_rate', 0.0005)
    num_epochs = config.get('num_epochs', 25)
    batch_size = config.get('batch_size', 32)

    optimizer_config = get_optimizer_config()
    loss_config = get_loss_config()

    # ==================== 2. SETUP TRAINING (GET DATALOADERS) ====================
    if method == 'split':
        print("Training method: FIXED SPLIT\n")
        patient_splits = get_split_patients()
        data_loader = Block1TrainingDataLoader()

        x_train, y_train = data_loader.prepare_dataset(
            patient_splits['training_patients'], "training"
        )
        x_valid, y_valid = data_loader.prepare_dataset(
            patient_splits['validation_patients'], "validation"
        )

        train_dataset = HRDataset(x_train, y_train)
        valid_dataset = HRDataset(x_valid, y_valid)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        dataset_config = patient_splits
        loaders_data = {
            'train_loader': train_loader,
            'valid_loader': valid_loader
        }

    elif method == 'loso':
        print("Training method: LOSO CROSS-VALIDATION\n")
        loader = Block1TrainingDataLoader()
        all_subjects = loader.get_all_subjects()
        dataset_config = {'loso_subjects': all_subjects}
        loaders_data = None  # LOSO prepare its own loaders per fold
    else:
        raise ValueError(f"Unknown training method: {method}. Must be 'split' or 'loso'")

    # ==================== 3. CREATE RUN DIRECTORY ====================
    run_dir = setup_run_directory(f"history/block_1/{version}")

    # ==================== 4. SAVE RUN CONFIGURATION ====================
    save_run_config(run_dir, config, learning_rate, num_epochs,
                    dataset_config, optimizer_config, loss_config,
                    training_method=method)

    # ==================== 5. DELEGATE TO TRAINING STRATEGY ====================
    trainer = TrainingStrategy(
        method=method,
        config=config,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        optimizer_config=optimizer_config,
        loss_config=loss_config,
        run_dir=run_dir,
        version=version,
        loaders_data=loaders_data
    )
    trainer.train()


def get_split_patients():
    """
    Returns the predefined patient splits for training, validation, and testing.

    Returns:
        dict: Dictionary containing patient splits from the data loader
    """
    loader = Block1TrainingDataLoader()
    return loader.get_patients()

if __name__ == "__main__":
    train("split")
