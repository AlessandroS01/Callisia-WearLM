"""
Training Strategy - Orchestrator for different training approaches.

RECEIVES ALL DATA FROM training_block_1 - No imports from training_block_1!

RESPONSIBILITIES:
- Implement strategy pattern (split vs LOSO)
- Core training functions: train_epoch(), validate(), test()
- Orchestrate training loop
- Orchestrate artifact saving

DEPENDENCY FLOW (ONE-WAY):
training_block_1.py (Entry point - prepares all data)
  → training_strategy.py (Receives all data as parameters)
  → block_1_data_loader.py (for LOSO patient splits)
  → evaluation_artifacts.py (for metrics & results)
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.dataset.hr_dataset import HRDataset
from src.models.architecture.hr_cnn import MultimodalHRNet
from src.models.evaluation_artifacts import EvaluationArtifacts
from src.models.training.block_1_data_loader import Block1TrainingDataLoader


# ==================== CORE TRAINING FUNCTIONS ====================

def train_epoch(model, train_loader, optimizer, loss_function, device):
    """Execute a single training epoch."""
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

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

    return epoch_loss / num_batches


def validate(model, valid_loader, loss_function, device):
    """Evaluate model on validation set."""
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

    return epoch_loss / num_batches


def test(model, test_loader, loss_function, device):
    """Evaluate model on test set and collect predictions."""
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

            preds_np = predictions.squeeze().cpu().numpy()
            targets_np = y_batch.squeeze().cpu().numpy()

            preds_list = np.atleast_1d(preds_np).tolist()
            targets_list = np.atleast_1d(targets_np).tolist()

            all_predictions.extend(preds_list)
            all_targets.extend(targets_list)

            if (batch_idx + 1) % 5 == 0:
                print(f"    Batch [{batch_idx + 1}/{len(test_loader)}] - Loss: {loss.item():.4f}")

    avg_loss = test_loss / num_batches
    print(f"Test completed - "
          f"Processed {num_batches} batches with {len(all_predictions)} total samples")

    return avg_loss, all_predictions, all_targets


# ==================== TRAINING STRATEGY CLASS ====================

class TrainingStrategy:
    """
    Implements Strategy pattern for training method selection.

    Receives all necessary data from training_block_1 as constructor parameters.
    Supports fixed split (fast) and LOSO cross-validation (rigorous) approaches.
    """

    def __init__(self, method: str = 'split', config: dict = None,
                 learning_rate: float = 0.0005, num_epochs: int = 25,
                 batch_size: int = 32, optimizer_config: dict = None,
                 loss_config: dict = None, run_dir: Optional[str] = None,
                 version: str = '5th_version', loaders_data: Optional[Dict] = None) -> None:
        """
        Initialize training strategy with all necessary data from training_block_1.

        Args:
            method: 'split' for fixed split or 'loso' for cross-validation
            config: Full configuration dictionary from config.yaml
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Batch size for dataloaders
            optimizer_config: Optimizer configuration dict
            loss_config: Loss function configuration dict
            run_dir: Directory for saving results
            version: Model version (e.g., '5th_version')
            loaders_data: Dict with train_loader, valid_loader, test_loader for split method
        """
        self.method = method
        if method not in ['split', 'loso']:
            raise ValueError(f"Method must be 'split' or 'loso', got '{method}'")

        self.config = config or {}
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer_config = optimizer_config or {}
        self.loss_config = loss_config or {}
        self.run_dir = run_dir
        self.version = version
        self.loaders_data = loaders_data  # For split method

    def train(self) -> None:
        """Execute training with selected method."""
        if self.method == 'split':
            self._train_split()
        else:
            self._train_loso()

    def _train_split(self) -> None:
        """Fixed split training (70% train, 15% val, 15% test)."""
        print("="*70)
        print("TRAINING WITH FIXED SPLIT METHOD")
        print("="*70 + "\n")

        # Get pre-loaded loaders from training_block_1
        train_loader = self.loaders_data['train_loader']
        valid_loader = self.loaders_data['valid_loader']
        test_loader = self.loaders_data['test_loader']

        # Initialize training components
        print("Initializing training components...")
        device, model, optimizer, loss_function, scheduler, model_run_dir = \
            self._initialize_training_components()
        print()

        # Train model with validation
        print("Training model...")
        _, _, _, epochs_data = \
            self._run_training_loop(model, train_loader, valid_loader, optimizer,
                                  loss_function, device, scheduler, model_run_dir)
        print()

        # Test model
        print("Testing model on held-out test set...")
        print("="*70)
        avg_test_loss, predictions, targets = test(model, test_loader, loss_function, device)
        print(f"✓ Test Loss: {avg_test_loss:.4f}\n")

        # Save artifacts
        print("Saving training artifacts and results...")
        self._save_training_results(epochs_data, predictions, targets)
        print("\n" + "="*70)
        print("SPLIT TRAINING COMPLETE!")
        print("="*70 + "\n")

    def _train_loso(self) -> None:
        """Leave-One-Subject-Out cross-validation."""
        print("="*70)
        print("TRAINING WITH LOSO CROSS-VALIDATION METHOD")
        print("="*70 + "\n")

        # Get all subjects
        loader = Block1TrainingDataLoader()
        all_subjects = loader.get_all_subjects()
        print(f"Total subjects: {len(all_subjects)}")
        print(f"Subjects: {', '.join(all_subjects)}\n")

        # Create main run directories
        models_base_dir = f"../../../models/block_1/{self.version}"
        run_name = os.path.basename(self.run_dir or "")
        models_run_dir = os.path.join(models_base_dir, run_name)
        os.makedirs(models_run_dir, exist_ok=True)
        print(f"History directory: {self.run_dir}")
        print(f"Models directory: {models_run_dir}\n")

        fold_results = []

        # Train each fold
        for fold_idx, test_subject in enumerate(all_subjects, 1):
            self._print_fold_header(fold_idx, len(all_subjects), test_subject)

            # Get fold-specific split
            split = self._get_loso_split(test_subject, all_subjects)

            # Prepare fold data
            print("Preparing data...")
            train_dataset, valid_dataset, test_dataset = \
                self._prepare_loso_datasets(loader, split)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            print(f"Train: {len(train_dataset)}, "
                  f"Val: {len(valid_dataset)}, Test: {len(test_dataset)}\n")

            # Create fold directories
            history_fold_dir = str(
                os.path.join(self.run_dir or "", f"fold_{fold_idx:02d}_{test_subject}")
            )
            models_fold_dir = str(
                os.path.join(models_run_dir, f"fold_{fold_idx:02d}_{test_subject}")
            )
            os.makedirs(history_fold_dir, exist_ok=True)
            os.makedirs(models_fold_dir, exist_ok=True)

            # Initialize components
            print("Initializing training components...")
            device, model, optimizer, loss_function, scheduler, _ = \
                self._initialize_training_components_for_fold(models_fold_dir)
            print()

            # Train
            print("Training...")
            _, _, _, epochs_data = \
                self._run_training_loop(model, train_loader, valid_loader, optimizer,
                                      loss_function, device, scheduler, models_fold_dir)
            print()

            # Test
            print("Testing...")
            avg_test_loss, predictions, targets = test(model, test_loader, loss_function, device)

            # Calculate metrics
            test_metrics = EvaluationArtifacts.calculate_metrics(
                np.array(predictions), np.array(targets)
            )
            test_metrics['num_samples'] = len(predictions)
            print(f"Test Loss: {avg_test_loss:.4f}")
            print(f"Test MAE: {test_metrics.get('mae', 0):.4f} bpm\n")

            # Save fold artifacts
            self._save_fold_artifacts(
                history_fold_dir, epochs_data, predictions, targets, test_metrics
            )
            print()

            fold_results.append({
                'subject': test_subject,
                'test_loss': avg_test_loss,
                'metrics': test_metrics
            })

        # Create ensemble model
        print("\n" + "="*70)
        print("Creating ensemble model from all folds...")
        print("="*70 + "\n")
        self._create_ensemble_model(models_run_dir, len(fold_results))

        print("="*70)
        print("LOSO CROSS-VALIDATION COMPLETE!")
        print("="*70 + "\n")

    def _initialize_optimizer(self, model) -> torch.optim.Optimizer:
        """Initialize optimizer from config."""
        if self.optimizer_config.get('name') == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=self.learning_rate,
                                   **self.optimizer_config.get('params', {}))

        raise ValueError(f"Unsupported optimizer: {self.optimizer_config.get('name')}")


    def _initialize_loss_and_scheduler(self, optimizer) -> Tuple:
        """Initialize loss function and scheduler. Returns (loss_function, scheduler)."""
        if self.loss_config.get('name') == 'HuberLoss':
            loss_function = torch.nn.HuberLoss(**self.loss_config.get('params', {}))
        elif self.loss_config.get('name') == 'SmoothL1Loss':
            loss_function = torch.nn.SmoothL1Loss(**self.loss_config.get('params', {}))
        else:
            loss_function = torch.nn.MSELoss()

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-7)
        return loss_function, scheduler

    def _initialize_training_components(self) -> Tuple:
        """Initialize model, optimizer, scheduler."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {device}\n")

        model = MultimodalHRNet().to(device)
        optimizer = self._initialize_optimizer(model)
        loss_function, scheduler = self._initialize_loss_and_scheduler(optimizer)

        model_run_dir = os.path.join(f"../../../models/block_1/{self.version}",
                                    os.path.basename(self.run_dir or ""))
        os.makedirs(model_run_dir, exist_ok=True)

        return device, model, optimizer, loss_function, scheduler, model_run_dir

    def _initialize_training_components_for_fold(self, models_fold_dir: str) -> Tuple:
        """Initialize model, optimizer, scheduler for a fold."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MultimodalHRNet().to(device)
        optimizer = self._initialize_optimizer(model)
        loss_function, scheduler = self._initialize_loss_and_scheduler(optimizer)

        return device, model, optimizer, loss_function, scheduler, models_fold_dir

    def _run_training_loop(self, model, train_loader, valid_loader, optimizer, loss_function,
                           device, scheduler, num_epochs_or_model_dir) -> Tuple:
        """Execute training and validation loop."""
        # Handle both cases: split (returns model_run_dir) and LOSO (returns model_run_dir)
        if isinstance(num_epochs_or_model_dir, str):
            model_run_dir = num_epochs_or_model_dir
            num_epochs = self.num_epochs
        else:
            num_epochs = num_epochs_or_model_dir
            model_run_dir = None

        best_val_loss, epochs_data = float('inf'), []

        print("="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")

        for epoch in range(num_epochs):
            print(f"{'─'*70}\nEpoch {epoch + 1}/{num_epochs}\n{'─'*70}")

            print("  [1/2] Training phase...")
            avg_train_loss = train_epoch(model, train_loader, optimizer, loss_function, device)

            print("\n  [2/2] Validation phase...")
            avg_val_loss = validate(model, valid_loader, loss_function, device)

            improvement = "↓" if avg_val_loss < best_val_loss else "↑"
            loss_diff = abs(avg_val_loss - best_val_loss)
            print(f"\n  Results:\n    Train Loss: {avg_train_loss:.4f}")
            print(f"    Val Loss:   {avg_val_loss:.4f} ({improvement} {loss_diff:.4f})")

            is_best = False
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if model_run_dir:
                    torch.save(model.state_dict(), os.path.join(model_run_dir, "best_model.pth"))
                is_best = True
                print(f"    ✓ Best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                print("    • No improvement")

            scheduler.step(avg_val_loss)
            epochs_data.append({'epoch': epoch + 1, 'train_loss': round(avg_train_loss, 6),
                               'val_loss': round(avg_val_loss, 6), 'best_model': is_best})
            print()

        return [], [], best_val_loss, epochs_data

    def _print_fold_header(self, fold_idx: int, total_folds: int, test_subject: str):
        """Print header for each LOSO fold."""
        print("\n" + "="*70)
        print(f"FOLD [{fold_idx}/{total_folds}]: Subject {test_subject} held out for testing")
        print("="*70 + "\n")

    def _get_loso_split(self, test_subject: str, all_subjects: List[str]) -> Dict[str, List[str]]:
        """Create LOSO split."""
        remaining_subjects = [s for s in all_subjects if s != test_subject]
        num_val = max(1, len(remaining_subjects) // 5)
        training_subjects = remaining_subjects[:-num_val]
        validation_subjects = remaining_subjects[-num_val:]

        return {
            'test_patients': [test_subject],
            'training_patients': training_subjects,
            'validation_patients': validation_subjects
        }

    def _prepare_loso_datasets(self, loader: Block1TrainingDataLoader,
                        split: Dict[str, List[str]]) -> Tuple[HRDataset, HRDataset, HRDataset]:
        """Prepare PyTorch datasets for one LOSO fold."""
        x_train, y_train = loader.prepare_dataset(split['training_patients'], "training")
        x_valid, y_valid = loader.prepare_dataset(split['validation_patients'], "validation")
        x_test, y_test = loader.prepare_dataset(split['test_patients'], "testing")

        return (
            HRDataset(x_train, y_train),
            HRDataset(x_valid, y_valid),
            HRDataset(x_test, y_test)
        )

    def _create_ensemble_model(self, models_run_dir: str, num_folds: int) -> None:
        """Create averaged model from all folds."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {num_folds} fold models...")
        fold_models = []

        for fold_idx in range(1, num_folds + 1):
            fold_name = None
            for entry in os.listdir(models_run_dir):
                if entry.startswith(f"fold_{fold_idx:02d}_"):
                    fold_name = entry
                    break

            if fold_name:
                model_path = os.path.join(models_run_dir, fold_name, "best_model.pth")
                if os.path.exists(model_path):
                    model = MultimodalHRNet().to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    fold_models.append(model)
                    print(f"  ✓ Loaded fold {fold_idx}")

        if not fold_models:
            print("✗ No fold models found")
            return

        print(f"\n✓ Loaded {len(fold_models)} models\n")
        print("Averaging weights...")

        averaged_model = MultimodalHRNet().to(device)
        avg_state_dict = averaged_model.state_dict()

        # Initialize with zeros, preserving dtype and device
        for key in avg_state_dict.keys():
            avg_state_dict[key] = torch.zeros_like(avg_state_dict[key], dtype=torch.float32)

        # Accumulate weights from all fold models
        for fold_model in fold_models:
            fold_state_dict = fold_model.state_dict()
            for key in avg_state_dict.keys():
                avg_state_dict[key] += fold_state_dict[key].float()

        # Average the accumulated weights
        for key in avg_state_dict.keys():
            avg_state_dict[key] /= len(fold_models)

        # Convert back to original dtype
        final_state_dict = {}
        for key in avg_state_dict.keys():
            original_dtype = averaged_model.state_dict()[key].dtype
            final_state_dict[key] = avg_state_dict[key].to(original_dtype)

        averaged_model.load_state_dict(final_state_dict)

        averaged_model_dir = os.path.join(models_run_dir, "averaged_model")
        os.makedirs(averaged_model_dir, exist_ok=True)
        averaged_model_path = os.path.join(averaged_model_dir, "best_model.pth")

        torch.save(averaged_model.state_dict(), averaged_model_path)
        print(f"✓ Averaged model saved to: {averaged_model_path}\n")

    def _save_fold_artifacts(self, fold_dir: str, epochs_data: List[Dict],
                             predictions: List, targets: List, test_metrics: Dict) -> None:
        """Save all artifacts for a LOSO fold."""
        EvaluationArtifacts.save_fold_artifacts(
            fold_dir, epochs_data, predictions, targets, test_metrics
        )

    def _save_training_results(self, epochs_data: List[Dict],
                              predictions: List, targets: List) -> None:
        """Save training results."""
        print("\n" + "="*70)
        print("SAVING TRAINING ARTIFACTS")
        print("="*70)

        metrics = EvaluationArtifacts.calculate_metrics(
            np.array(predictions), np.array(targets)
        )
        metrics['num_samples'] = len(predictions)

        metrics_csv = str(os.path.join(self.run_dir or "", "training_metrics.csv"))
        history_png = str(os.path.join(self.run_dir or "", "training_history.png"))

        EvaluationArtifacts.save_training_metrics(epochs_data, metrics_csv)
        EvaluationArtifacts.plot_training_history(metrics_csv, history_png)

        print("\n" + "="*70)
        print("SAVING TEST RESULTS & ANALYSIS")
        print("="*70)

        test_results_csv = str(os.path.join(self.run_dir or "", "test_results.csv"))
        test_analysis_png = str(os.path.join(self.run_dir or "", "test_analysis.png"))

        EvaluationArtifacts.save_test_results(predictions, targets, test_results_csv)
        EvaluationArtifacts.plot_test_results(predictions, targets, metrics, test_analysis_png)

        print("\n" + "─"*70)
        print("TEST PERFORMANCE METRICS")
        print("─"*70)
        EvaluationArtifacts.print_metrics_summary(metrics, metrics['num_samples'])
        print("─"*70)
