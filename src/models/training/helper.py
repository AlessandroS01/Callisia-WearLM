import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


class TrainingHelper:

    def __init__(self):
        pass

    # ==================== CORE TRAINING FUNCTIONS ====================

    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce loss tensor to scalar if needed (handles reduction='none' case)."""
        if loss.dim() > 0:
            return loss.mean()
        return loss

    def _compute_eval_batch_loss(self, predictions, targets, loss_function):
        """
        Compute and reduce loss for evaluation batches (validation/testing).

        Handles both reduction='none' and standard reduction cases.

        Args:
            predictions: Model predictions (will be squeezed)
            targets: Target values (will be squeezed)
            loss_function: Loss function to compute loss

        Returns:
            torch.Tensor: Scalar loss value
        """
        loss = loss_function(predictions.squeeze(), targets.squeeze())
        return self.reduce_loss(loss)

    def train_epoch(self, model, train_loader, optimizer, loss_function, device,
                    weight_multiplier: float = 1.5):
        """Execute a single training epoch with cost-sensitive weighted loss.
        
        Args:
            model: The neural network model
            train_loader: DataLoader for training data
            optimizer: Optimizer for training
            loss_function: Loss function to compute loss
            device: Device to run on (cuda or cpu)
            weight_multiplier: Multiplier for loss when target > 120 BPM (default: 1.5)
        """
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch).squeeze()
            target = y_batch.squeeze()

            # Cost-sensitive loss: penalize high heart rate predictions more
            base_loss = loss_function(pred, target)
            weights = torch.ones_like(target)
            weights[target > 120.0] = weight_multiplier
            weighted_loss = (base_loss * weights).mean()

            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] - "
                      f"Loss: {weighted_loss.item():.4f}")

        return epoch_loss / num_batches

    def validate(self, model, valid_loader, loss_function, device):
        """Evaluate model on validation set."""
        model.eval()
        epoch_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for _, (x_batch, y_batch) in enumerate(valid_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(x_batch)
                loss = self._compute_eval_batch_loss(predictions, y_batch, loss_function)

                epoch_loss += loss.item()
                num_batches += 1

            print(f"  Validation completed - Processed {num_batches} batches")

        return epoch_loss / num_batches

    def test(self, model, test_loader, loss_function, device):
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
                loss = self._compute_eval_batch_loss(predictions, y_batch, loss_function)

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

    def get_warmup_cosine_scheduler(self, optimizer, num_epochs: int, num_batches_per_epoch: int,
                                    warmup_epochs: int = 2):
        """
        Creates a unified warmup + cosine annealing scheduler for Transformer models.
        Updates per-batch (not per-epoch) for ultra-smooth learning rate adjustments.
        """
        total_steps = num_epochs * num_batches_per_epoch
        warmup_steps = warmup_epochs * num_batches_per_epoch

        def unified_lr_lambda(current_step: int) -> float:
            """Calculates the learning rate multiplier for the current batch step."""

            # PHASE 1: Linear Warmup (Ramp from 0.0 to 1.0)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            # PHASE 2: Cosine Annealing (Decay from 1.0 to 0.0)
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        # Return a SINGLE scheduler
        scheduler = LambdaLR(optimizer, unified_lr_lambda)

        return scheduler

    def apply_patch_masking(self, x_batch: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
        """Apply random patch masking for regularization (Transformer-specific).

        This is a form of data augmentation/regularization that randomly masks out
        entire time patches from the input, forcing the model to learn robust representations.

        Args:
            x_batch: Input tensor of shape (batch_size, channels, sequence_length)
            mask_ratio: Ratio of patches to mask out (default: 0.1 = 10%)

        Returns:
            torch.Tensor: Masked input batch with same shape as input
        """
        batch_size, channels, seq_len = x_batch.shape

        # For a typical setup: 512 sequence length / 16 patch size = 32 patches
        # So we randomly mask about 3-4 patches for 10% masking

        num_patches = seq_len // 16  # Assuming patch_size=16 from PatchHRNet
        num_masks = max(1, int(num_patches * mask_ratio))

        x_masked = x_batch.clone()

        for b in range(batch_size):
            # Randomly select patches to mask
            masked_indices = np.random.choice(num_patches, size=num_masks, replace=False).tolist()

            for patch_idx in masked_indices:
                patch_idx = int(patch_idx)
                start_idx = patch_idx * 16
                end_idx = start_idx + 16
                # Set masked patch to zero
                x_masked[b, :, start_idx:end_idx] = 0.0

        return x_masked
