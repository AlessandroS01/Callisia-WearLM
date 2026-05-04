import numpy as np
import torch


class TrainingHelper:

    def __init__(self):
        pass

    # ==================== CORE TRAINING FUNCTIONS ====================

    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
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
        return self._reduce_loss(loss)

    def train_epoch(self, model, train_loader, optimizer, loss_function, device):
        """Execute a single training epoch with cost-sensitive weighted loss."""
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
            weights[target > 120.0] = 3.0
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


