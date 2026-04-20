"""Neural network models for heart-rate estimation.

This module provides PyTorch model definitions used for heart-rate
estimation from bvp and acc wearable sensors. It currently contains a
placeholder `MultimodalHRNet` class to be implemented with convolutional
backbones and fusion logic.
"""

from torch import nn


class MultimodalHRNet(nn.Module):
    """Multi-modal 1D CNN for heart-rate estimation from wearable sensors.

    SIMPLIFIED ARCHITECTURE (Version 6):
    - Reduced parameters from 8.2M to ~80K (appropriate for small dataset)
    - Uses GlobalAvgPool for natural dimensionality reduction
    - Minimal FC layers to prevent overfitting
    - Normalized dropout and BatchNorm usage
    - Better suited for ~2500 training samples per LOSO fold

    The model learns robust HR patterns while maintaining generalization.

    Input shape
    -----------
    The model expects input tensors of shape ``(batch_size, in_channels,
    sequence_length)`` for 1D convolutional processing.

    Example
    -------
    >>> # model = MultimodalHRNet()
    >>> # x = torch.randn(8, 4, 512)
    >>> # out = model(x)
    """
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()

        # --- SIMPLIFIED CONV BLOCKS (Version 6) ---
        # Focused feature extraction with moderate capacity
        # 3 conv blocks: 4 → 16 → 32 → 64 channels

        # Block 1: Input (4 channels) -> Output (16 channels)
        # Kernel size 7 at 64Hz = ~110ms window (captures HR rhythm)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # Length: 512 -> 512 (padding) -> 256 (pool)
        )

        # Block 2: Input (16) -> Output (32)
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # Length: 256 -> 256 (padding) -> 128 (pool)
        )

        # Block 3: Input (32) -> Output (64)
        # Final feature extraction layer
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # Length: 128 -> 128 (padding) -> 64 (pool)
        )

        # --- GLOBAL AVERAGE POOLING ---
        # Reduces (batch, 64, 64) -> (batch, 64) naturally
        # Prevents overfitting by avoiding flatten-to-large-FC
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # --- LIGHTWEIGHT PREDICTION HEAD ---
        # Minimal FC layers appropriate for small dataset
        # 64 -> 32 -> 1
        self.fc_layers = nn.Sequential(
            # FC Layer 1: 64 -> 32
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            # Output Layer: 32 -> 1 (HR prediction)
            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, window_length).

        Returns
        -------
        torch.Tensor
            Output tensor containing heart-rate predictions.
        """
        # Conv feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Global average pooling: (batch, 64, 64) -> (batch, 64, 1)
        x = self.global_pool(x)

        # Flatten: (batch, 64, 1) -> (batch, 64)
        x = self.flatten(x)

        # FC layers for prediction
        out = self.fc_layers(x)

        return out.squeeze()  # Return shape (batch_size,) for regression output
