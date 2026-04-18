"""Neural network models for heart-rate estimation.

This module provides PyTorch model definitions used for heart-rate
estimation from bvp and acc wearable sensors. It currently contains a
placeholder `MultimodalHRNet` class to be implemented with convolutional
backbones and fusion logic.
"""

from torch import nn


class MultimodalHRNet(nn.Module):
    """Multi-modal 1D CNN for heart-rate estimation from wearable sensors.

    This class defines a skeleton PyTorch module that fuses multiple sensor
    channels (BVP and tri-axial ACC) and produces a per-window
    heart-rate prediction. The concrete network
    layers are not implemented in this file and should be added to the
    constructor and the `forward` method.

    Input shape
    -----------
    The model expects input tensors of shape ``(batch_size, in_channels,
    sequence_length)`` for 1D convolutional processing.

    Example
    -------
    >>> #model = MultimodalHRNet(in_channels=4, sequence_length=512)
    >>> #x = torch.randn(8, 4, 512)
    >>> #out = model(x)  
    """
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()

        # --- ENHANCED CONV BLOCKS (Version 5) ---
        # Progressive channel increase with better feature extraction
        # Reduced kernel size from 9 to 7 for finer HR detail capture

        # Block 1: Input (4 channels) -> Output (32 channels)
        # Kernel size 7 at 64Hz = ~110ms window (better for HR patterns)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # Length: 512 -> 512 (padding) -> 256 (pool)
        )

        # Block 2: Input (32) -> Output (64)
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate + 0.05),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # Length: 256 -> 256 (padding) -> 128 (pool)
        )

        # Block 3: Input (64) -> Output (128)
        # Increased channels for better feature extraction
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate + 0.1),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # Length: 128 -> 128 (padding) -> 64 (pool)
        )

        # --- ADAPTIVE POOLING FOR CONSISTENT SHAPE ---
        # Ensures we always get (batch, 128, 64) before flattening
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)

        # --- THE IMPROVED PREDICTION HEAD (Version 5) ---
        # Progressive feature reduction instead of aggressive reduction
        # Flattening size calculation: 128 channels * 64 remaining time steps = 8192

        self.flatten = nn.Flatten()

        # Improved FC layers with progressive reduction
        # 8192 -> 2048 -> 1024 -> 512 -> 256 -> 1
        self.fc_layers = nn.Sequential(
            # FC Layer 1: 8192 -> 2048
            nn.Linear(in_features=8192, out_features=2048),
            nn.BatchNorm1d(num_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # FC Layer 2: 2048 -> 1024
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            # FC Layer 3: 1024 -> 512
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # FC Layer 4: 512 -> 256
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.15),

            # Output Layer: 256 -> 1 (HR prediction)
            nn.Linear(in_features=256, out_features=1)
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
            Output tensor containing heart-rate predictions or embeddings.
        """
        # Conv feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Ensure consistent shape regardless of input size variations
        x = self.adaptive_pool(x)

        # Flatten and pass through FC layers
        x = self.flatten(x)
        out = self.fc_layers(x)

        return out.squeeze()  # Return shape (batch_size,) for regression output
