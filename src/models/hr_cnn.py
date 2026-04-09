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
    def __init__(self):
        super().__init__()

        # --- FEATURE EXTRACTION BLOCKS (The Convolutional Funnel) ---

        # Block 1: Input (4 channels) -> Output (16 channels)
        # Kernel set to 7 to catch those fast heartbeat slopes
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # Length drops from 512 to 256
        )

        # Block 2: Input (16) -> Output (32)
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # Length drops from 256 to 128
        )

        # Block 3: Input (32) -> Output (64)
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # Length drops from 128 to 64
        )

        # --- THE PREDICTION HEAD (From the Paper) ---

        # Flattening size calculation: 64 channels * 64 remaining time steps = 4096
        self.flatten = nn.Flatten()

        # Fully Connected Layer (n_fc1 in the paper)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=128),
            nn.ReLU(),
        )

        # Dropout (Exactly as paper specified: 0.5)
        self.dropout = nn.Dropout(p=0.5)

        # Final Fully Connected Layer (n_fc2 = 1 neuron in the paper)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.flatten(x)

        x = self.fc1(x)

        x = self.dropout(x)

        out = self.fc2(x)

        return out.squeeze()  # Return shape (batch_size,) for regression output
