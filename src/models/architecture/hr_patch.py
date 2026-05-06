"""Neural network models for heart-rate estimation.

This module provides PyTorch model definitions used for heart-rate
estimation from bvp and acc wearable sensors.
"""

import torch
from torch import nn


class PatchHRNet(nn.Module):
    """
    A Patch-Based Transformer Encoder for State-of-Art physiological time-series regression.

    This architecture abandons deep Convolutional and Recurrent layers in favor of a pure
    Sequence-to-Sequence Vision Transformer (ViT) approach adapted for 1D signals. It chops
    raw physiological waveforms into discrete "patches" to preserve high-frequency physical
    features (like the sharp morphological peaks of a heartbeat) that are typically blurred
    by standard deep CNNs.

    Args:
        in_channels (int): Number of input sensor modalities. Default is 4 (1x BVP, 3x ACC).
        seq_len (int): Total number of time steps in the input window.
                       Default is 512 (8 seconds at 64Hz).
        patch_size (int): The number of time steps per discrete patch.
                          Default is 16 (0.25 seconds).
                          Determines the resolution of the Transformer's tokenization.
        embed_dim (int): The latent dimension of the Transformer and Positional Encodings.
                         Default is 128.
        num_heads (int): Number of parallel attention heads in the Transformer blocks.
                         Default is 4.
        num_layers (int): The depth of the Transformer Encoder stack. Default is 4.
        dropout (float): Dropout probability for regularization. Default is 0.1.

    Inputs:
        x (Tensor): The raw continuous waveform tensor.
                    Expected shape: (Batch_Size, Channels, Sequence_Length).
                    Example: (32, 4, 512).

    Outputs:
        Tensor: A 1D tensor of the predicted Heart Rate (BPM).
                Shape: (Batch_Size,).

    Architecture Notes:
        - Tokenization: Utilizes a non-overlapping 1D Convolution (stride = kernel_size) to
          tokenize the continuous wave into 32 distinct patches.
        - Time Awareness: Injects learnable Positional Encodings, as Transformers are natively
          permutation-invariant and have no concept of sequential time.
        - Training Prerequisite: This model MUST be trained with a Learning Rate Warmup
          schedule. Standard static learning rates will cause immediate attention collapse.
    """
    def __init__(self,
                 in_channels=4,
                 seq_len=512,
                 patch_size=16,
                 embed_dim=128,
                 num_heads=4,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()

        # 1. The Patching Engine
        # 512 time steps / 16 (patch_size) = 32 patches.
        # Conv1d doesn't overlap. It strictly chops the sequence into 32 blocks of 0.25 seconds.
        self.patcher = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = seq_len // patch_size

        # 2. Positional Encoding
        # Transformers have no concept of time. Must inject the order of the patches.
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. The Pure Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. The Regression Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """Forward pass"""
        # x shape: (Batch, 4, 512)

        # 1. Create Patches: (Batch, 4, 512) -> (Batch, 128, 32)
        x = self.patcher(x)

        # 2. Tensor Gymnastics for Transformer: (Batch, 32, 128)
        x = x.permute(0, 2, 1)

        # 3. Add Positional Encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Transformer Magic (All 32 patches look at each other simultaneously)
        x = self.transformer(x)

        # 5. Global Average Pooling (Condense the 32 patches into one rich feature vector)
        x = x.mean(dim=1)
        x = self.norm(x)

        # 6. Predict HR
        out = self.head(x)
        return out.squeeze()


if __name__ == '__main__':
    model = PatchHRNet()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')
