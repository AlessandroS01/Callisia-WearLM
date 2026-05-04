"""Neural network models for heart-rate estimation.

This module provides PyTorch model definitions used for heart-rate
estimation from bvp and acc wearable sensors.
"""

from torch import nn

class TemporalAttentionBlock(nn.Module):
    def __init__(self, feature_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        
        # The core attention mechanism (batch_first=True matches our LSTM setup)
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Standard Transformer stabilization techniques
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x expected shape: (Batch, SeqLen, Features)
        
        # Self-Attention: Query, Key, and Value are all the same input sequence
        # We only care about the output tensor, we can discard the raw weights with `_` for now
        attn_out, _ = self.attention(query=x, key=x, value=x)
        
        # Residual Connection & Layer Normalization (Crucial to prevent vanishing gradients)
        x = self.norm(x + self.dropout(attn_out))
        
        return x

class ChannelAttention(nn.Module):
    """Lightweight channel attention module (Squeeze-and-Excitation).

    Enables the model to adaptively recalibrate channel-wise feature responses.
    Adds minimal parameters while improving feature representation.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Apply channel attention: (batch, channels, length) -> (batch, channels, length)"""
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ConvBlock(nn.Module):
    """Residual conv block with channel attention.

    Combines convolution, batch norm, activation, and channel attention.
    Includes optional residual connection for improved gradient flow.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 use_residual: bool = False, dropout_rate: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.use_residual = use_residual and (in_channels == out_channels)

        # Sequential layers: Conv -> BN -> ReLU -> Attention -> Dropout
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            ChannelAttention(out_channels),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        """Forward pass with optional residual connection."""
        out = self.layers(x)
        if self.use_residual:
            out = out + x
        return out


class MultimodalHRNet(nn.Module):
    """Multi-modal 1D CNN for heart-rate estimation from wearable sensors.

    IMPROVED ARCHITECTURE (Version 7):
    - ~113K parameters
    - Channel attention for adaptive feature weighting
    - Residual connections for improved gradient flow
    - Better channel capacity progression
    - 4 conv blocks → 32 → 64 → 128 → 128 channels
    - GlobalAvgPool + compact FC head
    - Optimized for ~2500 training samples per LOSO fold

    Key improvements without parameter explosion:
    1. Channel Attention: Significantly improves feature selection
    2. Residual Connections: Zero additional parameters, improves training dynamics
    3. Better Channel Scaling: More features at deeper layers where patterns emerge
    4. 4 Conv Blocks: Deeper feature extraction with manageable parameter count

    Input shape
    -----------
    (batch_size, 4, 512) for BVP + tri-axial ACC at 64Hz over 8 seconds

    Example
    -------
    >>> # model = MultimodalHRNet()
    >>> # x = torch.randn(8, 4, 512)
    >>> # out = model(x)
    """
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()

        # --- IMPROVED CONV BLOCKS (Version 7) ---
        # 4 blocks with channel attention and residual connections
        # Progressive channel increase: 4 → 32 → 64 → 128 → 128

        # Block 1: (4, 512) -> (32, 256)
        # Kernel=7 at 64Hz ≈ 110ms = captures HR rhythm dynamics
        self.block1 = nn.Sequential(
            ConvBlock(
                4, 32,
                kernel_size=7, use_residual=False, dropout_rate=dropout_rate
            ),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Block 2: (32, 256) -> (64, 128)
        # Kernel=5 captures intermediate frequency features
        self.block2 = nn.Sequential(
            ConvBlock(
                32, 64,
                kernel_size=5, use_residual=False, dropout_rate=dropout_rate
            ),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Block 3: (64, 128) -> (128, 64)
        # Kernel=3 for fine-grained pattern extraction
        self.block3 = nn.Sequential(
            ConvBlock(
                64, 128,
                kernel_size=3, use_residual=False, dropout_rate=dropout_rate
            ),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Block 4: (128, 64) -> (128, 64)
        # Additional depth with residual connection (same channels)
        self.block4 = ConvBlock(
            128, 128,
            kernel_size=3, use_residual=True, dropout_rate=dropout_rate)

        # --- TEMPORAL ATTENTION LAYER ---
        self.temporal_attention = TemporalAttentionBlock(feature_dim=128, num_heads=4)

        # --- RECURRENT MEMORY BLOCK ---
        # Feed these 64 steps into a Bidirectional LSTM to smooth out high peaks.
        # input_size=128 (from block4 channels), hidden_size=64.
        # Because it's bidirectional (looks forward and backward in time),
        # the output will be 64 * 2 = 128 features.
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=64,
            num_layers=1, batch_first=True, bidirectional=True
        )

        # --- GLOBAL AVERAGE POOLING ---
        # Naturally reduces (batch, 128, 64) -> (batch, 128, 1) -> (batch, 128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # --- COMPACT PREDICTION HEAD ---
        # 128 -> 64 -> 1 (keeps parameters manageable)
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, 4, 512) - 4 channels, 512 time steps

        Returns
        -------
        torch.Tensor
            Shape (batch_size,) - HR predictions in bpm
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # LSTM
        # Permute from (Batch, Channels, SeqLen) to (Batch, SeqLen, Channels)
        # (Batch, 128, 64) -> (Batch, 64, 128)
        x = x.permute(0, 2, 1)
        
        x = self.temporal_attention(x)
        # Apply Recurrent Memory
        # LSTM returns the output sequence and the hidden states (which we discard with `_`)
        # Output shape: (Batch, 64, 128)
        x, _ = self.lstm(x)

        # Permute back to (Batch, Channels, SeqLen): (Batch, 128, 64)
        x = x.permute(0, 2, 1)

        x = self.global_pool(x)
        x = self.flatten(x)
        out = self.fc_layers(x)

        return out.squeeze()


if __name__ == '__main__':
    model = MultimodalHRNet()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')
