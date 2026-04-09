import torch
import torch.nn as nn


class MultimodalHRNet(nn.Module):
    """Multi-modal 1D CNN for heart-rate estimation from wearable sensors.

    This class defines a skeleton PyTorch module that fuses multiple sensor
    channels (for example BVP and tri-axial ACC) and produces a per-window
    heart-rate prediction or downstream embedding. The concrete network
    layers are not implemented in this file and should be added to the
    constructor and the `forward` method.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 4). Typical channels are
        [BVP, ACC_X, ACC_Y, ACC_Z].
    window_length : int
        Length of the input window in samples (default 512).

    Input shape
    -----------
    The model expects input tensors of shape ``(batch_size, in_channels,
    window_length)`` for 1D convolutional processing.

    Example
    -------
    >>> model = MultimodalHRNet(in_channels=4, window_length=512)
    >>> x = torch.randn(8, 4, 512)
    >>> out = model(x)  # implement forward to return predictions
    """
    def __init__(self, in_channels = 4, window_length = 512):
        super().__init__()
        self.window_length = window_length
        self.in_channels = in_channels
