"""
PyTorch Dataset for Dalia and WESAD multimodal heart rate regression.

Accepts pre-fused 3D tensors containing BVP and ACC channels and formats
them for 1D Convolutional Neural Networks by applying the necessary transpositions.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class HRDataset(Dataset):
    """
    PyTorch Dataset for Dalia and WESAD multimodal heart rate regression.

    This dataset handles the formatting of
    synchronized time-series windows from Blood Volume Pulse (BVP) and
    Accelerometer (ACC) sensors. It prepares the data for 1D Convolutional
    Neural Networks (CNNs) by applying the necessary matrix transpositions
    to match PyTorch's strict 1D CNN input shape requirements.

    Args:
        x_windows (np.ndarray): A 3D array of shape (N, 512, 4) containing [BVP, ACC] channels.
        y_windows (np.ndarray): A 1D array of shape (N, 1) containing the HR targets.

    Attributes:
        x_windows (np.ndarray): A 3D array containing [BVP, ACCx, ACCy, ACCz] signal windows.
        labels (np.ndarray): The stored heart rate regression targets.

    Example:
        >>> #dataset = DaliaHRDataset()
        >>> #dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> #features, labels = next(iter(dataloader))
        >>> #features.shape
        torch.Size([32, 4, 512])  # (Batch, Channels, Sequence_Length)
    """
    def __init__(self, x_windows, y_windows):
        """
        Initializes the DaliaHRDataset with pre-fused sensor windows and corresponding labels.

        Args:
            x_windows (np.ndarray): Numpy array containing BVP and ACC channels
            y_windows (np.ndarray): Numpy array containing the values of HR
        """

        self.x_windows = x_windows
        self.labels = y_windows

    def __len__(self):
        """
        Returns the total number of time-windows in the dataset.

        Returns:
            int: The total number of windows in the dataset.
        """
        return len(self.x_windows)

    def __getitem__(self, idx):
        """
        Retrieves a single fused sensor window and its corresponding label.

        Extracts the BVP and ACC windows at the given index, and transposes the dimensions from
        (Sequence_Length, Channels) to (Channels, Sequence_Length).

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): The fused feature tensor of shape (4, sequence_length).
                - y (torch.Tensor): The scalar heart rate label of shape ().
        """

        x_sequence = self.x_windows[idx]
        y_value = self.labels[idx]

        x_sequence = np.transpose(x_sequence)

        x = torch.tensor(x_sequence, dtype=torch.float32)
        y = torch.tensor(y_value, dtype=torch.float32)

        return x, y
