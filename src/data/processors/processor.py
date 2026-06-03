"""Processing utilities for the DaLiA and WESAD datasets.

This module provides `WESADDaliaProcessor`, which standardizes wearable sensor
signals (BVP and ACC), applies sliding windows, and filters windows using
precomputed ECG signal-quality indices to produce ML-ready tensors.
"""

import os

import numpy as np
import pandas as pd

from src.utils.dalia_wesad_config import (
    BVP_SAMPLING_RATE,
    WINDOW_SIZE_SEC,
    STEP_SIZE_SEC,
)


class WESADDaliaProcessor:
    """
    Orchestrates the preprocessing of the PPG-DaLiA and WESAD datasets for 1D-CNN ingestion.

    This processor handles the temporal synchronization of multi-rate wearable sensors,
    applies sliding window extraction, and filters corrupted segments using a
    pre-computed ECG signal quality index. It transforms raw subject directories
    into processed ML-ready tensors.

    Attributes:
        subject_dir (str): Path to the base directory of a single DaLiA or WESAD subject.
        window_size (int): The size of the window for BVP and upsampled ACC data in samples.
        step_size (int): The step size for sliding windows in BVP and upsampled ACC data in samples.
        raw_x (pd.DataFrame): DataFrame containing BVP and ACC values. Shape: (N, 4)
        labels (pd.DataFrame): DataFrame containing label values. Shape: (N, 1)
        quality_mask (pd.DataFrame): DataFrame containing quality mask values. Shape: (N, 1)
        sqi_threshold (float): Threshold to determine if a specific time window is to be taken
    """

    def __init__(self, subject_dir: str, sqi_threshold: float = 0.45):
        """
        Initializes the DaliaProcessor with specific windowing parameters.
        """
        self.subject_dir = subject_dir

        self.window_size = WINDOW_SIZE_SEC * BVP_SAMPLING_RATE
        self.step_size = STEP_SIZE_SEC * BVP_SAMPLING_RATE

        bvp_data = self._retrieve_data("wrist/BVP.csv")
        acc_data = self._retrieve_data("wrist/ACC.csv")

        # Normalize each signal independently (z-score normalization)
        bvp_data = self._normalize_signal(bvp_data)
        acc_data = self._normalize_signal(acc_data)

        self.raw_x = self._align_and_upsample_acc(bvp_data, acc_data)

        self.labels = self._retrieve_data("label.csv")

        self.quality_mask = self._retrieve_data("features/signal_quality_index.csv")

        self.sqi_threshold = sqi_threshold

    def _align_and_upsample_acc(self, bvp_df: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Synchronizes the 32Hz Accelerometer data to match the 64Hz BVP data.

        Uses interpolation to upsample ACC data so that all sensor channels
        share the exact same temporal grid.

        Args:
            bvp_df (pd.DataFrame): Raw BVP data. Shape: (N, 1)
            acc_df (pd.DataFrame): Raw ACC data. Shape: (M, 3)

        Returns:
            pd.DataFrame: A fused dataframe containing [BVP, ACC_X, ACC_Y, ACC_Z]
                          sampled uniformly at 64Hz.
        """
        target_length = len(bvp_df)
        original_length = len(acc_df)

        # Create the time axes for interpolation
        x_original = np.linspace(0, 1, original_length)
        x_target = np.linspace(0, 1, target_length)

        # Crate a new array of (N, 3)
        upsampled_acc = np.zeros((target_length, 3))
        for col_idx in range(3):
            # np.interp does simple, straight-line guessing between points
            upsampled_acc[:, col_idx] = np.interp(
                x_target,
                x_original,
                acc_df.iloc[:, col_idx].values
            )

        # Convert back to DataFrame
        acc_upsampled_df = pd.DataFrame(
            upsampled_acc,
            columns=['ACC_X', 'ACC_Y', 'ACC_Z'],
            index=bvp_df.index
        )

        fused_df = pd.concat([bvp_df, acc_upsampled_df], axis=1)

        return fused_df

    def _apply_quality_mask(self, window_idx: int) -> bool:
        """
        Checks the signal_quality.parquet file to determine if a specific time window
        contains reliable ECG ground truth.

        Args:
            window_idx (int): The index of the window.

        Returns:
            bool: True if the window passes the quality threshold, False if it
                  should be discarded due to high motion artifact or lost ECG peaks.
        """
        try:
            current_sqi = self.quality_mask.iloc[window_idx].values[0]
            return current_sqi >= self.sqi_threshold

        except IndexError:
            # Safety catch: If the quality file is shorter than the label file
            print(f"Warning: No quality score found for window {window_idx}.")
            return False

    def get_standardized_windows(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts sliding windows of fused sensor data and their corresponding labels.

        Iterates through the subject's timeline, extracting data blocks defined by
        `window_size_sec`. It discards any windows that fail
        the `_apply_quality_mask` check.

        Returns:
            tuple: A tuple containing:

                - X (np.ndarray): The 3D feature tensor.

                  Shape: (num_valid_windows, sequence_length, num_channels)

                  Num channels must be 4 and represents [BVP, ACCx, ACCy, ACCz].
                - y (np.ndarray): The 1D label array for the signal quality index (SQI).
                  Shape: (num_valid_windows)
        """
        x_valid = []
        y_valid = []

        for window_idx in range(len(self.labels)):
            # Check quality for this window before extracting data
            if self._apply_quality_mask(window_idx):
                # Calculate the starting sample for this specific window
                step = window_idx * self.step_size

                # Extract the data
                x_window = self.raw_x.iloc[step: step + self.window_size]
                label = self.labels.iloc[window_idx]

                # Append to valid lists
                x_valid.append(x_window.values)
                y_valid.append(label.values)

        # Return as [BVP_array, ACC_array], Y_array for the later 1D CNN
        return np.array(x_valid), np.array(y_valid)


    def _retrieve_data(self, typology:str) -> pd.DataFrame:
        path = os.path.join(self.subject_dir, typology)
        return pd.read_csv(path)

    def process(self) -> tuple[np.ndarray, np.ndarray]:
        """Public convenience method that runs the full processing pipeline.

        This is a thin wrapper around `get_standardized_windows` kept for a
        clearer public API and to satisfy linters that expect more than one
        public method on processor classes.
        """

        return self.get_standardized_windows()

    def _normalize_signal(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the input signal DataFrame using z-score normalization.

        Applies per-column normalization: (x - mean) / std for each column independently.
        For multi-channel signals (e.g., ACC with X, Y, Z axes), each channel is normalized
        with its own statistics.

        Args:
            signal_df (pd.DataFrame): The input signal data.
            signal_name (str): A string indicating the type of signal (e.g., "BVP" or "ACC").

        Returns:
            pd.DataFrame: The normalized signal data with mean≈0 and std≈1 per channel.
        """
        # Calculate per-column mean and std
        means = signal_df.mean()
        stds = signal_df.std()

        # Apply z-score normalization: (x - mean) / std (broadcasted per column)
        normalized_df = (signal_df - means) / stds

        return pd.DataFrame(normalized_df)
