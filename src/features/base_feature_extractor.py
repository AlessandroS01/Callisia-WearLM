"""Base feature extraction class for ECG datasets.

This module provides `BaseFeatureExtractor` which defines common functionality
for feature extraction across different ECG datasets.
"""
import os
from abc import ABC, abstractmethod

import pandas as pd
import neurokit2 as nk

from src.utils.dalia_wesad_config import (
    ECG_SAMPLING_RATE,
    WINDOW_SIZE_SEC,
    STEP_SIZE_SEC,
)


class BaseFeatureExtractor(ABC):
    """Abstract base class for ECG feature extraction.

    Provides common functionality and interface for extracting features
    from ECG recordings across different datasets.
    """

    # Constants shared across all extractors
    ECG_SAMPLING_RATE = ECG_SAMPLING_RATE
    WINDOW_SIZE_SEC = WINDOW_SIZE_SEC
    STEP_SIZE_SEC = STEP_SIZE_SEC

    def __init__(self):
        """Initialize base feature extractor."""

    def _get_window_size(self):
        """Get window size in samples.

        Returns:
            int: Window size in samples
        """
        return int(self.WINDOW_SIZE_SEC * self.ECG_SAMPLING_RATE)

    def _get_step_size(self):
        """Get step size in samples.

        Returns:
            int: Step size in samples
        """
        return int(self.STEP_SIZE_SEC * self.ECG_SAMPLING_RATE)

    def _clean_ecg_signal(self, ecg_signal):
        """Clean ECG signal using neurokit2.

        Args:
            ecg_signal: Raw ECG signal array

        Returns:
            Cleaned ECG signal array
        """
        return nk.ecg_clean(
            ecg_signal,
            sampling_rate=self.ECG_SAMPLING_RATE
        )

    def _process_ecg_signal(self, ecg_signal):
        """Process ECG signal using neurokit2.

        Args:
            ecg_signal: ECG signal array (typically cleaned)

        Returns:
            Tuple of (signals DataFrame, info dict) from nk.ecg_process
        """
        return nk.ecg_process(
            ecg_signal,
            sampling_rate=self.ECG_SAMPLING_RATE
        )

    @abstractmethod
    def _get_rpeaks(self):
        """Get R-peak indices for the dataset.

        Returns:
            Array of R-peak indices

        This method must be implemented by subclasses to provide
        R-peaks either from pre-computed files or by detecting them
        from the ECG signal.
        """

    @abstractmethod
    def _get_output_dir(self):
        """Get the output directory for saving features.

        Returns:
            str: Path to the features output directory
        """

    def calculate_hrv_intervals(self):
        """Calculate HRV intervals (time and frequency domain) and save to CSV.

        Processes R-peaks to compute heart rate variability metrics
        using neurokit2 (both time and frequency domains).

        This method uses _get_rpeaks() which must be implemented by subclasses.
        """
        # Get R-peaks from subclass implementation
        rpeaks = self._get_rpeaks()

        # Calculate HRV in time and frequency domains
        hrv_time = nk.hrv_time(rpeaks, sampling_rate=self.ECG_SAMPLING_RATE)
        hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=self.ECG_SAMPLING_RATE)

        # Combine into final DataFrame
        hrv = pd.concat([hrv_time, hrv_freq], axis=1)

        # Define output directory and save
        output_path = os.path.join(self._get_output_dir(), "features")
        os.makedirs(output_path, exist_ok=True)
        hrv.to_csv(os.path.join(output_path, "hrv.csv"), index=False)

        print(f"HRV metrics saved to {os.path.join(output_path, 'hrv.csv')}")
