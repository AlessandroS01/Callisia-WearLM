"""Feature extraction module for the WESAD ECG dataset.

This module provides `WESADFeatureExtractor` which takes ECG and gives back
features like instantaneous heart rate (BPM) and heart rate variability (HRV) metrics.
"""
import os
from itertools import pairwise

import numpy as np
import pandas as pd

from src.features.base_feature_extractor import BaseFeatureExtractor
from src.utils.csv_saver import save_csv


class WESADFeatureExtractor(BaseFeatureExtractor):
    """Extract features from ECG recordings.

    Currently, supports RR-interval calculation and reuses
    `ECGQualityMeasure` for SQI computation.
    """
    def __init__(self, patient):
        """Initialize WESAD feature extractor."""
        super().__init__()
        self.folder = f"../../../data/processed/wesad/{patient}/"

    def _get_rpeaks(self):
        """Get R-peak indices by processing the ECG signal.

        Returns:
            Array of R-peak indices
        """
        ecg_path = os.path.join(self.folder, "chest/chest_ECG.csv")
        ecg_signal = pd.read_csv(ecg_path)['ECG'].values

        # Clean and process the entire ECG signal
        cleaned_ecg = self._clean_ecg_signal(ecg_signal)
        signals, info = self._process_ecg_signal(cleaned_ecg)

        # Extract R-peak indices from the processed signals
        return info['ECG_R_Peaks']

    def _get_output_dir(self):
        """Get the output directory for WESAD patient.

        Returns:
            str: Path to the output directory
        """
        return self.folder

    def calculate_rr_intervals(self):
        """Calculate RR intervals (in samples) from R-peak indices.

        Returns:
            List[int]: differences between consecutive R-peak indices.
        """
        rpeaks = self._get_rpeaks()
        rpeaks_list = sorted(rpeaks)
        couples = list(pairwise(rpeaks_list))

        rr_intervals = [b - a for a, b in couples]
        return rr_intervals

    def calculate_hr(self):
        """Compute HR per window from neuropeaks2 and save the list as CSV
            for each patient.
        """

        hr_values = []
        ecg_path = os.path.join(self.folder, "chest/chest_ECG.csv")

        ecg_signal = pd.read_csv(
            ecg_path,
        )['ECG'].values

        window_size = self._get_window_size()
        step_size = self._get_step_size()

        for step in range(0, len(ecg_signal) - window_size + 1, step_size):
            print(f"Calculating HR for window {step}")
            ecg_signal_chunk = ecg_signal[step:step + window_size]
            cleaned_ecg_chunk = self._clean_ecg_signal(ecg_signal_chunk)

            signals, info = self._process_ecg_signal(cleaned_ecg_chunk)

            hr_list_chunk = signals['ECG_Rate'].values
            hr_values.append(np.mean(hr_list_chunk))

        csv_path = os.path.join(self.folder, "features")

        save_csv(
            "hr",
            csv_path,
            hr_values
        )
