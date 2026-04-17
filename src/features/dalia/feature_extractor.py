"""Feature extraction utilities for the Dalia ECG dataset.

This module provides `DaliaFeatureExtractor` which wraps ECG quality
calculations and RR-interval feature extraction.
"""
import os

import numpy as np
import pandas as pd

from src.features.base_feature_extractor import BaseFeatureExtractor
from src.utils.csv_saver import save_csv


class DaliaFeatureExtractor(BaseFeatureExtractor):
    """Extract features from ECG recordings.

    Currently, supports RR-interval calculation and reuses
    `ECGQualityMeasure` for SQI computation.
    """
    def __init__(self, patient):
        super().__init__()
        self.folder = f"../../../data/processed/dalia/{patient}/"

    def _get_rpeaks(self):
        """Get R-peak indices from the Dalia quality measure.

        Returns:
            Array of R-peak indices
        """
        r_peaks_path = os.path.join(self.folder, "rpeaks.csv")

        return np.array(pd.read_csv(r_peaks_path)).flatten()

    def _get_output_dir(self):
        """Get the output directory.

        Returns:
            str: Path to the output directory
        """
        return self.folder

    def signal_quality_index_retrieval(self):
        """Compute Signal Quality Index (SQI) per window from neuropeaks2
        and save the list as a CSV file for each patient."""

        ecg_path = os.path.join(self.folder, "chest/chest_ECG.csv")
        ecg_signal = pd.read_csv(
            ecg_path,
        )['ECG'].values

        sqi = []

        window_size = self._get_window_size()
        step_size = self._get_step_size()

        for step in range(0, len(ecg_signal) - window_size + 1, step_size):
            print("Processing step: ", step)
            ecg_signal_chunk = ecg_signal[step:step + window_size]
            cleaned_ecg_chunk = self._clean_ecg_signal(ecg_signal_chunk)

            sqi.append(self.calculate_signal_quality(cleaned_ecg_chunk))

        output_path = os.path.join(self.folder, "features")

        save_csv(
            attribute="signal_quality_index",
            output_path=output_path,
            data=sqi,
        )
