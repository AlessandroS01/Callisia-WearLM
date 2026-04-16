"""Feature extraction utilities for the Dalia ECG dataset.

This module provides `DaliaFeatureExtractor` which wraps ECG quality
calculations and RR-interval feature extraction.
"""
import os
from itertools import pairwise

from typing_extensions import deprecated

import pandas as pd

import neurokit2 as nk

from src.features.dalia_ecg_sqi import DaliaECGQualityMeasure
from src.utils.csv_saver import save_csv
from src.utils.dalia_config import (
    ECG_SAMPLING_RATE,
    WINDOW_SIZE_SEC,
    STEP_SIZE_SEC,
)


class DaliaFeatureExtractor:
    """Extract features from ECG recordings.

    Currently, supports RR-interval calculation and reuses
    `ECGQualityMeasure` for SQI computation.
    """
    def __init__(self, r_peaks_path, ecg_signal_path):
        self.ecg_quality_measure = DaliaECGQualityMeasure(
            r_peaks_path=r_peaks_path,
            ecg_signal_path=ecg_signal_path
        )

    def calculate_rr_intervals(self, r_peaks_data: pd.DataFrame):
        """Return RR intervals (in samples) from R-peak indices.

        Args:
            r_peaks_data: DataFrame with a 'rpeaks' column of peak indices

        Returns:
            List[int]: differences between consecutive R-peak indices.
        """

        r_peaks_list = r_peaks_data["rpeaks"].tolist()
        couples = list(pairwise(r_peaks_list))

        rr_intervals = [b - a for a, b in couples]
        return rr_intervals

    def calculate_hrv_intervals(self, patient):
        """
        Return HRV intervals (in samples) from R-peak indices.

        Args:
            patient: patient identifier used to build output path.
            """

        peaks = self.ecg_quality_measure.true_peaks.flatten()
        # Calculate only Time and Frequency domains (runs in seconds)
        hrv_time = nk.hrv_time(peaks, sampling_rate=ECG_SAMPLING_RATE)
        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=ECG_SAMPLING_RATE)

        # Combine into your final DataFrame
        hrv = pd.concat([hrv_time, hrv_freq], axis=1)

        # Define your directory
        output_path = f"../../data/processed/dalia/{patient}/features"
        # Make sure the directory exists
        os.makedirs(output_path, exist_ok=True)
        # Save the DataFrame directly to CSV
        hrv.to_csv(os.path.join(output_path, "hrv.csv"), index=False)

    def signal_quality_index_retrieval(self, output_path):
        """Delegate SQI computation to `ECGQualityMeasure`.

        Args:
            output_path: directory where SQI CSV will be saved

        Returns:
            List of mean SQI values per window.
        """

        return self.ecg_quality_measure.signal_quality_index_retrieval(
            output_path
        )

    @deprecated("Deprecated: use SQI instead; peak F1 is not recommended")
    def calculate_peaks_f1_score(self):
        """Deprecated wrapper for peak F1 calculation.

        Returns:
            float: F1 score computed by `ECGQualityMeasure`.
        """

        return self.ecg_quality_measure.calculate_peak_f1()

    @deprecated("BPM ground truths already exist in labels.csv; this is deprecated")
    def calculate_bpm(self, patient):
        """Compute BPM per window from true R peaks and save as CSV.

        Args:
            patient: patient identifier used to build output path.
        """

        bpm_window = []

        peaks = self.ecg_quality_measure.true_peaks
        window_size = ECG_SAMPLING_RATE * WINDOW_SIZE_SEC
        step_size = STEP_SIZE_SEC

        max_index = peaks[-1][0]

        for step in range(0, max_index - window_size + 1, step_size):
            max_value = step + window_size
            # Count peaks that fall within the current window
            peaks_in_window = len([peak for peak in peaks if step <= peak <= max_value])

            # Calculate BPM for the current window
            bpm = peaks_in_window * (60 / WINDOW_SIZE_SEC)
            bpm_window.append(bpm)

            print(f"For step {step} the BPM is: {bpm}")

        save_csv(
            "BPM",
            f"datasets/dalia/{patient}/features",
            bpm_window
        )