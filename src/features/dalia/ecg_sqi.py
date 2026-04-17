"""ECG quality utilities.

This module provides the `DaliaECGQualityMeasure` class which computes signal
quality indices (SQI) for ECG data and contains a deprecated helper for
computing peak-based F1 scores. It uses NeuroKit2 for ECG processing.
"""



import neurokit2 as nk
import numpy as np
import pandas as pd
from typing_extensions import deprecated

from src.utils.csv_saver import save_csv
from src.utils.dalia_wesad_config import (
    ECG_SAMPLING_RATE,
    WINDOW_SIZE_SEC,
    STEP_SIZE_SEC,
)


class DaliaECGQualityMeasure:
    """Compute ECG signal quality and related utilities.

    This class provides methods to compute signal quality indices (SQI) on
    ECG recordings and includes a deprecated helper to compute peak-based
    F1 scores for diagnostic purposes.
    """
    def __init__(self, r_peaks_path=None, ecg_signal_path=None):
        """
        Constructor for the ECGQualityMeasure class.

        Initializes the time window parameters and loads the ECG signal and
        ground-truth R peaks from CSV files.

        Args:
            r_peaks_path: Path to the R_peaks ground truth file
            ecg_signal_path: Path to the ECG signal to be processed
        """
        self.ecg_signal = pd.read_csv(ecg_signal_path).iloc[:, 0]
        self.true_peaks = np.array(pd.read_csv(r_peaks_path))

    def signal_quality_index_retrieval(self, output_path):
        """
        Break the ECG signal into fixed-size chunks and compute SQI for each.

        Uses neurokit2.ecg_quality() on cleaned chunks.

        Args:
            output_path: Directory where the SQI values will be saved as a CSV file.

        Returns:
            A list with the mean SQI value for each processed chunk.
        """
        signal_quality_index = []
        window_size = WINDOW_SIZE_SEC * ECG_SAMPLING_RATE
        step_size = STEP_SIZE_SEC * ECG_SAMPLING_RATE

        for step in range(0, len(self.ecg_signal) - window_size + 1, step_size):
            print("Processing step: ", step)
            ecg_signal_chunk = self.ecg_signal.iloc[step:step + window_size]
            cleaned_ecg_chunk = self.clean_ecg(ecg_signal_chunk)

            # quality of each single point
            quality = nk.ecg_quality(cleaned_ecg_chunk, sampling_rate=ECG_SAMPLING_RATE)
            mean = np.mean(quality)

            signal_quality_index.append(mean)

        save_csv(
            attribute="signal_quality_index",
            output_path=output_path,
            data=signal_quality_index,
        )
        return signal_quality_index

    def clean_ecg(self, ecg_signal):
        """
        Clean the ECG signal using neurokit2.ecg_clean().

        Args:
            ecg_signal: ECG signal to be cleaned

        Returns:
            The cleaned ECG signal
        """
        return nk.ecg_clean(ecg_signal, sampling_rate=ECG_SAMPLING_RATE)

    @deprecated("Deprecated: use SQI instead; F1 is not recommended")
    def calculate_peak_f1(self, tolerance=int(0.05 * ECG_SAMPLING_RATE)):
        """
        Calculates the F1 score between detected and ground-truth peaks.

        Args:
            tolerance: Margin of error (in samples) when matching detected peaks to
                ground-truth peaks. Defaults to 5% of the sampling rate.
        """
        peaks_list, _ = nk.ecg_peaks(
            self.clean_ecg(self.ecg_signal),
            sampling_rate=ECG_SAMPLING_RATE
        )
        detected_peaks = np.where(peaks_list["ECG_R_Peaks"] == 1)[0]

        if len(self.true_peaks) == 0 and len(detected_peaks) == 0:
            return 1.0  # Perfect agreement (no peaks expected, none found)
        if len(self.true_peaks) == 0 or len(detected_peaks) == 0:
            return 0.0  # Complete failure

        matched_ground_truth = set()
        true_positives = 0
        false_positives = []

        for detected_peak in detected_peaks:
            # Distance from this detected peak to all true peaks
            distances = np.abs(self.true_peaks - detected_peak)

            # Find the closest true peak and its value
            closest_idx = np.argmin(distances)
            closest_true_peak_value = self.true_peaks[closest_idx].item()

            within_tol = distances[closest_idx] <= tolerance
            not_already_matched = closest_true_peak_value not in matched_ground_truth

            if within_tol and not_already_matched:
                true_positives += 1
                # Prevent the same true peak from being matched twice
                matched_ground_truth.add(closest_true_peak_value)
            else:
                false_positives.append(detected_peak)

        false_negatives = len(self.true_peaks) - true_positives

        if true_positives == 0:
            return 0.0

        f1_score =((2 * true_positives) /
                    ((2 * true_positives) + len(false_positives) + false_negatives))
        return f1_score
