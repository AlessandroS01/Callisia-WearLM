"""Feature extraction module for the WESAD ECG dataset.

This module provides `WESADFeatureExtractor` which takes ECG and gives back
features like instantaneous heart rate (BPM) and heart rate variability (HRV) metrics.
"""
import os
import pandas as pd
import numpy as np

import neurokit2 as nk

from src.utils.csv_saver import save_csv
from src.utils.dalia_wesad_config import (
    ECG_SAMPLING_RATE,
    WINDOW_SIZE_SEC,
    STEP_SIZE_SEC,
)


class WESADFeatureExtractor:
    """Extract features from ECG recordings.

    Currently, supports RR-interval calculation and reuses
    `ECGQualityMeasure` for SQI computation.
    """
    def __init__(self, patient):
        """Initialize WESAD feature extractor."""
        self.folder = f"../../../data/processed/wesad/{patient}/"


    def calculate_hr(self):
        """Compute HR per window from neuropeaks2 and save the list as CSV
            for each patient.
        """

        hr_values = []
        ecg_path = os.path.join(self.folder, "chest/chest_ECG.csv")

        ecg_signal = pd.read_csv(
            ecg_path,
        )['ECG'].values

        window_size = WINDOW_SIZE_SEC * ECG_SAMPLING_RATE
        step_size = STEP_SIZE_SEC * ECG_SAMPLING_RATE

        for step in range(0, len(ecg_signal) - window_size + 1, step_size):
            print(f"Calculating HR for window {step}")
            ecg_signal_chunk = ecg_signal[step:step + window_size]
            cleaned_ecg_chunk = nk.ecg_clean(ecg_signal_chunk, sampling_rate=ECG_SAMPLING_RATE)

            signals, info = nk.ecg_process(cleaned_ecg_chunk, sampling_rate=ECG_SAMPLING_RATE)

            hr_list_chunk = signals['ECG_Rate'].values
            hr_values.append(np.mean(hr_list_chunk))

        csv_path = os.path.join(self.folder, "features")

        save_csv(
            "hr",
            csv_path,
            hr_values
        )

def main():
    for i in range(2, 11):
        patient = f"S{i}"
        print(f"Processing patient {patient}")
        extractor = WESADFeatureExtractor(patient)

        extractor.calculate_hr()
if __name__ == "__main__":
    main()