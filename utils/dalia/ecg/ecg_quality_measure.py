import neurokit2 as nk
import numpy as np
import pandas as pd

from utils.dalia.configuration import ECG_SAMPLING_RATE


class ECGQualityMeasure:
    def __init__(self, n_seconds=10, ecg_signal_path: str = None):
        """
            Constructor for the ECGQualityMeasure class that initializes the number of seconds for the time window

            Args:
                n_seconds: The number of seconds in which the ECG signal should be chunked. Defaults to 10 seconds
                ecg_signal_path: Path to the ECG signal to be processed
        """
        self.n_seconds = n_seconds
        self.ecg_signal = pd.read_csv(ecg_signal_path).iloc[:, 0]

    def signal_quality_index_retrieval(self):
        """
            Breaks down the ECG signal into chunks of given time window and calculates the signal quality index (SQI)
            using the neurokit2 prebuilt function ecg_quality()

            Returns:
                A list of tuples, where each tuple contains the step and the corresponding signal quality index.
        """
        signal_quality_index = []
        window_size = self.n_seconds * ECG_SAMPLING_RATE
        step_size = window_size // 5  # 20% overlap
        for step in range(0, len(self.ecg_signal) - window_size + 1, step_size):
            print("Processing step: ", step)
            ecg_signal_chunk = self.ecg_signal.iloc[step:step + window_size]

            cleaned_ecg_chunk = nk.ecg_clean(ecg_signal_chunk, sampling_rate=ECG_SAMPLING_RATE)
            quality = nk.ecg_quality(cleaned_ecg_chunk, sampling_rate=ECG_SAMPLING_RATE)
            mean = np.mean(quality)
            signal_quality_index.append((step, quality, mean))

        return signal_quality_index
