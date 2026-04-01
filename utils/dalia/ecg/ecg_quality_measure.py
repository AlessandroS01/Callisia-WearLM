import neurokit2 as nk
import numpy as np
import pandas as pd

from utils.dalia.configuration import ECG_SAMPLING_RATE


class ECGQualityMeasure:
    def __init__(self, n_seconds=10, r_peaks_path=None, ecg_signal_path=None):
        """
            Constructor for the ECGQualityMeasure class that initializes the number of seconds for the time window

            Args:
                n_seconds: The number of seconds in which the ECG signal should be chunked. Defaults to 10 seconds
                r_peaks_path: Path to the R_peaks ground truth file
                ecg_signal_path: Path to the ECG signal to be processed
        """
        self.n_seconds = n_seconds
        self.ecg_signal = pd.read_csv(ecg_signal_path).iloc[:, 0]
        self.true_peaks = np.array(pd.read_csv(r_peaks_path))

    def calculate_peak_f1(self, tolerance=int(0.05 * ECG_SAMPLING_RATE)):
        """
            Calculates F1 score between detected and ground truth peaks.

            Args:
                  tolerance: Margin of error when comparing detected peaks with ground truth peaks. Default to 5% of the sampling rate
        """
        peaks_list, info = nk.ecg_peaks(self.clean_ecg(self.ecg_signal), sampling_rate=ECG_SAMPLING_RATE)
        detected_peaks = np.where(peaks_list["ECG_R_Peaks"] == 1)[0]

        if len(self.true_peaks) == 0 and len(detected_peaks) == 0:
            return 1.0  # Perfect agreement (no peaks expected, none found)
        if len(self.true_peaks) == 0 or len(detected_peaks) == 0:
            return 0.0  # Complete failure

        matched_ground_truth = set()
        true_positives = 0
        false_positives = []

        for detected_peak in detected_peaks:
            # Calculate the distance from THIS detected peak to ALL true peaks simultaneously
            distances = np.abs(self.true_peaks - detected_peak)

            # 2. Find the index of the closest true peak
            closest_idx = np.argmin(distances)
            closest_true_peak_value = self.true_peaks[closest_idx].item()

            # 3. If the closest peak is within tolerance AND hasn't already been matched to another detected peak
            if distances[closest_idx] <= tolerance and closest_true_peak_value not in matched_ground_truth:
                true_positives += 1
                # Add the true peak value (not the index) to the set so it can't be claimed twice
                matched_ground_truth.add(closest_true_peak_value)
            else:
                false_positives.append(detected_peak)

        false_negatives = len(self.true_peaks) - true_positives

        if true_positives == 0:
            return 0.0

        f1_score = (2 * true_positives) / (2 * true_positives + len(false_positives) + false_negatives)
        return f1_score

    def signal_quality_index_retrieval(self, filename):
        """
            Breaks down the ECG signal into chunks of given time window and calculates the signal quality index (SQI)
            using the neurokit2 prebuilt function ecg_quality().

            Args:
                filename: ECG filename to create

            Returns:
                A list of tuples, where each tuple contains the step, the corresponding signal quality index for each
                singular ecg value and the mean of the signal quality index for that chunk
        """
        signal_quality_index = []
        window_size = self.n_seconds * ECG_SAMPLING_RATE
        step_size = window_size // 3

        for step in range(0, len(self.ecg_signal) - window_size + 1, step_size):
            print("Processing step: ", step)
            ecg_signal_chunk = self.ecg_signal.iloc[step:step + window_size]
            cleaned_ecg_chunk = self.clean_ecg(ecg_signal_chunk)

            # quality of each single point
            quality = nk.ecg_quality(cleaned_ecg_chunk, sampling_rate=ECG_SAMPLING_RATE)
            mean = np.mean(quality)

            signal_quality_index.append((step, mean))


        df = pd.DataFrame(signal_quality_index, columns= ['step', 'mean_quality_nk'])
        df.to_parquet(filename, index=False)

        return signal_quality_index

    def clean_ecg(self, ecg_signal):
        return nk.ecg_clean(ecg_signal, sampling_rate=ECG_SAMPLING_RATE)