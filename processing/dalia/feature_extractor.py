from itertools import pairwise
from warnings import deprecated

import pandas as pd

from processing.dalia.utils.csv_saver import save_csv
from processing.dalia.utils.params.configuration import ECG_SAMPLING_RATE, WINDOW_SIZE_SEC, STEP_SIZE_SEC
from processing.dalia.ecg.ecg_quality_measure import ECGQualityMeasure


class FeatureExtractor:
    """
        A class to extract features from ECG data, specifically to calculate RR intervals from R-peak indices.
    """
    def __init__(self, r_peaks_path, ecg_signal_path):
        self.ecg_quality_measure = ECGQualityMeasure(
            r_peaks_path=r_peaks_path,
            ecg_signal_path=ecg_signal_path
        )

    def calculate_rr_intervals(self, r_peaks_data: pd.DataFrame, sampling_rate:int):
        """
            Calculate the time interval between each pair of peaks in r_peaks_list

        Args:
            r_peaks_data: List of R-peak indices
            sampling_rate: Sampling rate of the ECG data
        """

        r_peaks_list = r_peaks_data['rpeaks'].tolist()

        couples = list(pairwise(r_peaks_list))

        rr_intervals = []

        for couple in couples:
            rr_intervals.append(couple[1] - couple[0])

        return rr_intervals

    def signal_quality_index_retrieval(self, output_path):
        """
            Breaks down the ECG signal into chunks of given time window and calculates the signal quality index (SQI)
            using the neurokit2 prebuilt function ecg_quality().

            Args:
                output_path: Directory where the signal quality index will be saved as a csv file

            Returns:
                A list of tuples, where each tuple contains the step, the corresponding signal quality index for each
                singular ecg value and the mean of the signal quality index for that chunk
        """
        return self.ecg_quality_measure.signal_quality_index_retrieval(output_path)

    @deprecated("No longer needed as SQI gives already this information, and the F1 score is not a good measure for this task")
    def calculate_peaks_f1_score(self):
        """
            Calculate the F1 score between detected peaks and ground truth peaks.
            Returns:
                F1 score
        """
        return self.ecg_quality_measure.calculate_peak_f1()

    @deprecated("The ground truths of the heart rate are already present under the labels.csv file for each patient")
    def calculate_bpm(self, patient):
        """
            Calculate the BPM of the ECG signal and save it in a csv file.
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

        save_csv("BPM", f"datasets/dalia/converted/{patient}/features", bpm_window)

