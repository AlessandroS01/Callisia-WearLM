import os
from itertools import pairwise
import pandas as pd

from utils.dalia.ecg.ecg_quality_measure import ECGQualityMeasure


class FeatureExtractor:
    """
        A class to extract features from ECG data, specifically to calculate RR intervals from R-peak indices.
    """
    def __init__(self, n_seconds, r_peaks_path, ecg_signal_path):
        self.ecg_quality_measure = ECGQualityMeasure(
            n_seconds=n_seconds,
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

    def save_csv(self, attribute: str, output_path: str, data):
        """
            Save the given data to a csv file

            Args:
                attribute: Attribute of the file to be saved
                output_path: Directory of the file
                data: Data to be saved
        """
        os.makedirs(output_path, exist_ok=True)
        pd.DataFrame(data).to_csv(
            os.path.join(output_path, f"{attribute}.csv"), index=False
        )

    def calculate_peaks_f1_score(self):
        """
            Calculate the F1 score between detected peaks and ground truth peaks.
            Returns:
                F1 score
        """
        self.ecg_quality_measure.calculate_peak_f1()

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
        self.ecg_quality_measure.signal_quality_index_retrieval(filename)
