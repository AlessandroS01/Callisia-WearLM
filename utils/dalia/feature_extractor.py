from itertools import pairwise
import pandas as pd

class FeatureExtractor:
    """
        A class to extract features from ECG data, specifically to calculate RR intervals from R-peak indices.
    """
    def __init__(self):
        pass

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
