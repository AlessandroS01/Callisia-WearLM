"""
    Main entry point for Dalia dataset processing.
"""

import pandas as pd
from matplotlib import pyplot as plt

from processing.dalia.feature.feature_extractor import FeatureExtractor
from processing.dalia.processor import DaliaProcessor


def main():
    """
        Execute the main processing pipeline.
        Currently, runs window standardization for a specific patient.
    """
    #extractor()
    processor = DaliaProcessor("datasets/dalia/standardized/S1")
    processor.get_standardized_windows()



def make_histogram_quality_index():
    """
        Aggregate signal quality indices across all patients
        and plot the distribution as a histogram.
    """
    quality_scores = []

    for i in range(1, 16):
        patient = "S" + str(i)
        file_path = f"datasets/dalia/standardized/{patient}/features/signal_quality_index.csv"

        # Read the CSV
        df = pd.read_csv(file_path)

        # Extract the raw numbers and flatten them into a 1D array
        # If your CSV has a specific column name like 'SQI', use df['SQI'].values instead
        patient_scores = df.values.flatten()

        # Extend adds these numbers to our master pool (rather than appending a whole DataFrame)
        quality_scores.extend(patient_scores)
    plt.hist(quality_scores, bins=50)
    plt.title("Distribution of ECG Quality Scores")
    plt.xlabel("NeuroKit SQI")
    plt.show()

def extractor():
    """
        Iterate through patients S1 to S15, initialize the feature extractor,
        and retrieve/save the signal quality index for each.
    """
    for i in range(1, 16):
        patient = "S" + str(i)
        r_peaks_path = f"datasets/dalia/standardized/{patient}/rpeaks.csv"
        ecg_path = f"datasets/dalia/standardized/{patient}/chest/chest_ECG.csv"
        fe = FeatureExtractor(r_peaks_path=r_peaks_path, ecg_signal_path=ecg_path)
        print("Doing patient " + str(i))
        fe.signal_quality_index_retrieval(
            output_path=f"datasets/dalia/standardized/{patient}/features"
        )


if __name__ == '__main__':
    main()
