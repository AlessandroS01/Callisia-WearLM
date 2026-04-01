from utils.dalia.ecg.ecg_quality_measure import ECGQualityMeasure
from utils.dalia.feature_extractor import FeatureExtractor
from utils.dalia.dataset_handler import PPGDaliaDatasetHandler
from utils.plot_handler import PlotHandler

import pandas as pd


def main():
    """
    calculate_rr_intervals()"""
    path_ecg = "datasets/dalia/converted/S1/chest/chest_ECG.csv"
    r_peaks_path = "datasets/dalia/converted/S1/rpeaks.csv"

    compute_quality(path_ecg, r_peaks_path)

    #see_plot(path)

def compute_quality(ecg_path, r_peaks_path):
    measure = ECGQualityMeasure(10, ecg_signal_path=ecg_path, r_peaks_path=r_peaks_path)
    print(measure.calculate_peak_f1())
    #print(measure.signal_quality_index_retrieval("datasets/dalia/converted/S1/features/signal_quality.parquet"))


def open_pickle_dataset(path, patient):
    handler = PPGDaliaDatasetHandler(path)
    handler.extract_data(f'datasets/dalia/converted/{patient}')
    json_path = "datasets/dalia/converted/S1.json"
    handler.convert_pkl_json(json_path)

def see_plot(dataset):
    plot_handler = PlotHandler(dataset)
    plot_handler.create_ecg_plot()

def calculate_rr_intervals():
    extractor = FeatureExtractor()

    for i in range(1, 16):
        patient = f"S{i}"
        intervals = extractor.calculate_rr_intervals(pd.read_csv(f"datasets/dalia/converted/{patient}/rpeaks.csv"), 700)
        output_dir = f"datasets/dalia/converted/{patient}/features"
        extractor.save_csv("rr_intervals", output_dir, intervals)
        print(f"Patient: {patient} done")

if __name__ == '__main__':
    main()