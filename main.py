from utils.dalia.feature_extractor import FeatureExtractor
from utils.dalia.ppg_dalia_dataset_handler import PPGDaliaDatasetHandler
from utils.plot_handler import PlotHandler

import pandas as pd


def main():
    """path = "datasets/dalia/converted/S1/wrist/wrist_BVP.csv"

    see_plot(path)"""
    calculate_rr_intervals()

def open_pickle_dataset(path, patient):
    handler = PPGDaliaDatasetHandler(path)
    handler.extract_data(f'datasets/dalia/converted/{patient}')
    #json_path = "datasets/dalia/converted/S1.json"
    #handler.convert_pkl_json(json_path)

def see_plot(dataset):
    plot_handler = PlotHandler(dataset)
    plot_handler.create_plot()

def calculate_rr_intervals():
    extractor = FeatureExtractor()
    intervals = extractor.calculate_rr_intervals(pd.read_csv("datasets/dalia/converted/S1/rpeaks.csv"), 700)


if __name__ == '__main__':
    main()