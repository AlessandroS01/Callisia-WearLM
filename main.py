from utils.dalia.ppg_dalia_dataset_handler import PPGDaliaDatasetHandler
from utils.plot_handler import PlotHandler


def main():
    path = "datasets/dalia/converted/S1/chest/chest_ECG.csv"

    see_plot(path)

def open_pickle_dataset(path, patient):
    handler = PPGDaliaDatasetHandler(path)
    handler.extract_data(f'datasets/dalia/converted/{patient}')
    #json_path = "datasets/dalia/converted/S1.json"
    #handler.convert_pkl_json(json_path)

def see_plot(dataset):
    plot_handler = PlotHandler(dataset)
    plot_handler.create_plot()

if __name__ == '__main__':
    main()