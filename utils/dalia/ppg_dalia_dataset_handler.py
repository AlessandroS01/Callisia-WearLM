import json
import os
import pickle as pkl

import numpy as np
import pandas as pd


class PPGDaliaDatasetHandler:
    """
        A class to handle operations on the PPG Dalia dataset files.
    """

    def __init__(self, path):
        """
            Constructor for the DatasetHandler class

            Args:
                path: Path to the dataset
        """
        self.path = path

    def read_pkl_dataset(self):
        """
            Reads pickle file

            Returns:
                The data loaded from the pickle file, or None if an error occurs.
        """
        try:
            with open(self.path, 'rb') as file:
                data = pkl.load(file, encoding='latin1')

                self.print_pkl_data_shape(data)

            return data
        except FileNotFoundError:
            print(f"File not found: {self.path}")
        except pkl.UnpicklingError:
            print("Error: The file content is not a valid pickle format.")
        except EOFError:
            print("Error: The file is incomplete or corrupted.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return None

    def convert_pkl_json(self, json_path_name: str):
        """
            Converts the data from a pickle file to a JSON file and save it under the given path

            Args:
                json_path_name: JSON file path
        """
        pkl_data = self.read_pkl_dataset()

        # Helper to convert numpy arrays to lists so they are JSON serializable
        def default_serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(json_path_name, 'w') as f:
            json.dump(pkl_data, f, default=default_serialize, indent=4)


    def extract_and_split_data(self, output_dir: str):
        """
            Extracts data from the pickle file and saves it in a structured
            format in the specified output directory for each patient.

            Args:
                output_dir: Directory where the extracted files will be saved
        """
        os.mkdir(output_dir)

        pkl_data = self.read_pkl_dataset()
        metadata = {
            "subject": pkl_data.get("subject"),
            "questionnaire": pkl_data.get("questionnaire")
        }

        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

        for key in ['rpeaks', 'label', 'activity']:
            if key in pkl_data:
                # Convert array to DataFrame and save
                pd.DataFrame(pkl_data[key], columns=[key]).to_csv(
                    os.path.join(output_dir, f"{key}.csv"), index=False
                )

        if 'signal' in pkl_data:
            for signal_name, sensor_data in pkl_data['signal'].items():

                # takes chest data
                if signal_name == 'chest':
                    chest_folder = f'{output_dir}/chest'
                    os.mkdir(chest_folder)

                    for modality, data in sensor_data.items():
                        # ‘EDA’, ‘EMG’ and ‘Temp’ only include dummy data and are ignored
                        if str(modality).upper() in ['ECG', 'RESP']:
                            # Convert array to DataFrame and save
                            pd.DataFrame(data, columns=[modality]).to_csv(
                                os.path.join(chest_folder, f"{signal_name}_{modality}.csv"), index=False
                            )
                        if str(modality).upper() == 'ACC':
                            # Convert array to DataFrame and save
                            pd.DataFrame(data, columns=['x', 'y', 'z']).to_csv(
                                os.path.join(chest_folder, f"{signal_name}_{modality}.csv"), index=False
                            )

                # takes wrist data
                else:
                    wrist_folder = f'{output_dir}/wrist'
                    os.mkdir(wrist_folder)

                    for modality, data in sensor_data.items():
                        if str(modality).upper() == 'ACC':
                            # Convert array to DataFrame and save
                            pd.DataFrame(data, columns=['x', 'y', 'z']).to_csv(
                                os.path.join(wrist_folder, f"{signal_name}_{modality}.csv"), index=False
                            )
                        else:
                            pd.DataFrame(data, columns=[modality]).to_csv(
                                os.path.join(wrist_folder, f"{signal_name}_{modality}.csv"), index=False
                            )


    def print_pkl_data_shape(self, data):
        for key, value in data.items():
            print(f""
                  f"Key: {key}, "
                  f"Type: {type(value)}, "
                  f"Shape: {getattr(value, 'shape', len(value))}"
                  )
