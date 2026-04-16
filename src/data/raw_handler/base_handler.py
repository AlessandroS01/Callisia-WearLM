"""Base class for dataset handlers.

This module provides a common interface and shared functionality for handling
DALIA and WESAD datasets
"""

import json
import os
import pickle as pkl
from abc import ABC, abstractmethod

import pandas as pd


class BaseDatasetHandler(ABC):
    """
    Abstract base class for dataset handlers.
    Provides common functionality for extracting and processing biometric datasets.
    """

    def __init__(self, path):
        """
        Constructor for the BaseDatasetHandler class

        Args:
            path: Path to the dataset
        """
        self.path = path

    def print_pkl_data_shape(self, data):
        """Print a short summary (type and shape) for each item in the pickle.

        Args:
            data: Mapping-like object loaded from the pickle file.
        """
        for key, value in data.items():
            print(
                f"Key: {key}, Type: {type(value)}, "
                f"Shape: {getattr(value, 'shape', len(value))}"
            )

    @staticmethod
    def _save_signal_data(data, output_path: str, columns: list = None):
        """
        Helper method to save signal data to CSV.

        Args:
            data: The data array to save
            output_path: Path where the CSV file will be saved
            columns: Column names for the DataFrame (if None, uses default)
        """
        if columns is None:
            columns = ['data']
        pd.DataFrame(data, columns=columns).to_csv(output_path, index=False)

    def _process_chest_signals(
            self,
            chest_data: dict,
            output_dir: str,
            modalities_to_save: list
    ):
        """
        Process and save chest sensor data.

        Args:
            chest_data: Dictionary containing chest sensor modalities and data
            output_dir: Base output directory
            modalities_to_save: List of modalities to save (e.g., ['ECG', 'RESP'])
        """
        chest_folder = os.path.join(output_dir, 'chest')
        os.makedirs(chest_folder, exist_ok=True)

        for modality, data in chest_data.items():
            modality_upper = str(modality).upper()
            if modality_upper in modalities_to_save:
                filename = f"chest_{modality}.csv"
                out_path = os.path.join(chest_folder, filename)
                self._save_signal_data(data, out_path, columns=[modality])
            elif modality_upper == 'ACC':
                filename = f"chest_{modality}.csv"
                out_path = os.path.join(chest_folder, filename)
                self._save_signal_data(data, out_path, columns=['x', 'y', 'z'])

    def _process_wrist_signals(self, wrist_data: dict, output_dir: str):
        """
        Process and save wrist sensor data.

        Args:
            wrist_data: Dictionary containing wrist sensor modalities and data
            output_dir: Base output directory
        """
        wrist_folder = os.path.join(output_dir, 'wrist')
        os.makedirs(wrist_folder, exist_ok=True)

        for modality, data in wrist_data.items():
            modality_upper = str(modality).upper()
            filename = f"wrist_{modality}.csv"
            out_path = os.path.join(wrist_folder, filename)

            if modality_upper == 'ACC':
                self._save_signal_data(data, out_path, columns=['x', 'y', 'z'])
            else:
                self._save_signal_data(data, out_path, columns=[modality])

    def _save_metadata(self, output_dir: str, metadata: dict):
        """
        Save metadata to JSON file.

        Args:
            output_dir: Directory where metadata will be saved
            metadata: Dictionary containing metadata
        """
        with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

    def _load_pickle_file(self, file_path: str):
        """
        Load pickle file with common error handling.

        Args:
            file_path: Path to the pickle file

        Returns:
            The data loaded from the pickle file, or None if an error occurs.
        """
        try:
            with open(file_path, 'rb') as file:
                data = pkl.load(file, encoding='latin1')
                self.print_pkl_data_shape(data)
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except pkl.UnpicklingError:
            print("Error: The file content is not a valid pickle format.")
        except EOFError:
            print("Error: The file is incomplete or corrupted.")
        return None

    def _save_csv_columns(self, pkl_data: dict, output_dir: str, keys: list):
        """
        Save specified columns from pickle data as CSV files.

        Args:
            pkl_data: Dictionary containing the pickle data
            output_dir: Directory where CSV files will be saved
            keys: List of keys to extract and save as CSV
        """
        for key in keys:
            if key in pkl_data:
                pd.DataFrame(pkl_data[key], columns=[key]).to_csv(
                    os.path.join(output_dir, f"{key}.csv"), index=False
                )

    @abstractmethod
    def read_pkl_dataset(self, **kwargs):
        """
        Read pickle file. Must be implemented by subclasses.

        Returns:
            The data loaded from the pickle file, or None if an error occurs.
        """

    @abstractmethod
    def extract_data(self, output_dir: str, **kwargs):
        """
        Extract data from the dataset and save it in a structured format.
        Must be implemented by subclasses.

        Args:
            output_dir: Directory where the extracted files will be saved
        """
