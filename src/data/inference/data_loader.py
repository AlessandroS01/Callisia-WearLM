"""
Inference Data Loading Module.

This module provides the :class:`DataLoader` class, which handles the secure
and efficient ingestion of raw physiological sensor data for the production
inference pipeline. It is intentionally isolated from the training data loaders
to ensure the inference architecture remains lightweight and decoupled from
model training complexities (like batching or epoch management).

Typical usage example:

    from src.data.inference.data_loader import DataLoader

    # Initialize with the dataset root path
    loader = DataLoader(base_path="data/processed/dalia")

    # Fetch numpy arrays for a specific patient
    bvp_data, acc_data = loader.load_patient_signals(patient_id="S1")
"""

import os

import pandas as pd


class DataLoader:
    """
    A dedicated data ingestion module for loading physiological sensor signals.

    This class handles all file system operations required to read raw
    Blood Volume Pulse (BVP) and Accelerometer (ACC) data from a structured
    dataset directory. By isolating the file reading logic, it keeps the
    downstream signal processing pipelines agnostic to the underlying file structure.

    :ivar base_path: The root directory where the dataset is stored.
    :vartype base_path: str
    """

    def __init__(self, base_path, valid_leaf_columns):
        """
        Initializes the DataLoader with the dataset's root directory.

        :param base_path: The root directory of the dataset (e.g., passed from config.yaml).
        :param valid_leaf_columns: A list of expected column names in the LEAF device CSV files.
        """
        self.base_path = base_path
        self.valid_leaf_data = valid_leaf_columns

    def load_patient_signals(self, patient_id: str, timestamp) -> pd.DataFrame:
        """
        Loads raw BVP and ACC data for a specific patient.

        Constructs the expected file paths based on the patient ID and reads
        the sensor data from the 'wrist' subdirectory.

        :param patient_id: The specific patient identifier (e.g., 'S1').
        :return: A dataframe containing the signal data
        :raises FileNotFoundError: If signal data file is missing
                                   for the requested patient in the expected directory.
        """
        patient_dir = os.path.join(self.base_path, patient_id, "Leaf")

        for subdir in os.listdir(patient_dir):
            if "ppg_acc" in subdir and subdir.endswith(".csv"):
                signal_data = pd.read_csv(
                    os.path.join(patient_dir, subdir),
                    usecols=self.valid_leaf_data
                )

                return signal_data[
                    signal_data["timestamp_ms"].between(timestamp, timestamp + 2 * 60 * 1000)
                ]

        raise FileNotFoundError(
            f"No valid signal data file found for patient {patient_id} in {patient_dir}"
        )
