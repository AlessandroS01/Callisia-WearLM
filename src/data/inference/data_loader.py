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

import numpy as np
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

    def __init__(self, base_path):
        """
        Initializes the DataLoader with the dataset's root directory.

        :param base_path: The root directory of the dataset (e.g., passed from config.yaml).
        """
        self.base_path = base_path

    def load_patient_signals(self, patient_id: str) -> tuple:
        """
        Loads raw BVP and ACC data for a specific patient.

        Constructs the expected file paths based on the patient ID and reads
        the sensor data from the 'wrist' subdirectory.

        :param patient_id: The specific patient identifier (e.g., 'S1').
        :return: A tuple containing two numpy arrays (bvp_array, acc_array).
        :raises FileNotFoundError: If either the BVP or ACC CSV files are missing
                                   for the requested patient in the expected directory.
        """
        patient_dir = os.path.join(self.base_path, patient_id)

        bvp_path = os.path.join(patient_dir, "wrist/wrist_BVP.csv")
        acc_path = os.path.join(patient_dir, "wrist/wrist_ACC.csv")

        if not os.path.exists(bvp_path) or not os.path.exists(acc_path):
            raise FileNotFoundError(
                f"Missing sensor data for patient {patient_id} in {patient_dir}"
            )

        # Load data and immediately convert to numpy arrays for the model
        bvp_arr = np.array(pd.read_csv(bvp_path))
        acc_arr = np.array(pd.read_csv(acc_path))

        return bvp_arr, acc_arr
