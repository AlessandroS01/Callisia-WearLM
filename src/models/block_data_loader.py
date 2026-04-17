"""Base class for loading and preparing data for block-based models.

This module provides a reusable interface for data loading and dataset preparation
across different training and testing blocks, supporting multiple datasets.
"""
from abc import ABC, abstractmethod

import numpy as np

from src.data.processors.processor import WESADDaliaProcessor


class BlockDataLoader(ABC):
    """Abstract base class for loading and preparing data for model blocks.

    Provides common functionality for retrieving patient data and preparing
    datasets by combining multiple patients. Subclasses should implement
    dataset-specific paths and patient lists.
    """

    def retrieve_patient_data(self, patient: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts input features and labels for the specific patient.

        Params:
            patient: patient ID string, e.g. "S1", "S2", ..., "S15"

        Returns:
            tuple: A tuple containing:

                - x (ndarray): The input features for the model, typically
                  a 3D array of shape (num_samples, num_channels, sequence_length)
                  containing [BVP, ACCx, ACCy, ACCz]

                - y (ndarray): The target labels for the model, typically
                  a 1D array of shape (num_samples,) containing the heart rate values.
        """
        patient_path = self._get_patient_path(patient)
        processor = WESADDaliaProcessor(patient_path)
        x, y = processor.process()

        return x, y

    @abstractmethod
    def _get_patient_path(self, patient: str) -> str:
        """Get the path to patient data.

        Args:
            patient: patient ID string

        Returns:
            str: Path to the patient data directory

        This method must be implemented by subclasses to provide
        dataset-specific paths.
        """

    @abstractmethod
    def get_patients(self) -> dict:
        """Get patient splits for the dataset.

        Returns:
            dict: Dictionary with keys like 'training_patients', 'validation_patients',
                  'test_patients' containing lists of patient IDs
        """

    def prepare_dataset(self, patients: list, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares a dataset by combining data from multiple patients.

        Params:
            patients: list of patient IDs
            dataset_name: name of the dataset (for logging)

        Returns:
            tuple: Combined (x, y) arrays for the dataset
        """
        x_list, y_list = [], []

        for patient in patients:
            x, y = self.retrieve_patient_data(patient)
            x_list.append(x)
            y_list.append(y)
            print(f"{patient} has x shape: {x.shape} and y shape: {y.shape}")

        x_combined = np.concatenate(x_list, axis=0)
        y_combined = np.concatenate(y_list, axis=0)

        print(f"Combined {dataset_name} data with x shape: {x_combined.shape} and "
              f"y shape: {y_combined.shape}\n")

        return x_combined, y_combined

