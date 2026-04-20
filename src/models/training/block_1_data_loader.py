"""Data loader for Block 1 training pipeline.

This module handles:
- Patient ID retrieval and management
- Data loading and dataset preparation
- Split management for both standard and LOSO training
"""
from typing import List, Tuple

import numpy as np

from src.models.block_data_loader import BlockDataLoader


class Block1TrainingDataLoader(BlockDataLoader):
    """Data loader for Block 1 training with Dalia dataset."""

    def _get_patient_path(self, patient: str) -> str:
        """Get the path to Dalia patient data.

        Args:
            patient: patient ID string

        Returns:
            str: Path to the Dalia patient data directory
        """
        return f"../../../data/processed/dalia/{patient}"

    def get_patients(self) -> dict:
        """Get patient splits for Block 1 training.

        Returns:
            dict: Dictionary with training, and validation patient splits
        """
        return {
            'training_patients':
                ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S12", "S13", "S15"],
            'validation_patients':
                ["S11", "S14"]
        }

    def get_all_subjects(self) -> List[str]:
        """Get all subjects from all splits (for LOSO).

        Returns:
            List[str]: Sorted list of all subject IDs
        """
        patients_dict = self.get_patients()
        all_subjects = set()
        for split_type in ['training_patients', 'validation_patients']:
            if split_type in patients_dict:
                all_subjects.update(patients_dict[split_type])
        return sorted(list(all_subjects))

    def prepare_dataset(
            self, patients: List[str], split_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dataset by combining data from multiple patients.

        Args:
            patients: List of patient IDs to include
            split_type: Type of split ('training', 'validation')

        Returns:
            Tuple[np.ndarray, np.ndarray]: Combined (x, y) data from all patients
        """
        all_x = []
        all_y = []

        print(f"  Loading {split_type} data for {len(patients)} patients...")
        for patient in patients:
            x, y = self.retrieve_patient_data(patient)
            all_x.append(x)
            all_y.append(y)

        x_combined = np.concatenate(all_x, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        print(f"    ✓ {split_type} data shape: {x_combined.shape}")

        return x_combined, y_combined
