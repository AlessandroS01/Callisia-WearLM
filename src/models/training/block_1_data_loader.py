"""Data loader for Block 1 training pipeline.

Handles data loading and dataset preparation specifically for the Block 1 training module.
"""
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
            dict: Dictionary with training, validation, and test patient splits
        """
        return {
            'training_patients':
                ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"],
            'validation_patients':
                ["S11", "S12"],
            'test_patients':
                ["S13", "S14", "S15"]
        }
