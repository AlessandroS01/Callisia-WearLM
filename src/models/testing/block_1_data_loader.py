"""Data loader for Block 1 testing pipeline.

Handles data loading and dataset preparation specifically for the Block 1 testing module.
"""
from src.models.utils.block_data_loader import BlockDataLoader


class Block1TestingDataLoader(BlockDataLoader):
    """Data loader for Block 1 testing with WESAD dataset."""

    def _get_patient_path(self, patient: str) -> str:
        """Get the path to WESAD patient data.

        Args:
            patient: patient ID string

        Returns:
            str: Path to the WESAD patient data directory
        """
        return f"../../../data/processed/wesad/{patient}"

    def get_patients(self) -> dict:
        """Get patient list for Block 1 testing.

        Returns:
            dict: Dictionary with WESAD patient IDs
        """
        return {
            'testing_patients': [
                "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
                "S11", "S13", "S14", "S15", "S16", "S17"
            ]
        }
