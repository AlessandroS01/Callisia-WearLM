"""Helpers to extract and convert the original PPG-DaLiA dataset.

This module reads the distributed pickle file and exports processed CSV
representations for each subject (rpeaks, labels, sensor signals, etc.).
"""

import json
import os
import numpy as np
from typing_extensions import deprecated

from .base_handler import BaseDatasetHandler


class PPGDaliaDatasetHandler(BaseDatasetHandler):
    """
        Class to obtain a processed dataset from the original PPG Dalia dataset.
    """

    def read_pkl_dataset(self, **kwargs):
        """
        Reads pickle file

        Returns:
            The data loaded from the pickle file, or None if an error occurs.
        """
        return self._load_pickle_file(self.path)

    def extract_data(self, output_dir: str, **kwargs):
        """
        Extracts data from the pickle file and saves it in a structured
        format in the specified output directory for each patient.

        Args:
            output_dir: Directory where the extracted files will be saved
        """
        os.makedirs(output_dir, exist_ok=True)

        pkl_data = self.read_pkl_dataset()
        metadata = {
            "subject": pkl_data.get("subject"),
            "questionnaire": pkl_data.get("questionnaire")
        }

        self._save_metadata(output_dir, metadata)

        # Save rpeaks, label, and activity
        self._save_csv_columns(pkl_data, output_dir, ['rpeaks', 'label', 'activity'])

        # Process signal data
        if 'signal' in pkl_data:
            for signal_name, sensor_data in pkl_data['signal'].items():
                if signal_name == 'chest':
                    self._process_chest_signals(
                        sensor_data, output_dir, ['ECG', 'RESP']
                    )
                else:
                    self._process_wrist_signals(sensor_data, output_dir)

    @deprecated("Deprecated: kept for backward compatibility")
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

        with open(json_path_name, 'w', encoding='utf-8') as f:
            json.dump(pkl_data, f, default=default_serialize, indent=4)
