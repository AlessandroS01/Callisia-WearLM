"""Provide helpers to extract and convert the original WESAD dataset.

This module reads the pickle file and exports processed CSV
representations for each subject.
"""

import os
import re

from .base_handler import BaseDatasetHandler


class WESADDatasetHandler(BaseDatasetHandler):
    """
        Class to obtain a processed dataset from the original WESAD dataset.
    """

    def extract_patient_data(self, patient: str):
        """
        Reads the readme.txt file relative to the patient directory.

        Args:
            patient: Patient identifier (e.g., 'S1', 'S2', etc.)

        Returns:
            dict: a dictionary containing the patient data (age, height, weight, gender),
                  or None if an error occurs.
        """
        subject_data_path = os.path.join(self.path, f"{patient}_readme.txt")

        try:
            with open(subject_data_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Parse personal information from the text file
            personal_info = {}

            # Extract age
            age_match = self._extract_field(content, 'Age')
            if age_match:
                personal_info['age'] = int(age_match)

            # Extract height
            height_match = self._extract_field(content, 'Height')
            if height_match:
                personal_info['height'] = int(height_match)

            # Extract weight
            weight_match = self._extract_field(content, 'Weight')
            if weight_match:
                personal_info['weight'] = int(weight_match)

            # Extract gender
            gender_match = self._extract_field(content, 'Gender')
            if gender_match:
                personal_info['gender'] = gender_match


            return personal_info if personal_info else None
        except FileNotFoundError:
            print(f"File not found: {subject_data_path}")
        except ValueError as e:
            print(f"Error parsing data: {e}")

        return None

    def _extract_field(self, content: str, field_name: str):
        """
        Extracts a field value from the text content.

        Args:
            content: The text content to parse
            field_name: The name of the field to extract (e.g., 'Age', 'Height')

        Returns:
            str: The field value, or None if not found.
        """
        # Pattern to match "Field_name: value"
        # (handles various formats like "Height (cm): 175" or "Age: 27")
        pattern = rf"{field_name}\s*(?:\(.*?\))?\s*:\s*(\w+)"
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def read_pkl_dataset(self, **kwargs):
        """
        Reads pickle file

        Args:
            **kwargs: Must include 'patient' - Patient identifier (e.g., 'S1', 'S2', etc.)

        Returns:
            The data loaded from the pickle file, or None if an error occurs.

        Raises:
            ValueError: If 'patient' is not provided in kwargs
        """
        patient = kwargs.get('patient')
        if not patient or not isinstance(patient, str):
            raise ValueError(
                "'patient' parameter is required for WESAD dataset reading"
            )

        pkl_path = os.path.join(self.path, f"{patient}.pkl")
        return self._load_pickle_file(pkl_path)

    def extract_data(self, output_dir: str, **kwargs):
        """
        Extracts data from the pickle file and saves it in a structured
        format in the specified output directory for each patient.

        Args:
            output_dir: Directory where the extracted files will be saved
            **kwargs: Must include 'patient' - Patient identifier (e.g., 'S1', 'S2', etc.)

        Raises:
            ValueError: If 'patient' is not provided in kwargs
        """
        patient = kwargs.get('patient')
        if not patient or not isinstance(patient, str):
            raise ValueError(
                "'patient' parameter is required for WESAD dataset extraction"
            )

        os.makedirs(output_dir, exist_ok=True)

        pkl_data = self.read_pkl_dataset(patient=patient)
        metadata = {
            "subject": pkl_data.get("subject"),
            "questionnaire": self.extract_patient_data(patient)
        }

        self._save_metadata(output_dir, metadata)

        self._save_csv_columns(pkl_data, output_dir, ['label'])

        if 'signal' in pkl_data:
            for signal_name, sensor_data in pkl_data['signal'].items():
                if signal_name == 'chest':
                    self._process_chest_signals(
                        sensor_data,
                        output_dir,
        ['ECG', 'EMG', 'EDA', 'TEMP', 'RESP']
                    )
                else:
                    self._process_wrist_signals(sensor_data, output_dir)
