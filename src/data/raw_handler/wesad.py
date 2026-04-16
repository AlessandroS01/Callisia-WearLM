"""Provide helpers to extract and convert the original WESAD dataset.

This module reads the pickle file and exports processed CSV
representations for each subject.
"""

import json
import os
import pickle as pkl
import re
import pandas as pd


class WESADDatasetHandler:
    """
        Class to obtain a processed dataset from the original WESAD dataset.
    """

    def __init__(self, path):
        """
            Constructor for the DatasetHandler class

            Args:
                path: Path to the dataset
        """
        self.path = path

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
        # Pattern to match "Field_name: value" (handles various formats like "Height (cm): 175" or "Age: 27")
        pattern = rf"{field_name}\s*(?:\(.*?\))?\s*:\s*(\w+)"
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def read_pkl_dataset(self, patient: str):
        """
            Reads pickle file

            Args:
                patient: Patient identifier (e.g., 'S1', 'S2', etc.)

            Returns:
                The data loaded from the pickle file, or None if an error occurs.
        """
        pkl_path = os.path.join(self.path, f"{patient}.pkl")
        try:
            with open(pkl_path, 'rb') as file:
                data = pkl.load(file, encoding='latin1')

                self.print_pkl_data_shape(data)

            return data
        except FileNotFoundError:
            print(f"File not found: {self.path}")
        except pkl.UnpicklingError:
            print("Error: The file content is not a valid pickle format.")
        except EOFError:
            print("Error: The file is incomplete or corrupted.")

        return None

    def extract_data(self, output_dir: str, patient: str):
        """
            Extracts data from the pickle file and saves it in a structured
            format in the specified output directory for each patient.

            Args:
                output_dir: Directory where the extracted files will be saved
                patient: Patient identifier (e.g., 'S1', 'S2', etc.)
        """
        os.mkdir(output_dir)

        pkl_data = self.read_pkl_dataset(patient)
        metadata = {
            "subject": pkl_data.get("subject"),
            "questionnaire": self.extract_patient_data(patient)
        }

        with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        for key in ['label']:
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
                        if str(modality).upper() in ['ECG', "EMG", "EDA", "TEMP", "RESP"]:
                            # Convert array to DataFrame and save
                            filename = f"{signal_name}_{modality}.csv"
                            out_path = os.path.join(chest_folder, filename)
                            pd.DataFrame(data, columns=[modality]).to_csv(
                                out_path, index=False
                            )
                        if str(modality).upper() == 'ACC':
                            # Convert array to DataFrame and save
                            filename = f"{signal_name}_{modality}.csv"
                            out_path = os.path.join(chest_folder, filename)
                            pd.DataFrame(data, columns=['x', 'y', 'z']).to_csv(
                                out_path, index=False
                            )

                # takes wrist data
                else:
                    wrist_folder = f'{output_dir}/wrist'
                    os.mkdir(wrist_folder)

                    for modality, data in sensor_data.items():
                        if str(modality).upper() == 'ACC':
                            # Convert array to DataFrame and save
                            filename = f"{signal_name}_{modality}.csv"
                            out_path = os.path.join(wrist_folder, filename)
                            pd.DataFrame(data, columns=['x', 'y', 'z']).to_csv(
                                out_path, index=False
                            )
                        else:
                            filename = f"{signal_name}_{modality}.csv"
                            out_path = os.path.join(wrist_folder, filename)
                            pd.DataFrame(data, columns=[modality]).to_csv(
                                out_path, index=False
                            )


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
