"""
Signal Processing Pipeline Module.

This module provides the :class:`SignalProcessingPipeline` class, which handles
the first stage of the clinical insights architecture. It orchestrates the
loading of raw physiological sensor data and the execution of the machine
learning model to predict continuous heart rate.
"""
import numpy as np
import pandas as pd

from src.data.inference.data_loader import DataLoader
from src.models.hr_predictor import HRPredictor


class SignalProcessingPipeline:
    """
    Executes the signal processing and heart rate inference workflow.

    This pipeline sets up the necessary data loading and prediction tools
    upon initialization. It is designed to be reusable, allowing multiple
    patients to be processed sequentially by passing different patient IDs
    to the `run` method.

    :ivar base_path: The root directory for the dataset.
    :vartype base_path: str
    :ivar target_rate_freq: Sampling frequency of the LEAF device in Hz.
    :vartype target_rate_freq: int or float
    :ivar data_loader: The tool used to fetch raw sensor arrays.
    :vartype data_loader: DataLoader
    :ivar predictor: The tool used to infer heart rate from signals.
    :vartype predictor: HRPredictor
    """

    def __init__(self, config: dict):
        """
        Initializes the pipeline with configuration parameters and instantiates
        the required data loading and prediction tools.

        :param config: A dictionary containing 'base_path' and 'inference'
                               settings (sampling rates).
        """
        # Initialize config parameters
        self.base_path = config.get('base_path', "data/processed/dalia")
        self.valid_leaf_columns = config.get('valid_leaf_columns', [
            "timestamp_ms",
            "green",
            "acc_x",
            "acc_y",
            "acc_z"
        ])

        self.target_rate_freq = config.get(
            "inference", {}
        ).get("target_rate_freq", 40)

        # Initialize tools
        self.data_loader = DataLoader(
            base_path=self.base_path,
            valid_leaf_columns=self.valid_leaf_columns,
        )
        self.predictor = HRPredictor(
            bvp_freq=self.target_rate_freq,
            acc_freq=self.target_rate_freq
        )

    def _prepare_datetime_index(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the raw integer-based millisecond index into a pandas
        DatetimeIndex. Uses UTC not local time to avoid timezone complications.

        This is a prerequisite step for any time-aware operations. It ensures
        the DataFrame can handle the complex temporal alignment required for
        resampling and interpolation.

        :param df_data: A pandas DataFrame containing the raw data to be standardized.
        :return: A DataFrame with a validated pandas DatetimeIndex.
        """

        # first column represent timestamp in ms, we need to convert it to datetime index
        timestamp_ms = self.valid_leaf_columns[0]

        # preprocessing data: removing duplicates, sorting by time and creating index
        df_data = df_data.drop_duplicates(timestamp_ms)
        df_data = df_data.sort_values(timestamp_ms)
        df_data = df_data.set_index(timestamp_ms, drop=True)
        df_data.index = pd.to_datetime(df_data.index, unit='ms')

        return df_data

    def _standardize_sampling_rate(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes the temporal grid of the sensor data by correcting hardware
        jitter and bridging minor data gaps via linear interpolation.

        This method performs a three-stage harmonization:

        **Resampling**: Snaps raw timestamps to a strict target frequency (e.g., 40Hz).

        **Interpolation**: Applies a linear mathematical bridge across missing
        samples (limited to 1 second) to maintain signal continuity.

        **Cleaning**: Prunes segments with massive data dropouts where
        interpolation would be clinically unreliable.

        :param df_data: A pandas DataFrame of the data to be standardized.
        :return: A cleaned DataFrame locked to the target frequency, with
        datetime indices reverted to integer-based milliseconds.
        """

        # one sample each 25ms (1 second / 40 samples = 25ms per sample)
        resample_rate = f"{1000 / self.target_rate_freq}ms"

        resampled_df = df_data.resample(resample_rate).mean()
        # ensure interpolation does not exceed 1s of missing data (40 samples at 40Hz)
        df_data = resampled_df.interpolate(method='linear', limit=40)
        df_data.index = df_data.index.astype('int64')
        df_data = df_data.dropna()

        return df_data


    def _clean_data(self, signal_data: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates the transformation of raw sensor data into a standardized
        temporal format by chaining preparation and resampling procedures.

        This method serves as the high-level cleaning interface for the pipeline.
        It performs a two-stage harmonization:

        **Preparation**: Handles deduplication, temporal sorting, and the
        conversion of integer milliseconds into a valid DatetimeIndex.

        **Standardization**: Enforces a strict periodic grid (e.g., 40Hz)
        and applies linear interpolation to bridge minor hardware dropouts.

        :param signal_data: A pandas DataFrame containing raw physiological
        signal data with an associated timestamp column.

        :return: A cleaned and harmonized DataFrame, indexed by integer
        milliseconds, locked to the target frequency defined in the
        pipeline configuration.
        """

        signal_data = self._prepare_datetime_index(signal_data)
        return self._standardize_sampling_rate(signal_data)

    def run(self, patient_id: str, timestamp) -> tuple:
        """
        Executes the pipeline for a specific patient.

        This method coordinates the loading of the patient's raw signal data
        and the subsequent prediction of their continuous heart rate.

        :param patient_id: The unique identifier for the patient (e.g., 'S1').
        :return: A tuple (hr, idxs, bvp, acc) containing the predicted heart rate array, the
                corresponding time indices, the bvp and acc data
        """
        print(f"[{patient_id}] Loading signals...")

        # Load the data using the initialized tool
        signal_data = self.data_loader.load_patient_signals(patient_id, timestamp)

        print(f"[{patient_id}] Cleaning signals...")
        cleaned_data = self._clean_data(signal_data)

        # Predict Heart Rate
        print(f"[{patient_id}] Predicting heart rate...")
        bvp = np.array(cleaned_data[['green']])
        acc = np.array(cleaned_data[['acc_x', 'acc_y', 'acc_z']])

        hr, idxs = self.predictor.predict(bvp_data=bvp, acc_data=acc)

        return hr, idxs, bvp, acc
