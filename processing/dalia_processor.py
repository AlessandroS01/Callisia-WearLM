import os

import pandas as pd
import numpy as np
from pandas import DataFrame

from utils.dalia.configuration import BVP_SAMPLING_RATE, ACC_SAMPLING_RATE, WINDOW_SIZE_SEC


class DaliaProcessor:
    """
    Orchestrates the preprocessing of the PPG-DaLiA dataset for 1D-CNN ingestion.

    This processor handles the temporal synchronization of multi-rate wearable sensors,
    applies sliding window extraction, and filters corrupted segments using a
    pre-computed ECG signal quality index. It transforms raw subject directories
    into standardized ML-ready tensors.

    Attributes:
        subject_dir (str): Path to the base directory of a single DaLiA subject.
        window_size_bvp (int): The size of the window for BVP data in samples.
        window_size_acc (int): The size of the window for ACC data in samples.
        bvp_data (pd.DataFrame): Raw BVP data. Shape: (N, 1)
        acc_data (pd.DataFrame): Raw ACC data. Shape: (M, 3)
        sqi_data (pd.DataFrame): Raw SQI data. Shape: (M, 1)
        """

    def __init__(self, subject_dir: str):
        """
        Initializes the DaliaProcessor with specific windowing parameters.
        """
        super(DaliaProcessor, self).__init__()
        self.subject_dir = subject_dir

        self.window_size_bvp = WINDOW_SIZE_SEC * BVP_SAMPLING_RATE
        self.window_size_acc = WINDOW_SIZE_SEC * ACC_SAMPLING_RATE

        self.bvp_data = self._retrieve_data("wrist/wrist_BVP.csv", "csv")
        self.acc_data = self._retrieve_data("wrist/wrist_ACC.csv", "csv")
        self.sqi_data = self._retrieve_data("features/signal_quality.parquet", "parquet")

    def _align_and_resample(self, bvp_df: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Synchronizes the 32Hz Accelerometer data to match the 64Hz BVP data.

        Uses interpolation to upsample ACC data so that all sensor channels
        share the exact same temporal grid.

        Args:
            bvp_df (pd.DataFrame): Raw BVP data. Shape: (N, 1)
            acc_df (pd.DataFrame): Raw ACC data. Shape: (M, 3)

        Returns:
            pd.DataFrame: A fused dataframe containing [BVP, ACC_X, ACC_Y, ACC_Z]
                          sampled uniformly at 64Hz.
        """
        pass

    def _apply_quality_mask(self, window_start_sec: float) -> bool:
        """
        Checks the signal_quality.parquet file to determine if a specific time window
        contains reliable ECG ground truth.

        Args:
            window_start_sec (float): The absolute start time of the window in seconds.

        Returns:
            bool: True if the window passes the quality threshold, False if it
                  should be discarded due to high motion artifact or lost ECG peaks.
        """
        pass

    def get_standardized_windows(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts sliding windows of fused sensor data and their corresponding labels.

        Iterates through the subject's timeline, extracting data blocks defined by
        `window_size_sec`. It discards any windows that fail
        the `_apply_quality_mask` check.

        Returns:
            tuple: A tuple containing:

                - X (np.ndarray): The 3D feature tensor.

                  Shape: (num_valid_windows, sequence_length, num_channels)

                  Num channels must be 4 and represents [BVP, ACCx, ACCy, ACCz].
                - y (np.ndarray): The 1D label array for the signal quality index (SQI).
                  Shape: (num_valid_windows)
        """
        # x filling
        x_array_bvp = self._fill_arrays(feature="bvp")
        print(np.array(x_array_bvp).shape)
        x_array_acc = self._fill_arrays(feature="acc")
        print(np.array(x_array_acc).shape)
        x_array = []

        # y filling
        y_array = self._fill_arrays(feature="sqi")
        print(np.array(y_array).shape)

        return np.array(x_array), np.array(y_array)

    def _fill_arrays(self, feature) -> list:

        data = self.bvp_data if feature == "bvp" else self.acc_data if feature == "sqi" else self.sqi_data
        window_size = self.window_size_bvp if feature == "bvp" else self.window_size_acc
        step_size = window_size // 3

        array = []

        for step in range(0, len(data) - window_size + 1, step_size):
            array.append(data.iloc[step:step + window_size])

        return array

    def _retrieve_data(self, typology:str, data_type: str) -> DataFrame:
        path = os.path.join(self.subject_dir, typology)
        if data_type == "csv":
            return pd.read_csv(path)
        elif data_type == "parquet":
            return pd.read_parquet(path)
        return pd.DataFrame()