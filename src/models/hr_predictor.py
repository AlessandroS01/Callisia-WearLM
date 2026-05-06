"""
Heart Rate Prediction Module.

This module provides the :class:`HRPredictor` class, a standardized interface
for inferring continuous heart rate (HR) from Blood Volume Pulse (BVP) and
Accelerometer (ACC) signals. It is designed to be integrated into the broader
clinical insights pipeline, handling the necessary array validations and
interfacing with the underlying `beliefppg` inference engine.

Outputs are provided in Beats Per Minute (BPM) alongside their corresponding
time indices, which align with the midpoints of the inference sliding windows.

Typical usage example:
    from src.models.hr_predictor import HRPredictor

    # Initialize with sensor sampling frequencies
    predictor = HRPredictor(bvp_freq=64, acc_freq=32)

    # Predict heart rate from numpy arrays
    pred_hr, time_indices = predictor.predict(bvp_data, acc_data)
"""

import numpy as np
from beliefppg import infer_hr

class HRPredictor:
    """
    A predictive model wrapper for estimating heart rate from physiological signals.

    The `HRPredictor` class encapsulates the configuration and logic required to
    infer continuous heart rate (in BPM) using Blood Volume Pulse (BVP) and
    Accelerometer (ACC) sensor data. It acts as a standardized interface,
    storing the expected sampling frequencies and validating input dimensions
    before passing data to the underlying inference engine.

    :ivar bvp_freq: The sampling frequency of the Blood Volume Pulse (BVP)
                    sensor in Hertz (Hz).
    :vartype bvp_freq: int or float
    :ivar acc_freq: The sampling frequency of the Accelerometer (ACC)
                        sensor in Hertz (Hz).
    :vartype acc_freq: int or float
    """
    def __init__(self, bvp_freq, acc_freq):
        """
        Initializes the HRPredictor with the required sensor sampling frequencies.

        :param bvp_freq: Sampling frequency of the BVP sensor (e.g., 64).
        :param acc_freq: Sampling frequency of the ACC sensor (e.g., 32).
        """
        self.bvp_freq = bvp_freq
        self.acc_freq = acc_freq

    def predict(self, bvp_data: np.ndarray, acc_data: np.ndarray) -> tuple:
        """
        Predicts heart rate from Blood Volume Pulse (BVP) and Accelerometer (ACC) signal data.

        This method ensures the input signals are properly formatted 2D arrays before
        passing them to the underlying inference engine to compute the heart rate.

        :param bvp_data: A 2D numpy array of shape (n_samples, n_channels) containing the
                         Blood Volume Pulse / PPG signal data.
        :param acc_data: A 2D numpy array of shape (n_samples, n_channels) containing the
                         Accelerometer signal data.
        :return: A tuple (pred_hr, idx) where:
                 - pred_hr (np.ndarray): The predicted heart rate values (e.g., in BPM).
                 - idx (np.ndarray): The corresponding time indices (midpoints of the
                                     sliding windows) for the predictions.
        :raises ValueError: If either `bvp_data` or `acc_data` is not a 2-dimensional array.
        """
        if bvp_data.ndim != 2:
            raise ValueError(f"Expected ppg to be a 2D array, got shape {bvp_data.shape}"
                             f" with dimension {bvp_data.ndim}")
        if acc_data.ndim != 2:
            raise ValueError(f"Expected acc to be a 2D array, got shape {acc_data.shape}"
                             f" with dimension {acc_data.ndim} ")

        pred_hr, idx = infer_hr(
            ppg=bvp_data,
            ppg_freq=self.bvp_freq,
            acc=acc_data,
            acc_freq=self.acc_freq,
        )

        return pred_hr, idx
