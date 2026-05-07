"""
Clinical Aggregator Module.

This module provides the :class:`ClinicalAggregator` class, which transforms
high-frequency physiological arrays into structured statistical summaries
(like windowed variance) suitable for LLM context generation.
"""
import numpy as np


class ClinicalAggregator:
    """
    Transforms high-frequency BVP and ACC signals into aggregated statistical
    features suitable for LLM context generation.

    This class handles the synchronization of signals with different sampling
    rates through upsampling, calculates 3-axis accelerometer magnitudes,
    and computes vectorized windowed statistics to measure patient activity.

    :ivar window_sec: The size of each aggregation window in seconds.
    :vartype window_sec: int or float
    :ivar bvp_rate: The sampling frequency of the Blood Volume Pulse signal in Hz.
    :vartype bvp_rate: int
    :ivar acc_rate: The sampling frequency of the Accelerometer signal in Hz.
    :vartype acc_rate: int
    :ivar target_rate: The maximum frequency used to synchronize arrays.
    :vartype target_rate: int
    :ivar window_size: The exact number of array elements per time window.
    :vartype window_size: int
    """

    def __init__(self, config: dict):
        """
        Initializes the aggregator by loading window settings and frequency
        parameters from the global configuration dictionary.

        :param config: The master configuration dictionary containing 'params'
                               and 'inference' settings.
        """
        # initialize params
        self.window_sec = config.get('params', {}).get('seconds_per_window', 8)
        self.step_sec = config.get('params', {}).get('step_size', 2)
        self.bvp_rate = config.get('inference', {}).get('bvp_freq', 64)
        self.acc_rate = config.get('inference', {}).get('acc_freq', 32)

        # We calculate window size based on the highest frequency after upsampling
        self.target_rate = int(max(self.bvp_rate, self.acc_rate))
        self.window_size = self.target_rate * self.window_sec
        self.step_size = self.target_rate * self.step_sec

    def _upsample_signal(self, array: np.ndarray, factor: int) -> np.ndarray:
        """
        Upsamples a 1D signal by repeating its elements.

        :param array: The 1D numpy array to upsample.
        :param factor: The integer multiplier for upsampling.
        :return: The upsampled numpy array.
        """
        return np.repeat(array, factor)

    def _combine_acc(self, acc_array) -> np.ndarray:
        """
        Combines the 3-axis accelerometer data into a single magnitude array.

        :param acc_array: A 2D numpy array of shape (n_samples, 3) containing
                          the x, y, z accelerometer data.
        :return: A 1D numpy array of shape (n_samples,) containing the
                 combined magnitude of the accelerometer data.
        """
        return np.sqrt(np.sum(acc_array**2, axis=1))

    def _windowed_acc_variance(self, acc_array: np.ndarray) -> np.ndarray:
        """
        Calculates the variance of the accelerometer magnitude over discrete windows.

        :param acc_array: A 1D numpy array of accelerometer magnitudes.
        :return: A 1D numpy array of variances, one for each window.
        """
        variances = []

        for i in range(0, len(acc_array) - self.window_size + 1, self.step_size):
            # Grab the full 8-second context block
            current_window = acc_array[i: i + self.window_size]

            # Calculate the variance (movement score) for those entire 8 seconds
            score = np.var(current_window)
            variances.append(score)

        return np.array(variances)

    def aggregate(self,
                  bvp_array: np.ndarray,
                  acc_array: np.ndarray,
                  hr_prediction: np.ndarray
                  ) -> dict:
        """
        Coordinates the preprocessing and aggregation of raw sensor signals
        into an LLM-ready dictionary.

        This method safely upsamples the lower-frequency signal to align with
        the higher-frequency signal, extracts the movement magnitude, and
        calculates windowed variances.

        :param hr_prediction: A 2D numpy array containing predicted heart rates.
        :param bvp_array: A 1D numpy array containing raw BVP data.
        :param acc_array: A 2D numpy array containing raw 3-axis ACC data.
        :return: A dictionary containing the aggregated statistical features
                 (e.g., total windows, window size, movement variances).
        """
        acc_magnitude = self._combine_acc(acc_array)

        if self.bvp_rate < self.acc_rate:
            factor = int(self.acc_rate / self.bvp_rate)
            bvp_array = self._upsample_signal(bvp_array, factor)

        elif self.acc_rate < self.bvp_rate:
            factor = int(self.bvp_rate / self.acc_rate)
            acc_magnitude = self._upsample_signal(acc_magnitude, factor)

        acc_variance = self._windowed_acc_variance(acc_magnitude)

        return {
            "window_size_seconds": self.window_sec,
            "total_windows": len(acc_variance),
            "movement_variance_per_window": acc_variance.tolist()
        }
