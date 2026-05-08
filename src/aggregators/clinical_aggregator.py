"""
Clinical Aggregator Module.

This module provides the :class:`ClinicalAggregator` class, which transforms
high-frequency physiological arrays into structured statistical summaries
(like windowed variance) suitable for LLM context generation.
"""
import numpy as np
from scipy.stats import linregress


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

        # Run size of scheduled job
        self.total_duration = config.get("orchestrator", {}).get("run_interval_schedule", 120)

        # initialize thresholds
        self.resting_threshold = config.get('thresholds', {}).get('resting', 0.05)
        self.moving_threshold = config.get('thresholds', {}).get('moving', 0.5)

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

    def _cardiovascular_statistics(self, hr_predictions: np.ndarray) -> dict:
        """
        Calculates comprehensive clinical statistics and trend analysis from
        an array of predicted continuous heart rates.

        This method extracts standard descriptive statistics, categorizes the
        heart rate into clinical zones (Bradycardia, Normal, Tachycardia),
        and uses linear regression to determine the trajectory of the patient's
        heart rate over the recorded window.

        :param hr_predictions: A 1D numpy array of predicted heart rates (BPM).
        :return: A dictionary containing HR statistics, trend description,
                         and the distribution of beats across clinical zones.
        """

        total_samples = len(hr_predictions)

        # Standard descriptive stats
        hr_mean = float(np.mean(hr_predictions))
        hr_variance = float(np.var(hr_predictions))
        hr_std = float(np.std(hr_predictions))
        hr_min = float(np.min(hr_predictions))
        hr_max = float(np.max(hr_predictions))

        # Percentile Stats
        hr_25th = float(np.percentile(hr_predictions, 25))
        hr_median = float(np.median(hr_predictions))
        hr_75th = float(np.percentile(hr_predictions, 75))

        bradycardia_count = int(np.sum(hr_predictions < 60))
        normal_count = int(
            np.sum((hr_predictions >= 60) & (hr_predictions <= 100))
        )
        tachycardia_count = int(np.sum(hr_predictions > 100))

        x = np.arange(total_samples)
        slope, _, _, _, _ = linregress(x, hr_predictions)
        if slope > 0.5:
            trend = "rising rapidly"
        elif slope > 0.1:
            trend = "rising slightly"
        elif slope < -0.5:
            trend = "falling rapidly"
        elif slope < -0.1:
            trend = "falling slightly"
        else:
            trend = "stable"

        return {
            "standard_statistics": {
                "mean_hr": hr_mean,
                "min_hr": hr_min,
                "max_hr": hr_max,
                "standard_deviation_hr": hr_std,
                "variance_hr": hr_variance,
            },
            "baseline_percentiles": {
                "25th_percentile": hr_25th,
                "50th_percentile_median": hr_median,
                "75th_percentile": hr_75th
            },
            "trend": trend,
            "beats_distribution": {
                "under_60_bpm": bradycardia_count,
                "between_60_and_100_bpm": normal_count,
                "over_100_bpm": tachycardia_count
            }
        }

    def _movement_statistics(self, acc_variance_array: np.ndarray) -> dict:
        """
        Calculates physical activity and anomaly statistics based on windowed
        accelerometer variances.

        This method quantifies the patient's movement distribution (resting,
        light movement, active) and identifies sudden variance spikes that
        could indicate anomalies such as a fall, sudden exertion, or sensor drop.

        :param acc_variance_array: A 1D numpy array of calculated accelerometer
                                           variances per time window.
        :return: A dictionary containing the mean/peak variance, anomaly flags,
                         and the categorical distribution of movement.
        """
        mean_var = float(np.mean(acc_variance_array))
        max_var = float(np.max(acc_variance_array))

        resting_count = int(np.sum(acc_variance_array < self.resting_threshold))
        light_movement_count = int(
            np.sum(
                (acc_variance_array >= self.resting_threshold)
                &
                (acc_variance_array <= self.moving_threshold))
        )
        moving_count = int(np.sum(acc_variance_array > self.moving_threshold))

        # detect sudden extreme variance spike
        sudden_jolt_detected = bool(max_var > (mean_var * 10) and max_var > self.moving_threshold)

        return {
            "mean_variance": round(mean_var, 3),
            "peak_variance": round(max_var, 3),
            "sudden_jolt_detected": sudden_jolt_detected,
            "distribution": {
                "resting_windows_count": resting_count,
                "light_movement_windows_count": light_movement_count,
                "active_movement_windows_count": moving_count
            }
        }

    def _signal_correlation(
            self,
            hr_predictions: np.ndarray,
            acc_variance_array: np.ndarray
            ) -> dict:
        """
        Calculates the Pearson correlation coefficient between heart rate
        predictions and accelerometer variance to provide clinical context.

        Determines if an elevated heart rate is justified by exercise,
        or if it indicates an anomaly such as psychological stress, fever,
        or an arrhythmia (high HR with zero movement).

        :param hr_predictions: A 1D numpy array of predicted heart rates (BPM).
        :param acc_variance_array: A 1D numpy array of calculated accelerometer
                                           variances representing physical movement.
        :return: A dictionary containing the calculated correlation coefficient
                         and a human-readable string explaining the clinical context.
        """

        min_length = min(len(hr_predictions), len(acc_variance_array))
        hr_sync = hr_predictions[:min_length]
        acc_sync = acc_variance_array[:min_length]

        # correlation between hr and movement
        correlation = float(np.corrcoef(hr_sync, acc_sync)[0, 1])

        # Translate the math into clinical text for the LLM
        if correlation > 0.6:
            context = "Heart rate strongly driven by physical activity."
        elif correlation < 0.2:
            context = "Heart rate disconnected from physical movement."
        else:
            context = "Moderate correlation between movement and heart rate."

        return {
            "hr_movement_correlation": round(correlation, 3),
            "clinical_context": context
        }

    def _hr_volatility(self, hr_predictions: np.ndarray) -> dict:
        """
        Calculates the average absolute difference between successive heart rate
        predictions as a mathematical proxy for physiological volatility.

        Because continuous heart rate predictions lack the millisecond-level
        precision required for true Heart Rate Variability (HRV) metrics like
        RMSSD, this method measures the macroscopic "jumps" between sequential
        windows. A highly rigid heart rate (low volatility) can indicate
        sympathetic nervous system dominance (stress, focus, or fatigue),
        while a dynamic heart rate (high volatility) generally suggests
        parasympathetic activity (relaxation or recovery).

        :param hr_predictions: A 1D numpy array of predicted continuous heart
                               rates (BPM).
        :return: A dictionary containing the mean absolute difference between
                 consecutive predictions, representing the average jump in BPM.
        """
        # Calculate the absolute difference between every consecutive element
        successive_differences = np.abs(np.diff(hr_predictions))
        mean_volatility = float(np.mean(successive_differences))

        return {
            "average_beat_to_beat_jump": round(mean_volatility, 3)
        }


    def aggregate(self,
                  acc_array: np.ndarray,
                  hr_prediction: np.ndarray
                  ) -> dict:
        """
        Coordinates the post-processing and aggregation of raw sensor signals
        and ML predictions into an LLM-ready dictionary.

        This method extracts the movement magnitude from the raw accelerometer
        data, aligns its frequency to the baseline inference rate, and calculates
        windowed statistics to cross-reference with the predicted heart rate.

        :param acc_array: A 2D numpy array containing raw 3-axis ACC data.
        :param hr_prediction: A 1D numpy array containing predicted continuous
                              heart rates from the BeliefPPG model.
        :return: A comprehensive dictionary containing the aggregated clinical
                 features structured for natural language generation.
        """
        acc_magnitude = self._combine_acc(acc_array)

        if self.acc_rate < self.bvp_rate:
            factor = int(self.bvp_rate / self.acc_rate)
            acc_magnitude = self._upsample_signal(acc_magnitude, factor)

        # variance for each window
        acc_variance = self._windowed_acc_variance(acc_magnitude)

        print(f"Length of HR predictions: {len(hr_prediction)}")
        print(f"Length of ACC windows variances: {len(acc_variance)}")

        # generation individual feature dictionaries
        cardiovascular_summary_stats = self._cardiovascular_statistics(hr_prediction)
        movement_summary_stats = self._movement_statistics(acc_variance)
        volatility_stats = self._hr_volatility(hr_prediction)
        correlation_stats = self._signal_correlation(hr_prediction, acc_variance)

        payload = {
            "total_recording_duration_seconds": self.total_duration,
            "rolling_window_duration_seconds": self.window_sec,
            "rolling_window_step_seconds": self.step_sec,
            "array_elements_per_window": self.window_size,
            "cardiovascular_analysis": cardiovascular_summary_stats,
            "movement_analysis": movement_summary_stats,
            "autonomic_nervous_system_proxy": volatility_stats,
            "clinical_context": correlation_stats
        }

        return payload
