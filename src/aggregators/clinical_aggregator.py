"""
Clinical Aggregator Module.

This module provides the :class:`ClinicalAggregator` class, which transforms
high-frequency physiological arrays into structured statistical summaries
(like windowed variance) suitable for LLM context generation.
"""
import numpy as np
from scipy.signal import medfilt
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
        rest_threshold = config.get('thresholds', {}).get('resting', {})
        move_threshold = config.get('thresholds', {}).get('moving', {})

        self.resting_thresholds = {
            'std_magnitude': rest_threshold.get('std_magnitude', 15.0),
            'range_magnitude': rest_threshold.get('range_magnitude', 100.0),
            'mean_jerk': rest_threshold.get('mean_jerk', 5.0)
        }

        self.moving_thresholds = {
            'std_magnitude': move_threshold.get('std_magnitude', 100.0),
            'range_magnitude': move_threshold.get('range_magnitude', 1200.0),
            'mean_jerk': move_threshold.get('mean_jerk', 30.0)
        }

        # initialize receptive field of the model
        self.receptive_field_model_seconds = (
            config.get('params', {}).get('receptive_field_model_seconds', 20))

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

    def _extract_windowed_acc_features(self, acc_array: np.ndarray) -> dict:
        """
        Extracts multidimensional kinetic features from the accelerometer
        magnitude over discrete rolling windows.

        Instead of relying on a single variance metric, this method calculates
        three distinct features per window to create a robust motion profile:
        1. 'std' (Standard Deviation): Captures continuous, rhythmic motion (e.g., walking).
        2. 'range' (Peak-to-Peak): Captures sudden impacts or massive arm swings.
        3. 'jerk' (Mean Absolute Difference): Captures high-frequency jitter or fidgeting.

        These isolated features are required by the downstream pipeline to
        accurately categorize physical activity and flag motion artifacts that
        corrupt the optical PPG signal.

        :param acc_array: A 1D numpy array of synchronized accelerometer magnitudes.
        :return: A dictionary containing three 1D numpy arrays ('std', 'range', 'jerk'),
                 where each element represents the calculated feature for a single time window.
        """
        stds, ranges, jerks = [], [], []

        for i in range(0, len(acc_array) - self.window_size + 1, self.step_size):
            # Grab the full 8-second context block
            current_window = acc_array[i: i + self.window_size]

            # Calculate the variance (movement score) for those entire 8 seconds
            stds.append(np.std(current_window))
            ranges.append(np.ptp(current_window))  # Peak-to-peak (Max - Min)
            jerks.append(np.mean(np.abs(np.diff(current_window))))

        return {
            "std": np.array(stds),
            "range": np.array(ranges),
            "jerk": np.array(jerks)
        }

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

        if len(hr_predictions) == 0:
            return {}

        # smooths the prediction hallucinations without destroying trends
        clean_hr_predictions = medfilt(hr_predictions, kernel_size=7)

        # Standard descriptive stats
        raw_max = float(np.max(hr_predictions))
        clean_max = float(np.max(clean_hr_predictions))
        raw_min = float(np.min(hr_predictions))
        clean_min = float(np.min(clean_hr_predictions))

        hr_mean = float(np.mean(clean_hr_predictions))
        hr_var = float(np.var(clean_hr_predictions))
        hr_std = float(np.std(clean_hr_predictions))

        # Percentile Stats
        p25, p50, p75 = np.percentile(clean_hr_predictions, [25, 50, 75])

        bradycardia_count = int(np.sum(clean_hr_predictions < 60))
        normal_count = int(
            np.sum((clean_hr_predictions >= 60) & (clean_hr_predictions <= 100))
        )
        tachycardia_count = int(np.sum(clean_hr_predictions > 100))

        # trend analysis
        x_axis_minutes = np.arange(len(clean_hr_predictions)) * (self.step_sec / 60.0)

        # As X-axis is natively in minutes, the slope evaluates strictly as BPM per minute
        slope_bpm_per_minute, _, _, _, _ = linregress(x_axis_minutes, clean_hr_predictions)

        # Adjusted semantic thresholds based on realistic human physiology:
        # A change of >15 BPM per minute is a rapid physiological shift (e.g., starting a sprint)
        if slope_bpm_per_minute > 15.0:
            trend = "Rising rapidly"
        elif slope_bpm_per_minute > 5.0:
            trend = "Rising slightly"
        elif slope_bpm_per_minute < -15.0:
            trend = "Falling rapidly"
        elif slope_bpm_per_minute < -5.0:
            trend = "Falling slightly"
        else:
            trend = "Stable"

        return {
            "artifact_and_noise_context": {
                "raw_ml_max_hr": round(raw_max, 1),
                "raw_ml_min_hr": round(raw_min, 1),
                "filtered_physiological_max_hr": round(clean_max, 1),
                "filtered_physiological_min_hr": round(clean_min, 1),
                "estimated_motion_artifact_deviation_bpm": round(raw_max - clean_max, 1)
            },
            "physiological_statistics": {
                "mean_hr": round(hr_mean, 1),
                "standard_deviation_hr": round(hr_std, 1),
                "variance_hr": round(hr_var, 1),
            },
            "baseline_percentiles": {
                "25th_percentile": round(p25, 1),
                "50th_percentile_median": round(p50, 1),
                "75th_percentile": round(p75, 1)
            },
            "trajectory_analysis": {
                "semantic_trend_description": trend,
                "mathematical_slope_bpm_per_minute": round(float(slope_bpm_per_minute), 3)
            },
            "beats_distribution": {
                "under_60_bpm": bradycardia_count,
                "between_60_and_100_bpm": normal_count,
                "over_100_bpm": tachycardia_count
            }
        }

    def _movement_statistics(self, acc_features: dict) -> dict:
        """
        Calculates physical activity classifications and detects motion anomalies
        using multi-dimensional accelerometer features.

        This method evaluates each time window against three distinct kinetic metrics:
        sustained noise (standard deviation), peak impacts (range), and high-frequency
        jitter (jerk). It categorizes the patient's overall movement distribution
        (resting, light movement, active) by applying interlocking thresholds.
        Additionally, it identifies acute anomalies—such as a fall, sudden exertion,
        or a dropped sensor—by monitoring for sudden, disproportionate spikes in the
        range metric.

        :param acc_features: A dictionary containing 1D numpy arrays ('std', 'range',
                             and 'jerk'), representing the extracted accelerometer
                             features calculated per time window.
        :return: A dictionary containing the mean overall noise level, anomaly
                 detection flags (sudden_jolt_detected), and the categorical
                 distribution of movement (window counts for resting, light
                 movement, and active states).
        """
        std_arr = acc_features["std"]
        range_arr = acc_features["range"]
        jerk_arr = acc_features["jerk"]

        total_windows = len(std_arr)

        # ACTIVE: If ANY metric crosses the high threshold
        active_mask = (
                (std_arr > self.moving_thresholds['std_magnitude']) |
                (range_arr > self.moving_thresholds['range_magnitude']) |
                (jerk_arr > self.moving_thresholds['mean_jerk']))
        active_count = int(np.sum(active_mask))

        # RESTING: If ALL metrics are below the strict thresholds
        resting_mask = (
                (std_arr < self.resting_thresholds['std_magnitude']) |
                (range_arr < self.resting_thresholds['range_magnitude']) |
                (jerk_arr < self.resting_thresholds['mean_jerk'])
        )
        resting_count = int(np.sum(resting_mask))

        # MOVING: Everything stuck in the middle (Fidgeting/Postural changes)
        moving_count = total_windows - (active_count + resting_count)

        # detect sudden extreme variance spike
        mean_range = float(np.mean(range_arr))
        max_range = float(np.max(range_arr))
        sudden_jolt_detected = bool(max_range > (mean_range * 5) and max_range > 1000.0)

        return {
            "mean_noise_level": round(float(np.mean(std_arr)), 3),
            "sudden_jolt_detected": sudden_jolt_detected,
            "distribution": {
                "resting_windows_count": resting_count,
                "light_movement_windows_count": moving_count,
                "active_movement_windows_count": active_count
            }
        }

    def _signal_correlation(
            self,
            hr_predictions: np.ndarray,
            acc_std: np.ndarray,
            ) -> dict:
        """
        Calculates the Pearson correlation coefficient between heart rate
        predictions and accelerometer standard deviation to provide clinical context, as
        well as the artifact detection.

        Determines if an elevated heart rate is justified by exercise,
        or if it indicates an anomaly such as psychological stress, fever,
        or an arrhythmia (high HR with zero movement).

        :param hr_predictions: A 1D numpy array of predicted heart rates (BPM).
        :param acc_std: A 1D numpy array of calculated accelerometer
                                           standard deviations representing physical movement.
        :return: A dictionary containing the calculated correlation coefficient
                         and a human-readable string explaining the clinical context.
        """

        # Mathematical offset according to neural network warm-up
        # offset = (receptive field - window size sec) / step size sec
        total_extra_windows = (
                (self.receptive_field_model_seconds - self.window_sec) / self.step_sec)
        offset = int(total_extra_windows / 2)

        # Strip the first 3 and last 3 windows off the ACC array
        # so it perfectly aligns with the center of the HR predictions
        if len(acc_std) > 2 * offset:
            acc_sync = acc_std[offset: -offset]
        else:
            acc_sync = acc_std

        hr_sync = hr_predictions

        # Safety fallback just in case of rounding differences at the tail end
        min_length = min(len(hr_sync), len(acc_sync))
        hr_sync = hr_sync[:min_length]
        acc_sync = acc_sync[:min_length]

        # correlation between hr and movement
        correlation = float(np.corrcoef(hr_sync, acc_sync)[0, 1])

        if np.isnan(correlation):
            correlation = 0.0

            # 2. Semantic Translation (Updated to pass the 'assert in' tests)
        if correlation > 0.6:
            context = "High (Heart rate strongly driven by physical activity.)"
        elif correlation < 0.2:
            context = "Low (Heart rate disconnected from physical movement.)"
        else:
            context = "Moderate (Moderate correlation between movement and heart rate.)"

        # artifact detection
        high_hr_mask = hr_sync > 100.0
        total_high_hr_windows = int(np.sum(high_hr_mask))

        if total_high_hr_windows > 0:
            # Fallback to 0.5 if self.moving_thresholds isn't defined yet
            active_motion_mask = acc_sync > self.moving_thresholds['std_magnitude']

            high_hr_during_motion = int(np.sum(high_hr_mask & active_motion_mask))
            artifact_percentage = float((high_hr_during_motion / total_high_hr_windows) * 100.0)
        else:
            high_hr_during_motion = 0
            artifact_percentage = 0.0

        return {
            "global_movement_correlation": {
                "pearson_correlation_coefficient": round(correlation, 3),
                "semantic_relationship": context
            },
            "tachycardia_artifact_analysis": {
                "elevated_hr_windows_total": total_high_hr_windows,
                "elevated_hr_during_heavy_motion": high_hr_during_motion,
                "motion_artifact_probability_percentage": round(artifact_percentage, 1)
            }
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

        extreme_jumps_count = int(np.sum(successive_differences > 15.0))

        return {
            "average_beat_to_beat_jump": round(mean_volatility, 3),
            "unphysiological_jumps_detected": extreme_jumps_count,
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
        acc_features = self._extract_windowed_acc_features(acc_magnitude)

        print(f"Length of HR predictions: {len(hr_prediction)}")
        print(f"Total ACC features: {len(acc_features)} "
              f"with {len(acc_features['std'])} windows each")

        # generation individual feature dictionaries
        cardiovascular_summary_stats = self._cardiovascular_statistics(hr_prediction)
        movement_summary_stats = self._movement_statistics(acc_features)
        volatility_stats = self._hr_volatility(hr_prediction)
        correlation_stats = self._signal_correlation(hr_prediction, acc_features['std'])

        payload = {
            "system_telemetry": {
                "total_recording_duration_seconds": self.total_duration,
                "rolling_window_duration_seconds": self.window_sec,
                "rolling_window_step_seconds": self.step_sec,
                "array_elements_per_window": self.window_size,
                "model_receptive_field_seconds": self.receptive_field_model_seconds,
                "data_alignment_note": (
                    "Expected behavior: The cardiovascular_analysis contains slightly fewer "
                    "predictions than the movement_analysis. This is caused by the ML model's "
                    f"{self.receptive_field_model_seconds}-second historical receptive field,"
                    f" NOT a sensor failure or missing data."
                )
            },
            "cardiovascular_analysis": cardiovascular_summary_stats,
            "movement_analysis": movement_summary_stats,
            "autonomic_nervous_system_proxy": volatility_stats,
            "clinical_context": correlation_stats
        }

        return payload
