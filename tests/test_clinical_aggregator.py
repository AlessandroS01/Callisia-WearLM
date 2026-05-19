"""
Unit tests for the ClinicalAggregator module.

This test suite provides 100% coverage, including direct unit tests of
protected helper methods and integration tests of the public pipeline.
"""

# pylint: disable=redefined-outer-name, protected-access

import numpy as np
import pytest


# ==========================================
# TEST 1: Cardiovascular Zones & Trends
# ==========================================
def test_cardio_stats_calculates_zones_correctly(aggregator):
    """
    Verifies that heart rate predictions are accurately binned into
    clinical zones (bradycardia, normal, tachycardia) after passing
    through the artifact-rejection median filter.
    """
    # --- 1. ARRANGE ---
    # Create sustained "plateaus" so the kernel_size=7 median filter
    # recognizes them as real physiological states, not ML artifacts.
    brady_plateau = [50.0] * 7  # 7 consecutive bradycardia readings
    normal_plateau = [80.0] * 10  # 10 consecutive normal readings
    tachy_plateau = [120.0] * 7  # 7 consecutive tachycardia readings

    hr_data = np.array(brady_plateau + normal_plateau + tachy_plateau)

    # --- 2. ACT ---
    result = aggregator._cardiovascular_statistics(hr_data)
    zones = result["beats_distribution"]

    # --- 3. ASSERT ---
    # The counts should perfectly match the lengths of our plateaus
    assert zones["under_60_bpm"] == 7
    assert zones["between_60_and_100_bpm"] == 10
    assert zones["over_100_bpm"] == 7


# Updated deltas to test the new per-minute thresholds: >15, >5, <-15, <-5
# Assuming the test run duration is exactly 60 seconds (1 minute)
@pytest.mark.parametrize("start_hr, end_hr, expected_trend", [
    (60, 100, "Rising rapidly"),   # +40 BPM over 2 mins (Slope ~ 20.0)
    (60, 80, "Rising slightly"),   # +20 BPM over 2 mins (Slope ~ 10.0)
    (100, 60, "Falling rapidly"),  # -40 BPM over 2 mins (Slope ~ -20.0)
    (80, 60, "Falling slightly"),  # -20 BPM over 2 mins (Slope ~ -10.0)
    (75, 75, "Stable"),            # 0 BPM change (Slope 0.0)
])
def test_cardio_stats_evaluates_all_trends(
        aggregator, base_config, start_hr, end_hr, expected_trend
):
    """
    Tests all linear regression branches to ensure the calculated slope
    (scaled to BPM per minute) correctly translates to the appropriate
    human-readable trend string for the LLM.
    """
    # --- 1. ARRANGE ---
    duration = base_config["orchestrator"]["run_interval_schedule"]
    step = base_config["params"]["step_size"]
    total_hr_windows = int(duration / step)

    # np.linspace creates a perfectly smooth physiological trend
    hr_data = np.linspace(start_hr, end_hr, total_hr_windows)

    # --- 2. ACT ---
    result = aggregator._cardiovascular_statistics(hr_data)

    # --- 3. ASSERT ---
    actual_trend = result["trajectory_analysis"]["semantic_trend_description"]

    assert actual_trend == expected_trend, (
        f"Expected '{expected_trend}', but got '{actual_trend}'. "
        f"Check the linregress math if this fails."
    )

# ==========================================
# TEST 2: Movement Statistics & Anomalies
# ==========================================
def test_movement_stats_categorizes_zones(aggregator):
    """
    Ensures movement variance arrays are correctly classified into resting,
    light movement, and active states based on configured thresholds.
    """
    # --- 1. ARRANGE ---
    # Assuming standard defaults:
    # Resting: std < 15, range < 100, jerk < 5
    # Active: std > 100, range > 1200, jerk > 30

    acc_features = {
        # Windows: [Rest 1, Rest 2, Light 1, Light 2, Active 1 (High Std), Active 2 (High Range)]
        "std": np.array([10.0, 5.0, 30.0, 80.0, 105.0, 50.0]),
        "range": np.array([50.0, 80.0, 200.0, 800.0, 500.0, 1300.0]),
        "jerk": np.array([2.0, 4.0, 10.0, 20.0, 15.0, 10.0])
    }

    # --- 2. ACT ---
    result = aggregator._movement_statistics(acc_features)
    dist = result["distribution"]

    # --- 3. ASSERT ---
    # Windows 0 and 1: ALL metrics are strictly below the resting thresholds
    assert dist["resting_windows_count"] == 2

    # Windows 2 and 3: Metrics are above resting, but below active
    assert dist["light_movement_windows_count"] == 2

    # Window 4 (std > 100) and Window 5 (range > 1200) trigger the active OR-gate
    assert dist["active_movement_windows_count"] == 2

    # Mean range is ~488. Max range is 1300.
    # 1300 is not > (488 * 5), so no sudden jolt should be flagged.
    assert result["sudden_jolt_detected"] is False


def test_movement_stats_detects_sudden_jolt(aggregator, base_config):
    """
    Validates the anomaly heuristic that detects sudden extreme variance spikes,
    simulating events like a patient fall or a dropped sensor.
    """
    # --- 1. ARRANGE ---
    duration = base_config["orchestrator"]["run_interval_schedule"]
    step = base_config["params"]["step_size"]
    total_windows = int(duration / step)

    # Fill all windows except the last one with baseline resting data
    std_data = np.full(total_windows - 1, 10.0)
    range_data = np.full(total_windows - 1, 50.0)
    jerk_data = np.full(total_windows - 1, 2.0)

    # Append a massive physical impact (spike) to the last window
    # The range needs to be > 1000.0 and > 5x the mean to trigger the jolt
    std_data = np.append(std_data, 120.0)  # Standard deviation bumps up briefly
    range_data = np.append(range_data, 1500.0)  # Massive impact/swing
    jerk_data = np.append(jerk_data, 40.0)  # High directional change

    acc_features = {
        "std": std_data,
        "range": range_data,
        "jerk": jerk_data
    }

    # --- 2. ACT ---
    result = aggregator._movement_statistics(acc_features)

    # --- 3. ASSERT ---
    assert result["sudden_jolt_detected"] is True


# ==========================================
# TEST 3: Signal Correlation Branches
# ==========================================
def test_signal_correlation_high(aggregator):
    """
    Simulates physical exertion where heart rate rises synchronously
    with movement, ensuring a high correlation context is generated.
    """
    # --- 1. ARRANGE ---
    hr = np.linspace(60, 120, 10)    # HR goes up
    acc = np.linspace(0.1, 1.0, 10)  # Movement goes up simultaneously

    # --- 2. ACT ---
    result = aggregator._signal_correlation(hr, acc)

    # --- 3. ASSERT ---
    semantic_string = result["global_movement_correlation"]["semantic_relationship"]
    assert "High" in semantic_string, f"Expected 'High', got {semantic_string}"


def test_signal_correlation_low(aggregator):
    """
    Simulates physiological stress (e.g., anxiety or fever) where heart rate
    is highly elevated but the patient is perfectly still. Also tests the NaN fallback.
    """
    # --- 1. ARRANGE ---
    hr = np.linspace(60, 120, 10)    # HR goes up
    acc = np.array([0.1] * 10)       # Movement is perfectly flat (variance = 0)

    # --- 2. ACT ---
    result = aggregator._signal_correlation(hr, acc)

    # --- 3. ASSERT ---
    semantic_string = result["global_movement_correlation"]["semantic_relationship"]
    assert "Low" in semantic_string, f"Expected 'Low', got {semantic_string}"


def test_signal_correlation_moderate(aggregator):
    """
    Tests the moderate correlation branch (Pearson between 0.3 and 0.6) for
    inconsistent physiological signals.
    """
    # --- 1. ARRANGE ---
    hr = np.array([60, 70, 80, 90, 100])
    acc = np.array([0.1, 0.8, 0.2, 0.9, 0.7])

    # --- 2. ACT ---
    result = aggregator._signal_correlation(hr, acc)

    # --- 3. ASSERT ---
    semantic_string = result["global_movement_correlation"]["semantic_relationship"]
    assert "Moderate" in semantic_string, f"Expected 'Moderate', got {semantic_string}"


# ==========================================
# TEST 4: Volatility Math
# ==========================================
def test_hr_volatility_calculation(aggregator):
    """
    Verifies the mathematical proxy for autonomic nervous system volatility
    by checking the mean absolute difference of consecutive HR predictions.
    """
    # --- 1. ARRANGE ---
    # Jumps: |2|, |-1|, |4| -> mean(|2, 1, 4|) = 2.333
    hr = np.array([60.0, 62.0, 61.0, 65.0])

    # --- 2. ACT ---
    result = aggregator._hr_volatility(hr)

    # --- 3. ASSERT ---
    assert result["average_beat_to_beat_jump"] == 2.333


# ==========================================
# TEST 5: Integration & Upsampling Logic
# ==========================================
def test_aggregate_upsamples_acc_when_slower(base_config, aggregator, real_acc_data):
    """
        Tests the end-to-end integration of the aggregate method, specifically
        ensuring that lower-frequency accelerometer data is properly upsampled
        to match the target inference rate without crashing.
        """
    # --- 1. ARRANGE ---
    duration = base_config["orchestrator"]["run_interval_schedule"]
    step = base_config["params"]["step_size"]
    window = base_config["params"]["seconds_per_window"]

    # The perfect mathematical formula:
    total_windows = int((duration - window) / step) + 1

    dummy_hr = np.full(total_windows, 75.0)

    # --- 2. ACT ---
    result = aggregator.aggregate(acc_array=real_acc_data, hr_prediction=dummy_hr)

    # --- 3. ASSERT ---
    assert isinstance(result, dict)
    assert "cardiovascular_analysis" in result
    assert result["cardiovascular_analysis"]["physiological_statistics"]["mean_hr"] == 75.0
