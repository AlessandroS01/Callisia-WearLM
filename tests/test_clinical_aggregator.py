"""
Unit tests for the ClinicalAggregator module.

This test suite provides 100% coverage, including direct unit tests of
protected helper methods and integration tests of the public pipeline.
"""

# pylint: disable=redefined-outer-name, protected-access

import pytest
import numpy as np

from src.aggregators import ClinicalAggregator

@pytest.fixture
def base_config():
    """Provides a standard baseline configuration for testing."""
    return {
        "params": {"seconds_per_window": 8, "step_size": 2},
        "inference": {"bvp_freq": 64, "acc_freq": 32},
        "orchestrator": {"run_interval_schedule": 120},
        "thresholds": {"resting": 0.05, "moving": 0.5}
    }


@pytest.fixture
def aggregator(base_config):
    """Fixture to provide a reusable instance of the aggregator."""
    return ClinicalAggregator(base_config)


# ==========================================
# TEST 1: Cardiovascular Zones & Trends
# ==========================================
def test_cardio_stats_calculates_zones_correctly(aggregator):
    """
    Verifies that heart rate predictions are accurately binned into
    clinical zones (bradycardia, normal, tachycardia) based on standard
    medical thresholds (<60, 60-100, >100).
    """
    # --- 1. ARRANGE ---
    # Create a precise array: 2 under 60, 3 normal, 2 over 100
    hr_data = np.array([50.0, 55.0, 75.0, 80.0, 95.0, 110.0, 120.0])

    # --- 2. ACT ---
    result = aggregator._cardiovascular_statistics(hr_data)
    zones = result["beats_distribution"]

    # --- 3. ASSERT ---
    assert zones["under_60_bpm"] == 2
    assert zones["between_60_and_100_bpm"] == 3
    assert zones["over_100_bpm"] == 2


@pytest.mark.parametrize("start_hr, end_hr, expected_trend", [
    (60, 120, "rising rapidly"),  # Slope > 0.5
    (60, 75, "rising slightly"),  # 0.1 < Slope <= 0.5
    (120, 60, "falling rapidly"),  # Slope < -0.5
    (75, 60, "falling slightly"),  # -0.5 <= Slope < -0.1
    (75, 75, "stable"),  # Slope between -0.1 and 0.1
])
def test_cardio_stats_evaluates_all_trends(
        aggregator, base_config, start_hr, end_hr, expected_trend
):
    """
    Tests all linear regression branches to ensure the calculated slope
    correctly translates to the appropriate human-readable trend string.
    """
    # --- 1. ARRANGE ---
    # Dynamically calculate window sizes
    duration = base_config["orchestrator"]["run_interval_schedule"]
    step = base_config["params"]["step_size"]
    total_hr_windows = int(duration / step)

    # Create an array that perfectly matches the expected number of windows
    hr_data = np.linspace(start_hr, end_hr, total_hr_windows)

    # --- 2. ACT ---
    result = aggregator._cardiovascular_statistics(hr_data)

    # --- 3. ASSERT ---
    assert result["trend"] == expected_trend


# ==========================================
# TEST 2: Movement Statistics & Anomalies
# ==========================================
def test_movement_stats_categorizes_zones(aggregator):
    """
    Ensures movement variance arrays are correctly classified into resting,
    light movement, and active states based on configured thresholds.
    """
    # --- 1. ARRANGE ---
    # Create precise variance scores: 2 resting (<0.05), 2 light (0.05-0.5), 2 active (>0.5)
    variance_data = np.array([0.01, 0.04, 0.1, 0.4, 0.6, 1.2])

    # --- 2. ACT ---
    result = aggregator._movement_statistics(variance_data)
    dist = result["distribution"]

    # --- 3. ASSERT ---
    assert dist["resting_windows_count"] == 2
    assert dist["light_movement_windows_count"] == 2
    assert dist["active_movement_windows_count"] == 2
    assert result["sudden_jolt_detected"] is False


def test_movement_stats_detects_sudden_jolt(aggregator, base_config):
    """
    Validates the anomaly heuristic that detects sudden extreme variance spikes,
    simulating events like a patient fall or a dropped sensor.
    """
    # --- 1. ARRANGE ---
    duration = base_config["orchestrator"]["run_interval_schedule"]
    step = base_config["params"]["step_size"]
    total_hr_windows = int(duration / step)

    # Fill all windows except the last one with resting data, then add a massive spike
    variance_data = np.full(total_hr_windows - 1, 0.01)
    variance_data = np.append(variance_data, 5.0)

    # --- 2. ACT ---
    result = aggregator._movement_statistics(variance_data)

    # --- 3. ASSERT ---
    assert result["sudden_jolt_detected"] is True


# ==========================================
# TEST 3: Signal Correlation Branches
# ==========================================
def test_signal_correlation_high(aggregator):
    """
    Simulates physical exertion where heart rate rises synchronously
    with movement, ensuring a positive clinical context is generated.
    """
    # --- 1. ARRANGE ---
    hr = np.linspace(60, 120, 10)    # HR goes up
    acc = np.linspace(0.1, 1.0, 10)  # Movement goes up simultaneously

    # --- 2. ACT ---
    result = aggregator._signal_correlation(hr, acc)

    # --- 3. ASSERT ---
    assert "strongly driven by physical activity" in result["clinical_context"]


def test_signal_correlation_low(aggregator):
    """
    Simulates physiological stress (e.g., anxiety or fever) where heart rate
    is highly elevated but the patient is perfectly still.
    """
    # --- 1. ARRANGE ---
    hr = np.linspace(60, 120, 10)    # HR goes up
    acc = np.linspace(1.0, 0.1, 10)  # Movement goes down

    # --- 2. ACT ---
    result = aggregator._signal_correlation(hr, acc)

    # --- 3. ASSERT ---
    assert "disconnected from physical movement" in result["clinical_context"]


def test_signal_correlation_moderate(aggregator):
    """
    Tests the default fallback correlation branch (between 0.2 and 0.6) for
    inconsistent physiological signals.
    """
    # --- 1. ARRANGE ---
    hr = np.array([60, 65, 70, 75, 80])
    acc = np.array([0.1, 0.5, 0.2, 0.8, 0.3])

    # --- 2. ACT ---
    result = aggregator._signal_correlation(hr, acc)

    # --- 3. ASSERT ---
    assert "Moderate correlation" in result["clinical_context"]


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
def test_aggregate_upsamples_acc_when_slower(base_config):
    """
        Tests the end-to-end integration of the aggregate method, specifically
        ensuring that lower-frequency accelerometer data is properly upsampled
        to match the target inference rate without crashing.
        """
    # --- 1. ARRANGE ---
    aggregator = ClinicalAggregator(base_config)

    duration = base_config["orchestrator"]["run_interval_schedule"]
    step = base_config["params"]["step_size"]
    acc_freq = base_config["inference"]["acc_freq"]

    # Calculate exact dynamic array sizes
    total_hr_windows = int(duration / step)
    total_acc_samples = duration * acc_freq

    dummy_hr = np.full(total_hr_windows, 75.0)
    dummy_acc = np.asarray(np.random.uniform(low=-0.1, high=0.1, size=(total_acc_samples, 3)))

    # --- 2. ACT ---
    result = aggregator.aggregate(acc_array=dummy_acc, hr_prediction=dummy_hr)

    # --- 3. ASSERT ---
    assert isinstance(result, dict)
    assert "cardiovascular_analysis" in result
    assert result["cardiovascular_analysis"]["standard_statistics"]["mean_hr"] == 75.0
