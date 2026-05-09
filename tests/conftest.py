"""
Clinical Aggregator Test Fixtures and Data Loaders.

This module provides a centralized suite of `pytest` fixtures designed to support
both unit and integration testing for the `ClinicalAggregator` and related ML models
(e.g., BeliefPPG).

It establishes a standard baseline configuration and handles the dynamic loading
of real physical sensor data (Blood Volume Pulse and Accelerometer). The data
loaders intelligently slice the raw CSV files to perfectly match the expected
array lengths defined by the orchestrator's run interval and sampling frequencies.
"""

# pylint: disable=redefined-outer-name

from pathlib import Path

import pandas as pd
import pytest

from src.aggregators import ClinicalAggregator

# This finds the absolute root of your project dynamically
PROJECT_ROOT = Path(__file__).parent.parent
TEST_SAMPLES_DIR = PROJECT_ROOT / "data" / "test_samples"

@pytest.fixture
def base_config():
    """
    Provides a standard baseline configuration for testing across all files.

    Includes the base parameters, inference frequencies, and the 20-second
    receptive field required to accurately test the BeliefPPG model.
    """
    return {
        "params": {
            "seconds_per_window": 8,
            "step_size": 2,
            "receptive_field_model_seconds": 20
        },
        "inference": {"bvp_freq": 64, "acc_freq": 32},
        "orchestrator": {"run_interval_schedule": 120},
        "thresholds": {"resting": 0.05, "moving": 0.5}
    }


@pytest.fixture
def real_bvp_data(base_config):
    """
    Loads 120-seconds real Blood Volume Pulse (BVP) data for integration testing.

    Dynamically slices the physical CSV file into a 2D array exactly
    matching the length required by the orchestrator's scheduled interval.
    """
    run_interval = base_config["orchestrator"]["run_interval_schedule"]
    bvp_freq = base_config["inference"]["bvp_freq"]

    interval_recording_samples = run_interval * bvp_freq

    file_path = TEST_SAMPLES_DIR / "wrist_BVP.csv"

    if not file_path.exists():
        pytest.skip(f"Test data file not found at: {file_path}")


    data = pd.read_csv(file_path)
    # Returns a 2D array sliced perfectly to the required length (e.g., 7680 elements)
    return data.values[:interval_recording_samples, :1]


@pytest.fixture
def real_acc_data(base_config):
    """
    Loads real 120-seconds 3-axis Accelerometer (ACC) data for integration testing.

    Dynamically slices the physical CSV file into a 2D array exactly
    matching the length required by the orchestrator's scheduled interval.
    """
    run_interval = base_config["orchestrator"]["run_interval_schedule"]
    acc_freq = base_config["inference"]["acc_freq"]

    interval_recording_samples = run_interval * acc_freq

    file_path = TEST_SAMPLES_DIR / "wrist_ACC.csv"

    if not file_path.exists():
        pytest.skip(f"Test data file not found at: {file_path}")

    data = pd.read_csv(file_path)
    # Returns a 2D array sliced perfectly to the required length (e.g., 3840 elements)
    return data.values[:interval_recording_samples, :3]

@pytest.fixture
def aggregator(base_config):
    """
    Fixture to provide a reusable, pre-configured instance of the
    ClinicalAggregator for testing statistical methods.
    """
    return ClinicalAggregator(base_config)
