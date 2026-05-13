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
from src.models.hr_predictor import HRPredictor
from src.pipelines import LLMInsightsPipeline, ClinicalReportGeneratorPipeline
from src.schemas import ClinicalReportOutput

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
        "thresholds": {"resting": 0.05, "moving": 0.5},
        "llm": {"model_name": "gemini-3.1-flash-lite", "temperature": 0.1}
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

    file_path = TEST_SAMPLES_DIR / "BVP.csv"

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

    file_path = TEST_SAMPLES_DIR / "ACC.csv"

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

@pytest.fixture
def predictor(base_config):
    """
    Fixture to provide a reusable instance of the HRPredictor.

    Dynamically loads the required sampling frequencies (e.g., 64Hz for BVP,
    32Hz for ACC) directly from the centralized test configuration.
    """
    bvp_freq = base_config["inference"]["bvp_freq"]
    acc_freq = base_config["inference"]["acc_freq"]
    return HRPredictor(bvp_freq=bvp_freq, acc_freq=acc_freq)

@pytest.fixture
def predicting_hr(
        predictor,
        real_acc_data,
        real_bvp_data
):
    """
    Fixture to provide a reusable and truthful instance of the HR prediction output.
    """
    return predictor.predict(
        bvp_data= real_bvp_data,
        acc_data= real_acc_data
    )

@pytest.fixture
def llm_pipeline_generator(base_config) -> LLMInsightsPipeline:
    """
    Fixture to provide a reusable, pre-configured instance of the
    LLMInsightsPipeline for testing statistical methods.
    """
    return LLMInsightsPipeline(base_config)

@pytest.fixture
def report_generator_pipeline(base_config) -> ClinicalReportGeneratorPipeline:
    """
    Fixture to provide a reusable, pre-configured instance of the
    ClinicalReportGeneratorPipeline for testing statistical methods.
    """
    return ClinicalReportGeneratorPipeline(base_config)


@pytest.fixture
def mock_report() -> ClinicalReportOutput:
    """
    Mocks the creation of a ClinicalReportOutput instance from a validated json.
    :return:
    """
    raw_json_string = """{
      "internal_reasoning": "The patient exhibits a mean heart rate of 115.8 bpm with a high frequency of beats over 100 bpm (45 windows). Movement analysis shows 55 resting windows and only 2 light movement windows, indicating a state of physical inactivity. The correlation between HR and movement is low (0.349), suggesting that the elevated heart rate is not primarily driven by physical exertion. The HR trend is falling, but remains in a tachycardic range relative to resting state.",
      "primary_observation": "The patient exhibits sustained elevated heart rate despite a lack of physical activity, with a low correlation between cardiovascular output and movement.",
      "cardiovascular_state": "The heart rate is consistently elevated, with a mean of 115.8 bpm and a median of 119.4 bpm. The majority of the 120-second window (45/51 windows) shows heart rates exceeding 100 bpm, though the trend is currently falling.",
      "autonomic_tone": "The average beat-to-beat jump of 1.974 indicates moderate volatility, suggesting a lack of autonomic rigidity but a persistent sympathetic influence.",
      "movement_context": "The patient is primarily in a resting state, with 55 resting windows and no active movement detected. The elevated heart rate is decoupled from physical exertion.",
      "anomalies_detected": [
        "Sustained tachycardia during resting state",
        "Low correlation between heart rate and physical activity"
      ],
      "requires_attention": true,
      "recommended_system_action": "Monitor for sustained tachycardia and verify patient's subjective comfort level; ensure sensor is properly calibrated to rule out motion artifact.",
      "technical_notes": "Data alignment is consistent with the 20-second model receptive field; no sensor failure detected."
    }"""

    report_instance = ClinicalReportOutput.model_validate_json(raw_json_string)

    return report_instance
