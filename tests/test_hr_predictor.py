"""
Heart Rate Predictor Integration Tests.

This test suite validates the `HRPredictor` wrapper class. It ensures that
the system strictly enforces data dimension requirements to prevent upstream
crashes, and it performs a true end-to-end integration test against the
underlying machine learning engine (BeliefPPG).

By intentionally avoiding mock objects and passing real physical sensor data
directly into the ML model, this suite mathematically proves that the model's
internal receptive field aligns perfectly with the pipeline's
expected sliding window architecture.
"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from src.models.hr_predictor import HRPredictor


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

# ==========================================
# TEST 1: Initialization
# ==========================================
def test_hr_predictor_initializes_with_correct_frequencies(predictor):
    """
    Verifies that the constructor correctly assigns the sampling frequencies
    passed from the configuration dictionary.
    """
    # --- ASSERT ---
    assert predictor.bvp_freq == 64
    assert predictor.acc_freq == 32


# ==========================================
# TEST 2: Validation Errors (1D Arrays)
# ==========================================
def test_predict_raises_value_error_if_bvp_is_not_2d(predictor):
    """
    Ensures the predictor strictly enforces a 2D shape for the BVP input,
    preventing dimension mismatch crashes deep inside the BeliefPPG neural network.
    """
    # --- 1. ARRANGE ---
    bvp_1d = np.array([1.0, 2.0, 3.0])       # Invalid 1D array
    acc_2d = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]) # Valid 2D array

    # --- 2 & 3. ACT & ASSERT ---
    with pytest.raises(ValueError, match="Expected ppg to be a 2D array"):
        predictor.predict(bvp_data=bvp_1d, acc_data=acc_2d)


def test_predict_raises_value_error_if_acc_is_not_2d(predictor):
    """
    Ensures the predictor strictly enforces a 2D shape for the ACC input.
    """
    # --- 1. ARRANGE ---
    bvp_2d = np.array([[1.0], [2.0], [3.0]]) # Valid 2D array
    acc_1d = np.array([1.0, 2.0, 3.0])       # Invalid 1D array

    # --- 2 & 3. ACT & ASSERT ---
    with pytest.raises(ValueError, match="Expected acc to be a 2D array"):
        predictor.predict(bvp_data=bvp_2d, acc_data=acc_1d)


# ==========================================
# TEST 3: True ML Inference (Real Data Integration)
# ==========================================
def test_predict_returns_correct_shape_with_real_data(
        predictor, base_config, real_bvp_data, real_acc_data
):
    """
    INTEGRATION TEST: Passes real physical 120-second sensor arrays directly
    into the BeliefPPG model.

    Verifies that the model's output array lengths strictly correspond to its
    internal historical receptive field (e.g., 20 seconds), mathematically
    proving the model behaves exactly as the ClinicalAggregator expects.
    """
    # --- 1. ARRANGE ---
    # Calculate exactly how many windows the model SHOULD return based on our math
    duration = base_config["orchestrator"]["run_interval_schedule"]
    step = base_config["params"]["step_size"]
    receptive_field = base_config["params"]["receptive_field_model_seconds"]

    # Expected sliding windows for a 120s file with a 20s warm-up (51 windows)
    expected_hr_windows = int((duration - receptive_field) / step) + 1

    # --- 2. ACT ---
    # Trigger the real ML model!
    pred_hr, idx = predictor.predict(bvp_data=real_bvp_data, acc_data=real_acc_data)

    # --- 3. ASSERT ---
    assert isinstance(pred_hr, np.ndarray)
    assert isinstance(idx, np.ndarray)

    assert pred_hr.ndim == 1, f"Expected 1D HR array, got {pred_hr.ndim}D"
    assert idx.ndim == 1, f"Expected 1D Index array, got {idx.ndim}D"

    # Verify BeliefPPG outputs exactly 51 windows
    assert len(pred_hr) == expected_hr_windows, (
        f"Length Mismatch! Expected {expected_hr_windows} HR windows, "
        f"but BeliefPPG returned {len(pred_hr)}."
    )
    assert len(idx) == expected_hr_windows, (
        f"Length Mismatch! Expected {expected_hr_windows} Indices, "
        f"but BeliefPPG returned {len(idx)}."
    )
