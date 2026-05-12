"""
LLM Pipeline Integration Tests.
"""

import pytest

from src.schemas import ClinicalReportOutput


@pytest.mark.integration
def test_final_orchestrator(
        aggregator,
        predicting_hr,
        real_acc_data,
        llm_pipeline_generator
):
    """
    End-to-End Integration Test for the Clinical Insights Pipeline.

    This test validates the final handoff between the physical data processing
    layer (ClinicalAggregator) and the cognitive layer (LLMInsightsPipeline).
    It passes mathematically aggregated sensor data directly into the live LLM
    API to ensure the model strictly adheres to the requested JSON schema.

    Because this test makes a live network call to the LLM provider (e.g., Gemini),
    it is marked with `@pytest.mark.integration` and should generally be excluded
    from fast, local unit test runs to prevent unnecessary API latency and costs.

    :param aggregator: An initialized instance of the ClinicalAggregator.
    :param predicting_hr: A tuple containing the predicted heart rate array and indices.
    :param real_acc_data: A 2D numpy array of physical accelerometer data.
    :param llm_pipeline_generator: An initialized LLM pipeline bound to the Pydantic schema.

    :raises AssertionError: If the LLM output fails to parse into the strict
                                `ClinicalReportOutput` Pydantic model, or if the
                                expected data types do not match.
    """
    # --- 1. ARRANGE ---
    predicted_hr, _ = predicting_hr

    clinical_payload = aggregator.aggregate(
        hr_prediction=predicted_hr,
        acc_array=real_acc_data
    )
    print(f"Aggregated data: \n{clinical_payload}")

    # --- 2. ACT ---
    result = llm_pipeline_generator.run(payload=clinical_payload)
    print(f"Data interpretation: \n"
          f"{result.model_dump_json(indent=2)}")

    # --- 3. ASSERT ---
    assert isinstance(result, ClinicalReportOutput)
    assert isinstance(result.primary_observation, str)
    assert isinstance(result.requires_attention, bool)
    assert isinstance(result.anomalies_detected, list)
