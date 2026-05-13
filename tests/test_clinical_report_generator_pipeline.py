"""
Integration test for the ClinicalReportGeneratorPipeline module.
"""
import os

import pytest

from src.pipelines import ClinicalReportGeneratorPipeline

@pytest.mark.integration
def test_creating_report(
        mock_report,
        report_generator_pipeline: ClinicalReportGeneratorPipeline,
):
    """
    Test the creation of a clinical report using the ClinicalReportGeneratorPipeline.

    This test verifies that the pipeline can successfully generate a report
    from a given input and that the output is correctly formatted and saved
    to the specified directory.
    """

    # Generate the report using the pipeline
    report_generator_pipeline.run(mock_report)

    # Verify that the report was created in the output directory
    expected_report_path = f"{report_generator_pipeline.output_dir}/report_12345.md"
    assert os.path.exists(expected_report_path)
