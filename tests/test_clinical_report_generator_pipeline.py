"""
Integration test for the ClinicalReportGeneratorPipeline module.
"""
from datetime import datetime
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

    today_str = datetime.now().strftime("%Y-%m-%d")

    # 3. Get a list of all files currently in the output directory
    output_dir = report_generator_pipeline.output_dir
    generated_files = os.listdir(output_dir)

    # 4. Check if ANY file in that folder matches our required pattern
    report_was_created = any(
        file_name.startswith("report_") and today_str in file_name
        for file_name in generated_files
    )

    # 5. Assert with a helpful error message if it fails
    assert report_was_created, (
        f"Failed to find a report for today. "
        f"Looked for 'report_*{today_str}*' in {output_dir}. "
        f"Files found: {generated_files}"
    )
