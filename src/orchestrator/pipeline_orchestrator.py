"""
System Orchestrator Module.

This module provides the :class:`PipelineOrchestrator` class, which serves as
the master control program for the entire clinical insights architecture.
It is responsible for loading the global configuration file exactly once and
managing the execution flow between the heavy-compute signal processing
pipeline and the cognitive LLM reasoning pipeline.

Typical usage example:

    from src.orchestrators.system_orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(config_path="config.yaml")
    orchestrator.run_full_process()
"""
import json
import os
import time

from src.aggregators import ClinicalAggregator
from src.pipelines import (SignalProcessingPipeline,
                           LLMInsightsPipeline,
                           ClinicalReportGeneratorPipeline)


class PipelineOrchestrator:
    """
    Manages the end-to-end execution of the physiological data processing
    and clinical insights generation system.

    This class instantiates all necessary sub-pipelines using a single,
    centralized configuration dictionary to ensure data consistency
    across all stages of execution.

    :ivar config: The parsed configuration dictionary loaded from the YAML file.
    :vartype config: dict
    :ivar signal_pipeline: The pipeline responsible for sensor data ingestion
                           and heart rate inference.
    :vartype signal_pipeline: SignalProcessingPipeline
    :ivar llm_pipeline: The pipeline responsible for aggregating predictions
                        and generating clinical text.
    :vartype llm_pipeline: LLMInsightsPipeline
    """
    def __init__(self, config: dict):
        """
        Initializes the orchestrator by loading the YAML config and
        instantiating the sub-pipelines.

        :param config: The dict containing all configuration parameters.
        """
        # initialize config
        self.config = config

        # initialize pipelines
        self.signal_pipeline = SignalProcessingPipeline(config=self.config)
        self.llm_pipeline = LLMInsightsPipeline(config=self.config)
        self.report_generator = ClinicalReportGeneratorPipeline(config=self.config)

        # initialize aggregator
        self.aggregator = ClinicalAggregator(config=self.config)

    def run_full_process(self, timestamp):
        """
        Executes the complete system workflow.

        Currently, this method triggers the signal processing pipeline.
        It is designed to be expanded to handle the data handoff between
        the signal pipeline outputs and the LLM pipeline inputs.
        """

        # to change according to the database requirement
        patient_id = self.config["inference"]["patient_id"]

        # Step 1: Run the inference pipeline to process raw sensor data and generate predictions
        hr, _, _, acc = self.signal_pipeline.run(patient_id=patient_id, timestamp=timestamp)

        # Step 2: Run the aggregation
        print(f"[{patient_id}] Aggregating sensor features...")
        clinical_payload = self.aggregator.aggregate(hr_prediction=hr, acc_array=acc)

        print(f"[{patient_id}] LLM payload: \n{clinical_payload}")


        start_time = time.perf_counter()
        # Step 2: LLM feed to interpret the data
        data_interpretation = self.llm_pipeline.run(payload=clinical_payload)
        elapsed = time.perf_counter() - start_time

        print(f"[{patient_id}] Data interpretation: \n"
              f"{data_interpretation.model_dump_json(indent=2)}")

        folder_path = f"payloads/ollama/gemma-4/{timestamp}"
        os.makedirs(folder_path)
        with open(f"{folder_path}/payload.json", 'w', encoding="utf-8") as f:
            json.dump(clinical_payload, f, indent=2)
        with open(f"{folder_path}/interpretation.json", 'w', encoding="utf-8") as f:
            # Using model_dump_json ensures Pydantic objects are serialized correctly
            f.write(data_interpretation.model_dump_json(indent=2))
        metadata = {
            "execution_time_sec": round(elapsed, 4),
            "timestamp": timestamp
        }
        with open(f"{folder_path}/metadata.json", 'w', encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Step 3: Report generation
        print(f"[{patient_id}] Generating report...")

        output_path = self.report_generator.run(
            clinical_report=data_interpretation
        )

        print(f"[{patient_id}] Report: \n{output_path}")
