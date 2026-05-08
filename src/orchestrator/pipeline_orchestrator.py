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

from src.aggregators import ClinicalAggregator
from src.pipelines import SignalProcessingPipeline, LLMInsightsPipeline, ClinicalReportGenerator


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
        self.report_generator = ClinicalReportGenerator(config=self.config)

        # initialize aggregator
        self.aggregator = ClinicalAggregator(config=self.config)

    def run_full_process(self):
        """
        Executes the complete system workflow.

        Currently, this method triggers the signal processing pipeline.
        It is designed to be expanded to handle the data handoff between
        the signal pipeline outputs and the LLM pipeline inputs.
        """

        patient_id = self.config["inference"]["patient_id"]

        # Step 1: Run the inference pipeline to process raw sensor data and generate predictions.
        hr, _, bvp, acc = self.signal_pipeline.run(patient_id=patient_id)

        print(f"[{patient_id}] Aggregating sensor features...")
        llm_payload = self.aggregator.aggregate(hr_prediction=hr, acc_array=acc)

        print(f"[{patient_id}] LLM payload: \n{llm_payload}")

        # Step 2: Run the aggregation and LLM feed to interpret the data.
        self.llm_pipeline.run()
