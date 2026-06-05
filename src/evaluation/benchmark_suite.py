"""
Orchestration module for the Clinical LLM Evaluation Framework.

This module implements the BenchmarkSuite, a centralized controller designed to
quantify the performance of various LLM providers (e.g., Groq, Ollama, Mistral)
across three engineering pillars:
1. Reliability & Consistency: Stochastic stability of clinical reasoning.
2. Clinical Integrity: Auditor-driven safety and logic validation.
3. Decision Robustness: Perturbation analysis of decision boundaries.

The suite automates the retrieval of clinical artifacts (payloads, reports, and metadata)
and manages the lifecycle of the 'LLM-as-a-Judge' evaluation pipeline.
"""

import json
import time
from pathlib import Path
from random import randint

import numpy as np

from src.config_loader import load_config
from src.evaluation.clinical_auditor_evaluator import ClinicalAuditorEvaluator
from src.evaluation.consistency_evaluator import ConsistencyEvaluator
from src.evaluation.robustness_evaluator import RobustnessEvaluator
from src.pipelines import LLMInsightsPipeline


class BenchmarkSuite:
    """
    Centralized orchestrator for multi-provider clinical LLM benchmarking.

    The BenchmarkSuite manages the execution of specialized evaluators against
    standardized clinical datasets. It facilitates the comparison of 'Student'
    models (inference providers) against a 'Teacher' model (The Auditor) to
    identify logic drifts and safety violations in generated clinical reports.

    Attributes:
        config (dict): Global configuration including model parameters and judge settings.
        llm_pipeline (LLMInsightsPipeline): The production pipeline instance under test.
        clinical_auditor_eval (ClinicalAuditorEvaluator): The logic and safety auditor.
        consistency_eval (ConsistencyEvaluator): The semantic similarity engine.
         robustness_eval (RobustnessEvaluator): The perturbation/dial test engine.
        providers (list): Target inference providers (e.g., 'groq', 'ollama').
    """
    def __init__(self):
        """
        Initializes the benchmark environment and dependency injection for evaluators.
        """

        # Load the configuration once and initialize the LLM pipeline and evaluators with it
        self.config = load_config("config.yaml")
        self.llm_pipeline = LLMInsightsPipeline(config=self.config)

        # Initialize all evaluators with the same LLM pipeline and their specific configurations
        self.clinical_auditor_eval = ClinicalAuditorEvaluator(
            llm_pipeline=self.llm_pipeline,
            judge_config=self.config.get("evaluation")
        )
        self.consistency_eval = ConsistencyEvaluator(
            llm_pipeline=self.llm_pipeline,
        )
        self.robustness_eval = RobustnessEvaluator(
            llm_pipeline=self.llm_pipeline
        )

        # List of providers to evaluate
        self.providers = ["groq", "ollama", "mistral"]

    def _retrieve_data_provider(self, provider: str) -> dict:
        """
        Crawls the local artifact repository to reconstruct provider-specific data.

        Organizes data into a hierarchical structure mapping models to their
        respective timestamped clinical sessions.

        Args:
            provider (str): The name of the provider subdirectory to scan.

        Returns:
            dict: A nested mapping: {model_name: {timestamp: {interpretation/metadata/payload}}}.
        """

        base_dir = Path(f"payloads/{provider}")

        final_structure = {}

        for timestamp_path in base_dir.glob("*/*"):
            if not timestamp_path.is_dir():
                continue

            # 1. Grab the names directly from the path
            model_name = timestamp_path.parent.name
            folder_name = timestamp_path.name  # This is your timestamp string

            # 2. Ensure the model key exists
            if model_name not in final_structure:
                final_structure[model_name] = {}

            # 3. Load the specific clinical files
            content = {}
            for key in ["interpretation", "metadata", "payload"]:
                file_path = timestamp_path / f"{key}.json"

                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content[key] = json.load(f)
                else:
                    content[key] = {}

            # 4. Nest it: {model: {timestamp: {data}}}
            final_structure[model_name][folder_name] = content

        return final_structure

    def _retrieve_payloads(self, full_data):
        """
        Extracts and flattens all raw telemetry payloads from the provider dataset.

         Iterates through the hierarchical model/timestamp structure to isolate the
        deterministic statistical data used as the 'Ground Truth' for evaluation.

        Args:
            full_data (dict): The nested dictionary returned by _retrieve_data_provider.

        Returns:
            list[dict]: A flattened list of clinical telemetry payloads.
        """
        data = full_data
        return [
            folder_data["payload"]
            for model_contents in data.values()
            for folder_data in model_contents.values()
            if "payload" in folder_data
        ]

    def _retrieve_reports(self, full_data):
        """
        Extracts and flattens all AI-generated interpretations from the provider dataset.

        Isolates the clinical reports (stochastic hypotheses) generated by the
        target inference models for auditing and consistency checks.

        Args:
            full_data (dict): The nested dictionary returned by _retrieve_data_provider.

        Returns:
        A flattened list of generated clinical reports.
        """
        data = full_data
        return [
            folder_data["interpretation"]
            for model_contents in data.values()
            for folder_data in model_contents.values()
            if "interpretation" in folder_data
        ]

    def _retrieve_metadata(self, full_data):
        """
        Extracts technical and clinical metadata for each benchmark session.

        Retrieves context such as model parameters, prompt versions, or
        patient identifiers associated with each inference run.

        Args:
            full_data (dict): The nested dictionary returned by _retrieve_data_provider.

        Returns:
             list[dict]: A flattened list of session metadata objects.
        """
        data = full_data
        return [
            folder_data["metadata"]
            for model_contents in data.values()
            for folder_data in model_contents.values()
            if "metadata" in folder_data
        ]

    def _retrieve_model_name(self, full_data):
        """
        Identifies the distinct LLM models present in the provided dataset.

        Returns:
            list[str]: A list of unique model identifiers (the top-level keys).
        """
        data = full_data
        return list(data.keys())

    def run_consistency_evaluation(self, iterations = 10):
        """
        Executes stochastic trials to measure Functional Reliability and Semantic Consistency.

        This test identifies 'Inference Jitter' by running the same clinical
        payload through the pipeline multiple times. It quantifies:
        1. Functional Reliability: The API success rate (HTTP 200 vs Errors).
        2. Semantic Consistency: The cosine similarity of the 'Internal Reasoning'
            fields across successful runs.

        Args:
            iterations (int): Number of stochastic trials to perform.
        """
        reports = []
        provider = self.providers[2]

        provider_data = self._retrieve_data_provider(provider)
        payloads = self._retrieve_payloads(provider_data)

        index = randint(0, len(payloads) - 1)
        print(f"Chosen random index {index} for consistency evaluation")
        payload = payloads[index]

        print("="*50)
        print(f"Starting consistency evaluation on {payload} with 10 trials using {provider}")
        print("="*50)

        success_count = 0
        failure_count = 0

        for i in range(0, iterations):
            print(f"Run {i+1}/{iterations}")
            try:
                # If this fails (e.g., 400 error), it jumps to 'except'
                report = self.llm_pipeline.run(payload) # temperature to 0.7
                reports.append(report)
                success_count += 1
                print("SUCCESS")
            except Exception as e:
                failure_count += 1
                print(f"FAILED (Error: {type(e).__name__})")

        # Calculation of Reliability
        reliability = (success_count / iterations) * 100

        print("\n" + "=" * 50)
        print(f"RESULTS FOR PROVIDER: {provider}")
        print(f"Functional Reliability: {reliability:.2f}% ({success_count}/{iterations})")

        if success_count > 1:
            metrics = self.consistency_eval.evaluate(reports=reports)
            print(f"Semantic Consistency: \n{metrics}\n")
        else:
            print("Semantic Consistency: N/A (Not enough successful runs)")
        print("=" * 50)

    def run_clinical_audit_evaluation(self):
        """
        Orchestrates a batch audit of generated reports against deterministic ground truth.

        Utilizes the 'LLM-as-a-Judge' paradigm to perform a cross-reference audit
        between the raw statistical payloads and the generated narrative reports.
        It produces scores for logic alignment and safety guardrail compliance.
        """
        provider = self.providers[2]

        provider_data = self._retrieve_data_provider(provider)

        reports = self._retrieve_reports(provider_data)
        payloads = self._retrieve_payloads(provider_data)

        print("=" * 50)
        print(f"Retrieved {len(payloads)} payloads and {len(reports)} reports "
              f"from the provider:\n{provider}\n Starting clinical audit evaluation.")
        print("=" * 50)

        result = self.clinical_auditor_eval.evaluate(
            payloads=payloads,
            reports=reports,
        )

        print("=" * 50)
        print(f"RESULTS FOR PROVIDER: {provider}")
        print(f"Clinical Audit Evaluation: \n{result}\n")

    def run_robustness_evaluation(self):
        """
        Executes an adversarial perturbation sweep (The "Dial Test") using a baseline stress
        payload.

        This method establishes the "Ultimate Stress Test" payload—specifically injecting a highly
        critical clinical anomaly consisting of severe resting tachycardia (Mean HR = 144.2 BPM,
        Resting Windows = 55) paired with a pristine baseline noise environment (0.0% artifact
        probability). It forwards this payload to the `RobustnessEvaluator`, which
        programmatically alters the `motion_artifact_probability_percentage` vector from 0% to
        100% across multiple stochastic trials.

        This allows the benchmark to quantify the model's decision boundary stability,
        categorizing the target runtime framework into one of the four core behavior quadrants:
        robust (algorithmic obedience), fragile (clinical over-indexing), unstable
        (stochastic collapse), or failed. The comprehensive analysis metrics are output
        directly to the console.

        Raises:
            ValueError: If the underlying robustness evaluator or payload suite is uninitialized.
        """
        baseline_stress_payload = {
            "system_telemetry": {
                "total_recording_duration_seconds": 120,
                "rolling_window_duration_seconds": 8,
                "rolling_window_step_seconds": 2,
                "array_elements_per_window": 512,
                "model_receptive_field_seconds": 20,
                "data_alignment_note": (
                    "Expected behavior: The cardiovascular_analysis contains slightly fewer "
                    "predictions than the movement_analysis. This is caused by the ML model's "
                    "20-second historical receptive field."
                )
            },
            "cardiovascular_analysis": {
                "artifact_and_noise_context": {
                    "raw_ml_max_hr": 148.2,
                    "raw_ml_min_hr": 140.1,
                    "filtered_physiological_max_hr": 146.5,
                    "filtered_physiological_min_hr": 141.2,
                    "estimated_motion_artifact_deviation_bpm": 1.5
                },
                "physiological_statistics": {
                    "mean_hr": 144.2,
                    "standard_deviation_hr": 2.1,
                    "variance_hr": 4.4
                },
                "baseline_percentiles": {
                    "25th_percentile": 109.93,
                    "50th_percentile_median": 119.36,
                    "75th_percentile": 123.54
                },
                "trajectory_analysis": {
                    "semantic_trend_description": "Stable",
                    "mathematical_slope_bpm_per_minute": 0.05
                },
                "beats_distribution": {
                    "under_60_bpm": 0,
                    "between_60_and_100_bpm": 0,
                    "over_100_bpm": 51
                }
            },
            "movement_analysis": {
                "mean_noise_level": 0.013,
                "peak_variance": 0.079,
                "sudden_jolt_detected": False,
                "distribution": {
                    "resting_windows_count": 55,
                    "light_movement_windows_count": 2,
                    "active_movement_windows_count": 0
                }
            },
            "autonomic_nervous_system_proxy": {
                "average_beat_to_beat_jump": 0.53,
                "unphysiological_jumps_detected": 0
            },
            "clinical_context": {
                "global_movement_correlation": {
                    "pearson_correlation_coefficient": 0.3,
                    "semantic_relationship": "Moderate (Partial correlation "
                                             "between movement and heart rate.)"
                },
                "tachycardia_artifact_analysis": {
                    "elevated_hr_windows_total": 51,
                    "elevated_hr_during_heavy_motion": 0,
                    "motion_artifact_probability_percentage": 0.0
                }
            }
        }

        results = self.robustness_eval.evaluate(
            reports=None,
            payloads=[baseline_stress_payload]
        )

        print(f"Robustness Evaluation: "
              "\n"
              f"\n{results}")

    def run_timeline_evaluation(self):
        """
        Profiles and quantifies the mean computational execution latency across all active
        inference providers.

        This performance benchmark systematically evaluates the operational throughput of
        the pipeline architectures. It iterates through the collection of registered model
        providers (e.g., cloud-hosted engines via Groq vs. local runtime instances via Ollama),
        pulls the full metadata arrays for historical telemetry calls, and isolates the raw
        processing speeds (`execution_time_sec`).

        By calculating the arithmetic mean of these execution loops, it establishes the
        empirical framework needed to discuss the core trade-offs of the thesis: balancing
        logical accuracy against real-time, low-latency processing constraints for processing
        high-frequency health streams.

        Returns:
            dict[str, float]: A mapped collection where keys represent unique inference
                provider strings and values denote their calculated mean average execution
                latency in seconds.
        """
        mean_average_time = dict()

        for provider in self.providers:

            provider_data = self._retrieve_data_provider(provider)

            full_metadata = self._retrieve_metadata(provider_data)

            execution_times = [metadata['execution_time_sec'] for metadata in full_metadata]

            mean_average_time[provider] = np.mean(execution_times)

        print(f"Mean Average Execution Time per Provider: \n{mean_average_time}")
        return mean_average_time
