"""
Module for evaluating the inferential integrity and safety guardrail compliance
of clinical telemetry reports.

This module implements the 'ClinicalAuditorEvaluator', which leverages a
high-reasoning Judge LLM to perform automated peer-review of generated insights.
It focuses on detecting logical decoupling (where reasoning drifts from math)
and diagnostic leakage (violations of safety guardrails).
"""
import json
from typing import Optional, cast

from langchain_core.prompts import ChatPromptTemplate

from src.evaluation.base_evaluator import BaseEvaluator
from src.models.llm_creation import choice_model
from src.pipelines import LLMInsightsPipeline
from src.prompts import CLINICAL_AUDITOR_EVALUATION_PROMPT
from src.schemas import ClinicalReportOutput, ClinicalAuditEvaluation


class ClinicalAuditorEvaluator(BaseEvaluator):
    """
    Unified Auditor for Inferential Integrity and Safety Compliance.

    This evaluator implements an 'LLM-as-a-Judge' architecture to quantify the
    reliability of the clinical pipeline. It treats the generated report as a
    stochastic hypothesis and validates it against the deterministic 'ground truth'
    provided by the ClinicalAggregator.

    The audit assesses two primary engineering pillars:
    1. Inferential Integrity: Verifies that the internal reasoning field logically
       entails the final clinical flags based on the input telemetry.
    2. Safety Compliance: Scans for regulatory breaches, specifically the
       'No Diagnosis' and 'No Prescription' guardrails.
    """

    def __init__(self,
                 llm_pipeline: Optional[LLMInsightsPipeline] = None,
                 judge_config = None):
        """
        Initializes the auditor with a specific judge configuration and factory settings.

        Args:
            llm_pipeline (LLMInsightsPipeline, optional): The pipeline instance under test.
            judge_config: Configuration for the Judge LLM.
                Supported keys:
                - 'provider' (str): provider of the model
                - 'model_name' (str): Target model (e.g., "gemini-3.1-flash-lite").
        """
        super().__init__(llm_pipeline)

        if judge_config is None:
            judge_config = {}

        raw_judge_model = self._choice_judge_model(
            model_provider=judge_config.get("provider", "passau"),
            model_name=judge_config.get(
                "model_name", "qwen35-397b"
            ),
            temperature=judge_config.get("temperature", 0.1)
        )

        self.judge = raw_judge_model.with_structured_output(ClinicalAuditEvaluation)

        self.chat_template = ChatPromptTemplate.from_messages([
            ("system", CLINICAL_AUDITOR_EVALUATION_PROMPT),
            ("user", (
                "### INPUT 1: DETERMINISTIC TELEMETRY PAYLOAD\n"
                "```json\n{payload}\n```\n\n"
                "### INPUT 2: GENERATED CLINICAL REPORT\n"
                "```json\n{report}\n```\n\n"
                "Please execute the clinical audit now based on the grading anchors provided."
            ))
        ])

        self.chain = self.chat_template | self.judge


    def _choice_judge_model(
            self,
            model_provider: str,
            model_name: str,
            temperature: float = 0.0
    ):
        """
        Factory method to initialize the appropriate LLM provider.

        Ensures environment variables are loaded and sets temperature to 0.0
        to ensure deterministic auditing results.

        Args:
            model_provider (str): The cloud provider (google/openai).
            model_name (str): The specific model identifier.
            temperature (float): The temperature to set for deterministic audit. Default to 0.0

        Returns:
            A Chat-capable LLM instance.
        """

        return choice_model(
            model_provider=model_provider,
            model_name=model_name,
            temperature=temperature,
        )

    def _pipeline(self, payload: dict, report: ClinicalReportOutput) -> ClinicalAuditEvaluation:
        """
        Executes the internal audit logic.

        Serializes the payload and report before invoking the Judge LLM to
        calculate logic and safety scores.

        Args:
            payload (dict): The raw statistical data from the aggregator.
            report (ClinicalReportOutput): The generated LLM report object.

        Returns:
            ClinicalAuditEvaluation: A structured object containing scores and critiques.
        """
        formatted_payload = json.dumps(payload, indent=2)
        formatted_report = json.dumps(report, indent=2)

        evaluation_report = cast(
            ClinicalAuditEvaluation,
            self.chain.invoke({
                "payload": formatted_payload,
                "report": formatted_report
            })
        )

        return evaluation_report


    def evaluate(self,
                 reports: list[ClinicalReportOutput],
                 payloads: list[dict]
                 ) -> list[str]:
        """
        Entry point for the evaluation suite.

        Processes a paired list of generated reports and raw payloads to produce
        a serialized audit result.

        Args:
            reports (list[ClinicalReportOutput]): Batch of reports to be audited.
            payloads (list[dict]): Corresponding batch of raw aggregator payloads.

        Returns:
            list[str]: A list of JSON-formatted strings of the ClinicalAuditEvaluation,
                 including logic scores, safety scores, and clinical validity.

        Raises:
            ValueError: If the input lists are empty or mismatched.
        """
        if len(reports) == 0 or len(payloads) == 0:
            raise ValueError(
                "At least one report and one payload must be provided for evaluation."
            )
        if len(reports) != len(payloads):
            raise ValueError(
                "Provide the same amount of reports and payloads for evaluation."
            )

        results = []

        for item in zip(payloads, reports):
            payload = item[0]
            report = item[1]

            print(50*"-")
            print(f"Evaluating report: \n{report}\nfor the payload:\n{payload}\n\n\n")

            audit_result = self._pipeline(
                payload=payload,
                report=report
            )

            result = audit_result.model_dump_json(indent=2)

            print(f"Result: \n{result}\n")

            results.append(result)

        return results
