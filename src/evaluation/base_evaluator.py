"""
Base component for evaluating te response
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.pipelines import LLMInsightsPipeline
from src.schemas import ClinicalReportOutput


class BaseEvaluator(ABC):
    """
    Abstract Base Class for the Clinical Pipeline Evaluation Framework.

    This class defines the standard interface for systematic validation of
    the hybrid AI architecture. It is designed to facilitate the objective
    benchmarking of Large Language Models (e.g., Gemini 1.5 Flash vs. MedGemma 4B)
    by treating the LLMInsightsPipeline as a system-under-test.

    The evaluation framework is structured around four engineering pillars:
    1. Robustness (Decision Boundary Stability via perturbation).
    2. Reliability (Semantic Consistency via vector embeddings).
    3. Explainability (Inferential Integrity via LLM-as-a-Judge).
    4. Safety (Constraint Compliance/Diagnostic Leakage).

    Attributes:
        llm_pipeline (LLMInsightsPipeline): If needed, an instance of the LLM pipeline used
            to generate clinical reports.
    """
    def __init__(self, llm_pipeline: Optional[LLMInsightsPipeline] = None):
        if llm_pipeline:
            self.llm_pipeline = llm_pipeline

    @abstractmethod
    def evaluate(self,
                 reports: Optional[list[ClinicalReportOutput]],
                 payloads: Optional[list[dict]]
                 ) -> dict:
        """
        Executes the specific evaluation logic against a given clinical payload.

        This method must be overridden by specialized evaluators to implement
        deterministic checks, stochastic sampling, or judge-based audits.

        Args:
            reports (Optional[list[ClinicalReportOutput]]): If present, the structured outputs
                obtained from the pipeline representing a 120-second physiological window.
            payloads (Optional[list[dict]]): If present, the structured payloads relative to the
                given reports
        Returns:
            dict: A dictionary containing the quantified results of the evaluation.
                Common keys include 'score', 'raw_output', and 'metadata'
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
