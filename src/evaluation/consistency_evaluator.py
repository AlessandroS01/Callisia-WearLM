"""
Module for evaluating the consistency between clinical reports.
"""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluation.base_evaluator import BaseEvaluator
from src.pipelines import LLMInsightsPipeline
from src.schemas import ClinicalReportOutput


class ConsistencyEvaluator(BaseEvaluator):
    """
    Evaluator to quantify the Stochastic Reliability of the clinical LLM pipeline.

    This module measures semantic consistency across multiple generations of the
    same physiological data window. It ensures that the model's 'Cognitive Trace'
    (internal_reasoning) remains logically stable and invariant to stochastic
    sampling (temperature) by projecting reasoning into a medical vector space.

    Attributes:
        model_embedding (SentenceTransformer): Medically fine-tuned transformer
            (embeddinggemma-300m-medical) used to project text into 1024-dimensional
            clinical embeddings.
    """

    def __init__(self, llm_pipeline: Optional[LLMInsightsPipeline] = None):
        """
        Initializes the evaluator with a pipeline and the medical embedding engine.

        Args:
            llm_pipeline (LLMInsightsPipeline): The pipeline instance under evaluation.
        """
        super().__init__(llm_pipeline)

        self.model_embedding = SentenceTransformer(
            "sentence-transformers/embeddinggemma-300m-medical"
        )

    def evaluate(self,
                 reports: list[ClinicalReportOutput],
                 payloads: Optional[list[dict]] = None
                 ) -> dict:
        """
        Calculates the mean pairwise cosine similarity across a batch of reports.

        This method extracts the internal reasoning from N iterations of the same
        input payload and calculates the unique pairwise similarities using the
        upper triangle of the cosine matrix to avoid self-similarity bias.

        Args:
            reports (list[ClinicalReportOutput]): A list of N generated reports
                derived from the same physiological input window.
            payloads (list[dict]): A list of N generated payloads derived from

        Returns:
            dict: A dictionary containing the 'consistency_score' [0.0 - 1.0],
                representing the average semantic alignment of the model's logic.

        Raises:
            ValueError: If the number of reports is less than 1.
        """
        if len(reports) > 1:
            reasoning_texts = [query.internal_reasoning for query in reports]
        else:
            raise ValueError(
                "Not enough reports given to calculate the consistency score"
            )

        embeddings = self.model_embedding.encode(reasoning_texts)

        cosine_matrix = cosine_similarity(embeddings)

        rows, cols = np.triu_indices(len(cosine_matrix), k=1)

        pairwise_values = cosine_matrix[rows, cols]

        return {
            "reasoning_list": reasoning_texts,
            "cosine_matrix": cosine_matrix,
            "consistency_score": float(np.mean(pairwise_values))
        }
