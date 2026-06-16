"""
Module for evaluating the inter-model cross-consistency between clinical reports.
"""
import json
from typing import Optional, cast

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluation.base_evaluator import BaseEvaluator
from src.models.llm_creation import choice_model
from src.pipelines import LLMInsightsPipeline
from src.prompts import PAIRWISE_AUDITOR_EVALUATION_PROMPT
from src.schemas import ClinicalReportOutput, PairwiseAuditEvaluation


class CrossConsistencyEvaluator(BaseEvaluator):
    """
    Evaluator to quantify the Inter-Model Cross-Consistency of the clinical LLM pipeline.

    This module measures semantic, categorical, and logical consistency across two
    different models interpreting the same physiological data window. It ensures
    that the semantic meaning and clinical disposition are driven by the data
    rather than the specific architecture of the model.

    Attributes:
        model_embedding (SentenceTransformer): Medically fine-tuned transformer
            (embeddinggemma-300m-medical) used to project text into clinical embeddings.
    """

    def __init__(self,
                 llm_pipeline: Optional[LLMInsightsPipeline] = None,
                 judge_config= None):
        """
        Initializes the evaluator with a pipeline and the medical embedding engine.

        Args:
            llm_pipeline (LLMInsightsPipeline): The pipeline instance under evaluation.
            judge_config: Configuration for the Judge LLM.
                Supported keys:
                - 'provider' (str): provider of the model
                - 'model_name' (str): Target model (e.g., "gemini-3.1-flash-lite").
        """
        super().__init__(llm_pipeline)

        self.model_embedding = SentenceTransformer(
            "sentence-transformers/embeddinggemma-300m-medical"
        )

        raw_judge_model = self._choice_judge_model(
            model_provider=judge_config.get("provider", "passau"),
            model_name=judge_config.get("model_name", "qwen35-397b"),
            temperature=judge_config.get("temperature", 0.0)
        )

        # Enforces the structured JSON output schema for the pairwise audit
        self.judge = raw_judge_model.with_structured_output(PairwiseAuditEvaluation)

        self.chat_template = ChatPromptTemplate.from_messages([
            ("system", PAIRWISE_AUDITOR_EVALUATION_PROMPT),
            ("user", (
                "### INPUT 1: DETERMINISTIC TELEMETRY PAYLOAD\n"
                "```json\n{payload}\n```\n\n"
                "### INPUT 2: MODEL A OUTPUT\n"
                "```json\n{report_a}\n```\n\n"
                "### INPUT 3: MODEL B OUTPUT\n"
                "```json\n{report_b}\n```\n\n"
                "Please execute the pairwise clinical audit now."
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
        """
        return choice_model(
            model_provider=model_provider,
            model_name=model_name,
            temperature=temperature,
        )

    def _pipeline(self,
                  payload: dict,
                  report_a: ClinicalReportOutput,
                  report_b: ClinicalReportOutput) -> PairwiseAuditEvaluation:
        """
        Executes the internal pairwise audit logic.

        Serializes the payload and both reports before invoking the Judge LLM.

        Args:
            payload (dict): The raw statistical data from the aggregator.
            report_a (ClinicalReportOutput): The generated report from Model A.
            report_b (ClinicalReportOutput): The generated report from Model B.

        Returns:
            PairwiseAuditEvaluation: A structured object containing the comparative metrics.
        """
        formatted_payload = json.dumps(payload, indent=2)
        formatted_report_a = report_a.model_dump_json(indent=2)
        formatted_report_b = report_b.model_dump_json(indent=2)

        evaluation_report = cast(
            PairwiseAuditEvaluation,
            self.chain.invoke({
                "payload": formatted_payload,
                "report_a": formatted_report_a,
                "report_b": formatted_report_b
            })
        )

        return evaluation_report

    def evaluate(self,
                 reports_a: list[ClinicalReportOutput],
                 reports_b: list[ClinicalReportOutput],
                 payloads: Optional[list[dict]] = None
                 ) -> dict:
        """
        Calculates pairwise cross-consistency across a batch of paired reports.

        Args:
            reports_a (List[ClinicalReportOutput]): Generated reports from Model A.
            reports_b (List[ClinicalReportOutput]): Generated reports from Model B.
            payloads (Optional[List[dict]]): The original deterministic payloads.

        Returns:
            Dict[str, Any]: A dictionary containing semantic, categorical, and
                logical consistency metrics.

        Raises:
            ValueError: If the number of reports in A and B do not match.
        """
        if len(reports_a) != len(reports_b) or len(reports_a) == 0:
            raise ValueError(
                "Mismatched or empty report lists. Both models must evaluate "
                "the same number of payloads."
            )

        reports_a = [ClinicalReportOutput.model_validate(report) for report in reports_a]
        reports_b = [ClinicalReportOutput.model_validate(report) for report in reports_b]

        # 1. SEMANTIC LAYER (Cosine Similarity)
        print("Calculating semantic consistency via cosine similarity of internal reasoning...")
        reasoning_a = [report.internal_reasoning for report in reports_a]
        reasoning_b = [report.internal_reasoning for report in reports_b]

        embeddings_a = self.model_embedding.encode(reasoning_a)
        embeddings_b = self.model_embedding.encode(reasoning_b)

        semantic_similarities = [
            float(cosine_similarity([embeddings_a[i]], [embeddings_b[i]])[0][0])
            for i in range(len(reports_a))
        ]
        mean_semantic_score = float(np.mean(semantic_similarities))

        print(f"Mean Semantic Similarity: {mean_semantic_score:.4f}")

        # 2. CATEGORICAL LAYER (Boolean Agreement & Cohen's Kappa)
        print("Calculating categorical consistency via agreement rate and Cohen's Kappa...")
        attention_a = [report.requires_attention for report in reports_a]
        attention_b = [report.requires_attention for report in reports_b]

        agreement_rate = sum(a == b for a, b in zip(attention_a, attention_b)) / len(reports_a)

        try:
            # Force sklearn to recognize both boolean classes to prevent shape warnings
            kappa_score = float(cohen_kappa_score(attention_a, attention_b, labels=[False, True]))

            # If variance is zero (e.g., all 10 reports are 'False'), sklearn yields NaN.
            # In our clinical context, 100% agreement on a single state is perfect consistency.
            if np.isnan(kappa_score):
                kappa_score = 1.0 if agreement_rate == 1.0 else 0.0

        except Exception as e: # pylint: disable=broad-exception-caught
            kappa_score = float('nan')

        print(f"Kappa Score: {kappa_score:.4f}")

        # 3. LOGICAL LAYER (LLM-as-a-Judge)
        print("Calculating logical consistency via LLM-as-a-Judge pairwise audits...")
        judge_audits = []
        if payloads:
            for rep_a, rep_b, payload in zip(reports_a, reports_b, payloads):
                print(20*"=")
                print(f"Evaluating {rep_a} and {rep_b}")
                print(20*"=")
                # Execute the LangChain pipeline
                audit_result = self._pipeline(payload, rep_a, rep_b)
                # Dump the Pydantic model to a dict for the final JSON return
                judge_audits.append(audit_result.model_dump())
                print(f"Audit Result: {audit_result.model_dump_json(indent=2)}\n\n")

        return {
            "semantic_layer": {
                "mean_cosine_similarity": mean_semantic_score,
                "pairwise_similarities": semantic_similarities
            },
            "categorical_layer": {
                "agreement_rate": agreement_rate,
                "cohens_kappa": kappa_score
            },
            "logical_layer": {
                "judge_audits": judge_audits
            }
        }
