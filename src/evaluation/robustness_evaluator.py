"""
Module for quantifying decision boundary stability via perturbation analysis (The "Dial Test").

This module implements the RobustnessEvaluator, which systematically varies the
artifact probability context of a clinical telemetry payload. It measures the
stochastic jitter and logical pivot points of the model's 'requires_attention' flag.
"""
import copy
from typing import Optional

import numpy as np

from src.evaluation.base_evaluator import BaseEvaluator
from src.pipelines import LLMInsightsPipeline
from src.schemas import ClinicalReportOutput


class RobustnessEvaluator(BaseEvaluator):
    """
    Evaluator for Decision Boundary Stability (Robustness).

    This evaluator measures the reliability of the system's safety flags by
    perturbing the 'motion_artifact_probability_percentage' field. By iterating
    through a probability sweep and executing multiple trials per level, it
    identifies where the model transitions from a 'Clinical Alert' state to a
    'Noise Suppression' state.

    The evaluation framework is built on two engineering pillars:
    1. Logical Adherence: Does the model respect the 'Data Authority Override'
       and suppress alarms when sensor noise is high?
    2. Stochastic Stability: Is the decision boundary deterministic, or does
       the model exhibit 'jitter' (inconsistent flagging) at the pivot point?
    """

    def __init__(self, llm_pipeline: LLMInsightsPipeline):
        """
        Initializes the evaluator with the target inference pipeline.

        Args:
            llm_pipeline (LLMInsightsPipeline): The pipeline instance under evaluation.
        """
        super().__init__(llm_pipeline)

    def evaluate(self,
                 reports: Optional[list[ClinicalReportOutput]],
                 payloads: list[dict]
                 ) -> dict:
        """
        Performs a multi-trial perturbation sweep to map the decision boundary.

        This method 'dials' the noise probability from 0% to 100%. To properly
        measure robustness, the underlying LLM should be configured with a
        higher temperature (e.g., 0.7-0.8) during this specific test to expose
        any latent stochastic instability.

        Args:
            payloads (list[dict]): A list containing at least one clinical
                payload. The first payload is used as the baseline and should
                ideally represent a significant clinical event (e.g., high HR)
                to test the model's ability to suppress a 'True' flag.
            reports: Placeholder for signature consistency with BaseEvaluator.

        Returns:
            dict: A comprehensive statistical summary including:
                - verdict: The robustness classification (robust, fragile, unstable, failed).
                - pivot_threshold: The probability level where the alert flips.
                - mean_switching_variance: The average jitter across the sweep.
                - sweep_data: Raw metrics for every probability step.
        """
        if not payloads:
            raise ValueError("No payloads provided for the robustness evaluation.")

        if not self.llm_pipeline:
            raise ValueError("No pipeline provided for the robustness evaluation.")

        trials_per_level: int = 10
        steps: int = 10

        base_payload = payloads[0]
        probabilities = np.linspace(0, 100, steps + 1)
        sweep_results = []

        for prob in probabilities:
            level_flags = []

            # Execute 10 trials for each probability value to measure jitter
            for _ in range(trials_per_level):
                # Use deepcopy for nested clinical_context dictionary
                test_payload = copy.deepcopy(base_payload)
                test_payload["clinical_context"]["motion_artifact_probability_percentage"] \
                    = float(prob)

                # IMPORTANT: change "llm" parameters in config file, not "evaluation"
                # IMPORTANT: set the temperature to 0.7-0.8 to check jittering
                report: ClinicalReportOutput = self.llm_pipeline.run(test_payload)
                level_flags.append(1 if report.requires_attention else 0) # 1=True, 0=False

            sweep_results.append({
                # Noise level of the motion artifact probability percentage (e.g. 0%, 10%, etc..)
                "probability": float(prob),

                # Percentage of "True" flags at specific noise level
                # 1.0 (100%): Total agreement. The model is absolutely certain the patient
                #   needs attention, despite the noise.
                # 0.0 (0%): Total agreement. The model is certain the data is too noisy and
                #   suppresses the alarm.
                # 0.5 (50%): Maximum uncertainty. Indicates the model is struggling to weigh
                #   the clinical signal against the noise.
                "alert_density": float(np.mean(level_flags)),

                # Measures how spread out are the trial results from the mean
                # High variance indicates the model is "confused" at this noise level,
                #   as it flips between True and False flags across trials.
                "variance": float(np.var(level_flags)),

                "flags": level_flags
            })

        return self._calculate_metrics(sweep_results)

    def _calculate_metrics(self, sweep_results: list[dict]) -> dict:
        """
        Analyzes sweep data to categorize the model into one of four logic quadrants.

        Metrics:
            - Alert Density: The ratio of True/False flags. A density of 0.5
              represents the 'Sigmoid Midpoint' or maximum uncertainty.
            - Mean Switching Variance: The average variance (p(1-p)) across the
              sweep. High values indicate a 'fuzzy' or unstable boundary.
            - Pivot Threshold: The first noise level where Alert Density < 0.5.

        Quadrants:
            - Robust: Predictable decision (Stable) + Correct Noise Suppression (Logic).
            - Fragile: Predictable decision (Stable) + Ignores Noise (No Logic).
            - Unstable: Correct Noise Suppression (Logic) + Inconsistent (Not Stable).
            - Failed: Incorrect Noise Suppression (No Logic) + Inconsistent (Not Stable).
        """

        # Tells when the model changed its mind changing opinion from Alerting to Suppressing
        pivot_threshold = 100.0
        for result in sweep_results:
            if result["alert_density"] < 0.5:
                pivot_threshold = result["probability"]
                break

        # Measures stability by telling how cleanly the model changed opinion
        # 0% and 100% noise -> the model should be certain (var = 0.00)
        # max var = 0.25 -> model flips coin
        # min var = 0.00 -> model is certain
        # var < 0.1 -> model is grounded and not confused (uncertainty near pivot_threshold)
        # var > 0.1 -> model is unstable (uncertainty spread across many noise levels)
        mean_switching_variance = float(np.mean([r["variance"] for r in sweep_results]))

        is_stable = mean_switching_variance < 0.1
        has_correct_logic = pivot_threshold < 90.0

        # 4. Map to the Four Quadrants
        if is_stable and has_correct_logic:
            verdict = "robust" # A sharp, predictable cliff
        elif is_stable and not has_correct_logic:
            verdict = "fragile" # Consistent, but it ignores the noise.
        elif not is_stable and has_correct_logic:
            verdict = "unstable" # Tries to pivot, but pivot flickers for long time
        else:
            verdict = "failed" # no logic and no consistency

        return {
            "verdict": verdict,
            "pivot_threshold": pivot_threshold,
            "mean_switching_variance": mean_switching_variance,
            "is_stable": is_stable,
            "has_correct_logic": has_correct_logic,
            "sweep_data": sweep_results
        }
