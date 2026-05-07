"""
Aggregators Package.

This package contains modules responsible for post-processing and feature
engineering on machine learning predictions. It acts as the crucial bridge
between the mathematical signal processing pipelines and the cognitive
LLM reasoning pipelines.

Classes within this package transform high-frequency, raw prediction arrays
(such as continuous heart rate) into structured statistical summaries
(like windowed variances, maximums, and averages). This ensures the data
is compressed into a digestible, meaningful context that Large Language
Models can interpret without exceeding token limits or losing clinical fidelity.
"""

from .clinical_aggregator import ClinicalAggregator

__all__ = [
    "ClinicalAggregator"
]
