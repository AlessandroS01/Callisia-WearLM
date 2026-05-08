"""
Pipelines Package.

This package contains the isolated pipeline modules responsible for executing
specific stages of the clinical insights architecture. It houses the
heavy-compute machine learning pipelines (Signal Processing), the cognitive
reasoning pipelines (LLM Insights) and the report generator (Clinical Report).

By keeping these pipelines decoupled, the system ensures that mathematical
inference and natural language generation can scale, fail, and be tested
independently.
"""
from .clinical_report_generator import ClinicalReportGenerator
from .llm_insights_pipeline import LLMInsightsPipeline
from .signal_processing_pipeline import SignalProcessingPipeline

# The __all__ list explicitly defines what gets imported when someone uses:
# from src.pipelines import *
__all__ = [
    "SignalProcessingPipeline",
    "LLMInsightsPipeline",
    "ClinicalReportGenerator"
]
