"""
Orchestrators Package.

This package contains the master control modules that dictate the execution
flow of the clinical insights system. Orchestrators act as high-level managers;
they do not perform data manipulation themselves, but rather route configurations,
manage state, and handle the data handoffs between specialized sub-pipelines.
"""

from .pipeline_orchestrator import PipelineOrchestrator

__all__ = [
    "PipelineOrchestrator"
]
