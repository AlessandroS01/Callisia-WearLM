"""
Main Entry Point.

This script initializes the Clinical Insights Orchestrator and runs it
on a recurring schedule. It acts as the continuous monitoring engine
for the application.
"""

import time

from src.config_loader import load_config
from src.orchestrator.pipeline_orchestrator import PipelineOrchestrator


def job(config: dict, timestamp):
    """
    Executes a single, complete run of the clinical insights architecture.

    This function initializes the master orchestrator, which cascades the
    configuration to the sub-pipelines, processes the raw sensor data,
    and generates the final clinical report. It includes timestamped
    console logging for monitoring execution timing and success.

    :param config: The dict containing all configuration parameters.

    :raises Exception: Propagates any exceptions raised by the underlying
                       data loaders, mathematical models, or LLM APIs.
    """
    print("\n" + "=" * 50)
    print(f"Triggering Scheduled Run: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Initialize the orchestrator
    orchestrator = PipelineOrchestrator(config=config)

    # Run the full pipeline
    orchestrator.run_full_process(timestamp)

    print("Run Complete. Waiting for next interval...\n")

def main(timestamp):
    """
    Runs a single job with the provided timestamp.

    Executes the pipeline orchestrator once with the given timestamp.
    This allows external callers (like pipeline_test.py) to control
    timestamp increments and call main() multiple times.

    :param timestamp: The timestamp to use for this execution.
    """
    config = load_config('config.yaml')
    print(f"Starting Clinical Insights Service at timestamp: {timestamp}")
    job(config=config, timestamp=timestamp)

if __name__ == "__main__":
    # For direct execution, use current timestamp in milliseconds
    main(int(time.time() * 1000))
