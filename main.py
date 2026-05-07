"""
Main Entry Point.

This script initializes the Clinical Insights Orchestrator and runs it
on a recurring schedule. It acts as the continuous monitoring engine
for the application.
"""

import time
import schedule

from src.orchestrator.pipeline_orchestrator import PipelineOrchestrator


def job():
    """
    Executes a single, complete run of the clinical insights architecture.

    This function initializes the master orchestrator, which cascades the
    configuration to the sub-pipelines, processes the raw sensor data,
    and generates the final clinical report. It includes timestamped
    console logging for monitoring execution timing and success.

    :raises Exception: Propagates any exceptions raised by the underlying
                       data loaders, mathematical models, or LLM APIs.
    """
    print("\n" + "=" * 50)
    print(f"Triggering Scheduled Run: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Initialize the orchestrator
    orchestrator = PipelineOrchestrator()

    # Run the full pipeline
    orchestrator.run_full_process()

    print("Run Complete. Waiting for next interval...\n")

def main():
    """
    Initializes the job scheduler and keeps the main execution thread alive.

    Sets up the recurring execution interval, forces an immediate initial
    run so the system does not idle during the first cycle, and then enters
    a lightweight infinite loop to poll the schedule queue. Gracefully handles
    keyboard interrupts (Ctrl+C) for safe shutdowns.
    """
    interval_seconds = 120
    print(f"Starting Clinical Insights Service. Interval: {interval_seconds}s")

    # Run the job immediately upon startup (so you don't have to wait 2 mins to see if it works)
    job()

    # Schedule the recurring job
    schedule.every(interval_seconds).seconds.do(job)

    # Keep the script running forever
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)  # Check every second if a job is due

    except KeyboardInterrupt:
        print("\nService stopped by user.")

if __name__ == "__main__":
    main()
