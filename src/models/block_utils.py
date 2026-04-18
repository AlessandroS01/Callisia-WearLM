"""
Block Utilities - Minimal utility functions.

Provides:
- setup_run_directory(): Create numbered run directories

IMPORTANT: This module has NO DOMAIN DEPENDENCIES.
Only standard library.
"""

import os



def setup_run_directory(base_path: str) -> str:
    """
    Create a numbered run directory (run_001, run_002, etc.).

    Args:
        base_path: Base path where run directories will be created

    Returns:
        str: Path to the created run directory
    """
    os.makedirs(base_path, exist_ok=True)

    # Find the next run number
    existing_runs = []
    if os.path.exists(base_path):
        for entry in os.listdir(base_path):
            if entry.startswith('run_') and os.path.isdir(os.path.join(base_path, entry)):
                try:
                    run_num = int(entry.split('_')[1])
                    existing_runs.append(run_num)
                except (IndexError, ValueError):
                    pass

    next_run_num = max(existing_runs) + 1 if existing_runs else 1
    run_dir = os.path.join(base_path, f"run_{next_run_num:03d}")
    os.makedirs(run_dir, exist_ok=True)

    return run_dir
