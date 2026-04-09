"""CSV helpers for saving features and metrics.

Provides a small helper `save_csv` used across the processing pipeline to
persist arrays or lists as single-column CSV files.
"""

import os

import pandas as pd


def save_csv(attribute: str, output_path: str, data):
    """Save the given data to a CSV file.

    Args:
        attribute: Attribute/name of the column and resulting file (without ext).
        output_path: Directory where the CSV will be written.
        data: Iterable or array-like data to be saved as a single column.
    """

    os.makedirs(output_path, exist_ok=True)
    pd.DataFrame(data, columns=[attribute]).to_csv(
        os.path.join(output_path, f"{attribute}.csv"), index=False
    )
    