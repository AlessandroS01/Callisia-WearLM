import os

import pandas as pd


def save_csv(attribute: str, output_path: str, data):
    """
        Save the given data to a csv file

        Args:
            attribute: Attribute of the file to be saved
            output_path: Directory of the file
            data: Data to be saved
    """
    os.makedirs(output_path, exist_ok=True)
    pd.DataFrame(data, columns=[attribute]).to_csv(
        os.path.join(output_path, f"{attribute}.csv"), index=False
    )