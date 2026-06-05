"""
Module to extract the configuration parameters from the YAML file and
return them as a dictionary.
"""

import yaml


def load_config(config_path: str) -> dict:
    """
    Reads and parses the YAML configuration file.

    :param config_path: The file path to the YAML configuration file.
    :return: A dictionary containing all configuration parameters.
    :raises FileNotFoundError: If the specified config file does not exist.
    """
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config