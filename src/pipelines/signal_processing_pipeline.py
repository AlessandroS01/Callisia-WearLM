"""
Signal Processing Pipeline Module.

This module provides the :class:`SignalProcessingPipeline` class, which handles
the first stage of the clinical insights architecture. It orchestrates the
loading of raw physiological sensor data and the execution of the machine
learning model to predict continuous heart rate.
"""

from src.data.inference.data_loader import DataLoader
from src.models.hr_predictor import HRPredictor


class SignalProcessingPipeline:
    """
    Executes the signal processing and heart rate inference workflow.

    This pipeline sets up the necessary data loading and prediction tools
    upon initialization. It is designed to be reusable, allowing multiple
    patients to be processed sequentially by passing different patient IDs
    to the `run` method.

    :ivar base_path: The root directory for the dataset.
    :vartype base_path: str
    :ivar bvp_freq: Sampling frequency of the BVP sensor in Hz.
    :vartype bvp_freq: int or float
    :ivar acc_freq: Sampling frequency of the ACC sensor in Hz.
    :vartype acc_freq: int or float
    :ivar data_loader: The tool used to fetch raw sensor arrays.
    :vartype data_loader: DataLoader
    :ivar predictor: The tool used to infer heart rate from signals.
    :vartype predictor: HRPredictor
    """

    def __init__(self, config: dict):
        """
        Initializes the pipeline with configuration parameters and instantiates
        the required data loading and prediction tools.

        :param config: A dictionary containing 'base_path' and 'inference'
                               settings (sampling rates).
        """
        # Initialize config parameters
        self.base_path = config['base_path']

        self.bvp_freq = config['inference']['bvp_freq']
        self.acc_freq = config['inference']['acc_freq']

        # Initialize tools
        self.data_loader = DataLoader(base_path=self.base_path)
        self.predictor = HRPredictor(bvp_freq=self.bvp_freq, acc_freq=self.acc_freq)

    def run(self, patient_id: str) -> tuple:
        """
        Executes the pipeline for a specific patient.

        This method coordinates the loading of the patient's raw signal data
        and the subsequent prediction of their continuous heart rate.

        :param patient_id: The unique identifier for the patient (e.g., 'S1').
        :return: A tuple (hr, idxs) containing the predicted heart rate array
                 and the corresponding time indices.
        """
        print(f"[{patient_id}] Loading signals...")

        # Load the data using the initialized tool
        bvp, acc = self.data_loader.load_patient_signals(patient_id)

        # Predict Heart Rate
        print(f"[{patient_id}] Predicting heart rate...")
        hr, idxs = self.predictor.predict(bvp_data=bvp, acc_data=acc)

        return hr, idxs
