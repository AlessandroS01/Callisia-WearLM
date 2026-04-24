"""Module to test the first block of the model.

Takes BVP and ACC data and returns HR for a window of 8 seconds
and step size of 2 seconds
"""
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset.hr_dataset import HRDataset
from src.models.block_utils import setup_run_directory
from src.models.evaluation_artifacts import EvaluationArtifacts
from src.models.architecture.hr_cnn import MultimodalHRNet
from src.models.testing.block_1_data_loader import Block1TestingDataLoader


def load_config(config_path: str = "../../../config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return config


def create_config_file(run_dir: str, model_path: str) -> None:
    """
    Creates a config file in the run directory with model information.

    Params:
        run_dir: Directory where the config file will be saved
        model_path: Path to the model used for testing
    """
    config = {
        'model': model_path
    }

    config_path = os.path.join(run_dir, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✓ Config file created: {config_path}\n")


def load_test_data(patients_list):
    """Load test dataset and create data loader."""
    loader = Block1TestingDataLoader()
    x, y = loader.prepare_dataset(patients_list, 'testing')
    dataset = HRDataset(x, y)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"✓ Test batches: {len(test_loader)}\n")
    return test_loader


def load_model(device, model_path: str):
    """Load model from checkpoint."""
    print("Loading model from checkpoint...")
    model = MultimodalHRNet().to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded from: {model_path}\n")
        return model
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        return None


def run_inference(model, device, test_loader):
    """Run inference on test data."""
    print("="*70)
    print("RUNNING INFERENCE")
    print("="*70 + "\n")

    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)

            predictions.extend(outputs.squeeze().cpu().numpy().tolist())
            targets.extend(y_batch.squeeze().cpu().numpy().tolist())

            if (batch_idx + 1) % max(1, len(test_loader) // 5) == 0:
                print(f"  Processed batch [{batch_idx + 1}/{len(test_loader)}]")

    print(f"\n✓ Inference complete - {len(predictions)} samples processed\n")
    return np.array(predictions), np.array(targets)


def display_results(predictions_arr, targets_arr, test_metrics, run_dir):
    """Display and save test results."""
    EvaluationArtifacts.display_sample_predictions(predictions_arr, targets_arr, num_samples=10)

    print("\nGenerating test analysis plot...")
    EvaluationArtifacts.plot_test_results(predictions_arr, targets_arr, test_metrics,
                                         output_path=os.path.join(run_dir, "test_analysis.png"))

    EvaluationArtifacts.print_metrics_summary(test_metrics, test_metrics['num_samples'], run_dir)


def test():
    """
    Tests the trained Block 1 model on WESAD dataset.

    Loads the best model from the specified checkpoint and evaluates it
    on the test dataset, saving results to a run directory.
    """
    print("="*70)
    print("TESTING BLOCK 1 MODEL")
    print("="*70 + "\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}\n")

    # Load version from config
    config = load_config()
    version = config.get('version', '5th_version')

    # Setup run directory
    run_dir = setup_run_directory("history/block_1")

    # Define model path using version from config
    model_path = f"../../../models/block_1/{version}/run_002/best_model.pth"

    # Create config file in run directory with model info
    create_config_file(run_dir, model_path)

    # Load data
    print("Loading test data...")
    loader = Block1TestingDataLoader()
    patients_list = loader.get_patients()['testing_patients']
    test_loader = load_test_data(patients_list)

    # Load model
    model = load_model(device, model_path)
    if model is None:
        return

    # Run inference
    predictions_arr, targets_arr = run_inference(model, device, test_loader)

    # Calculate and save metrics
    print("\n" + "="*70)
    print("SAVING TEST RESULTS & ANALYSIS")
    print("="*70 + "\n")

    test_metrics = EvaluationArtifacts.calculate_metrics(predictions_arr, targets_arr)
    test_metrics['num_samples'] = len(predictions_arr)

    # Save test results CSV
    test_results_csv = os.path.join(run_dir, "test_results.csv")
    EvaluationArtifacts.save_test_results(
        predictions_arr.tolist(), targets_arr.tolist(), test_results_csv)

    if test_metrics:
        display_results(predictions_arr, targets_arr, test_metrics, run_dir)


if __name__ == "__main__":
    test()
