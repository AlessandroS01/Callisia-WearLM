"""Module to test the first block of the model.

Takes BVP and ACC data and returns HR for a window of 8 seconds
and step size of 2 seconds
"""
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset.hr_dataset import HRDataset
from src.models.hr_cnn import MultimodalHRNet
from src.models.testing.block_1_data_loader import Block1TestingDataLoader
from src.models.block_utils import setup_run_directory, save_test_results
from src.models.evaluation_utils import display_metrics, display_sample_predictions, plot_test_results



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
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✓ Config file created: {config_path}\n")


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

    # Setup run directory
    run_dir = setup_run_directory("history/block_1")

    # Define model path
    model_path = "../../../models/block_1/3rd_version/run_001/best_model.pth"

    # Create config file in run directory with model info
    create_config_file(run_dir, model_path)

    # Load data
    print("Loading test data...")
    loader = Block1TestingDataLoader()
    patients_list = loader.get_patients()['testing_patients']

    x, y = loader.prepare_dataset(patients_list, 'testing')

    dataset = HRDataset(x, y)
    print(f"✓ Test dataset size: {len(dataset)}\n")

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"✓ Test batches: {len(test_loader)}\n")

    # Load model
    print("Loading model from checkpoint...")
    model = MultimodalHRNet().to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded from: {model_path}\n")
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Test the model
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

    predictions_arr = np.array(predictions)
    targets_arr = np.array(targets)

    # Calculate and save metrics
    test_metrics = save_test_results(predictions_arr, targets_arr, run_dir)

    if test_metrics:
        # Display metrics
        display_metrics(test_metrics, test_metrics['num_samples'])
        display_sample_predictions(predictions_arr, targets_arr, num_samples=10)

        # Generate comprehensive analysis plot and save
        print("\nGenerating test analysis plot...")
        plot_test_results(predictions_arr, targets_arr, test_metrics,
                         output_path=os.path.join(run_dir, "test_analysis.png"))

        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"  MAE: {test_metrics['mae']:.4f} bpm")
        print(f"  RMSE: {test_metrics['rmse']:.4f} bpm")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        print(f"  Samples: {test_metrics['num_samples']}")
        print(f"  Results saved to: {run_dir}")
        print("="*70 + "\n")


if __name__ == "__main__":
    test()
