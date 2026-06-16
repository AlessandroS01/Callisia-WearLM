"""
Module for testing the framework and making the evaluation
"""

from main import main
from src.evaluation.benchmark_suite import BenchmarkSuite


def main_test():
    """
    Test the framework
    """
    initial_timestamp = 1777828389232

    n_repetition = 0
    two_mins = 2 * 60 * 1000
    for _ in range(10):
        timestamp = initial_timestamp + n_repetition * two_mins

        print(50*"-")
        print("Starting timestamp:", timestamp)
        print(50 * "-")

        n_repetition += 1
        main(timestamp)


def main_eval():
    """
    Make the evaluation
    """
    suite = BenchmarkSuite()

    suite.run_timeline_evaluation()

if __name__ == "__main__":
    main_eval()
