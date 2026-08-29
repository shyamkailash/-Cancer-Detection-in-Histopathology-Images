#!/usr/bin/env python
"""
Top-level CLI Script for running Federated Learning experiments (Baseline FedAvg & Privacy-Preserving DP).
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.training.federated_trainer import parse_args, run_federated_experiment


def main():
    args = parse_args()
    run_federated_experiment(args)


if __name__ == "__main__":
    main()

