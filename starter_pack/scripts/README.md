# `scripts/`

This folder contains the execution scripts for running experiments, generating data, and visualizing results. These scripts utilize the core implementation in the `src/` directory.

## Core Experiment Runners

- **`run_core_experiments.py`**: Executes the primary suite of experiments across all datasets (Linear Gaussian, Moons, and Digits) using both Softmax and Neural Network models.
- **`run_digits_5seeds.py`**: Runs the Digits experiments five times with different random seeds to compute statistical confidence intervals.
- **`run_digits_optimizers.py`**: Compares the performance of different optimizers (SGD, Momentum, Adam) on the Digits dataset.
- **`run_moons_ablation.py`**: Performs an ablation study on the Moons dataset, typically varying the hidden layer width of the neural network.

## Data & Visualization

- **`generate_synthetic.py`**: Regenerates the synthetic datasets: `data/linear_gaussian.npz` and `data/moons.npz`.
- **`make_digits_split.py`**: Generates reproducible train/val/test split indices for the Digits dataset.
- **`visualize_data.py`**: Generates exploratory visualizations of the datasets used in the project.

## Model Training & Debugging

- **`train_softmax.py`**: A standalone script for training and evaluating a Softmax regression model.
- **`train_nn_runner.py`**: A standalone script for training and evaluating a Neural Network model.
- **`sanity_checks.py`**: Performs critical implementation checks, such as gradient checking and verifying basic model convergence on toy problems.

## Logging

Most scripts in this directory use `src.experiment_logger` to automatically save their terminal output to the `results/logs/` directory with a unique timestamp.
