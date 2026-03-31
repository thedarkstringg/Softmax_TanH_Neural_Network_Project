# Training and Experiment Outputs

This folder contains files related to the training and evaluation pipeline.

Completed work includes:
- training loop integration
- mini-batch training support
- core experiments on:
  - linear Gaussian
  - moons
  - digits
- moons capacity ablation
- digits optimizer comparison
- digits 5-seed evaluation
- logging support for experiment runs

Main scripts used:
- `starter_pack/scripts/run_core_experiments.py`
- `starter_pack/scripts/run_moons_ablation.py`
- `starter_pack/scripts/run_digits_optimizers.py`
- `starter_pack/scripts/run_digits_5seeds.py`

Selected failure case:
- moons dataset with hidden width = 2

Notes:
- experiment runs generate log outputs
- this file documents the training-related work and saved outputs