# Experiments Registry

This directory contains the experiment registry for tracking all experiments conducted with this repository.

## Experiment Registry

The `experiment_registry.csv` file tracks all experiments with the following fields:

- `experiment_id`: Unique identifier for each experiment
- `status`: Status of the experiment (e.g., running, completed, failed)
- `task`: Task name (e.g., single_component_mp_scaffold)
- `split_type`: Type of data split (e.g., scaffold, random)
- `config_file`: Path to the configuration file used
- `code_entry`: Main code entry point (e.g., scripts/02_train.py)
- `git_commit`: Git commit hash at the time of running
- `output_dir`: Directory where output was saved
- `mean_mae`: Mean MAE across folds
- `std_mae`: Standard deviation of MAE across folds
- `notes`: Additional notes about the experiment

## Adding New Experiments

When running a new experiment, add a new entry to the registry with the experiment details.

## Tracking Results

Use the registry to track and compare results across different experiments.
