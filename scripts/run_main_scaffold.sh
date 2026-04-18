#!/bin/bash

# Script to run the main scaffold split experiment

# Set working directory
cd "$(dirname "$0")/.."

# Run the training script with main scaffold configuration
python scripts/02_train.py --config configs/main_scaffold.yaml
