#!/usr/bin/env python3
"""
Training script for single-component melting point prediction.
This is a thin wrapper around the legacy training script.
"""

import os
import sys
import argparse
import yaml
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for single-component melting point prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output_tag', type=str, default='', help='Additional tag for output directory')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set up output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = config.get('output_dir', './outputs')
    run_name = f"{config['task_name']}_{timestamp}"
    if args.output_tag:
        run_name = f"{run_name}_{args.output_tag}"
    full_output_dir = os.path.join(output_dir, run_name)
    
    # Build command to run legacy script
    legacy_script = os.path.join(os.path.dirname(__file__), 'legacy_main_train.py')
    cmd = [
        sys.executable, legacy_script,
        '--data_dir', config['data_dir'],
        '--outputs_root', output_dir,
        '--model_name', config['model_name_or_path'],
        '--seed', str(config['seed']),
        '--n_folds', str(config['n_folds']),
        '--batch_size', str(config['batch_size']),
        '--num_workers', str(config.get('num_workers', 8)),
        '--max_epochs', str(config['max_epochs']),
        '--freeze_bert_epochs', str(config['freeze_bert_epochs']),
        '--final_tune_epochs', str(config['final_tune_epochs']),
        '--output_tag', run_name,
    ]
    
    # Add optional parameters if present
    optional_params = [
        ('multi_gpu_mode', config.get('multi_gpu_mode', 'none')),
        ('gpu_ids', config.get('gpu_ids', '')),
        ('device', config.get('device', '')),
        ('use_frozen_split', config.get('use_frozen_split', False)),
        ('split_dir', config.get('split_dir', 'splits/scaffold')),
        ('split_manifest', config.get('split_manifest', 'splits/scaffold/split_manifest.csv')),
    ]
    
    for param_name, param_value in optional_params:
        if param_name == 'use_frozen_split':
            if param_value:
                cmd.extend([f'--{param_name}'])
        else:
            if param_value:
                cmd.extend([f'--{param_name}', str(param_value)])
    
    print(f"Running training with config: {args.config}")
    print(f"Output directory: {full_output_dir}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the legacy script
    os.system(' '.join(cmd))


if __name__ == '__main__':
    main()
