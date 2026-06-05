#!/usr/bin/env python3
"""
Training script for single-component melting point prediction.
This is a thin wrapper around the legacy training script.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for single-component melting point prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output_tag', type=str, default='', help='Additional tag for output directory')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def expand_config_value(value):
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def main():
    args = parse_args()
    config = {key: expand_config_value(value) for key, value in load_config(args.config).items()}
    model_name = config['model_name_or_path']
    if 'PATH_OR_HF_ID_TO_MOLFORMER' in model_name:
        model_name = os.environ.get('MOLFORMER_MODEL', model_name)
    if 'PATH_OR_HF_ID_TO_MOLFORMER' in model_name:
        raise ValueError(
            "Set model_name_or_path in the config, or export MOLFORMER_MODEL "
            "to a local MoLFormer checkpoint/Hugging Face model id."
        )
    
    # Set up output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = config.get('outputs_root', './outputs')
    run_name = f"{config['task_name']}_{timestamp}"
    if args.output_tag:
        run_name = f"{run_name}_{args.output_tag}"
    full_output_dir = os.path.join(output_dir, run_name)
    
    use_xtb = bool(config.get('use_xtb', True))
    legacy_script_name = 'legacy_main_train.py' if use_xtb else 'legacy_main_train_mlp_fusion.py'
    legacy_script = os.path.join(os.path.dirname(__file__), legacy_script_name)
    cmd = [
        sys.executable, legacy_script,
        '--data_dir', config.get('data_dir', 'data'),
        '--outputs_root', output_dir,
        '--model_name', model_name,
        '--seed', str(config['seed']),
        '--n_folds', str(config['n_folds']),
        '--batch_size', str(config['batch_size']),
        '--num_workers', str(config.get('num_workers', 8)),
        '--max_epochs', str(config['max_epochs']),
        '--freeze_bert_epochs', str(config['freeze_bert_epochs']),
        '--final_tune_epochs', str(config['final_tune_epochs']),
        '--output_tag', run_name,
    ]
    
    # Add optional parameters if present. These keep the YAML configs aligned
    # with the validation protocols described in the manuscript and SI.
    optional_params = [
        ('multi_gpu_mode', config.get('multi_gpu_mode', 'none')),
        ('gpu_ids', config.get('gpu_ids', '')),
        ('device', config.get('device', '')),
        ('use_frozen_split', config.get('use_frozen_split', False)),
        ('use_random_split', config.get('use_random_split', False)),
        ('split_dir', config.get('split_dir', 'splits/scaffold')),
        ('split_manifest', config.get('split_manifest', 'splits/scaffold/split_manifest.csv')),
        ('max_length', config.get('max_length', None)),
        ('grad_accum_steps', config.get('grad_accum_steps', None)),
        ('huber_delta', config.get('huber_delta', None)),
        ('clip_grad_norm', config.get('clip_grad_norm', None)),
        ('cache_graphs', config.get('cache_graphs', None)),
        ('dmpnn_layers', config.get('dmpnn_layers', None)),
        ('dmpnn_hidden_dim', config.get('dmpnn_hidden_dim', None)),
        ('dmpnn_output_dim', config.get('dmpnn_output_dim', None)),
        ('dmpnn_dropout', config.get('dmpnn_dropout', None)),
        ('common_hidden_dim', config.get('common_hidden_dim', None)),
        ('readout_dim', config.get('readout_dim', None)),
        ('main_branch_noise_std', config.get('main_branch_noise_std', None)),
    ]
    if use_xtb:
        optional_params.extend([
            ('xtb_hidden_dim', config.get('xtb_hidden_dim', None)),
            ('xtb_depth', config.get('xtb_depth', None)),
            ('use_rdkit_in_xtb', config.get('use_rdkit_in_xtb', None)),
            ('dynamic_weight_scale', config.get('dynamic_weight_scale', None)),
            ('bias_hidden_dim', config.get('bias_hidden_dim', None)),
            ('final_lr_scale', config.get('final_lr_scale', None)),
        ])
    else:
        optional_params.extend([
            ('scalar_hidden_dim', config.get('scalar_hidden_dim', config.get('bias_hidden_dim', None))),
        ])
    
    for param_name, param_value in optional_params:
        if param_value is None:
            continue
        if param_name in {'use_frozen_split', 'use_random_split'}:
            if param_value:
                cmd.extend([f'--{param_name}'])
        elif isinstance(param_value, bool):
            cmd.extend([f'--{param_name}', str(param_value).lower()])
        else:
            if param_value:
                cmd.extend([f'--{param_name}', str(param_value)])
    
    print(f"Running training with config: {args.config}")
    print(f"Output directory: {full_output_dir}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the legacy script.
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
