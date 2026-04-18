#!/usr/bin/env python3
"""
RDKit Volume Calculation Script

This script calculates molecular volumes using RDKit for molecules that have
already been processed by XTB. The volume is needed to complete the 17-dimensional
feature bundle.

Usage:
    python scripts/00c_compute_rdkit_volume.py \
        --input_smiles data/external/test_xtb_smiles_new.csv \
        --output_csv data/external/rdkit_volumes.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from tqdm import tqdm

from src.preprocessing.rdkit_features import batch_compute_volumes, get_feature_info


def main():
    parser = argparse.ArgumentParser(
        description='Compute molecular volumes using RDKit'
    )
    parser.add_argument(
        '--input_smiles',
        type=str,
        required=True,
        help='Path to input SMILES CSV file'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Path to save volume results'
    )
    parser.add_argument(
        '--smiles_column',
        type=str,
        default='SMILES',
        help='Name of SMILES column in input file'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("RDKit Volume Calculation")
    print("=" * 80)

    # Load input SMILES
    print(f"\nLoading SMILES from: {args.input_smiles}")
    input_df = pd.read_csv(args.input_smiles)
    smiles_list = input_df[args.smiles_column].tolist()
    print(f"Loaded {len(smiles_list)} molecules")

    # Get feature info
    feature_info = get_feature_info()
    print(f"\nVolume feature info:")
    print(f"  Feature: {feature_info['feature_name']}")
    print(f"  Unit: {feature_info['unit']}")
    print(f"  Method: {feature_info['method']}")
    print(f"  Note: {feature_info['limitations']}")

    # Compute volumes
    print("\nComputing volumes...")
    results = batch_compute_volumes(smiles_list)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate statistics
    valid_count = results_df['valid'].sum()
    invalid_count = len(results_df) - valid_count
    volume_count = results_df['molecular_volume_cm3_mol'].notnull().sum()
    missing_volume_count = len(results_df) - volume_count

    print("\n" + "=" * 60)
    print("Volume Calculation Summary")
    print("=" * 60)
    print(f"Total molecules: {len(results_df)}")
    print(f"Valid SMILES: {valid_count}")
    print(f"Invalid SMILES: {invalid_count}")
    print(f"Volume computed: {volume_count}")
    print(f"Volume missing: {missing_volume_count}")

    # Save results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to: {args.output_csv}")

    # Show sample results
    print("\nSample results:")
    sample_df = results_df.head(5)
    for _, row in sample_df.iterrows():
        volume = row['molecular_volume_cm3_mol']
        status = "✓" if row['valid'] and pd.notnull(volume) else "✗"
        print(f"  {status} {row['smiles']}: {volume:.4f} cm³/mol")

    print("\n" + "=" * 80)
    print("RDKit Volume Calculation Complete")
    print("=" * 80)
    print("Next step: Use 00d_merge_feature_bundle.py to merge with XTB features")


if __name__ == '__main__':
    main()
