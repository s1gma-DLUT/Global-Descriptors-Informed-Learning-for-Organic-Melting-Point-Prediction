#!/usr/bin/env python3
"""
Feature Bundle Merging Script

This script merges XTB features (16 dimensions) with RDKit volume features (1 dimension)
to create the complete 17-dimensional feature bundle that is compatible with the
original XTB_train.pth format.

Usage:
    python scripts/00d_merge_feature_bundle.py \
        --xtb_csv data/external/xtb_parsed/extracted_features.csv \
        --volume_csv data/external/rdkit_volumes.csv \
        --output_pth data/external/merged_features/XTB_feature_bundle.pth
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from src.preprocessing.merge_features import merge_feature_bundle, validate_feature_bundle


def main():
    parser = argparse.ArgumentParser(
        description='Merge XTB features with RDKit volume to create 17D feature bundle'
    )
    parser.add_argument(
        '--xtb_csv',
        type=str,
        required=True,
        help='Path to XTB features (CSV)'
    )
    parser.add_argument(
        '--volume_csv',
        type=str,
        required=True,
        help='Path to volume features (CSV)'
    )
    parser.add_argument(
        '--output_pth',
        type=str,
        required=True,
        help='Path to save merged features (.pth)'
    )
    parser.add_argument(
        '--existing_pth',
        type=str,
        help='Optional: path to existing features (.pth)'
    )
    parser.add_argument(
        '--validate',
        type=str,
        help='Optional: path to validate feature bundle'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Feature Bundle Merging")
    print("=" * 80)

    if args.validate:
        print(f"\nValidating feature bundle: {args.validate}")
        validation = validate_feature_bundle(args.validate)
        
        print("\n" + "=" * 60)
        print("Validation Results")
        print("=" * 60)
        print(f"Valid: {validation['valid']}")
        
        if validation['errors']:
            print("Errors:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        print(f"\nInfo: {validation['info']}")
        return

    # Create output directory
    os.makedirs(os.path.dirname(args.output_pth), exist_ok=True)

    print(f"\nMerging features:")
    print(f"  XTB features: {args.xtb_csv}")
    print(f"  Volume features: {args.volume_csv}")
    print(f"  Output: {args.output_pth}")

    if args.existing_pth:
        print(f"  Existing features: {args.existing_pth}")

    # Perform merge
    result = merge_feature_bundle(
        args.xtb_csv,
        args.volume_csv,
        args.output_pth,
        args.existing_pth
    )

    print("\n" + "=" * 60)
    print("Merge Results")
    print("=" * 60)
    print(f"Existing molecules: {result['existing_count']}")
    print(f"New molecules added: {result['new_added_count']}")
    print(f"Molecules with missing volume: {result['missing_volume_count']}")
    print(f"Total molecules: {result['total_count']}")
    print(f"Output saved to: {result['output_path']}")

    # Validate the output
    print("\nValidating merged feature bundle...")
    validation = validate_feature_bundle(args.output_pth)
    
    print(f"Validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")

    print("\n" + "=" * 80)
    print("Feature Bundle Merging Complete")
    print("=" * 80)
    print("The merged feature bundle is now ready for training.")
    print("\nSchema info:")
    print("  - 16 dimensions from XTB (direct parse + derived)")
    print("  - 1 dimension from RDKit (molecular volume)")
    print("  - Total: 17 dimensions (compatible with XTB_train.pth)")


if __name__ == '__main__':
    main()
