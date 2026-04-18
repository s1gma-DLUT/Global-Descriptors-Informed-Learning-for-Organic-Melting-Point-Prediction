#!/usr/bin/env python3
"""
Merge XTB Features for New Molecules

This script merges newly computed XTB features with the existing feature library.

Features are merged based on SMILES (with optional canonical SMILES matching)
to ensure molecules with different SMILES representations but same structure
are correctly identified as duplicates.

XTB Environment:
    - XTB binary is expected to be available via conda environment 'pxf_xtb'
    - Path: /home/liutao/.conda/envs/pxf_xtb/bin/xtb
    - To run: conda run -n pxf_xtb xtb ...

Workflow:
    1. Load existing XTB feature library (XTB_train.pth)
    2. Load newly computed XTB features from parsed outputs
    3. Identify molecules that are already in the library
    4. Add new molecules to the library
    5. Save merged feature library

Usage:
    # Check what would be merged without saving
    python scripts/00c_merge_xtb_features.py \\
        --existing_xtb data/processed/XTB_train.pth \\
        --new_features data/external/xtb_parsed/extracted_features.csv \\
        --mode check

    # Merge new features with existing library
    python scripts/00c_merge_xtb_features.py \\
        --existing_xtb data/processed/XTB_train.pth \\
        --new_features data/external/xtb_parsed/extracted_features.csv \\
        --output data/processed/XTB_train_merged.pth \\
        --mode merge

    # Merge with canonical SMILES matching (default)
    python scripts/00c_merge_xtb_features.py \\
        --existing_xtb data/processed/XTB_train.pth \\
        --new_features data/external/xtb_parsed/extracted_features.csv \\
        --output data/processed/XTB_train_merged.pth \\
        --mode merge \\
        --canonical
"""

import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
import torch


XTB_FEATURE_NAMES = [
    'N_Atoms',
    'N_Heavy_Atoms',
    'Molecular_Mass_amu',
    'Electronic_Energy_AU',
    'Electronic_Energy_kcal_mol',
    'HOMO_eV',
    'LUMO_eV',
    'HOMO_LUMO_Gap_eV',
    'Dipole_Total_Debye',
    'Dipole_Theta_deg',
    'Dipole_Phi_deg',
    'Charge_Min',
    'Charge_Max',
    'Charge_Mean',
    'Charge_STD',
    'Charge_Range',
    'Molecular_Volume_cm3_mol'
]

XTB_FEATURE_DIM = 17


def load_xtb_features(pth_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load XTB features from a .pth file.

    Args:
        pth_path: Path to the .pth file

    Returns:
        Tuple of (list of SMILES, feature numpy array)
    """
    data = torch.load(pth_path, weights_only=False)
    smiles_list = data['smiles']
    features = data['features']

    if isinstance(features, torch.Tensor):
        features = features.numpy()

    return smiles_list, features


def load_new_features(csv_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load newly computed features from CSV.

    Args:
        csv_path: Path to CSV file with features

    Returns:
        Tuple of (list of SMILES, feature numpy array)
    """
    df = pd.read_csv(csv_path)

    if 'SMILES' not in df.columns:
        raise ValueError(f"CSV must contain 'SMILES' column. Found: {df.columns.tolist()}")

    smiles_list = df['SMILES'].tolist()

    available_cols = [c for c in XTB_FEATURE_NAMES if c in df.columns]
    if len(available_cols) != XTB_FEATURE_DIM:
        missing = set(XTB_FEATURE_NAMES) - set(available_cols)
        raise ValueError(f"CSV missing required columns: {missing}")

    features = df[XTB_FEATURE_NAMES].values.astype(np.float32)

    return smiles_list, features


def canonicalize_mapping(smiles_list: List[str]) -> Dict[str, str]:
    """
    Build a mapping from original SMILES to canonical SMILES.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Dictionary mapping original -> canonical
    """
    from src.utils.smiles import canonicalize_smiles

    mapping = {}
    for smiles in smiles_list:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            mapping[smiles] = canonical
    return mapping


def find_duplicates(
    existing_smiles: List[str],
    new_smiles: List[str],
    use_canonical: bool = True
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Find molecules that are in both existing and new sets.

    Args:
        existing_smiles: List of existing SMILES
        new_smiles: List of new SMILES
        use_canonical: Whether to use canonical SMILES for matching

    Returns:
        Tuple of:
        - unique_existing: SMILES only in existing
        - unique_new: SMILES only in new
        - duplicates: List of (existing, new) pairs that are duplicates
    """
    if use_canonical:
        existing_canonical = {canonicalize_mapping(existing_smiles).get(s, s): s for s in existing_smiles}
        new_canonical = {canonicalize_mapping(new_smiles).get(s, s): s for s in new_smiles}

        common_canonical = set(existing_canonical.keys()) & set(new_canonical.keys())
        duplicates = [(existing_canonical[c], new_canonical[c]) for c in common_canonical]

        existing_only = [s for s in existing_smiles if canonicalize_mapping([s]).get(s, s) not in common_canonical]
        new_only = [s for s in new_smiles if canonicalize_mapping([s]).get(s, s) not in common_canonical]

        return existing_only, new_only, duplicates
    else:
        existing_set = set(existing_smiles)
        new_set = set(new_smiles)

        duplicates = [(s, s) for s in (existing_set & new_set)]
        existing_only = [s for s in existing_smiles if s not in new_set]
        new_only = [s for s in new_smiles if s not in existing_set]

        return existing_only, new_only, duplicates


def merge_features(
    existing_smiles: List[str],
    existing_features: np.ndarray,
    new_smiles: List[str],
    new_features: np.ndarray,
    use_canonical: bool = True
) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """
    Merge existing and new features, avoiding duplicates.

    Args:
        existing_smiles: List of existing SMILES
        existing_features: Array of existing features
        new_smiles: List of new SMILES
        new_features: Array of new features
        use_canonical: Whether to use canonical SMILES for matching

    Returns:
        Tuple of (merged_smiles, merged_features, stats_dict)
    """
    stats = {
        'existing_count': len(existing_smiles),
        'new_count': len(new_smiles),
        'duplicates_count': 0,
        'added_count': 0,
        'total_count': 0,
        'use_canonical': use_canonical
    }

    if use_canonical:
        existing_map = canonicalize_mapping(existing_smiles)
        existing_canonical_to_idx = {canonicalize_mapping([s]).get(s, s): i for s, i in zip(existing_smiles, range(len(existing_smiles)))}

        new_canonical_map = {}
        new_smiles_to_idx = {}
        for i, s in enumerate(new_smiles):
            canonical = canonicalize_mapping([s]).get(s, s)
            new_canonical_map[s] = canonical
            new_smiles_to_idx[s] = i

        existing_canonical_set = set(existing_canonical_to_idx.keys())
        new_canonical_set = set(new_canonical_map.values())

        duplicates_canonical = existing_canonical_set & new_canonical_set
        stats['duplicates_count'] = len(duplicates_canonical)

        merged_smiles = list(existing_smiles)
        merged_features = list(existing_features)

        for new_s in new_smiles:
            new_canonical = new_canonical_map[new_s]
            if new_canonical not in duplicates_canonical:
                merged_smiles.append(new_s)
                merged_features.append(new_features[new_smiles_to_idx[new_s]])
                stats['added_count'] += 1

    else:
        existing_set = set(existing_smiles)
        new_set = set(new_smiles)

        duplicates = existing_set & new_set
        stats['duplicates_count'] = len(duplicates)

        merged_smiles = list(existing_smiles)
        merged_features = list(existing_features)

        for i, new_s in enumerate(new_smiles):
            if new_s not in duplicates:
                merged_smiles.append(new_s)
                merged_features.append(new_features[i])
                stats['added_count'] += 1

    merged_features_array = np.stack(merged_features) if merged_features else np.array([]).reshape(0, XTB_FEATURE_DIM)
    stats['total_count'] = len(merged_smiles)

    return merged_smiles, merged_features_array, stats


def save_merged_features(
    smiles_list: List[str],
    features: np.ndarray,
    output_path: str,
    metadata: Dict[str, Any] = None
) -> None:
    """
    Save merged features to .pth file.

    Args:
        smiles_list: List of SMILES
        features: Feature array
        output_path: Path to save .pth file
        metadata: Optional metadata dictionary
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if isinstance(features, np.ndarray):
        features_tensor = torch.tensor(features, dtype=torch.float32)
    else:
        features_tensor = features

    data = {
        'features': features_tensor,
        'smiles': smiles_list,
        'feature_names': XTB_FEATURE_NAMES,
        'metadata': metadata or {}
    }

    torch.save(data, output_path)
    print(f"Saved merged features to: {output_path}")


def validate_pth_structure(pth_path: str) -> Dict[str, Any]:
    """
    Validate the structure of an XTB .pth file.

    Args:
        pth_path: Path to .pth file

    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }

    try:
        data = torch.load(pth_path, weights_only=False)

        if 'features' not in data:
            result['valid'] = False
            result['errors'].append("Missing 'features' key")
        else:
            features = data['features']
            if len(features.shape) != 2:
                result['valid'] = False
                result['errors'].append(f"Expected 2D features, got shape {features.shape}")
            elif features.shape[1] != XTB_FEATURE_DIM:
                result['warnings'].append(f"Expected {XTB_FEATURE_DIM} features, got {features.shape[1]}")

        if 'smiles' not in data:
            result['valid'] = False
            result['errors'].append("Missing 'smiles' key")
        else:
            result['info']['num_molecules'] = len(data['smiles'])

        if 'feature_names' in data:
            result['info']['feature_names'] = data['feature_names']

    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Failed to load: {str(e)}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Merge XTB features from new molecules with existing library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check merge without saving
    python scripts/00c_merge_xtb_features.py \\
        --existing_xtb data/processed/XTB_train.pth \\
        --new_features data/external/xtb_parsed/extracted_features.csv \\
        --mode check

    # Merge with canonical SMILES matching (default)
    python scripts/00c_merge_xtb_features.py \\
        --existing_xtb data/processed/XTB_train.pth \\
        --new_features data/external/xtb_parsed/extracted_features.csv \\
        --output data/processed/XTB_train_merged.pth

    # Merge with raw SMILES matching (faster, less robust)
    python scripts/00c_merge_xtb_features.py \\
        --existing_xtb data/processed/XTB_train.pth \\
        --new_features data/external/xtb_parsed/extracted_features.csv \\
        --output data/processed/XTB_train_merged.pth \\
        --no_canonical

Feature Naming Recommendations:
    - Original: XTB_train.pth
    - After merge: XTB_train_extended_YYYYMMDD.pth
    - Examples:
        XTB_train_20260418.pth
        XTB_train_plus_1500_molecules.pth
        XTB_train_v2_baseline.pth
        """
    )
    parser.add_argument('--existing_xtb', type=str, required=True,
                        help='Path to existing XTB feature library (.pth)')
    parser.add_argument('--new_features', type=str, required=True,
                        help='Path to new features (CSV format)')
    parser.add_argument('--output', type=str,
                        help='Path to save merged features (.pth)')
    parser.add_argument('--mode', type=str, default='merge',
                        choices=['merge', 'check'],
                        help='Mode: merge (save), or check (report only)')
    parser.add_argument('--canonical', action='store_true', default=True,
                        help='Use canonical SMILES for matching (default: True)')
    parser.add_argument('--no_canonical', action='store_false', dest='canonical',
                        help='Disable canonical SMILES matching')

    args = parser.parse_args()

    print("=" * 60)
    print("XTB Feature Merging")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Canonical SMILES matching: {args.canonical}")
    print()

    print(f"Loading existing XTB library: {args.existing_xtb}")
    existing_valid = validate_pth_structure(args.existing_xtb)
    if not existing_valid['valid']:
        print(f"  WARNING: Existing library has issues:")
        for err in existing_valid['errors']:
            print(f"    - {err}")
    else:
        print(f"  ✓ Valid structure")
        print(f"    Molecules: {existing_valid['info'].get('num_molecules', 'unknown')}")

    existing_smiles, existing_features = load_xtb_features(args.existing_xtb)
    print(f"  Loaded {len(existing_smiles)} molecules")
    print(f"  Feature shape: {existing_features.shape}")

    print(f"\nLoading new features: {args.new_features}")
    new_smiles, new_features = load_new_features(args.new_features)
    print(f"  Loaded {len(new_smiles)} molecules")
    print(f"  Feature shape: {new_features.shape}")

    existing_only, new_only, duplicates = find_duplicates(
        existing_smiles, new_smiles, use_canonical=args.canonical
    )

    print("\n" + "=" * 60)
    print("Merge Analysis")
    print("=" * 60)
    print(f"Existing molecules: {len(existing_smiles)}")
    print(f"New molecules: {len(new_smiles)}")
    print(f"Duplicates (in both): {len(duplicates)}")
    print(f"Unique to existing: {len(existing_only)}")
    print(f"Unique to new (will be added): {len(new_only)}")
    print(f"Matching method: {'canonical SMILES' if args.canonical else 'raw SMILES'}")
    print()

    if duplicates:
        print("First 5 duplicates:")
        for i, (existing, new) in enumerate(duplicates[:5]):
            if existing != new:
                print(f"  {i+1}. existing='{existing[:40]}...' <-> new='{new[:40]}...'")
            else:
                print(f"  {i+1}. '{existing[:60]}...'")

    if new_only:
        print(f"\nFirst 5 molecules to be added:")
        for i, smiles in enumerate(new_only[:5]):
            print(f"  {i+1}. {smiles[:60]}...")

    if args.mode == 'check':
        print("\n[Check mode - no file written]")
        return {
            'existing_count': len(existing_smiles),
            'new_count': len(new_smiles),
            'duplicates_count': len(duplicates),
            'added_count': len(new_only),
            'total_count': len(existing_smiles) + len(new_only),
            'duplicates': duplicates,
            'new_only': new_only
        }

    if len(new_only) == 0:
        print("\nNo new molecules to add. Existing library is complete.")
        return {
            'existing_count': len(existing_smiles),
            'new_count': len(new_smiles),
            'duplicates_count': len(duplicates),
            'added_count': 0,
            'total_count': len(existing_smiles)
        }

    merged_smiles, merged_features, stats = merge_features(
        existing_smiles, existing_features,
        new_smiles, new_features,
        use_canonical=args.canonical
    )

    metadata = {
        'original_library': args.existing_xtb,
        'new_features': args.new_features,
        'merge_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'molecules_added': stats['added_count'],
        'molecules_duplicates': stats['duplicates_count'],
        'total_molecules': stats['total_count'],
        'use_canonical': args.canonical
    }

    if args.output:
        save_merged_features(merged_smiles, merged_features, args.output, metadata)

        print("\n" + "=" * 60)
        print("Merge Complete")
        print("=" * 60)
        print(f"Output file: {args.output}")
        print(f"Total molecules: {stats['total_count']}")
        print(f"New molecules added: {stats['added_count']}")
        print(f"Duplicates skipped: {stats['duplicates_count']}")
        print(f"Feature shape: {merged_features.shape}")
        print(f"Feature dimension: {XTB_FEATURE_DIM}")

        print("\nOutput .pth structure:")
        print("  {")
        print(f"    'features': torch.Tensor (shape: {merged_features.shape})")
        print(f"    'smiles': List[str] (length: {len(merged_smiles)})")
        print(f"    'feature_names': List[str] (length: {XTB_FEATURE_DIM})")
        print(f"    'metadata': Dict with merge info")
        print("  }")

        print("\nNext steps:")
        print(f"  1. Update config to use new feature file:")
        print(f"     xtb_feature_path: \"{args.output}\"")
        print(f"  2. Run training:")
        print(f"     python scripts/02_train.py --config configs/main_scaffold.yaml")
        print(f"  3. Record in experiment registry:")
        print(f"     - XTB feature file: {os.path.basename(args.output)}")
        print(f"     - Molecules added: {stats['added_count']}")
        print(f"     - Total molecules: {stats['total_count']}")

    return {
        'existing_count': len(existing_smiles),
        'new_count': len(new_smiles),
        'duplicates_count': stats['duplicates_count'],
        'added_count': stats['added_count'],
        'total_count': stats['total_count'],
        'output': args.output
    }


if __name__ == '__main__':
    main()
