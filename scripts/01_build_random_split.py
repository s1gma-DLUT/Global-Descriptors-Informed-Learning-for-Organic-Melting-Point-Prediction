#!/usr/bin/env python3
"""
Build frozen random cross-validation splits.

This script creates random 5-fold CV splits:
1. Aligns samples from multimodal_train.csv, rdkit3d_train.npy, and XTB_train.pth
2. Randomly shuffles samples and splits into n_folds
3. Each fold serves as validation set once

Outputs:
- splits/random/split_manifest.csv: Complete split manifest
- splits/random/split_summary.json: Split statistics
- splits/random/fold{1-5}_train.csv: Train samples for each fold
- splits/random/fold{1-5}_val.csv: Validation samples for each fold
"""

import os
import json
import argparse
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.splits import DEFAULT_SEED, build_random_folds


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles or not isinstance(smiles, str):
        return None
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_aligned_multimodal_data(csv_path: str, xtb_path: str) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """
    Load and align samples from multiple sources.

    Returns:
        Tuple of (aligned_df, smiles_list, sample_ids)
    """
    multimodal_df = pd.read_csv(csv_path)

    import torch

    xtb_data = torch.load(xtb_path, weights_only=False)
    xtb_smiles = {
        canonicalize_smiles(str(smiles))
        for smiles in xtb_data['smiles']
        if canonicalize_smiles(str(smiles)) is not None
    }
    multimodal_df = multimodal_df.copy()
    multimodal_df['canonical_smiles'] = multimodal_df['SMILES'].map(canonicalize_smiles)

    # Filter samples
    # 1. SMILES non-empty
    # 2. MP non-empty
    # 3. SMILES exists in XTB_train.pth
    mask = (
        multimodal_df['SMILES'].notna() &
        multimodal_df['MP'].notna() &
        multimodal_df['canonical_smiles'].isin(xtb_smiles)
    )

    aligned_df = multimodal_df[mask].reset_index(drop=True)
    aligned_df['SMILES'] = aligned_df['canonical_smiles']
    aligned_df = aligned_df.drop(columns=['canonical_smiles'])
    smiles_list = aligned_df['SMILES'].tolist()
    sample_ids = list(range(len(aligned_df)))

    return aligned_df, smiles_list, sample_ids


def generate_split_manifest(
    aligned_df: pd.DataFrame,
    smiles_list: List[str],
    sample_ids: List[int],
    fold_indices: List[List[int]]
) -> pd.DataFrame:
    """
    Generate split manifest.

    Args:
        aligned_df: Aligned dataframe
        smiles_list: List of SMILES
        sample_ids: List of sample IDs
        fold_indices: List of fold sample indices

    Returns:
        Split manifest dataframe
    """
    # Create sample to fold mapping
    sample_to_fold = {}
    for fold_idx, indices in enumerate(fold_indices, start=1):
        for idx in indices:
            sample_to_fold[idx] = fold_idx

    # Build manifest
    manifest_rows = []
    for i, sample_id in enumerate(sample_ids):
        smiles = smiles_list[i]
        assigned_val_fold = sample_to_fold.get(i, 0)

        manifest_rows.append({
            'sample_id': sample_id,
            'smiles': smiles,
            'assigned_val_fold': assigned_val_fold,
        })

    return pd.DataFrame(manifest_rows)


def generate_fold_files(
    manifest: pd.DataFrame,
    output_dir: str,
    n_folds: int = 5
) -> Dict[int, Dict[str, int]]:
    """
    Generate fold train/val files.

    Args:
        manifest: Split manifest
        output_dir: Output directory
        n_folds: Number of folds

    Returns:
        Dictionary of fold sizes
    """
    fold_sizes = {}

    for fold in range(1, n_folds + 1):
        # Get validation indices for this fold
        val_mask = (manifest['assigned_val_fold'] == fold)
        val_df = manifest[val_mask]

        # Get train indices (all except val)
        train_mask = (manifest['assigned_val_fold'] != fold)
        train_df = manifest[train_mask]

        # Assert no overlap between train and val
        val_sample_ids = set(val_df['sample_id'])
        train_sample_ids = set(train_df['sample_id'])
        assert len(val_sample_ids & train_sample_ids) == 0, f"Overlap between train and val for fold {fold}"

        # Save fold files
        val_df.to_csv(os.path.join(output_dir, f'fold{fold}_val.csv'), index=False)
        train_df.to_csv(os.path.join(output_dir, f'fold{fold}_train.csv'), index=False)

        fold_sizes[fold] = {
            'train_size': len(train_df),
            'val_size': len(val_df)
        }

    return fold_sizes


def generate_split_summary(
    aligned_df: pd.DataFrame,
    fold_sizes: Dict[int, Dict[str, int]],
    n_folds: int = 5,
    seed: int = DEFAULT_SEED
) -> Dict[str, Any]:
    """
    Generate split summary.

    Args:
        aligned_df: Aligned dataframe
        fold_sizes: Fold sizes
        n_folds: Number of folds
        seed: Random seed

    Returns:
        Split summary dictionary
    """
    total_aligned_samples = len(aligned_df)

    fold_val_sizes = [fold_sizes[fold]['val_size'] for fold in range(1, n_folds + 1)]
    fold_train_sizes = [fold_sizes[fold]['train_size'] for fold in range(1, n_folds + 1)]

    return {
        'total_aligned_samples': total_aligned_samples,
        'n_folds': n_folds,
        'seed': seed,
        'split_type': 'frozen_random_cv',
        'split_version': 'v1',
        'fold_val_sizes': fold_val_sizes,
        'fold_train_sizes': fold_train_sizes,
        'generated_by': '01_build_random_split.py',
        'generated_at': datetime.now().isoformat()
    }


def main():
    """
    Main function to build frozen random splits.
    """
    parser = argparse.ArgumentParser(
        description='Build frozen random cross-validation splits'
    )
    parser.add_argument(
        '--input_csv',
        type=str,
        default='data/raw/cleaned/data_set.csv',
        help='CSV with SMILES and MP columns'
    )
    parser.add_argument(
        '--xtb_pth',
        type=str,
        default='data/raw/cleaned/XTB_train.pth',
        help='Feature bundle used to align the training rows'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='splits/random',
        help='Output directory for split files'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Building Frozen Random Splits")
    print("=" * 80)

    # Step 1: Load and align data
    print("1. Loading and aligning data...")
    aligned_df, smiles_list, sample_ids = load_aligned_multimodal_data(args.input_csv, args.xtb_pth)
    print(f"   Total aligned samples: {len(aligned_df)}")

    # Step 2: Build random folds
    print("2. Building random folds...")
    print(f"   Random seed: {args.seed}")
    fold_indices = build_random_folds(
        len(aligned_df), n_folds=args.n_folds, seed=args.seed
    )
    for fold_idx, indices in enumerate(fold_indices, start=1):
        print(f"   Fold {fold_idx}: {len(indices)} samples")

    # Step 3: Generate split manifest
    print("3. Generating split manifest...")
    manifest = generate_split_manifest(
        aligned_df,
        smiles_list,
        sample_ids,
        fold_indices
    )
    manifest_path = os.path.join(args.output_dir, 'split_manifest.csv')
    manifest.to_csv(manifest_path, index=False)
    print(f"   Split manifest saved to: {manifest_path}")

    # Step 4: Generate fold files
    print("4. Generating fold files...")
    fold_sizes = generate_fold_files(manifest, args.output_dir, n_folds=args.n_folds)
    for fold, sizes in fold_sizes.items():
        print(f"   Fold {fold}: train={sizes['train_size']}, val={sizes['val_size']}")

    # Step 5: Generate split summary
    print("5. Generating split summary...")
    summary = generate_split_summary(
        aligned_df,
        fold_sizes,
        n_folds=args.n_folds,
        seed=args.seed
    )
    summary_path = os.path.join(args.output_dir, 'split_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Split summary saved to: {summary_path}")

    print("=" * 80)
    print("Frozen Random Split Build Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
