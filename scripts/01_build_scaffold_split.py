#!/usr/bin/env python3
"""
Build frozen scaffold-based cross-validation splits.

This script strictly reproduces the split logic from the original training code:
1. Aligns samples from multimodal_train.csv, rdkit3d_train.npy, and XTB_train.pth
2. Computes Murcko scaffolds for each molecule
3. Builds balanced 5-fold CV splits based on scaffolds
4. Ensures none-scaffold samples are train-only

Outputs:
- splits/scaffold/split_manifest.csv: Complete split manifest
- splits/scaffold/split_summary.json: Split statistics
- splits/scaffold/fold{1-5}_train.csv: Train samples for each fold
- splits/scaffold/fold{1-5}_val.csv: Validation samples for each fold
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def load_aligned_multimodal_data() -> Tuple[pd.DataFrame, List[str], List[int]]:
    """
    Load and align samples from multiple sources.
    
    This reproduces the load_aligned_multimodal_data() logic from the original training code.
    
    Returns:
        Tuple of (aligned_df, smiles_list, sample_ids)
    """
    # Load multimodal train data
    multimodal_path = 'data/raw/multimodal_train.csv'
    multimodal_df = pd.read_csv(multimodal_path)
    
    # Load XTB features
    xtb_path = 'data/processed/XTB_train.pth'
    xtb_data = torch.load(xtb_path, weights_only=False)
    xtb_smiles = set(xtb_data['smiles'])
    
    # Filter samples
    # 1. SMILES non-empty
    # 2. MP non-empty
    # 3. SMILES exists in XTB_train.pth
    mask = (
        multimodal_df['SMILES'].notna() &
        multimodal_df['MP'].notna() &
        multimodal_df['SMILES'].isin(xtb_smiles)
    )
    
    aligned_df = multimodal_df[mask].reset_index(drop=True)
    smiles_list = aligned_df['SMILES'].tolist()
    sample_ids = list(range(len(aligned_df)))
    
    return aligned_df, smiles_list, sample_ids


def get_scaffold(smiles: str) -> Optional[str]:
    """
    Get Murcko scaffold for a molecule.
    
    This reproduces the get_scaffold() logic from the original training code.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Scaffold string or None if parsing fails or scaffold is empty
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if not scaffold:
            return None
        
        return scaffold
    except Exception:
        return None


def build_balanced_scaffold_folds(smiles_list: List[str], n_folds: int = 5) -> Tuple[Dict[str, List[int]], List[int], List[List[int]]]:
    """
    Build balanced scaffold-based folds.
    
    This reproduces the build_balanced_scaffold_folds() logic from the original training code.
    
    Args:
        smiles_list: List of SMILES strings
        n_folds: Number of folds
        
    Returns:
        Tuple of (scaffold_dict, none_idx, fold_scaffold_indices)
    """
    # Compute scaffolds for all samples
    scaffolds = [get_scaffold(smiles) for smiles in smiles_list]
    
    # Build scaffold dictionary: {scaffold: list of sample indices}
    scaffold_dict: Dict[str, List[int]] = {}
    none_idx: List[int] = []
    
    for i, scaffold in enumerate(scaffolds):
        if scaffold is None:
            none_idx.append(i)
        else:
            if scaffold not in scaffold_dict:
                scaffold_dict[scaffold] = []
            scaffold_dict[scaffold].append(i)
    
    # Sort scaffold groups by size (descending)
    scaffold_groups = sorted(scaffold_dict.values(), key=len, reverse=True)
    
    # Initialize fold indices and sizes
    fold_scaffold_indices = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    
    # Assign each scaffold group to the smallest fold
    for group in scaffold_groups:
        smallest_fold = min(range(n_folds), key=lambda x: fold_sizes[x])
        fold_scaffold_indices[smallest_fold].extend(group)
        fold_sizes[smallest_fold] += len(group)
    
    return scaffold_dict, none_idx, fold_scaffold_indices


def generate_split_manifest(
    aligned_df: pd.DataFrame,
    smiles_list: List[str],
    sample_ids: List[int],
    scaffold_dict: Dict[str, List[int]],
    none_idx: List[int],
    fold_scaffold_indices: List[List[int]]
) -> pd.DataFrame:
    """
    Generate split manifest.
    
    Args:
        aligned_df: Aligned dataframe
        smiles_list: List of SMILES
        sample_ids: List of sample IDs
        scaffold_dict: Scaffold to sample indices mapping
        none_idx: List of none-scaffold sample indices
        fold_scaffold_indices: Fold to sample indices mapping
        
    Returns:
        Split manifest dataframe
    """
    # Compute scaffolds for all samples
    scaffolds = [get_scaffold(smiles) for smiles in smiles_list]
    
    # Create scaffold to fold mapping
    scaffold_to_fold = {}
    for fold_idx, indices in enumerate(fold_scaffold_indices):
        for idx in indices:
            scaffold = scaffolds[idx]
            if scaffold is not None:
                scaffold_to_fold[scaffold] = fold_idx + 1  # folds are 1-based
    
    # Build manifest
    manifest_rows = []
    for i, sample_id in enumerate(sample_ids):
        smiles = smiles_list[i]
        scaffold = scaffolds[i]
        is_none_scaffold = (scaffold is None)
        
        if is_none_scaffold:
            assigned_val_fold = 0  # none-scaffold samples are train-only
        else:
            assigned_val_fold = scaffold_to_fold.get(scaffold, 0)
        
        manifest_rows.append({
            'sample_id': sample_id,
            'smiles': smiles,
            'scaffold': scaffold,
            'assigned_val_fold': assigned_val_fold,
            'is_none_scaffold': is_none_scaffold
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
        
        # Get train indices (all except val, plus none-scaffold)
        train_mask = (manifest['assigned_val_fold'] != fold) | (manifest['is_none_scaffold'])
        train_df = manifest[train_mask]
        
        # Assert no overlap between train and val
        val_sample_ids = set(val_df['sample_id'])
        train_sample_ids = set(train_df['sample_id'])
        assert len(val_sample_ids & train_sample_ids) == 0, f"Overlap between train and val for fold {fold}"
        
        # Assert no none-scaffold in val
        assert not val_df['is_none_scaffold'].any(), f"None-scaffold samples in val for fold {fold}"
        
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
    scaffold_dict: Dict[str, List[int]],
    none_idx: List[int],
    fold_sizes: Dict[int, Dict[str, int]],
    n_folds: int = 5
) -> Dict[str, any]:
    """
    Generate split summary.
    
    Args:
        aligned_df: Aligned dataframe
        scaffold_dict: Scaffold to sample indices mapping
        none_idx: List of none-scaffold sample indices
        fold_sizes: Fold sizes
        n_folds: Number of folds
        
    Returns:
        Split summary dictionary
    """
    total_aligned_samples = len(aligned_df)
    valid_scaffold_samples = total_aligned_samples - len(none_idx)
    none_scaffold_samples_train_only = len(none_idx)
    unique_valid_scaffolds = len(scaffold_dict)
    
    fold_valid_scaffold_val_sizes = [fold_sizes[fold]['val_size'] for fold in range(1, n_folds + 1)]
    
    return {
        'total_aligned_samples': total_aligned_samples,
        'valid_scaffold_samples': valid_scaffold_samples,
        'none_scaffold_samples_train_only': none_scaffold_samples_train_only,
        'unique_valid_scaffolds': unique_valid_scaffolds,
        'fold_valid_scaffold_val_sizes': fold_valid_scaffold_val_sizes,
        'n_folds': n_folds,
        'split_type': 'frozen_scaffold_cv',
        'split_version': 'v1',
        'generated_by': '01_build_scaffold_split.py',
        'generated_at': datetime.now().isoformat()
    }


def main():
    """
    Main function to build frozen scaffold splits.
    """
    parser = argparse.ArgumentParser(
        description='Build frozen scaffold-based cross-validation splits'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='splits/scaffold',
        help='Output directory for split files'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Building Frozen Scaffold Splits")
    print("=" * 80)
    
    # Step 1: Load and align data
    print("1. Loading and aligning data...")
    aligned_df, smiles_list, sample_ids = load_aligned_multimodal_data()
    print(f"   Total aligned samples: {len(aligned_df)}")
    
    # Step 2: Build scaffold folds
    print("2. Building scaffold-based folds...")
    scaffold_dict, none_idx, fold_scaffold_indices = build_balanced_scaffold_folds(
        smiles_list, n_folds=args.n_folds
    )
    print(f"   Valid scaffold samples: {len(aligned_df) - len(none_idx)}")
    print(f"   None-scaffold samples (train-only): {len(none_idx)}")
    print(f"   Unique valid scaffolds: {len(scaffold_dict)}")
    
    # Step 3: Generate split manifest
    print("3. Generating split manifest...")
    manifest = generate_split_manifest(
        aligned_df,
        smiles_list,
        sample_ids,
        scaffold_dict,
        none_idx,
        fold_scaffold_indices
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
        scaffold_dict,
        none_idx,
        fold_sizes,
        n_folds=args.n_folds
    )
    summary_path = os.path.join(args.output_dir, 'split_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Split summary saved to: {summary_path}")
    
    print("=" * 80)
    print("Frozen Scaffold Split Build Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
