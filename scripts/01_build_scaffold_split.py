#!/usr/bin/env python3
"""
Script to build scaffold-based splits for single-component melting point prediction.
"""

import os
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Build scaffold-based splits')
    parser.add_argument('--data_dir', type=str, default='./data/processed', help='Directory containing processed data')
    parser.add_argument('--splits_dir', type=str, default='./splits', help='Directory to save splits')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=114514, help='Random seed')
    return parser.parse_args()


def get_scaffold(smiles):
    """
    Get Murcko scaffold for a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold is None or scaffold == '':
            return None
        return scaffold
    except Exception:
        return None


def build_balanced_scaffold_folds(smiles_list, n_folds=5):
    """
    Build balanced scaffold-based folds.
    """
    scaffold_dict = defaultdict(list)
    none_idx = []
    
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        if scaffold is None:
            none_idx.append(idx)
        else:
            scaffold_dict[scaffold].append(idx)
    
    scaffold_groups = sorted(scaffold_dict.values(), key=len, reverse=True)
    fold_scaffold_indices = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    
    for group in scaffold_groups:
        smallest_fold = min(range(n_folds), key=lambda x: fold_sizes[x])
        fold_scaffold_indices[smallest_fold].extend(group)
        fold_sizes[smallest_fold] += len(group)
    
    # Distribute molecules without scaffolds evenly
    for idx, fold_idx in enumerate(range(len(none_idx))):
        target_fold = idx % n_folds
        fold_scaffold_indices[target_fold].append(none_idx[idx])
        fold_sizes[target_fold] += 1
    
    return fold_scaffold_indices


def build_splits(data_dir, splits_dir, n_folds=5, seed=114514):
    """
    Build scaffold-based splits.
    """
    os.makedirs(os.path.join(splits_dir, 'scaffold'), exist_ok=True)
    os.makedirs(os.path.join(splits_dir, 'random'), exist_ok=True)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Building scaffold splits...")
    print(f"Data directory: {data_dir}")
    print(f"Splits directory: {splits_dir}")
    print(f"Number of folds: {n_folds}")
    
    # TODO: Load data
    # For example:
    # data = pd.read_csv(os.path.join(data_dir, 'multimodal_train.csv'))
    # smiles_list = data['SMILES'].tolist()
    
    # TODO: Generate scaffold splits
    # fold_indices = build_balanced_scaffold_folds(smiles_list, n_folds)
    
    # TODO: Save splits
    # for fold in range(n_folds):
    #     train_indices = [i for f in range(n_folds) if f != fold for i in fold_indices[f]]
    #     val_indices = fold_indices[fold]
    #     np.save(os.path.join(splits_dir, 'scaffold', f'train_fold{fold}.npy'), train_indices)
    #     np.save(os.path.join(splits_dir, 'scaffold', f'val_fold{fold}.npy'), val_indices)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scaffold splits built successfully!")


def main():
    args = parse_args()
    build_splits(args.data_dir, args.splits_dir, args.n_folds, args.seed)


if __name__ == '__main__':
    main()
