"""
Feature Merging Module

This module provides functionality to merge XTB features with RDKit-derived volume
to create the complete 17-dimensional feature bundle.

It handles:
    - Loading existing feature libraries (.pth format)
    - Merging 16D XTB features with 1D RDKit volume
    - Ensuring compatibility with old XTB_train.pth format
    - Deduplication by SMILES
    - Output in training-compatible format
"""

import os
import argparse
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch

from .schema import (
    XTB_PARSED_16D_NAMES,
    RDKIT_EXTRA_1D_NAMES,
    FULL_17D_FEATURE_NAMES,
    XTB_PARSED_16D_DIM,
    RDKIT_EXTRA_1D_DIM,
    FULL_17D_FEATURE_DIM
)

FEATURE_NAMES = FULL_17D_FEATURE_NAMES
XTB_FEATURE_NAMES = XTB_PARSED_16D_NAMES
VOLUME_FEATURE_NAME = RDKIT_EXTRA_1D_NAMES[0]


def load_xtb_features(pth_path: str) -> Tuple[List[str], torch.Tensor]:
    """
    Load XTB features from a .pth file.

    Args:
        pth_path: Path to the .pth file

    Returns:
        Tuple of (list of SMILES, feature tensor)
    """
    data = torch.load(pth_path, weights_only=False)
    smiles_list = data['smiles']
    features = data['features']
    return smiles_list, features


def load_volume_features(csv_path: str) -> Dict[str, float]:
    """
    Load volume features from a CSV file.

    Args:
        csv_path: Path to volume features CSV

    Returns:
        Dict mapping SMILES to volume value
    """
    df = pd.read_csv(csv_path)
    volume_dict = {}
    
    for _, row in df.iterrows():
        smiles = row.get('smiles', row.get('SMILES', None))
        volume = row.get('molecular_volume_cm3_mol', row.get('Molecular_Volume_cm3_mol', None))
        if smiles and volume is not None:
            volume_dict[smiles] = float(volume)
    
    return volume_dict


def merge_feature_bundle(
    xtb_features_csv: str,
    volume_features_csv: str,
    output_pth_path: str,
    existing_pth_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Merge XTB features with RDKit volume to create complete 17D feature bundle.

    Args:
        xtb_features_csv: Path to XTB features (CSV)
        volume_features_csv: Path to volume features (CSV)
        output_pth_path: Path to save merged features (.pth)
        existing_pth_path: Optional path to existing features (.pth)

    Returns:
        Dictionary with merge statistics
    """
    # Load XTB features
    xtb_df = pd.read_csv(xtb_features_csv)
    xtb_smiles = xtb_df['SMILES'].tolist()
    
    # Validate XTB feature columns
    missing_xtb_columns = [col for col in XTB_FEATURE_NAMES if col not in xtb_df.columns]
    if missing_xtb_columns:
        raise ValueError(f"Missing XTB feature columns: {missing_xtb_columns}")
    
    # Load volume features
    volume_dict = load_volume_features(volume_features_csv)
    
    # Load existing features if provided
    existing_dict = {}
    existing_count = 0
    if existing_pth_path:
        existing_smiles, existing_features = load_xtb_features(existing_pth_path)
        existing_dict = {s: existing_features[i] for i, s in enumerate(existing_smiles)}
        existing_count = len(existing_smiles)
    
    # Prepare merged features
    merged_smiles = list(existing_dict.keys())
    merged_features = [existing_dict[s] for s in merged_smiles]
    
    new_added = 0
    missing_volume = 0
    
    # Process new molecules
    for idx, row in xtb_df.iterrows():
        smiles = row['SMILES']
        
        # Skip if already exists
        if smiles in existing_dict:
            continue
        
        # Get XTB features (16 dimensions)
        xtb_feats = []
        for feat_name in XTB_FEATURE_NAMES:
            if feat_name in row:
                xtb_feats.append(float(row[feat_name]))
            else:
                xtb_feats.append(0.0)
        
        # Validate XTB feature length
        if len(xtb_feats) != XTB_PARSED_16D_DIM:
            raise ValueError(f"Expected {XTB_PARSED_16D_DIM} XTB features, got {len(xtb_feats)}")
        
        # Get volume feature
        if smiles in volume_dict:
            volume = volume_dict[smiles]
        else:
            volume = 0.0
            missing_volume += 1
        
        # Create complete 17D feature vector
        full_feats = xtb_feats + [volume]
        
        # Validate full feature length
        if len(full_feats) != FULL_17D_FEATURE_DIM:
            raise ValueError(f"Expected {FULL_17D_FEATURE_DIM} features, got {len(full_feats)}")
        
        feat_vec = torch.tensor(full_feats, dtype=torch.float32)
        
        merged_smiles.append(smiles)
        merged_features.append(feat_vec)
        new_added += 1
    
    # Create tensor
    merged_features_tensor = torch.stack(merged_features) if merged_features else torch.tensor([])
    
    # Create merged data
    merged_data = {
        'features': merged_features_tensor,
        'smiles': merged_smiles,
        'feature_names': FEATURE_NAMES,
        'source_files': {
            'xtb_features': xtb_features_csv,
            'volume_features': volume_features_csv,
            'existing': existing_pth_path
        },
        'schema_info': {
            'description': 'Mixed-source physicochemical feature bundle',
            'xtb_features': XTB_PARSED_16D_DIM,
            'rdkit_features': RDKIT_EXTRA_1D_DIM,
            'total_dimensions': FULL_17D_FEATURE_DIM,
            'xtb_feature_names': XTB_FEATURE_NAMES,
            'rdkit_feature_names': RDKIT_EXTRA_1D_NAMES
        }
    }
    
    # Save to .pth
    torch.save(merged_data, output_pth_path)
    
    return {
        'existing_count': existing_count,
        'new_added_count': new_added,
        'total_count': len(merged_smiles),
        'missing_volume_count': missing_volume,
        'output_path': output_pth_path
    }


def validate_feature_bundle(pth_path: str) -> Dict[str, Any]:
    """
    Validate a feature bundle .pth file.

    Args:
        pth_path: Path to feature bundle .pth

    Returns:
        Validation results
    """
    data = torch.load(pth_path, weights_only=False)
    
    validation = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {
            'num_molecules': len(data['smiles']),
            'feature_shape': data['features'].shape,
            'feature_names': data.get('feature_names', []),
            'schema_info': data.get('schema_info', {})
        }
    }
    
    # Check feature dimensions
    if data['features'].shape[1] != FULL_17D_FEATURE_DIM:
        validation['valid'] = False
        validation['errors'].append(f"Expected {FULL_17D_FEATURE_DIM} features, got {data['features'].shape[1]}")
    
    # Check feature names
    if 'feature_names' in data:
        if data['feature_names'] != FEATURE_NAMES:
            validation['warnings'].append('Feature names do not match expected schema')
    else:
        validation['warnings'].append('Missing feature_names')
    
    # Check schema info
    if 'schema_info' not in data:
        validation['warnings'].append('Missing schema_info')
    
    return validation


def main():
    parser = argparse.ArgumentParser(description='Merge feature bundle (XTB + RDKit volume)')
    parser.add_argument('--xtb_csv', type=str, required=True, help='Path to XTB features (CSV)')
    parser.add_argument('--volume_csv', type=str, required=True, help='Path to volume features (CSV)')
    parser.add_argument('--output_pth', type=str, required=True, help='Path to save merged features (.pth)')
    parser.add_argument('--existing_pth', type=str, help='Optional: path to existing features (.pth)')
    parser.add_argument('--validate', type=str, help='Optional: path to validate feature bundle')
    args = parser.parse_args()
    
    if args.validate:
        validation = validate_feature_bundle(args.validate)
        print("=" * 60)
        print("Feature Bundle Validation")
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
        print(f"Info: {validation['info']}")
        return
    
    result = merge_feature_bundle(
        args.xtb_csv,
        args.volume_csv,
        args.output_pth,
        args.existing_pth
    )
    
    print("=" * 60)
    print("Feature Bundle Merge Complete")
    print("=" * 60)
    print(f"Existing molecules: {result['existing_count']}")
    print(f"New molecules added: {result['new_added_count']}")
    print(f"Molecules with missing volume: {result['missing_volume_count']}")
    print(f"Total molecules: {result['total_count']}")
    print(f"Output saved to: {result['output_path']}")


if __name__ == '__main__':
    main()

