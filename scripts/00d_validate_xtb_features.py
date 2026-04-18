#!/usr/bin/env python3
"""
XTB Feature Alignment Validation Script

This script validates that newly computed XTB features are consistent with
the existing XTB_train.pth feature library.

Usage:
    python scripts/00d_validate_xtb_features.py \\
        --xtb_dir data/external/xtb_jobs/alignment_test/outputs \\
        --xtb_pth /path/to/XTB_train.pth \\
        --output_dir reports/tables
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from src.preprocessing.xtb_extract import (
    parse_xtb_output,
    XTB_FEATURE_NAMES,
    xtb_result_to_feature_vector
)


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize a SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)
    except:
        return smiles


def load_xtb_train_pth(pth_path: str) -> dict:
    """Load XTB_train.pth and build a SMILES index."""
    data = torch.load(pth_path, weights_only=False)

    smiles_list = data['smiles']
    canonical_to_idx = {}
    for idx, smi in enumerate(smiles_list):
        canonical = canonicalize_smiles(smi)
        canonical_to_idx[canonical] = idx

    return {
        'features': data['features'],
        'smiles': smiles_list,
        'targets': data.get('targets', None),
        'feature_names': data.get('feature_names', XTB_FEATURE_NAMES),
        'canonical_to_idx': canonical_to_idx
    }


def parse_xtb_log_dir(xtb_dir: str) -> dict:
    """Parse all XTB log files in a directory."""
    results = {}

    for filename in os.listdir(xtb_dir):
        if not (filename.endswith('.log') or filename.endswith('.out')):
            continue

        filepath = os.path.join(xtb_dir, filename)
        smiles = filename.replace('.log', '').replace('.out', '').replace('mol_000000_', '')

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        result = parse_xtb_output(content, smiles)
        canonical = canonicalize_smiles(smiles)

        results[canonical] = {
            'smiles': smiles,
            'canonical': canonical,
            'filename': filename,
            'result': result,
            'features': xtb_result_to_feature_vector(result)
        }

    return results


def validate_alignment(
    new_features: dict,
    old_data: dict,
    tolerance_energy: float = 0.01,
    tolerance_homo_lumo: float = 0.5,
    tolerance_dipole: float = 0.1
) -> pd.DataFrame:
    """
    Validate alignment between new and old features.

    Returns a DataFrame with comparison results.
    """
    comparison_records = []

    for canonical, new_info in new_features.items():
        if canonical not in old_data['canonical_to_idx']:
            comparison_records.append({
                'smiles': new_info['smiles'],
                'canonical': canonical,
                'filename': new_info['filename'],
                'status': 'NOT_IN_OLD_LIBRARY',
                'notes': 'Molecule not found in XTB_train.pth'
            })
            continue

        idx = old_data['canonical_to_idx'][canonical]
        old_feat = old_data['features'][idx].numpy()
        new_feat = new_info['features']

        result = new_info['result']

        old_energy = old_feat[3]
        new_energy = new_feat[3]
        energy_diff = abs(old_energy - new_energy)

        old_homo = old_feat[5]
        new_homo = new_feat[5]
        homo_diff = abs(old_homo - new_homo)

        old_lumo = old_feat[6]
        new_lumo = new_feat[6]
        lumo_diff = abs(old_lumo - new_lumo)

        old_gap = old_feat[7]
        new_gap = new_feat[7]
        gap_diff = abs(old_gap - new_gap)

        old_dipole = old_feat[8]
        new_dipole = new_feat[8]
        dipole_diff = abs(old_dipole - new_dipole)

        old_volume = old_feat[16]
        new_volume = new_feat[16]

        if energy_diff < tolerance_energy:
            energy_status = 'MATCH'
        elif energy_diff < tolerance_energy * 10:
            energy_status = 'ACCEPTABLE'
        else:
            energy_status = 'MISMATCH'

        if homo_diff < tolerance_homo_lumo and lumo_diff < tolerance_homo_lumo:
            orbital_status = 'MATCH'
        elif homo_diff < tolerance_homo_lumo * 2 and lumo_diff < tolerance_homo_lumo * 2:
            orbital_status = 'ACCEPTABLE'
        else:
            orbital_status = 'MISMATCH'

        if dipole_diff < tolerance_dipole:
            dipole_status = 'MATCH'
        elif dipole_diff < tolerance_dipole * 5:
            dipole_status = 'ACCEPTABLE'
        else:
            dipole_status = 'MISMATCH'

        if energy_status == 'MATCH' and orbital_status == 'MATCH' and dipole_status == 'MATCH':
            overall_status = 'FULL_MATCH'
        elif 'MISMATCH' not in f"{energy_status}{orbital_status}{dipole_status}":
            overall_status = 'ACCEPTABLE'
        else:
            overall_status = 'MISMATCH'

        comparison_records.append({
            'smiles': new_info['smiles'],
            'canonical': canonical,
            'filename': new_info['filename'],
            'status': overall_status,
            'notes': f"E={energy_status}({energy_diff:.4f}), HOMO/LUMO={orbital_status}({homo_diff:.2f}/{lumo_diff:.2f}), Dipole={dipole_status}({dipole_diff:.4f})",
            'old_energy': old_energy,
            'new_energy': new_energy,
            'energy_diff': energy_diff,
            'energy_status': energy_status,
            'old_homo': old_homo,
            'new_homo': new_homo,
            'homo_diff': homo_diff,
            'old_lumo': old_lumo,
            'new_lumo': new_lumo,
            'lumo_diff': lumo_diff,
            'old_dipole': old_dipole,
            'new_dipole': new_dipole,
            'dipole_diff': dipole_diff,
            'old_volume': old_volume,
            'new_volume': new_volume,
            'n_atoms': result.n_heavy_atoms,
            'field_status': '; '.join([f"{k}={v}" for k, v in result.field_status.items() if v != 'direct_parse'])
        })

    return pd.DataFrame(comparison_records)


def main():
    parser = argparse.ArgumentParser(
        description='Validate XTB feature alignment between new calculations and XTB_train.pth'
    )
    parser.add_argument(
        '--xtb_dir',
        type=str,
        required=True,
        help='Directory containing XTB .log files'
    )
    parser.add_argument(
        '--xtb_pth',
        type=str,
        default='/home/liutao/pxf/MP_new/data/XTB_train.pth',
        help='Path to XTB_train.pth'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='reports/tables',
        help='Output directory for validation reports'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("XTB Feature Alignment Validation")
    print("=" * 80)

    print(f"\nLoading XTB_train.pth from: {args.xtb_pth}")
    old_data = load_xtb_train_pth(args.xtb_pth)
    print(f"  Loaded {len(old_data['smiles'])} molecules")

    print(f"\nParsing XTB logs from: {args.xtb_dir}")
    new_features = parse_xtb_log_dir(args.xtb_dir)
    print(f"  Parsed {len(new_features)} log files")

    print("\nValidating alignment...")
    df = validate_alignment(new_features, old_data)

    summary_csv = os.path.join(args.output_dir, 'xtb_feature_alignment.csv')
    df.to_csv(summary_csv, index=False)
    print(f"\nSaved comparison table to: {summary_csv}")

    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)

    status_counts = df['status'].value_counts()
    print(f"\nStatus distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    print(f"\nDetailed results:")
    for _, row in df.iterrows():
        print(f"\n  {row['smiles']} ({row['canonical']})")
        print(f"    Status: {row['status']}")
        print(f"    Notes: {row['notes']}")

    if 'MISMATCH' in status_counts.index:
        print(f"\n*** WARNING: {status_counts.get('MISMATCH', 0)} molecules show MISMATCH ***")
        mismatches = df[df['status'] == 'MISMATCH']
        for _, row in mismatches.iterrows():
            print(f"  - {row['smiles']}: {row['notes']}")

    print("\n" + "=" * 80)
    print("Field Status Summary")
    print("=" * 80)

    for canonical, info in new_features.items():
        result = info['result']
        unresolved = [k for k, v in result.field_status.items() if v == 'unresolved']
        if unresolved:
            print(f"\n{canonical}: Unresolved fields: {unresolved}")

    print("\nValidation complete!")


if __name__ == '__main__':
    main()