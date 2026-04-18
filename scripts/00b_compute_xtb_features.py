#!/usr/bin/env python3
"""
Compute XTB Features for New Molecules

This script handles the workflow for computing XTB features for molecules
that are not yet in the existing feature library.

XTB Environment:
    - XTB binary is expected to be available via conda environment 'pxf_xtb'
    - Path: /home/liutao/.conda/envs/pxf_xtb/bin/xtb
    - To run: conda run -n pxf_xtb xtb ...

Workflow:
    1. Load new molecules from input file (CSV with SMILES column)
    2. Load existing XTB feature library
    3. Identify which molecules are missing from the library
    4. Prepare XTB calculation jobs for missing molecules
    5. Generate shell scripts or batch commands for XTB calculations

Usage:
    # Step 1: Identify missing molecules
    python scripts/00b_compute_xtb_features.py \\
        --input data/external/new_molecules.csv \\
        --existing_xtb data/processed/XTB_train.pth \\
        --output_dir data/external/xtb_jobs \\
        --step identify

    # Step 2: Generate XTB calculation commands
    python scripts/00b_compute_xtb_features.py \\
        --input data/external/new_molecules.csv \\
        --existing_xtb data/processed/XTB_train.pth \\
        --output_dir data/external/xtb_jobs \\
        --step generate_cmds

    # Step 3: Check XTB availability
    python scripts/00b_compute_xtb_features.py --check_xtb

Note:
    This script generates the input files and commands for XTB calculations,
    but does NOT run the actual XTB calculations. You need to execute the
    generated commands separately using the pxf_xtb conda environment.
"""

import os
import sys
import argparse
import subprocess
import shutil
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import numpy as np


XTB_FEATURE_NAMES = [
    'N_Atoms', 'N_Heavy_Atoms', 'Molecular_Mass_amu',
    'Electronic_Energy_AU', 'Electronic_Energy_kcal_mol',
    'HOMO_eV', 'LUMO_eV', 'HOMO_LUMO_Gap_eV',
    'Dipole_Total_Debye', 'Dipole_Theta_deg', 'Dipole_Phi_deg',
    'Charge_Min', 'Charge_Max', 'Charge_Mean', 'Charge_STD', 'Charge_Range',
    'Molecular_Volume_cm3_mol'
]

XTB_CONDA_ENV = "pxf_xtb"
XTB_BINARY_DEFAULT = "/home/liutao/.conda/envs/pxf_xtb/bin/xtb"


def check_xtb_installation() -> Dict[str, Any]:
    """
    Check if XTB is installed and available.

    Returns:
        Dictionary with 'available', 'path', 'version', 'method', 'error'
    """
    result = {
        'available': False,
        'path': None,
        'version': None,
        'method': None,
        'error': None
    }

    # Method 1: Try conda run
    try:
        proc = subprocess.run(
            ['conda', 'run', '-n', XTB_CONDA_ENV, 'xtb', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if proc.returncode == 0:
            result['available'] = True
            result['method'] = 'conda run'
            result['version'] = proc.stdout.strip() if proc.stdout else 'unknown'
            try:
                proc2 = subprocess.run(
                    ['conda', 'run', '-n', XTB_CONDA_ENV, 'which', 'xtb'],
                    capture_output=True, text=True, timeout=10
                )
                if proc2.returncode == 0:
                    result['path'] = proc2.stdout.strip()
            except:
                pass
            return result
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        result['error'] = f"conda run method failed: {e}"

    # Method 2: Try direct path
    if os.path.exists(XTB_BINARY_DEFAULT):
        try:
            proc = subprocess.run(
                [XTB_BINARY_DEFAULT, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if proc.returncode == 0:
                result['available'] = True
                result['path'] = XTB_BINARY_DEFAULT
                result['method'] = 'direct path'
                result['version'] = proc.stdout.strip() if proc.stdout else 'unknown'
                return result
        except Exception as e:
            result['error'] = f"direct path method failed: {e}"

    # Method 3: Try which
    try:
        proc = subprocess.run(
            ['which', 'xtb'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if proc.returncode == 0:
            result['path'] = proc.stdout.strip()
            result['available'] = True
            result['method'] = 'which'
            return result
    except Exception as e:
        result['error'] = f"which method failed: {e}"

    return result


def load_existing_xtb(xtb_pth_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load existing XTB feature library.

    Args:
        xtb_pth_path: Path to XTB .pth file

    Returns:
        Tuple of (list of SMILES, feature array)
    """
    data = torch.load(xtb_pth_path, weights_only=False)
    smiles_list = data['smiles']
    features = data['features']

    if isinstance(features, torch.Tensor):
        features = features.numpy()

    return smiles_list, features


def load_new_molecules(input_csv: str) -> List[str]:
    """
    Load new molecule SMILES from input CSV.

    Args:
        input_csv: Path to CSV file with SMILES column

    Returns:
        List of SMILES strings
    """
    df = pd.read_csv(input_csv)
    if 'SMILES' not in df.columns:
        raise ValueError(f"Input CSV must contain 'SMILES' column. Found: {df.columns.tolist()}")
    return df['SMILES'].tolist()


def identify_missing_molecules(
    new_smiles: List[str],
    existing_smiles: List[str],
    use_canonical: bool = True
) -> Dict[str, Any]:
    """
    Identify which molecules are missing from the XTB feature library.

    Args:
        new_smiles: List of new molecule SMILES
        existing_smiles: List of existing SMILES from feature library
        use_canonical: Whether to use canonical SMILES for matching

    Returns:
        Dictionary with missing, found, and statistics
    """
    from src.utils.smiles import canonicalize_smiles, check_smiles_in_library

    if use_canonical:
        results = check_smiles_in_library(new_smiles, existing_smiles, use_canonical=True)
        missing = [s for s, found in results.items() if not found]
        found = [s for s, found in results.items() if found]
    else:
        existing_set = set(existing_smiles)
        missing = [s for s in new_smiles if s not in existing_set]
        found = [s for s in new_smiles if s in existing_set]

    return {
        'missing': missing,
        'found': found,
        'total_new': len(new_smiles),
        'already_exists': len(found),
        'need_compute': len(missing),
        'use_canonical': use_canonical
    }


def generate_xtb_input_file(smiles: str, output_dir: str, mol_id: int) -> str:
    """
    Generate XTB input file for a single molecule.

    Args:
        smiles: SMILES string
        output_dir: Directory to save input file
        mol_id: Numeric ID for the molecule

    Returns:
        Path to generated input file
    """
    os.makedirs(output_dir, exist_ok=True)

    smiles_safe = smiles.replace('/', '_').replace('\\', '_').replace(' ', '_')[:50]
    input_file = os.path.join(output_dir, f"mol_{mol_id:06d}_{smiles_safe}.smiles")
    with open(input_file, 'w') as f:
        f.write(smiles)

    return input_file


def generate_xtb_commands(
    missing_smiles: List[str],
    output_dir: str,
    xtb_binary: str = None,
    conda_env: str = XTB_CONDA_ENV,
    method: str = "GFN2-xTB"
) -> Tuple[str, List[str]]:
    """
    Generate XTB calculation commands for missing molecules.

    Args:
        missing_smiles: List of SMILES missing from library
        output_dir: Directory for input/output files
        xtb_binary: Path to XTB executable (if None, uses conda run)
        conda_env: Conda environment name for XTB
        method: XTB method (GFN2-xTB, GFN1-xTB, etc.)

    Returns:
        Tuple of (batch script path, list of individual commands)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'inputs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'outputs'), exist_ok=True)

    batch_script = os.path.join(output_dir, 'run_xtb_batch.sh')
    commands = []

    run_prefix = f"conda run -n {conda_env}" if xtb_binary is None else ""

    for i, smiles in enumerate(missing_smiles):
        smiles_safe = smiles.replace('/', '_').replace('\\', '_').replace(' ', '_')[:50]
        input_file = os.path.join(output_dir, 'inputs', f"mol_{i:06d}_{smiles_safe}.smiles")
        output_dir_mol = os.path.join(output_dir, 'outputs', f"mol_{i:06d}_{smiles_safe}")

        with open(input_file, 'w') as f:
            f.write(smiles)

        if run_prefix:
            cmd = f"{run_prefix} xtb {input_file} --{method.lower()} -o {output_dir_mol}"
        else:
            cmd = f"{xtb_binary} {input_file} --{method.lower()} -o {output_dir_mol}"

        commands.append(cmd)

    with open(batch_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# XTB Batch Calculation Script\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total molecules: {len(missing_smiles)}\n")
        f.write(f"# XTB method: {method}\n")
        f.write(f"# Conda environment: {conda_env}\n")
        f.write(f"#\n")
        f.write(f"# Usage:\n")
        f.write(f"#   conda run -n {conda_env} bash {batch_script}\n")
        f.write(f"#   OR\n")
        f.write(f"#   bash {batch_script}  (if xtb is in PATH)\n")
        f.write(f"#\n\n")
        for cmd in commands:
            f.write(f"{cmd}\n")

    os.chmod(batch_script, 0o755)

    return batch_script, commands


def main():
    parser = argparse.ArgumentParser(
        description='Compute XTB features for new molecules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check XTB installation
    python scripts/00b_compute_xtb_features.py --check_xtb

    # Identify missing molecules (with canonical SMILES matching)
    python scripts/00b_compute_xtb_features.py \\
        --input data/external/new_molecules.csv \\
        --existing_xtb data/processed/XTB_train.pth \\
        --output_dir data/external/xtb_jobs \\
        --step identify

    # Identify missing molecules (raw SMILES matching, faster but less robust)
    python scripts/00b_compute_xtb_features.py \\
        --input data/external/new_molecules.csv \\
        --existing_xtb data/processed/XTB_train.pth \\
        --output_dir data/external/xtb_jobs \\
        --step identify \\
        --no_canonical

    # Generate XTB calculation commands
    python scripts/00b_compute_xtb_features.py \\
        --input data/external/new_molecules.csv \\
        --existing_xtb data/processed/XTB_train.pth \\
        --output_dir data/external/xtb_jobs \\
        --step generate_cmds

XTB Requirements:
    - XTB must be installed in conda environment 'pxf_xtb'
    - To run calculations: conda run -n pxf_xtb bash run_xtb_batch.sh
        """
    )
    parser.add_argument('--check_xtb', action='store_true',
                        help='Check XTB installation and availability')
    parser.add_argument('--input', type=str,
                        help='Path to input CSV with SMILES column')
    parser.add_argument('--existing_xtb', type=str,
                        help='Path to existing XTB feature library (.pth)')
    parser.add_argument('--output_dir', type=str,
                        help='Directory for output files')
    parser.add_argument('--step', type=str,
                        choices=['identify', 'generate_cmds'],
                        help='Step to execute')
    parser.add_argument('--canonical', action='store_true', default=True,
                        help='Use canonical SMILES for matching (default: True)')
    parser.add_argument('--no_canonical', action='store_false', dest='canonical',
                        help='Disable canonical SMILES matching')
    parser.add_argument('--method', type=str, default='GFN2-xTB',
                        help='XTB method (default: GFN2-xTB)')

    args = parser.parse_args()

    if args.check_xtb:
        print("=" * 60)
        print("Checking XTB Installation")
        print("=" * 60)

        result = check_xtb_installation()

        if result['available']:
            print(f"✓ XTB is available")
            print(f"  Method: {result['method']}")
            print(f"  Path: {result['path']}")
            print(f"  Version: {result['version']}")
            print()
            print("To run XTB calculations:")
            print(f"  conda run -n {XTB_CONDA_ENV} xtb [options]")
            print(f"  OR")
            print(f"  conda run -n {XTB_CONDA_ENV} bash run_xtb_batch.sh")
        else:
            print(f"✗ XTB is NOT available")
            if result['error']:
                print(f"  Error: {result['error']}")
            print()
            print("To install XTB:")
            print(f"  conda create -n {XTB_CONDA_ENV} -c conda-forge xtb")
            print(f"  OR")
            print(f"  conda activate {XTB_CONDA_ENV}")
            print(f"  # Then install xtb via your preferred method")

        return

    print("=" * 60)
    print("XTB Feature Computation for New Molecules")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Canonical SMILES matching: {args.canonical}")
    print()

    print(f"Checking XTB availability...")
    xtb_check = check_xtb_installation()
    if xtb_check['available']:
        print(f"  ✓ XTB available ({xtb_check['method']})")
    else:
        print(f"  ✗ XTB NOT available")
        print(f"    {xtb_check.get('error', 'Unknown error')}")
        print()
        print("You can still generate commands, but need XTB installed to run them.")

    print(f"\nLoading existing XTB library: {args.existing_xtb}")
    existing_smiles, _ = load_existing_xtb(args.existing_xtb)
    print(f"  Loaded {len(existing_smiles)} molecules")

    print(f"\nLoading new molecules from: {args.input}")
    new_smiles = load_new_molecules(args.input)
    print(f"  Loaded {len(new_smiles)} molecules")

    if args.step == 'identify':
        result = identify_missing_molecules(new_smiles, existing_smiles, use_canonical=args.canonical)

        print("\n" + "=" * 60)
        print("Missing Molecule Identification Results")
        print("=" * 60)
        print(f"Total new molecules: {result['total_new']}")
        print(f"Already in library: {result['already_exists']}")
        print(f"Need XTB computation: {result['need_compute']}")
        print(f"Matching method: {'canonical SMILES' if result['use_canonical'] else 'raw SMILES'}")

        if result['need_compute'] > 0:
            print(f"\nFirst 10 missing molecules:")
            for i, smiles in enumerate(result['missing'][:10]):
                print(f"  {i+1}. {smiles[:60]}...")
            if len(result['missing']) > 10:
                print(f"  ... and {len(result['missing']) - 10} more")

        os.makedirs(args.output_dir, exist_ok=True)
        missing_csv = os.path.join(args.output_dir, 'missing_molecules.csv')
        missing_df = pd.DataFrame({'SMILES': result['missing']})
        missing_df.to_csv(missing_csv, index=False)
        print(f"\nMissing molecules saved to: {missing_csv}")

        found_csv = os.path.join(args.output_dir, 'found_molecules.csv')
        found_df = pd.DataFrame({'SMILES': result['found']})
        found_df.to_csv(found_csv, index=False)
        print(f"Already computed molecules saved to: {found_csv}")

        return result

    elif args.step == 'generate_cmds':
        result = identify_missing_molecules(new_smiles, existing_smiles, use_canonical=args.canonical)

        if result['need_compute'] == 0:
            print("\nAll molecules already exist in the library!")
            print("No XTB calculations needed.")
            return result

        print(f"\nGenerating XTB commands for {result['need_compute']} molecules...")
        batch_script, commands = generate_xtb_commands(
            result['missing'],
            args.output_dir,
            conda_env=XTB_CONDA_ENV,
            method=args.method
        )

        print(f"\nGenerated batch script: {batch_script}")
        print(f"Total commands: {len(commands)}")
        print(f"XTB method: {args.method}")
        print(f"Conda environment: {XTB_CONDA_ENV}")

        print("\nFirst 5 commands:")
        for i, cmd in enumerate(commands[:5]):
            print(f"  {i+1}. {cmd[:80]}...")
        if len(commands) > 5:
            print(f"  ... and {len(commands) - 5} more")

        print("\n" + "=" * 60)
        print("Next Steps")
        print("=" * 60)
        print(f"1. Review the generated commands in: {args.output_dir}")
        print(f"2. Run XTB calculations:")
        print(f"   conda run -n {XTB_CONDA_ENV} bash {batch_script}")
        print(f"3. Wait for XTB calculations to complete")
        print(f"4. Extract features:")
        print(f"   python -m src.preprocessing.xtb_extract --xtb_dir {args.output_dir}/outputs")
        print(f"5. Merge features:")
        print(f"   python scripts/00c_merge_xtb_features.py \\")
        print(f"      --existing_xtb data/processed/XTB_train.pth \\")
        print(f"      --new_features {args.output_dir}/parsed/extracted_features.csv \\")
        print(f"      --output data/processed/XTB_train_extended.pth")
        print("=" * 60)

        return result


if __name__ == '__main__':
    main()
