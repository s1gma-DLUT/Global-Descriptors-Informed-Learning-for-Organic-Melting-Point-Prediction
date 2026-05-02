"""
XTB Feature Extraction Module

This module provides functionality to extract 16-dimensional features from XTB calculation outputs.

FEATURE FIELD STATUS (16 dimensions):
    0.  N_Atoms - Heavy atom count (excluding H) - DIRECT PARSE
    1.  N_Heavy_Atoms - Same as N_Atoms for consistency - DIRECT PARSE
    2.  Molecular_Mass_amu - Calculated from atom counts - DERIVED
    3.  Electronic_Energy_AU - Total energy in Hartree - DIRECT PARSE
    4.  Electronic_Energy_kcal_mol - Energy in kcal/mol - DERIVED (AU * 627.509)
    5.  HOMO_eV - HOMO energy in eV - DIRECT PARSE
    6.  LUMO_eV - LUMO energy in eV - DIRECT PARSE
    7.  HOMO_LUMO_Gap_eV - HOMO-LUMO gap in eV - DIRECT PARSE
    8.  Dipole_Total_Debye - Total dipole moment - DIRECT PARSE
    9.  Dipole_Theta_deg - Dipole theta angle - DERIVED
    10. Dipole_Phi_deg - Dipole phi angle - DERIVED
    11. Charge_Min - Minimum atomic charge - DIRECT PARSE
    12. Charge_Max - Maximum atomic charge - DIRECT PARSE
    13. Charge_Mean - Mean atomic charge - DIRECT PARSE
    14. Charge_STD - Standard deviation of charges - DIRECT PARSE
    15. Charge_Range - Range of charges (max - min) - DERIVED

NOTE:
    Molecular_Volume_cm3_mol is NOT included here. It is computed separately by RDKit.

USAGE:
    For new molecules, use the workflow:
        1. scripts/00b_compute_xtb_features.py --step generate_cmds
        2. Run the generated XTB batch script in your XTB environment
        3. Extract 16D features: this module
        4. Compute RDKit volume: scripts/00c_compute_rdkit_volume.py
        5. Merge to 17D: scripts/00d_merge_feature_bundle.py
"""

import os
import re
import argparse
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from math import sqrt, atan2, degrees

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .schema import XTB_PARSED_16D_NAMES, XTB_PARSED_16D_DIM

ATOMIC_MASSES = {
    1: 1.008,    # H
    6: 12.011,  # C
    7: 14.007,  # N
    8: 15.999,  # O
    9: 18.998,  # F
    17: 35.45,  # Cl
    35: 79.904, # Br
    53: 126.90, # I
}


@dataclass
class XTBResult:
    smiles: str
    n_atoms: float = 0.0
    n_heavy_atoms: float = 0.0
    molecular_mass_amu: float = 0.0
    electronic_energy_au: float = 0.0
    electronic_energy_kcal_mol: float = 0.0
    homo_ev: float = 0.0
    lumo_ev: float = 0.0
    homo_lumo_gap_ev: float = 0.0
    dipole_total_debye: float = 0.0
    dipole_theta_deg: float = 0.0
    dipole_phi_deg: float = 0.0
    charge_min: float = 0.0
    charge_max: float = 0.0
    charge_mean: float = 0.0
    charge_std: float = 0.0
    charge_range: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    raw_parsing_notes: List[str] = field(default_factory=list)
    field_status: Dict[str, str] = field(default_factory=dict)


def parse_xtb_output(xtb_output_text: str, smiles: str) -> XTBResult:
    """
    Parse XTB output text and extract all 16 features.

    This parser extracts features directly from XTB output where possible.
    Fallback/estimation is clearly marked in field_status.

    Args:
        xtb_output_text: Raw text content of XTB output
        smiles: SMILES string for this molecule

    Returns:
        XTBResult object containing parsed features
    """
    result = XTBResult(smiles=smiles)
    result.field_status = {name: 'unparsed' for name in XTB_PARSED_16D_NAMES}

    try:
        lines = xtb_output_text.split('\n')

        atom_info = _parse_atom_counts(lines, result)
        _parse_total_energy(lines, result)
        _parse_homo_lumo(lines, result)
        _parse_dipole(lines, result)
        _parse_charges(lines, result)
        _calculate_derived_fields(result)

        # Consider the parse successful when enough feature fields were
        # recovered to build a meaningful vector. Missing fields remain marked
        # in field_status for downstream validation.
        successful_fields = [k for k, v in result.field_status.items() if v not in ['unparsed', 'unresolved']]

        if len(successful_fields) >= 5:
            result.success = True
            result.raw_parsing_notes.append(f"Parse completed successfully with {len(successful_fields)} fields")
        else:
            result.raw_parsing_notes.append(f"Parse incomplete: only {len(successful_fields)} fields parsed")

    except Exception as e:
        result.error_message = f"Parse error: {str(e)}"
        result.raw_parsing_notes.append(f"Error: {e}")

    return result


def _parse_atom_counts(lines: List[str], result: XTBResult) -> Dict[int, int]:
    """
    Parse atom counts from XTB output.

    Note: N_Atoms in XTB_train.pth is actually HEAVY ATOM COUNT (excluding H)!
    This only counts heavy atoms (Z > 1), not total atoms including H.
    """
    atom_counts = {}
    total_heavy_atoms = 0

    for i, line in enumerate(lines):
        if 'ID    Z sym.   atoms' in line or 'ID    Z sym.' in line:
            for j in range(i+1, min(i+20, len(lines))):
                atom_line = lines[j].strip()
                if not atom_line or atom_line.startswith('---') or 'Calculation' in atom_line:
                    break

                parts = atom_line.split()
                if len(parts) >= 4:
                    try:
                        atom_num = int(parts[0])
                        z = int(parts[1])
                        atoms_range = parts[3]

                        if z == 1:
                            continue

                        if '-' in atoms_range:
                            start, end = map(int, atoms_range.split('-'))
                            count = end - start + 1
                        else:
                            count = 1

                        atom_counts[z] = atom_counts.get(z, 0) + count
                        total_heavy_atoms += count

                    except (ValueError, IndexError):
                        pass
            break

    atom_count_source = 'direct_parse' if total_heavy_atoms > 0 else 'unresolved'

    # Fall back to the input SMILES when the XTB atom table is not present.
    if total_heavy_atoms == 0 and result.smiles:
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(result.smiles)
            if mol:
                total_heavy_atoms = mol.GetNumHeavyAtoms()
                atom_count_source = 'calculated_from_smiles'
                mass = 0.0
                for atom in mol.GetAtoms():
                    z = atom.GetAtomicNum()
                    mass += ATOMIC_MASSES.get(z, 0.0)
                result.molecular_mass_amu = mass
                result.raw_parsing_notes.append(f"Calculated heavy atoms from SMILES: {total_heavy_atoms}, Mass: {mass:.4f} amu")
        except Exception as e:
            result.raw_parsing_notes.append(f"Error calculating atoms from SMILES: {e}")

    result.n_heavy_atoms = float(total_heavy_atoms)
    result.n_atoms = float(total_heavy_atoms)

    mass = 0.0
    for z, count in atom_counts.items():
        mass += ATOMIC_MASSES.get(z, 0.0) * count
    if mass > 0:
        result.molecular_mass_amu = mass

    result.field_status['N_Atoms'] = atom_count_source
    result.field_status['N_Heavy_Atoms'] = atom_count_source
    result.field_status['Molecular_Mass_amu'] = 'derived' if mass > 0 else atom_count_source

    result.raw_parsing_notes.append(f"Heavy atoms: {total_heavy_atoms}, Mass: {result.molecular_mass_amu:.4f} amu")

    return atom_counts


def _parse_total_energy(lines: List[str], result: XTBResult) -> None:
    """Parse total energy from XTB output."""
    energy_au = None

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Format: "energy: -45.215485223988 gnorm: 0.000654437166 ..."
        if line_stripped.startswith('energy:'):
            parts = line_stripped.split()
            if len(parts) >= 2:
                try:
                    energy_str = parts[1]
                    energy_au = float(energy_str)
                    result.raw_parsing_notes.append(f"Found total energy (xtb format): {energy_au} Eh")
                    break
                except (ValueError, IndexError):
                    pass

        # Format: ":: total energy             -45.215485224507 Eh"
        if ':: total energy' in line_stripped:
            parts = line_stripped.split()
            for j, part in enumerate(parts):
                if 'Eh' in part:
                    try:
                        energy_str = parts[j-1]
                        energy_au = float(energy_str)
                        result.raw_parsing_notes.append(f"Found total energy (:: format): {energy_au} Eh")
                        break
                    except (ValueError, IndexError):
                        pass
            if energy_au is not None:
                break

        # Format: "| TOTAL ENERGY              -45.219193089762 Eh   |"
        if '| TOTAL ENERGY' in line_stripped:
            parts = line_stripped.split()
            for j, part in enumerate(parts):
                if 'Eh' in part:
                    try:
                        energy_str = parts[j-1]
                        energy_au = float(energy_str)
                        result.raw_parsing_notes.append(f"Found total energy (| format): {energy_au} Eh")
                        break
                    except (ValueError, IndexError):
                        pass
            if energy_au is not None:
                break

        # Format: "total energy  :   -19.7268295 Eh ..."
        if 'total energy' in line_stripped and ':' in line_stripped:
            parts = line_stripped.split(':', 1)
            if len(parts) == 2:
                value_part = parts[1].strip()
                value_parts = value_part.split()
                for j, part in enumerate(value_parts):
                    if 'Eh' in part:
                        try:
                            energy_str = value_parts[j-1]
                            energy_au = float(energy_str)
                            result.raw_parsing_notes.append(f"Found total energy (total energy: format): {energy_au} Eh")
                            break
                        except (ValueError, IndexError):
                            pass
                if energy_au is not None:
                    break

    if energy_au is not None and energy_au != 0:
        result.electronic_energy_au = energy_au
        result.electronic_energy_kcal_mol = energy_au * 627.509
        result.field_status['Electronic_Energy_AU'] = 'direct_parse'
        result.field_status['Electronic_Energy_kcal_mol'] = 'derived'
    else:
        result.field_status['Electronic_Energy_AU'] = 'unresolved'
        result.field_status['Electronic_Energy_kcal_mol'] = 'unresolved'
        result.raw_parsing_notes.append("WARNING: Could not parse total energy")


def _parse_homo_lumo(lines: List[str], result: XTBResult) -> None:
    """Parse HOMO and LUMO energies from XTB output."""
    homo_ev = None
    lumo_ev = None
    gap_ev = None

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Format: "HL-Gap 0.1511286 Eh 4.1124 eV"
        if gap_ev is None and 'HL-Gap' in line_stripped:
            parts = line_stripped.split()
            for j, p in enumerate(parts):
                if 'eV' in p:
                    try:
                        gap_str = p.replace('eV', '').strip()
                        if gap_str:
                            gap_ev = float(gap_str)
                            result.homo_lumo_gap_ev = gap_ev
                            result.field_status['HOMO_LUMO_Gap_eV'] = 'direct_parse'
                            result.raw_parsing_notes.append(f"Found HOMO-LUMO gap: {gap_ev} eV")
                            break
                    except ValueError:
                        pass

                    # If "eV" is a separate token, use the previous token.
                    if j > 0:
                        try:
                            gap_ev = float(parts[j-1])
                            result.homo_lumo_gap_ev = gap_ev
                            result.field_status['HOMO_LUMO_Gap_eV'] = 'direct_parse'
                            result.raw_parsing_notes.append(f"Found HOMO-LUMO gap (from previous part): {gap_ev} eV")
                            break
                        except ValueError:
                            pass

        # Format: "... -11.1260 (HOMO)"
        if homo_ev is None and '(HOMO)' in line_stripped:
            parts = line_stripped.split()
            for j, p in enumerate(parts):
                if '(HOMO)' in p:
                    for k in range(j-1, max(j-5, -1), -1):
                        try:
                            homo_ev = float(parts[k])
                            result.homo_ev = homo_ev
                            result.field_status['HOMO_eV'] = 'direct_parse'
                            result.raw_parsing_notes.append(f"Found HOMO: {homo_ev} eV")
                            break
                        except ValueError:
                            continue
                    break

        # Format: "... -7.0135 (LUMO)"
        if lumo_ev is None and '(LUMO)' in line_stripped:
            parts = line_stripped.split()
            for j, p in enumerate(parts):
                if '(LUMO)' in p:
                    for k in range(j-1, max(j-5, -1), -1):
                        try:
                            lumo_ev = float(parts[k])
                            result.lumo_ev = lumo_ev
                            result.field_status['LUMO_eV'] = 'direct_parse'
                            result.raw_parsing_notes.append(f"Found LUMO: {lumo_ev} eV")
                            break
                        except ValueError:
                            continue
                    break

    if homo_ev is None and lumo_ev is None and gap_ev is not None and gap_ev > 0:
        homo_ev = -8.0
        lumo_ev = homo_ev + gap_ev
        result.homo_ev = homo_ev
        result.lumo_ev = lumo_ev
        result.field_status['HOMO_eV'] = 'estimated_from_gap'
        result.field_status['LUMO_eV'] = 'estimated_from_gap'
        result.raw_parsing_notes.append(f"Estimated HOMO/LUMO from gap (HOMO={homo_ev}, LUMO={lumo_ev})")

    if homo_ev is None:
        result.field_status['HOMO_eV'] = 'unresolved'
    if lumo_ev is None:
        result.field_status['LUMO_eV'] = 'unresolved'


def _parse_dipole(lines: List[str], result: XTBResult) -> None:
    """Parse dipole moment from XTB output."""
    dipole_x = None
    dipole_y = None
    dipole_z = None
    dipole_tot = None

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Format: "full: -0.676 0.883 0.231 2.886"
        if 'full:' in line_stripped:
            parts = line_stripped.split()
            if len(parts) >= 4:
                try:
                    dipole_x = float(parts[1])
                    dipole_y = float(parts[2])
                    dipole_z = float(parts[3])
                    if len(parts) >= 5:
                        dipole_tot = float(parts[4])

                    # Some outputs omit total; derive it from the vector.
                    if dipole_tot is None and dipole_x is not None and dipole_y is not None and dipole_z is not None:
                        dipole_tot = sqrt(dipole_x**2 + dipole_y**2 + dipole_z**2)

                    if dipole_tot is not None:
                        result.dipole_total_debye = dipole_tot
                        result.field_status['Dipole_Total_Debye'] = 'direct_parse'

                        if dipole_x is not None and dipole_y is not None and dipole_z is not None and dipole_tot > 0:
                            theta = degrees(atan2(dipole_z, sqrt(dipole_x**2 + dipole_y**2)))
                            phi = degrees(atan2(dipole_y, dipole_x))
                            result.dipole_theta_deg = theta
                            result.dipole_phi_deg = phi
                            result.field_status['Dipole_Theta_deg'] = 'derived'
                            result.field_status['Dipole_Phi_deg'] = 'derived'
                            result.raw_parsing_notes.append(f"Dipole: x={dipole_x}, y={dipole_y}, z={dipole_z}, total={dipole_tot}")
                    break
                except (ValueError, IndexError) as e:
                    result.raw_parsing_notes.append(f"Error parsing dipole: {e}")

    if dipole_tot is None:
        result.field_status['Dipole_Total_Debye'] = 'unresolved'
        result.field_status['Dipole_Theta_deg'] = 'unresolved'
        result.field_status['Dipole_Phi_deg'] = 'unresolved'


def _parse_charges(lines: List[str], result: XTBResult) -> None:
    """Parse atomic charges from XTB output."""
    charges = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Charge table header, for example: "# Z covCN q C6AA alpha(0)".
        if '#   Z          covCN         q' in line_stripped or '#   Z          covCN         q      C6AA' in line_stripped:
            for j in range(i+1, min(i+200, len(lines))):
                charge_line = lines[j].strip()
                if not charge_line or charge_line.startswith('#') or 'Mol. C6AA' in charge_line:
                    break

                parts = charge_line.split()
                if len(parts) >= 4:
                    try:
                        charge_str = parts[3].strip()
                        charge = float(charge_str)
                        charges.append(charge)
                    except (ValueError, IndexError):
                        pass

    if charges:
        result.charge_min = min(charges)
        result.charge_max = max(charges)
        result.charge_mean = float(np.mean(charges))
        result.charge_std = float(np.std(charges))
        result.charge_range = result.charge_max - result.charge_min

        result.field_status['Charge_Min'] = 'direct_parse'
        result.field_status['Charge_Max'] = 'direct_parse'
        result.field_status['Charge_Mean'] = 'direct_parse'
        result.field_status['Charge_STD'] = 'direct_parse'
        result.field_status['Charge_Range'] = 'derived'

        result.raw_parsing_notes.append(f"Found {len(charges)} charges, min={result.charge_min:.4f}, max={result.charge_max:.4f}")
    else:
        result.field_status['Charge_Min'] = 'unresolved'
        result.field_status['Charge_Max'] = 'unresolved'
        result.field_status['Charge_Mean'] = 'unresolved'
        result.field_status['Charge_STD'] = 'unresolved'
        result.field_status['Charge_Range'] = 'unresolved'


def _calculate_derived_fields(result: XTBResult) -> None:
    """Calculate derived fields."""
    if result.electronic_energy_au != 0:
        result.electronic_energy_kcal_mol = result.electronic_energy_au * 627.509
        result.field_status['Electronic_Energy_kcal_mol'] = 'derived'

    if result.charge_range == 0 and result.charge_max != 0:
        result.charge_range = result.charge_max - result.charge_min
        result.field_status['Charge_Range'] = 'derived'

    # Note: Molecular_Volume_cm3_mol is computed separately by RDKit


def xtb_result_to_feature_vector(result: XTBResult) -> np.ndarray:
    """Convert XTBResult to numpy feature vector (16 dimensions)."""
    return np.array([
        result.n_atoms,
        result.n_heavy_atoms,
        result.molecular_mass_amu,
        result.electronic_energy_au,
        result.electronic_energy_kcal_mol,
        result.homo_ev,
        result.lumo_ev,
        result.homo_lumo_gap_ev,
        result.dipole_total_debye,
        result.dipole_theta_deg,
        result.dipole_phi_deg,
        result.charge_min,
        result.charge_max,
        result.charge_mean,
        result.charge_std,
        result.charge_range
    ], dtype=np.float32)


def extract_features_from_directory(xtb_dir: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """Extract features from all XTB output files in a directory."""
    results = []

    for filename in os.listdir(xtb_dir):
        if filename.endswith('.log') or filename.endswith('.out'):
            filepath = os.path.join(xtb_dir, filename)
            smiles = filename.replace('.log', '').replace('.out', '').replace('mol_000000_', '')

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            result = parse_xtb_output(content, smiles)
            row = {
                'SMILES': result.smiles,
                'success': result.success,
                'error': result.error_message,
                **{name: getattr(result, _field_name(name)) for name in XTB_PARSED_16D_NAMES}
            }
            for fname, status in result.field_status.items():
                row[f'{fname}_status'] = status
            results.append(row)

    df = pd.DataFrame(results)

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def _field_name(feature_name: str) -> str:
    """Convert feature name to XTBResult field name."""
    mapping = {
        'N_Atoms': 'n_atoms',
        'N_Heavy_Atoms': 'n_heavy_atoms',
        'Molecular_Mass_amu': 'molecular_mass_amu',
        'Electronic_Energy_AU': 'electronic_energy_au',
        'Electronic_Energy_kcal_mol': 'electronic_energy_kcal_mol',
        'HOMO_eV': 'homo_ev',
        'LUMO_eV': 'lumo_ev',
        'HOMO_LUMO_Gap_eV': 'homo_lumo_gap_ev',
        'Dipole_Total_Debye': 'dipole_total_debye',
        'Dipole_Theta_deg': 'dipole_theta_deg',
        'Dipole_Phi_deg': 'dipole_phi_deg',
        'Charge_Min': 'charge_min',
        'Charge_Max': 'charge_max',
        'Charge_Mean': 'charge_mean',
        'Charge_STD': 'charge_std',
        'Charge_Range': 'charge_range'
    }
    return mapping.get(feature_name, feature_name)


def convert_csv_to_pth(input_csv: str, output_pth: str, smiles_list: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convert XTB features from CSV to .pth format."""
    df = pd.read_csv(input_csv)

    if smiles_list is not None:
        smiles_set = set(smiles_list)
        df = df[df['SMILES'].isin(smiles_set)]

    features = df[XTB_PARSED_16D_NAMES].values.astype(np.float32)
    smiles_matched = df['SMILES'].tolist()

    features_tensor = torch.tensor(features, dtype=torch.float32)

    data = {
        'features': features_tensor,
        'smiles': smiles_matched,
        'feature_names': XTB_PARSED_16D_NAMES,
        'note': 'Converted from CSV via xtb_extract (16D XTB only)'
    }

    torch.save(data, output_pth)

    return {
        'features_shape': features_tensor.shape,
        'num_molecules': len(smiles_matched),
        'output_path': output_pth
    }


def validate_xtb_pth(pth_path: str) -> Dict[str, Any]:
    """Validate the structure of an XTB .pth file."""
    data = torch.load(pth_path, weights_only=False)

    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }

    if 'features' not in data:
        result['valid'] = False
        result['errors'].append("Missing 'features' key")
    else:
        features = data['features']
        if len(features.shape) != 2:
            result['valid'] = False
            result['errors'].append(f"Expected 2D features, got shape {features.shape}")
        elif features.shape[1] != XTB_PARSED_16D_DIM:
            result['warnings'].append(f"Expected {XTB_PARSED_16D_DIM} features, got {features.shape[1]}")

    if 'smiles' not in data:
        result['valid'] = False
        result['errors'].append("Missing 'smiles' key")
    else:
        result['info']['num_molecules'] = len(data['smiles'])

    if 'feature_names' in data:
        result['info']['feature_names'] = data['feature_names']

    return result


def compare_features(result1: XTBResult, result2: XTBResult) -> Dict[str, Any]:
    """
    Compare two XTBResult objects field by field.

    Returns comparison results with differences.
    """
    comparison = {
        'field': [],
        'value1': [],
        'value2': [],
        'abs_diff': [],
        'rel_diff_pct': []
    }

    for fname in XTB_PARSED_16D_NAMES:
        v1 = getattr(result1, _field_name(fname))
        v2 = getattr(result2, _field_name(fname))

        abs_diff = abs(v1 - v2) if v1 != 0 or v2 != 0 else 0.0
        rel_diff = abs(abs_diff / v1 * 100) if v1 != 0 else float('inf') if v2 != 0 else 0.0

        comparison['field'].append(fname)
        comparison['value1'].append(v1)
        comparison['value2'].append(v2)
        comparison['abs_diff'].append(abs_diff)
        comparison['rel_diff_pct'].append(rel_diff if rel_diff != float('inf') else None)

    return comparison


def main():
    parser = argparse.ArgumentParser(description='XTB Feature Extraction')
    parser.add_argument('--xtb_dir', type=str, help='Directory with XTB .log files')
    parser.add_argument('--csv', type=str, help='Path to processed CSV')
    parser.add_argument('--output_pth', type=str, help='Path to save .pth file')
    parser.add_argument('--validate', type=str, help='Path to .pth file to validate')
    args = parser.parse_args()

    if args.validate:
        result = validate_xtb_pth(args.validate)
        print(f"Valid: {result['valid']}")
        if result['errors']:
            print(f"Errors: {result['errors']}")
        if result['warnings']:
            print(f"Warnings: {result['warnings']}")
        print(f"Info: {result['info']}")
        return


if __name__ == '__main__':
    main()
