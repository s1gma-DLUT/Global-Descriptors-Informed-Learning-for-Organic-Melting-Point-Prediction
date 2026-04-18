"""
RDKit Features Module

This module provides functionality to compute molecular features using RDKit,
primarily for the `Molecular_Volume_cm3_mol` field that is not available from
standard XTB output.

Key features:
- Molecular volume calculation (approximate)
- SMILES validation and canonicalization
- Error handling for invalid molecules
"""

from typing import Optional, Dict, List, Any
from rdkit import Chem
from rdkit.Chem import Descriptors

from .schema import RDKIT_EXTRA_1D_NAMES


def compute_molecular_volume_cm3_mol(smiles: str) -> Optional[float]:
    """
    Compute molecular volume (cm³/mol) using RDKit descriptors.
    
    This is an approximate calculation using molar refractivity as a proxy for volume.
    
    Args:
        smiles: SMILES string for the molecule
    
    Returns:
        float: Molecular volume in cm³/mol, or None if calculation fails
    """
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens to get complete molecule
        mol = Chem.AddHs(mol)
        
        # Calculate molar refractivity (correlates with volume)
        molar_refractivity = Descriptors.MolMR(mol)
        
        # Convert to approximate volume (cm³/mol)
        # Based on empirical relationship: Volume ≈ 1.2 * molar_refractivity
        volume = molar_refractivity * 1.2
        
        return volume
    except Exception as e:
        return None


def validate_smiles(smiles: str) -> Dict[str, Any]:
    """
    Validate SMILES string and return validation information.
    
    Args:
        smiles: SMILES string to validate
    
    Returns:
        Dict with validation results
    """
    result = {
        'smiles': smiles,
        'valid': False,
        'canonical': None,
        'error': None
    }
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            result['valid'] = True
            result['canonical'] = Chem.MolToSmiles(mol)
    except Exception as e:
        result['error'] = str(e)
    
    return result


def batch_compute_volumes(smiles_list: List[str]) -> List[Dict[str, Any]]:
    """
    Compute volumes for a batch of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
    
    Returns:
        List of dictionaries with SMILES and volume information
    """
    results = []
    
    for smiles in smiles_list:
        validation = validate_smiles(smiles)
        volume = compute_molecular_volume_cm3_mol(smiles) if validation['valid'] else None
        
        results.append({
            'smiles': smiles,
            'canonical_smiles': validation['canonical'],
            'valid': validation['valid'],
            'molecular_volume_cm3_mol': volume,
            'error': validation['error']
        })
    
    return results


def get_feature_name() -> str:
    """
    Get the feature name for molecular volume.
    
    Returns:
        str: Feature name
    """
    return 'Molecular_Volume_cm3_mol'


def get_feature_info() -> Dict[str, Any]:
    """
    Get information about the RDKit-derived volume feature.
    
    Returns:
        Dict with feature information
    """
    return {
        'feature_name': 'Molecular_Volume_cm3_mol',
        'unit': 'cm³/mol',
        'source': 'RDKit-derived',
        'description': 'Approximate molecular volume calculated from molar refractivity',
        'method': 'Volume = 1.2 * molar_refractivity (empirical conversion)',
        'limitations': 'Approximate value, not a precise physical measurement',
        'dtype': 'float32'
    }
