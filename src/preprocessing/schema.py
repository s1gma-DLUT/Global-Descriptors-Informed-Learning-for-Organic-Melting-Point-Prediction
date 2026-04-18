"""
Feature Schema Definitions

This module defines the feature schema constants for the mixed-source feature bundle.

Constants:
- XTB_PARSED_16D_NAMES: 16-dimensional features directly parsed from XTB output
- RDKIT_EXTRA_1D_NAMES: 1-dimensional feature computed by RDKit
- FULL_17D_FEATURE_NAMES: Complete 17-dimensional feature bundle (XTB + RDKit)
"""

# XTB direct parse or derived features (16 dimensions)
XTB_PARSED_16D_NAMES = [
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
    'Charge_Range'
]

# RDKit-derived feature (1 dimension)
RDKIT_EXTRA_1D_NAMES = [
    'Molecular_Volume_cm3_mol'
]

# Complete feature bundle (17 dimensions)
FULL_17D_FEATURE_NAMES = XTB_PARSED_16D_NAMES + RDKIT_EXTRA_1D_NAMES

# Feature dimensions
XTB_PARSED_16D_DIM = len(XTB_PARSED_16D_NAMES)
RDKIT_EXTRA_1D_DIM = len(RDKIT_EXTRA_1D_NAMES)
FULL_17D_FEATURE_DIM = len(FULL_17D_FEATURE_NAMES)

# Validation constants
EXPECTED_XTB_DIM = 16
EXPECTED_RDKIT_DIM = 1
EXPECTED_FULL_DIM = 17
