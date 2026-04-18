"""
Preprocessing module for single-component melting point prediction.

This module provides utilities for:
    - XTB feature extraction from calculation outputs
    - RDKit feature computation
    - Feature merging and alignment
    - Data preprocessing pipelines
"""

from .schema import (
    XTB_PARSED_16D_NAMES,
    RDKIT_EXTRA_1D_NAMES,
    FULL_17D_FEATURE_NAMES,
    XTB_PARSED_16D_DIM,
    RDKIT_EXTRA_1D_DIM,
    FULL_17D_FEATURE_DIM
)
from .xtb_extract import (
    XTBResult,
    parse_xtb_output,
    extract_features_from_directory,
    convert_csv_to_pth,
    validate_xtb_pth,
    compare_features
)
from .rdkit_features import (
    compute_molecular_volume_cm3_mol,
    validate_smiles,
    batch_compute_volumes,
    get_feature_info
)
from .merge_features import (
    load_xtb_features,
    load_volume_features,
    merge_feature_bundle,
    validate_feature_bundle
)

__all__ = [
    # Schema
    'XTB_PARSED_16D_NAMES',
    'RDKIT_EXTRA_1D_NAMES',
    'FULL_17D_FEATURE_NAMES',
    'XTB_PARSED_16D_DIM',
    'RDKIT_EXTRA_1D_DIM',
    'FULL_17D_FEATURE_DIM',
    # XTB
    'XTBResult',
    'parse_xtb_output',
    'extract_features_from_directory',
    'convert_csv_to_pth',
    'validate_xtb_pth',
    'compare_features',
    # RDKit
    'compute_molecular_volume_cm3_mol',
    'validate_smiles',
    'batch_compute_volumes',
    'get_feature_info',
    # Merge
    'load_xtb_features',
    'load_volume_features',
    'merge_feature_bundle',
    'validate_feature_bundle'
]
