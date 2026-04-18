"""
Preprocessing module for single-component melting point prediction.

This module provides utilities for:
    - XTB feature extraction from calculation outputs
    - RDKit feature computation
    - Feature merging and alignment
    - Data preprocessing pipelines
"""

from .xtb_extract import (
    XTB_FEATURE_NAMES,
    XTBResult,
    parse_xtb_output,
    extract_features_from_directory,
    convert_csv_to_pth,
)
from .merge_features import (
    load_xtb_features,
    find_missing_smiles,
    merge_xtb_features,
)

__all__ = [
    'XTB_FEATURE_NAMES',
    'XTBResult',
    'parse_xtb_output',
    'extract_features_from_directory',
    'convert_csv_to_pth',
    'load_xtb_features',
    'find_missing_smiles',
    'merge_xtb_features',
]
