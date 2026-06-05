"""
Utility modules for single-component melting point prediction.
"""

from .splits import DEFAULT_SEED, build_random_folds, get_random_fold_indices

_SMILES_EXPORTS = {
    'canonicalize_smiles',
    'canonicalize_smiles_list',
    'build_canonical_map',
    'find_unique_smiles',
    'check_smiles_in_library',
}


def __getattr__(name):
    if name in _SMILES_EXPORTS:
        from . import smiles

        return getattr(smiles, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'canonicalize_smiles',
    'canonicalize_smiles_list',
    'build_canonical_map',
    'find_unique_smiles',
    'check_smiles_in_library',
    'DEFAULT_SEED',
    'build_random_folds',
    'get_random_fold_indices',
]
