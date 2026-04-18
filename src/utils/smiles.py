"""
SMILES Canonicalization Utilities

This module provides utilities for SMILES canonicalization and matching.
It helps ensure robust matching between SMILES from different sources
by converting them to a canonical form before comparison.

Usage:
    from src.utils.smiles import canonicalize_smiles, canonicalize_smiles_list

    # Single SMILES
    canonical = canonicalize_smiles("CCO")  # Returns canonical form or None

    # Batch processing
    smiles_list = ["CCO", "OCC", "c1ccccc1"]
    canonical_list, failed = canonicalize_smiles_list(smiles_list)
"""

import argparse
from typing import List, Tuple, Optional, Dict, Set
from functools import lru_cache

from rdkit import Chem


@lru_cache(maxsize=10000)
def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize a SMILES string using RDKit.

    This converts a SMILES to its canonical form, which ensures that
    the same molecule will always produce the same canonical SMILES
    regardless of the input format.

    Args:
        smiles: Input SMILES string (may be non-canonical)

    Returns:
        Canonical SMILES string, or None if the input is invalid

    Examples:
        >>> canonicalize_smiles("CCO")  # ethanol
        'CCO'
        >>> canonicalize_smiles("OCC")  # same as above, different format
        'CCO'
        >>> canonicalize_smiles("InvalidSMILES")
        None
    """
    if not smiles or not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return canonical
    except Exception:
        return None


def canonicalize_smiles_list(
    smiles_list: List[str],
    return_failed: bool = True
) -> Tuple[List[str], List[Tuple[str, Optional[str]]]]:
    """
    Canonicalize a list of SMILES strings.

    Args:
        smiles_list: List of input SMILES strings
        return_failed: If True, return list of (original, None) for failed conversions

    Returns:
        Tuple of (successful_canonical_list, failed_list)
        - successful_canonical_list: List of successfully canonicalized SMILES
        - failed_list: List of (original, None) tuples for failed conversions

    Examples:
        >>> smiles = ["CCO", "OCC", "Invalid"]
        >>> canonical, failed = canonicalize_smiles_list(smiles)
        >>> canonical
        ['CCO', 'CCO']
        >>> len(failed)
        1
    """
    successful = []
    failed = []

    for smiles in smiles_list:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            successful.append(canonical)
        else:
            failed.append((smiles, None))

    return successful, failed


def build_canonical_map(
    smiles_list: List[str]
) -> Dict[str, str]:
    """
    Build a mapping from original SMILES to canonical SMILES.

    Args:
        smiles_list: List of input SMILES strings

    Returns:
        Dictionary mapping original SMILES to canonical SMILES
        Only includes successfully canonicalized entries

    Examples:
        >>> smiles = ["CCO", "OCC", "Invalid"]
        >>> build_canonical_map(smiles)
        {'CCO': 'CCO', 'OCC': 'CCO'}
    """
    mapping = {}
    for smiles in smiles_list:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            mapping[smiles] = canonical
    return mapping


def find_unique_smiles(
    smiles_list: List[str],
    use_canonical: bool = True
) -> List[str]:
    """
    Find unique SMILES from a list, optionally using canonical form.

    Args:
        smiles_list: List of input SMILES strings
        use_canonical: If True, use canonical form for uniqueness check

    Returns:
        List of unique SMILES (in original order)

    Examples:
        >>> smiles = ["CCO", "OCC", "c1ccccc1", "C1=CC=CC=C1"]
        >>> find_unique_smiles(smiles, use_canonical=True)
        ['CCO', 'c1ccccc1']  # OCC duplicates CCO, benzene duplicates itself
    """
    seen: Set[str] = []
    unique = []

    for smiles in smiles_list:
        to_check = canonicalize_smiles(smiles) if use_canonical else smiles
        if to_check is not None and to_check not in seen:
            seen.add(to_check)
            unique.append(smiles)  # Keep original form

    return unique


def check_smiles_in_library(
    query_smiles: List[str],
    library_smiles: List[str],
    use_canonical: bool = True
) -> Dict[str, bool]:
    """
    Check which query SMILES are in the library.

    Args:
        query_smiles: List of SMILES to check
        library_smiles: List of SMILES in the library
        use_canonical: If True, use canonical form for matching

    Returns:
        Dictionary mapping each query SMILES to True/False indicating
        whether it's found in the library

    Examples:
        >>> query = ["CCO", "methane"]
        >>> library = ["CCO", "c1ccccc1"]
        >>> check_smiles_in_library(query, library, use_canonical=True)
        {'CCO': True, 'methane': False}
    """
    if use_canonical:
        canonical_library = set(canonicalize_smiles(s) for s in library_smiles if canonicalize_smiles(s) is not None)
        result = {}
        for q in query_smiles:
            canonical_q = canonicalize_smiles(q)
            result[q] = canonical_q in canonical_library if canonical_q else False
        return result
    else:
        library_set = set(library_smiles)
        return {q: q in library_set for q in query_smiles}


def main():
    parser = argparse.ArgumentParser(description='SMILES Canonicalization Utilities')
    parser.add_argument('--smiles', type=str, help='Single SMILES to canonicalize')
    parser.add_argument('--file', type=str, help='File with SMILES (one per line)')
    parser.add_argument('--check', type=str, help='Check SMILES against library')
    parser.add_argument('--library', type=str, help='Library SMILES file')
    args = parser.parse_args()

    if args.smiles:
        canonical = canonicalize_smiles(args.smiles)
        print(f"Input:  {args.smiles}")
        print(f"Output: {canonical}")
        if canonical is None:
            print("WARNING: Invalid SMILES")

    if args.file:
        with open(args.file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        canonical_list, failed = canonicalize_smiles_list(smiles_list)
        print(f"Processed: {len(smiles_list)}")
        print(f"Successful: {len(canonical_list)}")
        print(f"Failed: {len(failed)}")
        if failed:
            print("Failed SMILES:")
            for orig, _ in failed[:5]:
                print(f"  {orig}")

    if args.check and args.library:
        with open(args.check, 'r') as f:
            query = [line.strip() for line in f if line.strip()]
        with open(args.library, 'r') as f:
            library = [line.strip() for line in f if line.strip()]

        results = check_smiles_in_library(query, library)
        in_library = [s for s, found in results.items() if found]
        not_in_library = [s for s, found in results.items() if not found]

        print(f"Query count: {len(query)}")
        print(f"Library count: {len(library)}")
        print(f"In library: {len(in_library)}")
        print(f"Not in library: {len(not_in_library)}")
        if not_in_library:
            print("First 5 not in library:")
            for s in not_in_library[:5]:
                print(f"  {s}")


if __name__ == '__main__':
    main()
