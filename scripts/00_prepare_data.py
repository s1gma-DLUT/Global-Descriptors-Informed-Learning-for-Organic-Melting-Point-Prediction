#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from typing import Optional

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and normalize a melting-point CSV for training."
    )
    parser.add_argument(
        "--input_csv",
        default="data/raw/cleaned/data_set.csv",
        help="Input CSV containing SMILES and MP columns.",
    )
    parser.add_argument(
        "--output_csv",
        default="data/raw/cleaned/data_set_prepared.csv",
        help="Output CSV with canonical SMILES and numeric MP.",
    )
    parser.add_argument(
        "--min_mp",
        type=float,
        default=-150.0,
        help="Lower accepted melting-point bound in degrees Celsius.",
    )
    parser.add_argument(
        "--max_mp",
        type=float,
        default=350.0,
        help="Upper accepted melting-point bound in degrees Celsius.",
    )
    return parser.parse_args()


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles or not isinstance(smiles, str):
        return None
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def prepare_data(input_csv: str, output_csv: str, min_mp: float, max_mp: float) -> None:
    """Validate the public CSV and write a normalized copy."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preparing data")
    print(f"Input CSV: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {"SMILES", "MP"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prepared = df.copy()
    prepared["SMILES"] = prepared["SMILES"].astype(str).str.strip()
    prepared["MP"] = pd.to_numeric(prepared["MP"], errors="coerce")
    prepared["canonical_smiles"] = prepared["SMILES"].map(canonicalize_smiles)

    before = len(prepared)
    prepared = prepared[
        prepared["canonical_smiles"].notna()
        & prepared["MP"].notna()
        & prepared["MP"].between(min_mp, max_mp)
    ].copy()
    prepared["SMILES"] = prepared["canonical_smiles"]
    prepared = prepared.drop(columns=["canonical_smiles"])
    # Deduplicate by canonical SMILES with a 30 degC consistency threshold:
    #   - entries whose MP spread is <= 30 degC are averaged
    #   - groups exceeding 30 degC are discarded entirely
    dup_threshold = 30.0
    grouped = prepared.groupby("SMILES")["MP"]
    mp_range = grouped.transform(lambda s: s.max() - s.min())
    mp_mean = grouped.transform("mean")
    within_mask = mp_range <= dup_threshold
    # Keep one row per SMILES: averaged MP for consistent groups
    prepared = prepared.loc[within_mask].copy()
    prepared["MP"] = mp_mean[within_mask]
    prepared = prepared.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    prepared.to_csv(output_csv, index=False)

    print(f"Rows in: {before}")
    print(f"Rows out: {len(prepared)}")
    print(f"Output CSV: {output_csv}")


def main():
    args = parse_args()
    prepare_data(args.input_csv, args.output_csv, args.min_mp, args.max_mp)


if __name__ == '__main__':
    main()
