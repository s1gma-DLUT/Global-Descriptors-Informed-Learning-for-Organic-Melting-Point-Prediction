#!/usr/bin/env python3
"""
Legacy utility for converting a parsed XTB CSV into train/test .pth bundles.

This file is kept for historical reproducibility. Prefer the newer feature
bundle scripts in `scripts/00c_merge_xtb_features.py` and
`scripts/00d_merge_feature_bundle.py` for new work.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


FEATURE_COLS = [
    "N_Atoms",
    "N_Heavy_Atoms",
    "Molecular_Mass_amu",
    "Electronic_Energy_AU",
    "Electronic_Energy_kcal_mol",
    "HOMO_eV",
    "LUMO_eV",
    "HOMO_LUMO_Gap_eV",
    "Dipole_Total_Debye",
    "Dipole_Theta_deg",
    "Dipole_Phi_deg",
    "Charge_Min",
    "Charge_Max",
    "Charge_Mean",
    "Charge_STD",
    "Charge_Range",
    "Molecular_Volume_cm3_mol",
]


def build_bundle(split_df: pd.DataFrame, smiles_to_features: dict[str, np.ndarray]) -> dict:
    features = []
    targets = []
    smiles_matched = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="matching"):
        smiles = row["SMILES"]
        if smiles not in smiles_to_features:
            continue
        features.append(smiles_to_features[smiles])
        targets.append(row["MP"])
        smiles_matched.append(smiles)

    return {
        "features": torch.tensor(np.asarray(features), dtype=torch.float32),
        "targets": torch.tensor(np.asarray(targets), dtype=torch.float32),
        "smiles": smiles_matched,
        "feature_names": FEATURE_COLS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert parsed XTB CSV to train/test .pth bundles")
    parser.add_argument("--xtb_csv", default="data/processed/xtb_processed.csv")
    parser.add_argument("--train_csv", default="data/raw/multimodal_train.csv")
    parser.add_argument("--test_csv", default="data/raw/multimodal_test.csv")
    parser.add_argument("--train_out", default="data/processed/XTB_train.pth")
    parser.add_argument("--test_out", default="data/processed/XTB_test.pth")
    args = parser.parse_args()

    xtb_df = pd.read_csv(args.xtb_csv)
    missing = [col for col in FEATURE_COLS + ["SMILES"] if col not in xtb_df.columns]
    if missing:
        raise ValueError(f"Missing required XTB columns: {missing}")

    smiles_to_features = {
        row["SMILES"]: row[FEATURE_COLS].to_numpy(dtype=np.float32)
        for _, row in xtb_df.iterrows()
    }

    train_bundle = build_bundle(pd.read_csv(args.train_csv), smiles_to_features)
    test_bundle = build_bundle(pd.read_csv(args.test_csv), smiles_to_features)

    torch.save(train_bundle, args.train_out)
    torch.save(test_bundle, args.test_out)
    print(f"Saved train bundle: {args.train_out} ({len(train_bundle['smiles'])} molecules)")
    print(f"Saved test bundle: {args.test_out} ({len(test_bundle['smiles'])} molecules)")


if __name__ == "__main__":
    main()
