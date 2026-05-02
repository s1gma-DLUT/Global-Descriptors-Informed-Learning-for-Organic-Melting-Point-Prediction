# Data Directory

This directory documents the expected data layout. Most data files are not
tracked because they are large, generated, or privately sourced.

## Expected Layout

```text
data/
|-- README.md
|-- raw/
|   |-- multimodal_train.csv
|   `-- multimodal_test.csv
|-- processed/
|   |-- XTB_train.pth
|   |-- XTB_test.pth
|   |-- rdkit3d_train.npy
|   `-- rdkit3d_test.npy
|-- metadata/
`-- external/
    |-- new_molecules.csv
    |-- xtb_jobs/
    |-- xtb_parsed/
    `-- merged_features/
```

## Main Files

| File | Description | Tracked |
| --- | --- | --- |
| `raw/multimodal_train.csv` | Training molecules and melting points | No |
| `raw/multimodal_test.csv` | Test molecules and melting points | No |
| `processed/rdkit3d_train.npy` | RDKit descriptor array for training | No |
| `processed/rdkit3d_test.npy` | RDKit descriptor array for test | No |
| `processed/XTB_train.pth` | 17D mixed-source feature bundle for training | No |
| `processed/XTB_test.pth` | 17D mixed-source feature bundle for test | No |
| `metadata/` | Lightweight provenance notes | Yes, when small |

Expected CSV columns:

- `SMILES`: molecule string.
- `MP`: melting point target, normally in degrees Celsius.

## XTB Bundle Format

The `.pth` feature bundle is expected to contain:

```python
{
    "features": torch.Tensor,     # shape: (n_molecules, 17)
    "smiles": list[str],
    "feature_names": list[str],
}
```

The 17 features are:

1. `N_Atoms`
2. `N_Heavy_Atoms`
3. `Molecular_Mass_amu`
4. `Electronic_Energy_AU`
5. `Electronic_Energy_kcal_mol`
6. `HOMO_eV`
7. `LUMO_eV`
8. `HOMO_LUMO_Gap_eV`
9. `Dipole_Total_Debye`
10. `Dipole_Theta_deg`
11. `Dipole_Phi_deg`
12. `Charge_Min`
13. `Charge_Max`
14. `Charge_Mean`
15. `Charge_STD`
16. `Charge_Range`
17. `Molecular_Volume_cm3_mol`

The first 16 dimensions come from XTB parsing or XTB-derived values. The last
dimension is computed separately with RDKit.

## New Molecule Workflow

1. Put molecules in `data/external/new_molecules.csv`.
2. Generate XTB jobs with `scripts/00b_compute_xtb_features.py`.
3. Run the generated shell script on a system with XTB installed.
4. Parse outputs with `python -m src.preprocessing.xtb_extract`.
5. Compute RDKit volume with `scripts/00c_compute_rdkit_volume.py`.
6. Merge to a 17D bundle with `scripts/00d_merge_feature_bundle.py`.

Keep generated job folders and feature tensors out of git unless a small,
explicit fixture is added for tests.
