# Feature Bundle Schema

## Overview

The historical files named `XTB_train.pth` and `XTB_test.pth` contain a
17-dimensional mixed-source physicochemical feature bundle. They are not pure
XTB outputs:

- 16 dimensions are parsed or derived from XTB output.
- 1 dimension, `Molecular_Volume_cm3_mol`, is computed separately with RDKit.

## Feature Definitions

| Index | Field name | Unit | Source | Implementation | Note |
| --- | --- | --- | --- | --- | --- |
| 0 | `N_Atoms` | count | XTB parse | Heavy atom count, Z > 1 | Historical name; excludes H |
| 1 | `N_Heavy_Atoms` | count | XTB parse | Same as `N_Atoms` | Kept for compatibility |
| 2 | `Molecular_Mass_amu` | amu | XTB-derived | Sum of atomic masses | Derived from parsed atom counts |
| 3 | `Electronic_Energy_AU` | Hartree | XTB parse | Total energy in Eh | Direct parse |
| 4 | `Electronic_Energy_kcal_mol` | kcal/mol | XTB-derived | AU * 627.509 | Unit conversion |
| 5 | `HOMO_eV` | eV | XTB parse | Orbital line containing `(HOMO)` | Direct parse when available |
| 6 | `LUMO_eV` | eV | XTB parse | Orbital line containing `(LUMO)` | Direct parse when available |
| 7 | `HOMO_LUMO_Gap_eV` | eV | XTB parse | `HL-Gap` line | Direct parse |
| 8 | `Dipole_Total_Debye` | Debye | XTB parse | `full: x y z total` | Direct parse or vector norm |
| 9 | `Dipole_Theta_deg` | degrees | XTB-derived | `atan2(z, sqrt(x*x + y*y))` | Derived |
| 10 | `Dipole_Phi_deg` | degrees | XTB-derived | `atan2(y, x)` | Derived |
| 11 | `Charge_Min` | e | XTB parse | Min atomic charge | Direct parse |
| 12 | `Charge_Max` | e | XTB parse | Max atomic charge | Direct parse |
| 13 | `Charge_Mean` | e | XTB parse | Mean atomic charge | Direct parse |
| 14 | `Charge_STD` | e | XTB parse | Standard deviation | Direct parse |
| 15 | `Charge_Range` | e | XTB-derived | `Charge_Max - Charge_Min` | Derived |
| 16 | `Molecular_Volume_cm3_mol` | cm3/mol | RDKit-derived | `1.2 * MolMR` | Approximate volume |

## Schema Layers

The schema constants are defined in `src/preprocessing/schema.py`:

- `XTB_PARSED_16D_NAMES`: the 16 XTB and XTB-derived fields.
- `RDKIT_EXTRA_1D_NAMES`: the RDKit volume field.
- `FULL_17D_FEATURE_NAMES`: the complete feature bundle.

## Implementation Modules

- `src/preprocessing/xtb_extract.py`: parses 16D XTB features.
- `src/preprocessing/rdkit_features.py`: computes RDKit molecular volume.
- `src/preprocessing/merge_features.py`: merges 16D + 1D into a 17D bundle.
- `scripts/00d_validate_xtb_features.py`: validates feature dimensions and
  schema consistency.

## New Molecule Workflow

1. Run XTB calculations for the new molecules.
2. Parse XTB outputs with `python -m src.preprocessing.xtb_extract`.
3. Compute RDKit volume with `scripts/00c_compute_rdkit_volume.py`.
4. Merge the two sources with `scripts/00d_merge_feature_bundle.py`.

## Versioning

Feature bundles should record lightweight provenance when possible:

```python
schema_info = {
    "description": "Mixed-source physicochemical feature bundle",
    "xtb_features": 16,
    "rdkit_features": 1,
    "total_dimensions": 17,
}
```

When regenerating features, prefer versioned filenames and keep a small
provenance note under `data/metadata/`.
