# Data Directory

This directory contains all data files for the single-component melting point prediction project.

## Directory Structure

```
data/
├── README.md                    # This file
├── raw/                        # Raw input data
│   ├── multimodal_train.csv     # Training data (SMILES + MP)
│   └── multimodal_test.csv      # Test data (SMILES + MP)
├── processed/                   # Processed features
│   ├── rdkit3d_train.npy       # RDKit 3D features (training)
│   ├── rdkit3d_test.npy        # RDKit 3D features (test)
│   ├── XTB_train.pth            # XTB features (training)
│   └── XTB_test.pth            # XTB features (test)
├── metadata/                    # Metadata files
├── external/                    # External/new molecule data
│   ├── new_molecules.csv       # New molecules to compute XTB for
│   ├── xtb_jobs/               # XTB calculation jobs
│   │   ├── inputs/             # XTB input files
│   │   └── outputs/            # XTB output files
│   ├── xtb_parsed/             # Parsed XTB features
│   │   └── extracted_features.csv
│   └── merged_features/        # Merged feature libraries
```

## Data Files

### Training Input Files

| File | Description | Format | Can Regenerate |
|------|-------------|--------|----------------|
| `multimodal_train.csv` | Training molecules with melting points | CSV (SMILES, MP) | No - original data |
| `multimodal_test.csv` | Test molecules with melting points | CSV (SMILES, MP) | No - original data |
| `rdkit3d_train.npy` | RDKit 3D features for training | NumPy array | Yes - via compute_rdkit_3d_features.py |
| `rdkit3d_test.npy` | RDKit 3D features for test | NumPy array | Yes - via compute_rdkit_3d_features.py |
| `XTB_train.pth` | XTB quantum features for training | PyTorch dict | Yes - via XTB computation + extraction |
| `XTB_test.pth` | XTB quantum features for test | PyTorch dict | Yes - via XTB computation + extraction |

### XTB Feature Format

The XTB feature files (.pth) contain:
```python
{
    'features': torch.Tensor,     # Shape: (n_molecules, 17)
    'smiles': List[str],          # List of SMILES strings
    'feature_names': List[str]    # Names of the 17 features
}
```

XTB Feature Dimensions (17 total):
1. N_Atoms
2. N_Heavy_Atoms
3. Molecular_Mass_amu
4. Electronic_Energy_AU
5. Electronic_Energy_kcal_mol
6. HOMO_eV
7. LUMO_eV
8. HOMO_LUMO_Gap_eV
9. Dipole_Total_Debye
10. Dipole_Theta_deg
11. Dipole_Phi_deg
12. Charge_Min
13. Charge_Max
14. Charge_Mean
15. Charge_STD
16. Charge_Range
17. Molecular_Volume_cm3_mol

### RDKit Feature Format

The RDKit feature files (.npy) contain:
- Shape: (n_molecules, 42) - 42 molecular descriptors

## Extending to New Molecules

### Overview

If you have a set of new molecules that are not in the existing XTB feature library,
follow these steps to:
1. Compute XTB features for the new molecules
2. Merge them with the existing feature library
3. Use the extended library for training or inference

### Step-by-Step Process

#### Step 0: Prepare New Molecules

Place your new molecules in `data/external/new_molecules.csv`:

```csv
SMILES
CCO
c1ccccc1
...
```

The CSV must have a column named `SMILES`.

#### Step 1: Identify Missing Molecules

Check which molecules are missing from the existing XTB library:

```bash
python scripts/00b_compute_xtb_features.py \
    --input data/external/new_molecules.csv \
    --existing_xtb data/processed/XTB_train.pth \
    --output_dir data/external/xtb_jobs \
    --step identify
```

This will:
- Report how many molecules are already in the library
- Report how many need new XTB computation
- Save the missing molecules list to `data/external/xtb_jobs/missing_molecules.csv`

#### Step 2: Generate XTB Calculation Commands

Generate shell scripts for XTB calculations:

```bash
python scripts/00b_compute_xtb_features.py \
    --input data/external/new_molecules.csv \
    --existing_xtb data/processed/XTB_train.pth \
    --output_dir data/external/xtb_jobs \
    --step generate_cmds
```

This will:
- Generate `data/external/xtb_jobs/run_xtb_batch.sh`
- Create input files in `data/external/xtb_jobs/inputs/`
- Prepare for output files in `data/external/xtb_jobs/outputs/`

#### Step 3: Run XTB Calculations

Execute the XTB calculations on a system with XTB installed:

```bash
cd data/external/xtb_jobs
bash run_xtb_batch.sh
```

Note: This step requires:
- XTB software installed
- Proper license (if required)
- Sufficient computational resources

#### Step 4: Extract Features from XTB Outputs

After XTB calculations complete, extract features:

```bash
python -m src.preprocessing.xtb_extract \
    --xtb_dir data/external/xtb_jobs/outputs \
    --output_csv data/external/xtb_parsed/extracted_features.csv
```

Or use the wrapper script:

```bash
python scripts/00c_merge_xtb_features.py \
    --existing_xtb data/processed/XTB_train.pth \
    --new_features data/external/xtb_parsed/extracted_features.csv \
    --mode check
```

#### Step 5: Merge Features

Merge new features with the existing library:

```bash
python scripts/00c_merge_xtb_features.py \
    --existing_xtb data/processed/XTB_train.pth \
    --new_features data/external/xtb_parsed/extracted_features.csv \
    --output data/processed/XTB_train_extended.pth
```

This will create a new feature file containing both original and new molecules.

#### Step 6: Use Extended Features for Training

Update your config to use the extended feature file:

```yaml
# In configs/main_scaffold.yaml
xtb_feature_path: "data/processed/XTB_train_extended.pth"
```

Then run training:
```bash
python scripts/02_train.py --config configs/main_scaffold.yaml
```

### Directory Usage Summary

| Directory | Purpose | Managed By |
|-----------|---------|------------|
| `data/raw/` | Original train/test CSV files | Do not modify |
| `data/processed/` | Computed features (RDKit, XTB) | Regeneratable |
| `data/external/new_molecules.csv` | User-provided new molecules | User |
| `data/external/xtb_jobs/` | XTB calculation I/O | XTB computation |
| `data/external/xtb_parsed/` | Parsed XTB features | xtb_extract module |
| `data/external/merged_features/` | Final merged feature libraries | User decisions |

### Avoiding Feature Library Confusion

To prevent mixing up different versions of feature libraries:

1. **Version your feature files**: Use descriptive names
   - `XTB_train.pth` - Original
   - `XTB_train_v1.pth` - After first extension
   - `XTB_train_20260418.pth` - Dated version

2. **Track provenance**: Keep notes in `data/metadata/` about:
   - When features were computed
   - Which molecules were added
   - XTB method and parameters used

3. **Update configs**: Always specify exact feature file paths in config files

4. **Document in experiment registry**: Record which feature file was used for each experiment

## Legacy Scripts

Legacy scripts for data processing are preserved in `scripts/legacy/`:
- `convert_xtb_to_pth.py` - Original XTB to PTH conversion
- `compute_rdkit_3d_features.py` - Original RDKit feature computation
- `check_data_leakage.py` - Data leakage checking
- `split_multimodal_fixed.py` - Scaffold-based data splitting
- `verify_split.py` - Split verification

These are kept for reference and backward compatibility but are not actively used in the new workflow.
