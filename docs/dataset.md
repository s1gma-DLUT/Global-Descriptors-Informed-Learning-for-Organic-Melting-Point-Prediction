# Dataset Documentation

## Overview

This document describes the dataset used for single-component melting point prediction, including XTB feature generation and extension to new molecules.

## Dataset Sources

The dataset consists of single-component molecules with experimentally measured melting points.

### Data Format

#### Main Data Files

1. **Training/Test CSV files** (`data/raw/`):
   - `multimodal_train.csv`: Training molecules with melting points
   - `multimodal_test.csv`: Test molecules with melting points
   - Columns: `SMILES`, `MP` (melting point in Celsius)

2. **RDKit features** (`data/processed/`):
   - `rdkit3d_train.npy`: NumPy array of RDKit 3D features (training)
   - `rdkit3d_test.npy`: NumPy array of RDKit 3D features (test)
   - Shape: (n_molecules, 42) - 42 molecular descriptors

3. **XTB features** (`data/processed/`):
   - `XTB_train.pth`: PyTorch file containing mixed-source physicochemical features
   - `XTB_test.pth`: PyTorch file containing mixed-source physicochemical features
   - Structure:
     ```python
     {
         'features': torch.Tensor,      # Shape: (n_molecules, 17)
         'smiles': List[str],           # List of SMILES strings
         'feature_names': List[str],     # Names of the 17 features
         'schema_info': Dict            # Information about feature sources
     }
     ```

   **Important**: Despite the name, this is a **mixed-source feature bundle**:
   - **16 dimensions** from XTB quantum mechanical calculations
   - **1 dimension** (Molecular_Volume_cm3_mol) from RDKit
   - See `docs/xtb_feature_schema.md` for detailed schema information

## XTB Feature Details

### XTB Role in the Model

XTB (eXtended Tight Binding) features provide quantum mechanical properties that complement:
- **MoLFormer**: Captures sequential SMILES patterns
- **D-MPNN**: Captures graph connectivity
- **RDKit**: Captures traditional molecular descriptors

XTB features include:
- Electronic properties (HOMO/LUMO energies, gap)
- Molecular geometry (atom counts, volume)
- Charge distribution statistics
- Dipole moments

### Schema Layers

The feature bundle is organized into three distinct schema layers:

#### 1. XTB_PARSED_16D_NAMES (16 dimensions)
- Directly parsed from XTB output or derived from XTB parsed values
- Does NOT include molecular volume
- Defined in: `src/preprocessing/schema.py`

#### 2. RDKIT_EXTRA_1D_NAMES (1 dimension)
- `Molecular_Volume_cm3_mol` - Computed separately by RDKit
- Defined in: `src/preprocessing/schema.py`

#### 3. FULL_17D_FEATURE_NAMES (17 dimensions)
- Complete feature bundle: XTB_PARSED_16D_NAMES + RDKIT_EXTRA_1D_NAMES
- Compatible with old XTB_train.pth format
- Defined in: `src/preprocessing/schema.py`

### XTB Feature Dimensions (17 total)

#### 16D XTB Features (from XTB calculations):
1. N_Atoms - Total number of atoms
2. N_Heavy_Atoms - Number of heavy atoms
3. Molecular_Mass_amu - Molecular mass in atomic mass units
4. Electronic_Energy_AU - Electronic energy in atomic units
5. Electronic_Energy_kcal_mol - Electronic energy in kcal/mol
6. HOMO_eV - Highest occupied molecular orbital energy (eV)
7. LUMO_eV - Lowest unoccupied molecular orbital energy (eV)
8. HOMO_LUMO_Gap_eV - HOMO-LUMO gap (eV)
9. Dipole_Total_Debye - Total dipole moment (Debye)
10. Dipole_Theta_deg - Dipole theta angle (degrees)
11. Dipole_Phi_deg - Dipole phi angle (degrees)
12. Charge_Min - Minimum partial charge
13. Charge_Max - Maximum partial charge
14. Charge_Mean - Mean partial charge
15. Charge_STD - Standard deviation of partial charges
16. Charge_Range - Range of partial charges

#### 1D RDKit Feature (separately computed):
17. Molecular_Volume_cm3_mol - Approximate molecular volume based on molar refractivity

### XTB Computation Method

- Method: GFN2-xTB
- The XTB calculations must be performed externally using the xtb software
- Results are then parsed and converted to the .pth format

## Data Preparation

### Steps

1. **Raw Data Collection**
   - Gather experimentally measured melting point data
   - Ensure data quality and consistency

2. **SMILES Validation**
   - Validate SMILES strings using RDKit
   - Remove invalid or problematic molecules

3. **Feature Generation**
   - Generate RDKit features using `scripts/legacy/compute_rdkit_3d_features.py`
   - Generate XTB features using external XTB software

4. **Data Splitting**
   - Generate scaffold-based splits for main evaluation
   - Generate random splits for baseline comparison

## Split Information

### Scaffold Split

- **Purpose**: Evaluate generalization to chemically diverse molecules
- **Method**: Murcko scaffold-based splitting
- **Folds**: 5-fold cross-validation

### Random Split

- **Purpose**: Provide baseline performance
- **Method**: Random splitting
- **Folds**: 5-fold cross-validation

## Extending to New Molecules

### Overview

If you have a set of new molecules that are not in the existing XTB feature library, you can:

1. Compute XTB features for the new molecules
2. Merge them with the existing feature library
3. Use the extended library for training or inference

### Recommended Workflow

#### Step 1: Prepare New Molecules

Create `data/external/new_molecules.csv`:
```csv
SMILES
CCO
c1ccccc1
NC(=O)c1ccccc1
...
```

#### Step 2: Identify Missing Molecules

```bash
python scripts/00b_compute_xtb_features.py \
    --input data/external/new_molecules.csv \
    --existing_xtb data/processed/XTB_train.pth \
    --output_dir data/external/xtb_jobs \
    --step identify
```

This reports:
- How many molecules are already in the library
- How many need new XTB computation
- Saves missing list to `data/external/xtb_jobs/missing_molecules.csv`

#### Step 3: Generate XTB Commands

```bash
python scripts/00b_compute_xtb_features.py \
    --input data/external/new_molecules.csv \
    --existing_xtb data/processed/XTB_train.pth \
    --output_dir data/external/xtb_jobs \
    --step generate_cmds
```

This generates:
- Batch script: `data/external/xtb_jobs/run_xtb_batch.sh`
- Input files in: `data/external/xtb_jobs/inputs/`
- Output directories in: `data/external/xtb_jobs/outputs/`

#### Step 4: Run XTB Calculations

Execute on a system with XTB installed:
```bash
cd data/external/xtb_jobs
bash run_xtb_batch.sh
```

#### Step 5: Extract XTB Features

After XTB calculations complete:
```bash
python -m src.preprocessing.xtb_extract \
    --xtb_dir data/external/xtb_jobs/outputs \
    --output_csv data/external/xtb_parsed/extracted_features.csv
```

#### Step 6: Calculate RDKit Volume

Compute molecular volumes using RDKit:
```bash
python scripts/00c_compute_rdkit_volume.py \
    --input_smiles data/external/xtb_jobs/missing_molecules.csv \
    --output_csv data/external/rdkit_volumes.csv
```

#### Step 7: Merge Feature Bundle

Combine XTB features with RDKit volume:
```bash
python scripts/00d_merge_feature_bundle.py \
    --xtb_csv data/external/xtb_parsed/extracted_features.csv \
    --volume_csv data/external/rdkit_volumes.csv \
    --output_pth data/processed/XTB_train_extended.pth \
    --existing_pth data/processed/XTB_train.pth
```

#### Step 8: Use Extended Features

Update config:
```yaml
xtb_feature_path: "./data/processed/XTB_train_extended.pth"
```

### Avoiding Feature Library Confusion

To prevent mixing up different versions of feature libraries:

1. **Version your feature files**: Use descriptive names
   - `XTB_train.pth` - Original
   - `XTB_train_20260418.pth` - Dated version
   - `XTB_train_v2.pth` - Numbered version

2. **Track provenance**: Keep notes in `data/metadata/` about:
   - When features were computed
   - Which molecules were added
   - XTB method and parameters used

3. **Update configs**: Always specify exact feature file paths in config files

4. **Document in experiment registry**: Record which feature file was used for each experiment

## Data Usage Guidelines

1. **Training**
   - Use the scaffold split for main model development
   - Use the random split for baseline comparison

2. **Evaluation**
   - Report results on both scaffold and random splits
   - Emphasize scaffold split results for generalization

3. **Ablation Studies**
   - Use the same splits for all ablation studies
   - Ensure fair comparison across different model variants

## Legacy Scripts

Legacy scripts for data processing are preserved in `scripts/legacy/`:
- `convert_xtb_to_pth.py` - Original XTB to PTH conversion
- `compute_rdkit_3d_features.py` - Original RDKit feature computation
- `check_data_leakage.py` - Data leakage checking
- `split_multimodal_fixed.py` - Scaffold-based data splitting
- `verify_split.py` - Split verification

These are kept for reference and backward compatibility.

## References

- Dataset sources and references will be added here
- XTB software: https://xtb-python.readthedocs.io/
- GFN2-xTB method: https://doi.org/10.1021/acs.jctc.9b00141
