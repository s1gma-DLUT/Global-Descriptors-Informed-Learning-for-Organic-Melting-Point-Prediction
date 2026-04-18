# Single-Component Melting Point Prediction

## Project Overview

This repository implements a state-of-the-art model for single-component melting point (MP) prediction using multimodal fusion techniques. The model integrates multiple molecular representations to achieve accurate predictions.

## Method Overview

The core approach combines:
- **MoLFormer** (SMILES encoder) - for sequential representation of molecular structures
- **D-MPNN** (Directed Message Passing Neural Network) - for graph-based molecular representation
- **Mixed-source XTB+RDKit features** - for quantum mechanical properties and molecular volume
- **Residual boosting / dynamic readout** fusion head - for effective integration of multiple modalities

## Repository Structure

```
single_component_mp_prediction/
├── README.md              # Project documentation
├── LICENSE                # License file
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies (pip format)
├── environment.yml        # Conda environment file
├── data/                  # Data directory
│   ├── README.md          # Data documentation
│   ├── raw/               # Raw data (SMILES + MP)
│   ├── processed/         # Processed features (RDKit, XTB)
│   ├── metadata/           # Metadata
│   └── external/           # External/new molecule data
│       ├── new_molecules.csv    # New molecules to compute
│       ├── xtb_jobs/            # XTB calculation jobs
│       ├── xtb_parsed/         # Parsed XTB features
│       └── merged_features/     # Merged feature libraries
├── splits/                # Data splits
│   ├── scaffold/          # Scaffold-based splits
│   └── random/            # Random splits
├── configs/               # Configuration files
│   ├── main_scaffold.yaml     # Main experiment configuration
│   ├── random_baseline.yaml   # Random split baseline
│   ├── ablation_no_xtb.yaml   # Ablation without XTB features
│   └── ablation_no_dmpnn.yaml # Ablation without D-MPNN
├── src/                   # Source code
│   ├── __init__.py
│   ├── preprocessing/         # Data preprocessing
│   │   ├── __init__.py
│   │   ├── xtb_extract.py     # XTB feature extraction
│   │   └── merge_features.py  # Feature merging
│   ├── data/              # Data loading and processing
│   ├── models/            # Model definitions
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation utilities
│   └── utils/             # Helper functions
│       ├── __init__.py
│       └── smiles.py       # SMILES canonicalization
├── scripts/               # Scripts for data preparation, training, and evaluation
│   ├── legacy/                 # Legacy scripts (preserved)
│   │   ├── convert_xtb_to_pth.py
│   │   ├── compute_rdkit_3d_features.py
│   │   ├── check_data_leakage.py
│   │   ├── split_multimodal_fixed.py
│   │   └── verify_split.py
│   ├── 00_prepare_data.py        # Data preparation
│   ├── 00b_compute_xtb_features.py # XTB computation for new molecules
│   ├── 00c_merge_xtb_features.py   # Merge XTB features
│   ├── 01_build_scaffold_split.py # Scaffold split generation
│   ├── 02_train.py               # Training script
│   ├── 03_eval_cv.py             # Cross-validation evaluation
│   ├── 04_make_tables.py          # Table generation
│   ├── 05_make_figures.py         # Figure generation
│   ├── run_main_scaffold.sh       # Main experiment runner
│   └── legacy_main_train.py       # Legacy training script (snapshot)
├── experiments/           # Experiment registry
│   ├── experiment_registry.csv    # Experiment tracking
│   └── README.md                  # Experiment documentation
├── outputs/               # Output directory (not tracked by git)
├── reports/               # Reports and figures
│   ├── figures/           # Generated figures
│   ├── tables/            # Generated tables
│   └── paper_notes/       # Paper-related notes
└── docs/                  # Documentation
    ├── method.md          # Method description
    ├── dataset.md         # Dataset documentation
    ├── reproducibility.md # Reproducibility guidelines
    └── changelog.md       # Changelog
```

## Environment Setup

### XTB Installation

XTB (eXtended Tight Binding) is required for computing quantum mechanical features.

**Your system has XTB installed in conda environment: `pxf_xtb`**

To check XTB availability:
```bash
python scripts/00b_compute_xtb_features.py --check_xtb
```

To install XTB if needed:
```bash
conda create -n pxf_xtb -c conda-forge xtb
conda activate pxf_xtb
```

### Python Environment

```bash
conda env create -f environment.yml
conda activate single_component_mp
```

Or using pip:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Place your raw data in `data/raw/`
2. Run the data preparation script:
   ```bash
   python scripts/00_prepare_data.py
   ```
3. Generate scaffold splits:
   ```bash
   python scripts/01_build_scaffold_split.py
   ```

## Training

### Main Experiment (Scaffold Split)

```bash
bash scripts/run_main_scaffold.sh
```

### Custom Training

```bash
python scripts/02_train.py --config configs/main_scaffold.yaml
```

## Evaluation

```bash
python scripts/03_eval_cv.py --output_dir outputs/your_experiment
```

## Extending to New Molecules

This repository supports computing XTB features for new molecules and merging them with the existing feature library.

### Why Extend XTB Features?

If you have a new set of molecules without XTB features, you can:
1. Compute XTB features for these molecules
2. Merge them with the existing feature library
3. Use the extended library for training or inference

### Workflow Overview

```
New Molecules (CSV)
    │
    ▼
┌─────────────────────────────────────────┐
│ 00b_compute_xtb_features.py            │
│   --step identify                       │
│   --existing_xtb data/processed/        │
│           XTB_train.pth                 │
└─────────────────────────────────────────┘
    │  Identifies which molecules are missing
    ▼
┌─────────────────────────────────────────┐
│ 00b_compute_xtb_features.py            │
│   --step generate_cmds                  │
└─────────────────────────────────────────┘
    │  Generates run_xtb_batch.sh
    ▼
┌─────────────────────────────────────────┐
│ conda run -n pxf_xtb bash               │
│     data/external/xtb_jobs/             │
│     run_xtb_batch.sh                    │
└─────────────────────────────────────────┘
    │  Runs XTB calculations (requires pxf_xtb env)
    ▼
┌─────────────────────────────────────────┐
│ src/preprocessing/xtb_extract.py       │
│   --xtb_dir xtb_jobs/outputs/           │
│   --output_csv xtb_parsed/              │
│         extracted_features.csv          │
└─────────────────────────────────────────┘
    │  Extracts 16D XTB features
    ▼
┌─────────────────────────────────────────┐
│ 00c_compute_rdkit_volume.py            │
│   --input_smiles missing_molecules.csv  │
│   --output_csv rdkit_volumes.csv        │
└─────────────────────────────────────────┘
    │  Calculates 1D volume feature
    ▼
┌─────────────────────────────────────────┐
│ 00d_merge_feature_bundle.py            │
│   --xtb_csv extracted_features.csv      │
│   --volume_csv rdkit_volumes.csv        │
│   --output_pth XTB_train_extended.pth  │
└─────────────────────────────────────────┘
    │  Creates 17D feature bundle
    ▼
Extended Feature Library (17D)
    │
    ▼
Training / Inference
```

### Step-by-Step Guide

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

This will tell you:
- How many molecules are already in the library
- How many need new XTB computation

#### Step 3: Generate XTB Commands

```bash
python scripts/00b_compute_xtb_features.py \
    --input data/external/new_molecules.csv \
    --existing_xtb data/processed/XTB_train.pth \
    --output_dir data/external/xtb_jobs \
    --step generate_cmds
```

This generates:
- `data/external/xtb_jobs/run_xtb_batch.sh` - Batch script
- Input files in `data/external/xtb_jobs/inputs/`

#### Step 4: Run XTB Calculations

```bash
conda run -n pxf_xtb bash data/external/xtb_jobs/run_xtb_batch.sh
```

**Important**: Requires the `pxf_xtb` conda environment.

#### Step 5: Extract XTB Features

```bash
python -m src.preprocessing.xtb_extract \
    --xtb_dir data/external/xtb_jobs/outputs \
    --output_csv data/external/xtb_parsed/extracted_features.csv
```

#### Step 6: Calculate RDKit Volume

```bash
python scripts/00c_compute_rdkit_volume.py \
    --input_smiles data/external/xtb_jobs/missing_molecules.csv \
    --output_csv data/external/rdkit_volumes.csv
```

#### Step 7: Merge Feature Bundle

```bash
python scripts/00d_merge_feature_bundle.py \
    --xtb_csv data/external/xtb_parsed/extracted_features.csv \
    --volume_csv data/external/rdkit_volumes.csv \
    --output_pth data/processed/XTB_train_extended.pth \
    --existing_pth data/processed/XTB_train.pth
```

#### Step 8: Train with Extended Features

Update `configs/main_scaffold.yaml`:
```yaml
xtb_feature_path: "./data/processed/XTB_train_extended.pth"
```

Then run training:
```bash
python scripts/02_train.py --config configs/main_scaffold.yaml
```

### About Canonical SMILES Matching

By default, the system uses **canonical SMILES matching** to:
- Avoid treating different SMILES representations of the same molecule as different molecules
- Correctly identify duplicates when merging feature libraries

For example, `CCO`, `OCC`, and `c1cc(O)ccc1` might all represent ethanol with different SMILES formats.

To disable canonical matching (faster but less robust):
```bash
python scripts/00b_compute_xtb_features.py ... --no_canonical
python scripts/00c_merge_xtb_features.py ... --no_canonical
```

## XTB Feature Details

XTB features provide 17 quantum mechanical properties:
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

## Results

Results will be saved in the `outputs/` directory, including:
- Model checkpoints
- Prediction files
- Evaluation metrics
- Training logs

## Reproducibility

To ensure reproducibility:
1. Use the provided split files in `splits/`
2. Use the exact configuration files
3. Set the random seed as specified in the config
4. Use the same environment setup
5. Track which XTB feature library version was used in `experiments/experiment_registry.csv`

## Current Status

- Repository structure is initialized
- Configuration templates are created
- Legacy training script is preserved
- XTB feature extraction and merging modules are implemented
- Data preparation scripts are in place
- Canonical SMILES matching is supported
- XTB dependency checking is implemented
- Ready for main training runs and new molecule extensions

## Citation

If you use this code, please cite our paper (to be added).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
