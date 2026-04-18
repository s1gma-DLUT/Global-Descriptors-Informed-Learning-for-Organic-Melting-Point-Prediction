# Reproducibility Guidelines

## Overview

This document provides guidelines for ensuring reproducibility of the results presented in our paper on single-component melting point prediction.

## Data, Code, and Configuration

### Data
- Raw data should be placed in `data/raw/`
- Processed data will be saved in `data/processed/`
- Data splits are stored in `splits/` and are tracked by git to ensure consistency
- XTB feature libraries are stored in `data/processed/` with explicit versioning in filenames

### Code
- All code is version-controlled using git
- Each experiment should be run from a specific git commit to ensure reproducibility
- The main training entry point is `scripts/02_train.py`
- Legacy code is preserved in `scripts/legacy_main_train.py` and `scripts/legacy/` for reference

### Configuration
- Configuration files are stored in `configs/`
- Each experiment should use a specific configuration file
- Configuration files define all hyperparameters and experimental settings
- Always specify exact data paths (including XTB feature library version) in config files

## Split Management

### Why Freeze Splits?
- Splits are frozen to ensure that all experiments use the same train/validation/test partitions
- This allows for fair comparison between different models and ablation studies
- Split files are tracked by git to prevent accidental changes

### Split Types
- `splits/scaffold/`: Scaffold-based splits for main experiments
- `splits/random/`: Random splits for baseline comparisons

## Mixed-Source Feature Bundle Versioning

The feature bundle is a hybrid of XTB and RDKit features, which requires careful versioning.

### Current Feature Files

| File | Description | Version |
|------|-------------|---------|
| `XTB_train.pth` | Original training set mixed-source features | v1 (original) |
| `XTB_test.pth` | Original test set mixed-source features | v1 (original) |

### Feature Sources

- **16 dimensions**: XTB quantum mechanical calculations
- **1 dimension**: RDKit molecular volume (Molecular_Volume_cm3_mol)

### Versioning Requirements

To ensure reproducibility, record the following components:

1. **XTB Parser Version**: Track changes to `src/preprocessing/xtb_extract.py`
2. **RDKit Volume Calculation Version**: Track changes to `src/preprocessing/rdkit_features.py`
3. **Merge Schema Version**: Track changes to `src/preprocessing/merge_features.py`
4. **Feature Bundle Version**: Include version info in the output .pth file

### Extended Feature Libraries

When extending to new molecules, create versioned filenames:

```
XTB_train_YYYYMMDD.pth      # Dated version
XTB_train_extended.pth      # Symlink to latest
XTB_train_v2.pth            # Numbered version
```

### Versioning Best Practices

1. **Never overwrite original feature files**: Always create new files when extending
2. **Use descriptive names**: Include date or version number
3. **Track provenance**: Record in `data/metadata/` which molecules were added
4. **Update configs**: Always specify exact feature file path in experiment configs
5. **Register experiments**: Record which feature file was used in `experiments/experiment_registry.csv`

### Example Feature Extension Record

Create `data/metadata/XTB_extension_log.md`:
```markdown
# XTB Feature Extension Log

## Extension 1 (2026-04-18)
- Original: XTB_train.pth (268745 molecules)
- New molecules: 1500
- Extended file: XTB_train_20260418.pth (270245 molecules)
- XTB method: GFN2-xTB
- Config used: configs/main_scaffold_extended.yaml
```

## Split Management

### Why Freeze Splits?
- Splits are frozen to ensure that all experiments use the same train/validation/test partitions
- This allows for fair comparison between different models and ablation studies
- Split files are tracked by git to prevent accidental changes

### Split Types
- `splits/scaffold/`: Scaffold-based splits for main experiments
- `splits/random/`: Random splits for baseline comparisons

## Result Tracking

### Experiment Registry
All experiments are recorded in `experiments/experiment_registry.csv`. Each entry includes:
- Experiment ID
- Status
- Task name
- Split type
- Configuration file
- Code entry point
- Git commit
- Output directory
- XTB feature library version used (via path)
- Performance metrics
- Notes

### Output Organization
All outputs are saved in `outputs/` directory:
- Each experiment gets its own subdirectory with a timestamp
- Outputs include:
  - Model checkpoints
  - Prediction files
  - Evaluation metrics
  - Training logs

### Merged Feature Files

When using merged XTB features, the merged file should be tracked:

```
merged_xtb_feature_path: "./data/external/merged_features/XTB_merged_20260418.pth"
```

The merge operation records:
- Original library source
- New features source
- Merge date
- Number of molecules added

## Environment Setup

### Conda Environment
```bash
conda env create -f environment.yml
conda activate single_component_mp
```

### Pip Environment
```bash
pip install -r requirements.txt
```

## Running Experiments

### Main Experiment
```bash
bash scripts/run_main_scaffold.sh
```

### Custom Experiments
```bash
python scripts/02_train.py --config configs/your_config.yaml
```

## Extending XTB Features for New Molecules

### Complete Workflow

1. **Prepare new molecules** in `data/external/new_molecules.csv`

2. **Identify missing molecules**:
   ```bash
   python scripts/00b_compute_xtb_features.py \
       --input data/external/new_molecules.csv \
       --existing_xtb data/processed/XTB_train.pth \
       --output_dir data/external/xtb_jobs \
       --step identify
   ```

3. **Generate XTB commands**:
   ```bash
   python scripts/00b_compute_xtb_features.py \
       --input data/external/new_molecules.csv \
       --existing_xtb data/processed/XTB_train.pth \
       --output_dir data/external/xtb_jobs \
       --step generate_cmds
   ```

4. **Run XTB calculations**:
   ```bash
   cd data/external/xtb_jobs
   bash run_xtb_batch.sh
   ```

5. **Extract XTB features**:
   ```bash
   python -m src.preprocessing.xtb_extract \
       --xtb_dir data/external/xtb_jobs/outputs \
       --output_csv data/external/xtb_parsed/extracted_features.csv
   ```

6. **Calculate RDKit volume**:
   ```bash
   python scripts/00c_compute_rdkit_volume.py \
       --input_smiles data/external/xtb_jobs/missing_molecules.csv \
       --output_csv data/external/rdkit_volumes.csv
   ```

7. **Merge feature bundle**:
   ```bash
   python scripts/00d_merge_feature_bundle.py \
       --xtb_csv data/external/xtb_parsed/extracted_features.csv \
       --volume_csv data/external/rdkit_volumes.csv \
       --output_pth data/processed/XTB_train_20260418.pth \
       --existing_pth data/processed/XTB_train.pth
   ```

8. **Update config** with new feature path:
   ```yaml
   xtb_feature_path: "./data/processed/XTB_train_20260418.pth"
   ```

9. **Register experiment** in `experiments/experiment_registry.csv`

### Avoiding Feature Library Confusion

1. **Never modify original files**: Always create new extended versions
2. **Version your files**: Use dates or version numbers in filenames
3. **Track in metadata**: Keep logs in `data/metadata/`
4. **Update configs**: Always specify exact file paths
5. **Document in registry**: Record which feature file each experiment used

## Result Reproduction

To reproduce the results from our paper:
1. Checkout the specific git commit mentioned in the paper
2. Set up the environment using the provided environment files
3. Place the required data in `data/raw/`
4. Run the data preparation script
5. Generate the splits (or use the provided ones)
6. Run the main experiment using the provided configuration
7. Evaluate the results using `scripts/03_eval_cv.py`

## Troubleshooting

If you encounter any issues with reproducibility:
- Ensure you're using the exact same git commit
- Verify that the splits are identical
- Check that the XTB feature library version is correct
- Verify the config file paths point to the correct data
- Check that the environment is set up correctly
