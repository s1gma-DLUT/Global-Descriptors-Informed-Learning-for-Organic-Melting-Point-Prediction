# Single-Component Melting Point Prediction

This repository contains the core code needed to reproduce a multimodal model
for single-component melting point prediction. The model combines SMILES,
molecular graph, XTB-derived quantum features, and RDKit descriptors.

## What Is Included

```text
single_component_mp_prediction/
|-- configs/                 Experiment configurations
|-- data/                    Data placeholders and schema notes
|-- docs/                    Method, dataset, reproducibility, and schema docs
|-- scripts/                 Data preparation, training, inference, evaluation
|-- splits/                  Frozen split indices when available
|-- src/                     Reusable preprocessing and utility code
|-- environment.yml          Conda environment
|-- requirements.txt         Pip dependency list
|-- LICENSE                  MIT license
`-- README.md
```

Large data files, model checkpoints, XTB job outputs, and generated reports are
not tracked by git. See [data/README.md](data/README.md) for the expected file
layout.

## Model Overview

The main architecture uses four information sources:

- MoLFormer encoder for SMILES sequence representations.
- D-MPNN encoder for molecular graph representations.
- XTB feature bundle for electronic, charge, and dipole properties.
- RDKit descriptors for molecular volume and auxiliary descriptors.

The main training implementation currently lives in
[`scripts/legacy_main_train.py`](scripts/legacy_main_train.py). The cleaner
entry point [`scripts/02_train.py`](scripts/02_train.py) loads a YAML config and
dispatches to that training implementation.

## Environment

### Conda

```bash
conda env create -f environment.yml
conda activate single_component_mp
```

### Pip

```bash
pip install -r requirements.txt
```

XTB itself is usually easiest to install from conda-forge:

```bash
conda create -n xtb -c conda-forge xtb
conda run -n xtb xtb --version
```

The scripts default to `conda run -n xtb xtb` where possible. If your XTB
environment has another name, pass the relevant command-line option or edit the
config for your local machine.

## Required Data Layout

Place private or regenerated data under `data/` without committing it:

```text
data/
|-- raw/
|   |-- multimodal_train.csv
|   `-- multimodal_test.csv
|-- processed/
|   |-- XTB_train.pth
|   |-- XTB_test.pth
|   |-- rdkit3d_train.npy
|   `-- rdkit3d_test.npy
`-- external/
    `-- new_molecules.csv
```

Expected CSV columns are:

- `SMILES`: molecule string.
- `MP`: melting point target, normally in degrees Celsius.

The XTB feature bundle is a 17-dimensional mixed-source feature file:

- 16 dimensions parsed or derived from XTB output.
- 1 dimension, `Molecular_Volume_cm3_mol`, computed with RDKit.

See [docs/xtb_feature_schema.md](docs/xtb_feature_schema.md) for the exact
schema.

## Reproduction Workflow

1. Prepare the environment.

   ```bash
   conda env create -f environment.yml
   conda activate single_component_mp
   ```

2. Put the raw CSV and feature files in `data/raw/` and `data/processed/`.

3. Build or verify frozen splits.

   ```bash
   python scripts/01_build_scaffold_split.py
   ```

4. Train the main scaffold model.

   ```bash
   python scripts/02_train.py --config configs/main_scaffold.yaml
   ```

5. Evaluate or run inference with the produced checkpoint directory.

   ```bash
   python scripts/inference_test_set.py \
     --run_root outputs/YOUR_RUN \
     --test_csv data/raw/multimodal_test.csv \
     --xtb_path data/processed/XTB_test.pth \
     --rdkit_path data/processed/rdkit3d_test.npy \
     --output_csv outputs/test_predictions.csv
   ```

## Computing XTB Features For New Molecules

1. Create `data/external/new_molecules.csv`.

   ```csv
   SMILES
   CCO
   c1ccccc1
   ```

2. Identify molecules missing from an existing feature bundle.

   ```bash
   python scripts/00b_compute_xtb_features.py \
     --input data/external/new_molecules.csv \
     --existing_xtb data/processed/XTB_train.pth \
     --output_dir data/external/xtb_jobs \
     --step identify
   ```

3. Generate XTB input files and a batch script.

   ```bash
   python scripts/00b_compute_xtb_features.py \
     --input data/external/new_molecules.csv \
     --existing_xtb data/processed/XTB_train.pth \
     --output_dir data/external/xtb_jobs \
     --step generate_cmds
   ```

4. Run the generated XTB jobs on a machine with XTB installed.

   ```bash
   bash data/external/xtb_jobs/run_xtb_batch.sh
   ```

5. Parse XTB output, compute RDKit volume, and merge to the 17D bundle.

   ```bash
   python -m src.preprocessing.xtb_extract \
     --xtb_dir data/external/xtb_jobs/outputs \
     --output_csv data/external/xtb_parsed/extracted_features.csv

   python scripts/00c_compute_rdkit_volume.py \
     --input_smiles data/external/xtb_jobs/missing_molecules.csv \
     --output_csv data/external/rdkit_volumes.csv

   python scripts/00d_merge_feature_bundle.py \
     --xtb_csv data/external/xtb_parsed/extracted_features.csv \
     --volume_csv data/external/rdkit_volumes.csv \
     --output_pth data/external/merged_features/XTB_merged.pth \
     --existing_pth data/processed/XTB_train.pth
   ```

## Configuration Notes

The YAML configs use relative data paths. Set `model_name_or_path` to either a
local MoLFormer checkpoint directory or a Hugging Face model identifier that is
compatible with your checkpoints.

Because large artifacts are intentionally ignored, a clean clone needs data and
checkpoint files supplied separately before full reproduction can run.

## Repository Hygiene

- Do not commit raw data, processed feature tensors, checkpoints, logs, or
  generated reports.
- Commit frozen split files and lightweight manifests needed to reproduce
  reported numbers.
- Record the git commit, config file, data version, and checkpoint directory for
  every reported experiment.

See [docs/reproducibility.md](docs/reproducibility.md) for more detail.
