# Melting Point Prediction for Single-Component Systems

This repository contains the training code for a multimodal melting point
prediction model. The model uses SMILES text, molecular graphs, XTB features,
and RDKit descriptors.

## Repository Layout

```text
configs/      Training configurations
data/         CSV data and local feature-file placeholders
docs/         Short notes on data, model, and reproduction
scripts/      Training and preprocessing entry points
splits/       Frozen split indices
src/          Reusable preprocessing utilities
```

The main training entry point is:

```bash
python scripts/02_train.py --config configs/main_scaffold.yaml
```

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate single_component_mp
```

Or install with pip:

```bash
pip install -r requirements.txt
```

XTB is required only if you need to regenerate XTB features. The training script
expects precomputed feature files.

## Data

The cleaned CSV dataset is included at:

```text
data/raw/cleaned/data_set.csv
```

It contains:

- `SMILES`
- `MP`

Training also requires precomputed feature files:

```text
data/raw/cleaned/XTB_train.pth
data/raw/cleaned/rdkit3d_train.npy
```

CSV files under `data/raw/` are tracked. Feature tensors and checkpoints are
ignored by git.

## Model Path

Set `model_name_or_path` in the config to a local MoLFormer checkpoint or a
compatible Hugging Face model id. You can also export:

```bash
export MOLFORMER_MODEL=/path/to/MoLFormer
```

On Windows PowerShell:

```powershell
$env:MOLFORMER_MODEL="C:\path\to\MoLFormer"
```

## Training

Main scaffold split:

```bash
python scripts/02_train.py --config configs/main_scaffold.yaml
```

Random split baseline:

```bash
python scripts/02_train.py --config configs/main_random.yaml
```

The training wrapper loads the YAML config and calls
`scripts/legacy_main_train.py`, which contains the model and training loop.

## Configs

- `configs/main_scaffold.yaml`: main frozen scaffold split training.
- `configs/main_random.yaml`: random split baseline.
- `configs/ablation_no_xtb.yaml`: ablation without XTB features.
- `configs/ablation_no_dmpnn.yaml`: ablation without D-MPNN.

## Notes

- Keep feature tensors, checkpoints, logs, and generated outputs out of git.
- Frozen split indices under `splits/canonical_v2_scaffold/` are tracked for
  reproducible training.
- See `docs/` for compact details about the model and data schema.
