# Reproducibility

To reproduce a training run, keep the following fixed:

- Git commit.
- Config file.
- CSV data files.
- XTB and RDKit feature files.
- Split directory.
- MoLFormer checkpoint or model id.

## Minimal Run

```bash
conda env create -f environment.yml
conda activate single_component_mp
python scripts/02_train.py --config configs/main_scaffold.yaml
```

Before training, place the required feature files under `data/raw/cleaned/`, and
set `model_name_or_path` or `MOLFORMER_MODEL`.

For a local MoLFormer checkpoint:

```bash
export MOLFORMER_MODEL=/path/to/MoLFormer
```

On Windows PowerShell:

```powershell
$env:MOLFORMER_MODEL="C:\path\to\MoLFormer"
```

## Tracked Splits

The scaffold fold indices in `splits/scaffold/` are committed.
Use `configs/main_scaffold.yaml` to train with those frozen folds.

The default public random seed is `516`. Scaffold runs rely on the committed
indices; random-split runs use the configured seed to generate folds.

## Suggested Checks

Before launching a long run:

- Confirm that `data/raw/cleaned/data_set.csv` contains `SMILES` and `MP`.
- Confirm that feature files exist under `data/raw/cleaned/`.
- Confirm that the MoLFormer path resolves locally.
- Run `git status` and record the commit hash used for the experiment.

## What Is Not Tracked

The repository tracks CSV files under `data/raw/`. It does not track processed
feature tensors, checkpoints, scalers, logs, or generated outputs.
