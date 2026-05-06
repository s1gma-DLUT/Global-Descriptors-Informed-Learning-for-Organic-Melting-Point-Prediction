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

## Tracked Splits

The scaffold fold indices in `splits/scaffold/` are committed.
Use `configs/main_scaffold.yaml` to train with those frozen folds.

## What Is Not Tracked

The repository tracks CSV files under `data/raw/`. It does not track processed
feature tensors, checkpoints, scalers, logs, or generated outputs.
