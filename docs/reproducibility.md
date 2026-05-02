# Reproducibility

To reproduce a training run, keep the following fixed:

- Git commit.
- Config file.
- Raw data files.
- XTB and RDKit feature files.
- Split directory.
- MoLFormer checkpoint or model id.

## Minimal Run

```bash
conda env create -f environment.yml
conda activate single_component_mp
python scripts/02_train.py --config configs/main_scaffold.yaml
```

Before training, place the required files under `data/raw/` and
`data/processed/`, and set `model_name_or_path` or `MOLFORMER_MODEL`.

## Tracked Splits

The scaffold fold indices in `splits/canonical_v2_scaffold/` are committed.
Use `configs/main_scaffold.yaml` to train with those frozen folds.

## What Is Not Tracked

The repository does not track raw datasets, processed feature tensors,
checkpoints, scalers, logs, or generated outputs.
