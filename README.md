# Global Descriptors Informed Learning for Organic Melting-Point Prediction

This repository supports the ChemComm manuscript and ESI for a
descriptor-conditioned multimodal model for organic molecular melting-point
prediction. The Global-informed model combines:

- MoLFormer-c3-1.1B SMILES representations
- D-MPNN molecular graph representations
- xTB/RDKit-derived global descriptors used as a conditioning branch

The code also includes the topology-only ablation used in the ESI, where the
xTB/RDKit descriptor branch is removed and the MoLFormer + D-MPNN representation
is passed directly to an MLP regression head.

## Manuscript Alignment

The repository is organized around the validation protocol reported in the main
text and ESI:

- Curated dataset: 230,775 organic melting-point entries
- Curation threshold: records are retained from -150 to 350 degC
- Final observed dataset range: -149.10 to 350.00 degC
- Duplicate handling: canonical-SMILES duplicates with MP spread <= 30 degC are
  averaged; inconsistent groups are discarded
- Main validation: five-fold scaffold validation with acyclic/no-scaffold
  molecules kept in the training pool only
- Contextual comparison: five-fold random validation
- Ablation: topology-only model without the xTB/RDKit descriptor branch
- External comparison: independently provided `multimodal_test` style data can
  be evaluated with the inference script

Reported headline results from the manuscript/ESI are:

| Setting | MAE | RMSE | Notes |
| --- | ---: | ---: | --- |
| Global-informed, random fold 5 | 21.35 | 27.82 | Main-text comparison value |
| Global-informed, scaffold fold 5 | 23.31 | 29.98 | Main-text scaffold value |
| Global-informed, random CV mean | 21.45 +/- 0.08 | 27.88 +/- 0.08 | ESI Table S2 |
| Global-informed, scaffold CV mean | 23.06 +/- 0.25 | 29.74 +/- 0.24 | ESI Table S3 |
| Topology-only, random CV mean | 21.86 +/- 0.11 | 28.50 +/- 0.11 | ESI Table S1 |
| Topology-only, scaffold CV mean | 23.67 +/- 0.22 | 30.52 +/- 0.20 | ESI Table S4 |

All temperatures are in degC/K differences, so MAE and RMSE values have the same
numeric magnitude in degC and K.

## Quick Start

Create the environment and point the training code to a MoLFormer checkpoint or
Hugging Face model id:

```bash
conda env create -f environment.yml
conda activate single_component_mp
export MOLFORMER_MODEL=/path/to/MoLFormer
```

Run the Global-informed model with scaffold validation:

```bash
python scripts/02_train.py --config configs/main_scaffold.yaml
```

Run the Global-informed model with random validation:

```bash
python scripts/02_train.py --config configs/main_random.yaml
```

Run the topology-only scaffold ablation:

```bash
python scripts/02_train.py --config configs/ablation_no_xtb.yaml
```

Run the topology-only random ablation:

```bash
python scripts/02_train.py --config configs/ablation_no_xtb_random.yaml
```

## Repository Layout

```text
configs/    YAML configurations for scaffold, random, and ablation runs
data/       Curated CSV and local descriptor tensors
scripts/    Data preparation, split construction, training, evaluation, tables, figures
splits/     Frozen scaffold split indices and train-only no-scaffold indices
src/        Preprocessing utilities and shared SMILES/split helpers
```

## Data And Features

Tracked cleaned data:

- `data/raw/cleaned/data_set.csv`: curated SMILES/MP table
- `data/raw/cleaned/XTB_train.pth`: 17D descriptor bundle
- `data/raw/cleaned/rdkit3d_train.npy`: 25D RDKit descriptor matrix

The 17D descriptor bundle contains 16 parsed or derived GFN2-xTB features plus
one RDKit molecular-volume feature. The schema is defined in
`src/preprocessing/schema.py`.

The 25D RDKit descriptor matrix contains common 2D molecular descriptors used by
the model-side descriptor branch. During training, xTB/RDKit descriptors are
imputed and scaled within each fold using training-fold statistics only.

## Data Curation

The curation script validates SMILES, canonicalizes structures with RDKit,
filters melting points outside the accepted -150 to 350 degC range, resolves
duplicate canonical SMILES with the 30 degC consistency threshold, and writes a
normalized CSV. The cleaned dataset may have a slightly narrower observed range
if no molecule lies exactly at the threshold; in the ESI dataset the lowest
value is -149.10 degC.

```bash
python scripts/00_prepare_data.py \
  --input_csv data/raw/cleaned/data_set.csv \
  --output_csv data/raw/cleaned/data_set_prepared.csv
```

## Scaffold Splitting

The scaffold split uses Bemis-Murcko scaffolds. Molecules with the same scaffold
are assigned to the same validation fold. Acyclic molecules or molecules without
a valid ring scaffold are written to `splits/scaffold/none_idx.npy` and kept in
the training pool for every scaffold fold.

```bash
python scripts/01_build_scaffold_split.py \
  --input_csv data/raw/cleaned/data_set.csv \
  --xtb_pth data/raw/cleaned/XTB_train.pth \
  --output_dir splits/scaffold
```

## Model Summary

Global-informed model:

- SMILES branch: MoLFormer-c3-1.1B with the last four layers fine-tuned at a
  reduced learning rate
- Graph branch: D-MPNN with four message-passing steps
- Primary representation: concatenated SMILES and graph features projected to a
  16D molecular representation
- Descriptor branch: xTB/RDKit descriptors passed through an MLP to produce a
  16D weight vector and scalar bias
- Readout: descriptor-conditioned representation plus bias correction

Topology-only ablation:

- Keeps the same MoLFormer and D-MPNN branches
- Removes the full xTB/RDKit conditioning branch
- Uses a direct MLP regression head for prediction

## Prediction And Postprocessing

Use trained fold checkpoints for single-molecule or CSV prediction:

```bash
python scripts/03_eval_cv.py \
  --mode both \
  --model_dir_3d outputs/YOUR_GLOBAL_RUN \
  --model_dir_no3d outputs/YOUR_ABLATION_RUN \
  --smiles "Cc1ccccc1"
```

Generate compact result tables and scatter figures from output CSV files:

```bash
python scripts/04_make_tables.py --results_dir outputs
python scripts/05_make_figures.py --results_dir outputs
```

## Notes

- Default random seed: `516`
- Frozen scaffold files under `splits/scaffold/` are intended to reproduce the
  scaffold validation protocol in the manuscript and ESI
- xTB must be available for new-molecule Global-informed inference unless
  precomputed descriptors are supplied
- Source code and data availability statement:
  <https://github.com/s1gma-DLUT/Global-Descriptors-Informed-Learning-for-Organic-Melting-Point-Prediction>
