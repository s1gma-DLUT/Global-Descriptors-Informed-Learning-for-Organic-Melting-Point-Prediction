# Dataset

The training code expects single-component molecules with experimental melting
points.

## Included CSV

```text
data/raw/cleaned/data_set.csv
```

Raw CSV columns:

- `SMILES`: molecular representation.
- `MP`: melting point target.

Rows with missing `SMILES`, missing `MP`, or missing feature entries are not
used by the training loader. The loader aligns samples by SMILES against the
feature bundle before constructing folds.

## Feature Files

Training requires precomputed feature files placed locally under `data/`.
The default local layout is:

```text
data/raw/cleaned/XTB_train.pth
data/raw/cleaned/XTB_test.pth
data/raw/cleaned/rdkit3d_train.npy
data/raw/cleaned/rdkit3d_test.npy
```

`XTB_train.pth` and `XTB_test.pth` contain 17 physicochemical features per
molecule. The first 16 are parsed or derived from XTB output, and the final
feature is RDKit-derived molecular volume.

`rdkit3d_train.npy` and `rdkit3d_test.npy` contain RDKit descriptor arrays used
by the training script.

All feature files must be aligned with the SMILES order used by the dataset
loader.

Expected bundle contents:

- `XTB_train.pth`: dictionary-like object with `features` and `smiles`.
- `rdkit3d_train.npy`: numeric array with one descriptor row per aligned
  molecule.
- Test-set files follow the same convention when used for downstream
  prediction.

The current training implementation expects 17 XTB/RDKit bundle features and
25 RDKit descriptor features.

## Splits

The main experiment uses frozen scaffold splits under:

```text
splits/scaffold/
```

These split files are tracked so the same folds can be reused across runs.

The `.npy` files contain validation indices for each fold plus `none_idx.npy`
for molecules without a usable scaffold. None-scaffold samples are kept out of
validation folds in the scaffold experiment.
