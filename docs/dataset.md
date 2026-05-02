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

## Feature Files

Training requires precomputed feature files placed locally under `data/`.

`XTB_train.pth` and `XTB_test.pth` contain 17 physicochemical features per
molecule. The first 16 are parsed or derived from XTB output, and the final
feature is RDKit-derived molecular volume.

`rdkit3d_train.npy` and `rdkit3d_test.npy` contain RDKit descriptor arrays used
by the training script.

All feature files must be aligned with the SMILES order used by the dataset
loader.

## Splits

The main experiment uses frozen scaffold splits under:

```text
splits/canonical_v2_scaffold/
```

These split files are tracked so the same folds can be reused across runs.
