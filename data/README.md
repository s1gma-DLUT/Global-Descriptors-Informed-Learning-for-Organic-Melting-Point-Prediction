# Data

This directory is a placeholder for local data. Large files are not committed.

Expected files:

```text
data/raw/multimodal_train.csv
data/raw/multimodal_test.csv
data/processed/XTB_train.pth
data/processed/XTB_test.pth
data/processed/rdkit3d_train.npy
data/processed/rdkit3d_test.npy
```

CSV files should contain:

- `SMILES`
- `MP`

`XTB_*.pth` files are expected to contain a 17-dimensional feature matrix and
aligned SMILES strings. `rdkit3d_*.npy` files should be aligned with the same
training/test molecules used by the training script.

Do not commit raw data, processed feature tensors, checkpoints, or generated
outputs.
