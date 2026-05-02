# Data

This directory stores CSV data and local feature files.

Included CSV:

```text
data/raw/cleaned/data_set.csv
```

CSV files under `data/raw/` are tracked when they contain plain tabular data.
They should contain:

- `SMILES`
- `MP`

Training requires local feature files such as:

```text
data/processed/XTB_train.pth
data/processed/XTB_test.pth
data/processed/rdkit3d_train.npy
data/processed/rdkit3d_test.npy
```

Do not commit processed feature tensors, checkpoints, or generated outputs.
