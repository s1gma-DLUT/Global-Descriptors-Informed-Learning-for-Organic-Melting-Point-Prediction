# Method

The model predicts melting point from four molecular representations:

- MoLFormer SMILES encoder.
- D-MPNN molecular graph encoder.
- XTB physicochemical feature vector.
- RDKit descriptor vector.

The public training entry point is `scripts/02_train.py`.

## Training

The model is trained as a regression model with Huber loss. The optimizer uses
separate learning rates for the pretrained text encoder, graph encoder, and
fusion layers.

The default scaffold configuration trains 5 folds using the frozen indices in
`splits/scaffold/`.

## Inputs

Each sample uses:

- `SMILES`
- melting point target `MP`
- 17D XTB/RDKit feature bundle
- RDKit descriptor array

## Metrics

The training loop computes standard regression metrics such as MAE, RMSE, and
R^2.
