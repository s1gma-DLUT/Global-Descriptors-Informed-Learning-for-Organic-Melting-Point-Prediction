# Method

The model predicts melting point from four molecular representations:

- MoLFormer SMILES encoder.
- D-MPNN molecular graph encoder.
- XTB physicochemical feature vector.
- RDKit descriptor vector.

The public training entry point is `scripts/02_train.py`.

## Architecture

The model has three main branches:

- A MoLFormer text encoder consumes tokenized SMILES.
- A directed message-passing graph encoder consumes molecular graphs built from
  RDKit molecules.
- A tabular feature branch consumes the XTB/RDKit feature bundle and RDKit
  descriptors.

The branch outputs are fused by a small neural head that predicts melting point
as a scalar regression target.

## Training

The model is trained as a regression model with Huber loss. The optimizer uses
separate learning rates for the pretrained text encoder, graph encoder, and
fusion layers.

The default scaffold configuration trains 5 folds using the frozen indices in
`splits/scaffold/`.

The training schedule can freeze and unfreeze the pretrained text encoder,
switch batch size in later stages, and adjust regularization according to the
configuration. These settings are exposed in the YAML configs rather than being
hard-coded into the wrapper script.

## Inputs

Each sample uses:

- `SMILES`
- melting point target `MP`
- 17D XTB/RDKit feature bundle
- RDKit descriptor array

## Metrics

The training loop computes standard regression metrics such as MAE, RMSE, and
R^2.

Per-fold outputs include checkpoints, scalers, prediction tables, and metric
summaries under the configured output directory.
