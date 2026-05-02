# Contributing

Thanks for improving this repository. Please keep contributions focused on
reproducibility, correctness, and clear experiment tracking.

## Development Guidelines

- Keep generated data, checkpoints, logs, and reports out of git.
- Prefer small, reviewable changes.
- Record any behavior-changing experiment updates in `docs/changelog.md`.
- Keep config paths relative when possible.
- Document any new required artifact in `data/README.md`.

## Before Opening A Pull Request

Run the lightweight checks that are possible in your environment:

```bash
python -m compileall src scripts
git diff --check
```

For model changes, also report:

- Config file used.
- Git commit.
- Dataset and feature-bundle version.
- Split directory.
- Checkpoint/output directory.
- Validation or test metrics.

## Large Artifacts

Do not commit raw data, `.pth` feature bundles, checkpoints, scaler files, or
generated report artifacts. Share those through an external artifact store and
document the expected placement under `data/` or `outputs/`.
