# Contributing

Keep changes focused on the training pipeline.

Before submitting changes, run:

```bash
python -m compileall src scripts
git diff --check
```

Do not commit raw data, processed feature files, checkpoints, logs, or generated
outputs.
