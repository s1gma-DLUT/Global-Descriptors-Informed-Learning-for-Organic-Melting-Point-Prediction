#!/usr/bin/env bash
set -euo pipefail

python scripts/02_train.py --config configs/main_scaffold.yaml "$@"
