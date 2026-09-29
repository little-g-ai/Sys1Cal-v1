#!/usr/bin/env bash
set -euo pipefail

python scripts/evaluate_sys1cal.py \
  --dataset data/v0.1.0_tiny_cleanvars_parallel_primitives.jsonl \
  --predictions examples/predictions_minimal.jsonl
