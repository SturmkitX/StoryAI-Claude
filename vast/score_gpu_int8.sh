#!/usr/bin/env bash
# Score the original and fine-tuned models on the GPU, both as INT8 copies.
# Run after training has finished. Rerun after an interruption to continue.
set -euo pipefail

cd "$(dirname "$0")/.."
BASE=HuggingFaceTB/SmolLM2-360M-Instruct
RUN=runs/smollm2-360m
STORIES="${STORIES:-10}"
PY=.venv/bin/python

$PY -c "import bitsandbytes" 2>/dev/null || .venv/bin/pip install --quiet -r requirements.txt

[ -f "$RUN/final/model.safetensors" ] || { echo "Training hasn't finished: $RUN/final is missing" >&2; exit 1; }
[ -f runs/baseline-360m/gpu-int8/config.json ] || $PY export/export_gpu_int8.py --model "$BASE" --output runs/baseline-360m/gpu-int8
[ -f "$RUN/gpu-int8/config.json" ] || $PY export/export_gpu_int8.py --model "$RUN/final" --output "$RUN/gpu-int8"

$PY eval/continuity_suite.py --model runs/baseline-360m/gpu-int8 --device cuda \
  --output results/finetune-gpu/baseline.jsonl --stories "$STORIES"
$PY eval/continuity_suite.py --model "$RUN/gpu-int8" --device cuda \
  --output results/finetune-gpu/finetuned.jsonl --stories "$STORIES"
