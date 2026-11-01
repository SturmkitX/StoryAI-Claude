#!/usr/bin/env bash
# Prepare the data, fine-tune SmolLM2-360M, export to INT8, then score it against
# the unmodified model on the same stories. Rerun after an interruption to continue.
set -euo pipefail

cd "$(dirname "$0")/.."
BASE=HuggingFaceTB/SmolLM2-360M-Instruct
RUN=runs/smollm2-360m
STORIES="${STORIES:-10}"
PY=.venv/bin/python

[ -f data/processed/train.jsonl ] || { $PY data/download.py; $PY data/prepare.py; }

[ -f "$RUN/final/model.safetensors" ] || $PY train/train_finetune.py --model "$BASE" --output "$RUN" "$@"
[ -f "$RUN/int8/model.onnx" ] || $PY export/export_onnx.py --base "$BASE" --model "$RUN/final" --output "$RUN/int8"

$PY eval/continuity_suite.py --model "$BASE" --onnx-file "$BASE:onnx/model_int8.onnx" \
  --output results/finetune/baseline.jsonl --stories "$STORIES"
$PY eval/continuity_suite.py --model "$RUN/final" --onnx-file "$RUN/int8/model.onnx" \
  --output results/finetune/finetuned.jsonl --stories "$STORIES"
