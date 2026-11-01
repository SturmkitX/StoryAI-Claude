#!/usr/bin/env bash
# Score all three unmodified models at INT8 on CPU, as the phone will run them.
# Rerun after an interruption to continue from the last finished story.
set -euo pipefail

cd "$(dirname "$0")/.."

screen_model() {
  .venv/bin/python eval/continuity_suite.py --model "$2" --onnx-file "$3" --output "results/screening/$1.jsonl"
}

screen_model smollm2-135m HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-135M-Instruct:onnx/model_int8.onnx
screen_model smollm2-360m HuggingFaceTB/SmolLM2-360M-Instruct HuggingFaceTB/SmolLM2-360M-Instruct:onnx/model_int8.onnx
screen_model qwen3-0.6b Qwen/Qwen3-0.6B onnx-community/Qwen3-0.6B-ONNX:onnx/model_int8.onnx
