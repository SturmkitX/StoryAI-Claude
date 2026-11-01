#!/usr/bin/env bash
# Run once on a fresh Vast.ai instance using a PyTorch image with CUDA.
set -euo pipefail

cd "$(dirname "$0")/.."
python -m venv --system-site-packages .venv
.venv/bin/pip install --quiet -r requirements.txt
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'No GPU visible'; print(torch.cuda.get_device_name(0))"
.venv/bin/python -m pytest --quiet eval data train
