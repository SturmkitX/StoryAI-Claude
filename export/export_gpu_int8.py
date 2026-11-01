"""Save an INT8 copy of a model that runs on a GPU, for fast scoring.

The phone still uses the ONNX export. This one rounds weights differently, so
use it to compare models with each other, not to predict exact phone scores.

    python export/export_gpu_int8.py --model runs/smollm2-360m/final \
        --output runs/smollm2-360m/gpu-int8
"""

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": 0},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.output)

    int8_bytes = sum(p.numel() for p in model.parameters() if p.dtype.is_floating_point is False)
    other_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.dtype.is_floating_point)
    print(f"{int8_bytes / 1e6:.0f} MB of INT8 weights, {other_bytes / 1e6:.0f} MB left as floats, saved to {args.output}")


if __name__ == "__main__":
    main()
