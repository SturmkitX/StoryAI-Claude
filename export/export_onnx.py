"""Export a fine-tuned model to an INT8 ONNX folder for the coherence test and the phone.

    python export/export_onnx.py --base HuggingFaceTB/SmolLM2-360M-Instruct \
        --model runs/smollm2-360m/final --output runs/smollm2-360m/int8
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

EMBEDDING = "model.embed_tokens.weight"
BLOCK_SIZE = 128  # Larger blocks store fewer scale values, so the file is smaller.


def build_int8(base, model, output, cache):
    subprocess.run(
        [
            sys.executable, "-m", "onnxruntime_genai.models.builder",
            "-m", base, "-i", str(model), "-o", str(output), "-c", str(cache),
            "-p", "int8", "-e", "cpu",
            "--extra_options", f"int4_block_size={BLOCK_SIZE}",
        ],
        check=True,
    )


def quantize_embeddings(graph):
    """The builder only quantizes matrix multiplications, leaving this table as float."""
    original = next(item for item in graph.initializer if item.name == EMBEDDING)
    weights = numpy_helper.to_array(original)
    scale = np.float32(np.abs(weights).max() / 127)
    quantized = np.round(weights / scale).clip(-127, 127).astype(np.int8)
    graph.initializer.remove(original)
    graph.initializer.extend(
        [
            numpy_helper.from_array(quantized, EMBEDDING + "_quantized"),
            numpy_helper.from_array(np.array(scale, dtype=np.float32), EMBEDDING + "_scale"),
            numpy_helper.from_array(np.array(0, dtype=np.int8), EMBEDDING + "_zero"),
        ]
    )
    node = helper.make_node(
        "DequantizeLinear",
        [EMBEDDING + "_quantized", EMBEDDING + "_scale", EMBEDDING + "_zero"],
        [EMBEDDING],
        name="/model/embed_tokens/DequantizeLinear",
    )
    graph.node.insert(0, node)


def finish_folder(source, output):
    output.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source / "model.onnx")
    quantize_embeddings(model.graph)
    onnx.save_model(
        model,
        output / "model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=1024,
    )
    # The tokenizer and runtime config are needed next to the model.
    for path in source.iterdir():
        if not path.name.startswith("model.onnx"):
            shutil.copy2(path, output / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Model id the fine-tune started from")
    parser.add_argument("--model", required=True, type=Path, help="Folder holding the fine-tuned weights")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as work:
        work = Path(work)
        build_int8(args.base, args.model, work / "build", work / "cache")
        finish_folder(work / "build", args.output)
    size = sum(path.stat().st_size for path in args.output.glob("model.onnx*")) / 1e6
    print(f"Saved {args.output} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
