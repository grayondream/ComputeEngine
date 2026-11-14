#!/usr/bin/env python3
"""Generate the ResNet50 ONNX model and ONNX Runtime reference logits.

Torchvision's untrained ResNet50 (fixed seed) is exported with a static
1x3x224x224 all-ones input. The legacy exporter emits pure-alias ``Identity``
nodes for duplicated constants when constant folding is disabled; the script
rewires those aliases away (semantics-preserving, no extra ONNX op needed).
ONNX Runtime then runs the resulting model on the same input and the 1000
float32 logits are stored as a raw little endian binary. The C++ parity test
reads both files through environment variables, so it never needs Python,
PyTorch, ONNX, or ONNX Runtime:

    IE_RESNET50_ONNX=/tmp/opencode/resnet50/resnet50.onnx \
    IE_RESNET50_REFERENCE=/tmp/opencode/resnet50/resnet50_logits.bin \
        ./build/tests/ie_tests --gtest_filter='ResNet50ParityTest.*'

The ONNX file is roughly 100 MB; it and the logits binary are local test
assets and must not be committed. The generation machine needs torch and
torchvision on the default path plus onnx/onnxruntime, which on the
development machine live in a separate venv:

    PYTHONPATH=/tmp/opencode/onnxvenv/lib/python3.14/site-packages \
        python3 tests/reference/gen_resnet50_onnx.py \
            --out-dir /tmp/opencode/resnet50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from torchvision.models import resnet50

SEED = 20261114
INPUT_SHAPE = (1, 3, 224, 224)
DEFAULT_OUT_DIR = Path("/tmp/opencode/resnet50")


def strip_constant_identity_nodes(model) -> None:
    """Remove pure-alias Identity nodes emitted by the no-folding export.

    With ``do_constant_folding=False`` the legacy TorchScript exporter
    deduplicates equal constants through ``Identity`` nodes (ResNet50's
    default BatchNorm parameters are all equal). Those nodes only alias
    initializers, so pointing every consumer at the original constant is
    semantics-preserving and keeps the C++ ONNX subset free of an extra op.
    """
    graph = model.graph
    initializers = {tensor.name for tensor in graph.initializer}
    aliases = {
        node.output[0]: node.input[0]
        for node in graph.node
        if node.op_type == "Identity" and node.input[0] in initializers
    }
    if not aliases:
        return

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in aliases and name not in seen:
            seen.add(name)
            name = aliases[name]
        return name

    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = resolve(name)
    for output in graph.output:
        output.name = resolve(output.name)

    remaining = [node for node in graph.node if node.output[0] not in aliases]
    del graph.node[:]
    graph.node.extend(remaining)
    onnx.checker.check_model(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory for resnet50.onnx and resnet50_logits.bin",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "resnet50.onnx"
    logits_path = out_dir / "resnet50_logits.bin"

    torch.manual_seed(SEED)
    model = resnet50(weights=None).eval()
    x = torch.ones(*INPUT_SHAPE, dtype=torch.float32)

    torch.onnx.export(
        model,
        (x,),
        str(onnx_path),
        opset_version=17,
        dynamo=False,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["logits"],
    )

    exported = onnx.load(str(onnx_path))
    strip_constant_identity_nodes(exported)
    onnx.save(exported, str(onnx_path))

    import onnxruntime as ort  # generation-only dependency

    with torch.no_grad():
        torch_logits = model(x).numpy()

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    (ort_logits,) = session.run(["logits"], {"input": x.numpy()})
    # The C++ parity contract is 1e-3 against the ONNX Runtime logits below.
    # This check only proves the export is faithful; ORT and PyTorch use
    # different convolution kernels, so their 50-layer accumulation differs
    # by a few times 1e-5 rather than 1e-5 relative.
    np.testing.assert_allclose(ort_logits, torch_logits, rtol=1e-4, atol=1e-4)
    assert ort_logits.shape == (1, 1000), ort_logits.shape
    assert ort_logits.dtype == np.float32, ort_logits.dtype

    # Raw float32 logits in row-major order, matching the C++ test's reader.
    ort_logits.astype("<f4", copy=False).tofile(logits_path)

    top1 = int(np.argmax(ort_logits[0]))
    print(f"wrote {onnx_path}")
    print(f"wrote {logits_path}")
    print(f"onnxruntime logits: shape={ort_logits.shape} top1={top1}")
    print("run the parity test with:")
    print(f"  IE_RESNET50_ONNX={onnx_path} \\")
    print(f"  IE_RESNET50_REFERENCE={logits_path} \\")
    print("    ./build/tests/ie_tests --gtest_filter='ResNet50ParityTest.*'")


if __name__ == "__main__":
    main()
