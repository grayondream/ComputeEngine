#!/usr/bin/env python3
"""Generate the committed ONNX fixtures used by ComputeEngine tests.

Tests never invoke Python; this script is run only when the fixtures need to be
regenerated:

    PYTHONPATH=/tmp/opencode/onnxvenv/lib/python3.14/site-packages \
        python3 tests/reference/gen_onnx_fixtures.py
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build_tiny_add() -> onnx.ModelProto:
    """y = Add(x, w) with x [2, 2] F32 and initializer w [2, 2] F32."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    w = helper.make_tensor("w", TensorProto.FLOAT, [2, 2], [1.0, 2.0, 3.0, 4.0])
    node = helper.make_node("Add", inputs=["x", "w"], outputs=["y"], name="add")
    graph = helper.make_graph(
        nodes=[node],
        name="tiny_add",
        inputs=[x],
        outputs=[y],
        initializer=[w],
    )
    return helper.make_model(
        graph, producer_name="compute_engine", opset_imports=[helper.make_opsetid("", 17)]
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "tiny_add.onnx"
    onnx.save(build_tiny_add(), path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
