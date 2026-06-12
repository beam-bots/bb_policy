# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Generate a tiny, deterministic ONNX model used as a test fixture.

The model is a single linear layer: action = observation @ W + b, with a fixed
weight/bias so tests can assert exact outputs. Input/output use static shapes
(batch 1), mirroring how a real ACT-class policy is exported (see PROJECT_PLAN
R2/D5). Run inside the nix onnx env:

    nix-shell -p 'python3.withPackages(ps: [ps.onnx])' \
      --run "python3 test/fixtures/generate_linear.py"
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

IN, OUT = 3, 2

# action = obs @ W + b ; W is (IN, OUT), b is (OUT,)
W = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float32)
b = np.array([0.5, -0.5], dtype=np.float32)

obs = helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, IN])
act = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, OUT])

W_init = numpy_helper.from_array(W, name="W")
b_init = numpy_helper.from_array(b, name="b")

matmul = helper.make_node("MatMul", ["observation", "W"], ["xw"])
add = helper.make_node("Add", ["xw", "b"], ["action"])

graph = helper.make_graph(
    [matmul, add], "linear_policy", [obs], [act], initializer=[W_init, b_init]
)
model = helper.make_model(
    graph, producer_name="bb_policy_test", opset_imports=[helper.make_opsetid("", 13)]
)
model.ir_version = 9  # compatible with onnxruntime in ort 2.0-rc
onnx.checker.check_model(model)
onnx.save(model, "test/fixtures/linear_policy.onnx")
print("wrote test/fixtures/linear_policy.onnx")
