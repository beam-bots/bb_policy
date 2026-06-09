# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Generate a tiny ONNX model that outputs an action *chunk*, for testing the
receding-horizon queue and temporal ensembling.

Output shape is [1, chunk_len, action_dim] = [1, 2, 2]. The model computes a
linear map then tiles it across the chunk with a per-step offset, so the two
rows of a chunk differ (row1 = row0 + 1.0), making ensemble blending
observable. Run inside the nix onnx env (see generate_linear.py header)."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

IN, OUT, CHUNK = 3, 2, 2

W = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)  # (IN, OUT)
# offsets per chunk step, shape (CHUNK, OUT): row0 += 0, row1 += 1
offset = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

obs = helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, IN])
act = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, CHUNK, OUT])

W_init = numpy_helper.from_array(W, name="W")
off_init = numpy_helper.from_array(offset.reshape(1, CHUNK, OUT), name="offset")

# base = obs @ W -> [1, OUT]; reshape to [1,1,OUT]; broadcast-add offset [1,CHUNK,OUT]
matmul = helper.make_node("MatMul", ["observation", "W"], ["base"])
shape_const = numpy_helper.from_array(np.array([1, 1, OUT], dtype=np.int64), name="newshape")
reshape = helper.make_node("Reshape", ["base", "newshape"], ["base3"])
add = helper.make_node("Add", ["base3", "offset"], ["action"])

graph = helper.make_graph(
    [matmul, reshape, add], "chunk_policy", [obs], [act],
    initializer=[W_init, off_init, shape_const],
)
model = helper.make_model(
    graph, producer_name="bb_policy_test", opset_imports=[helper.make_opsetid("", 13)]
)
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, "test/fixtures/chunk_policy.onnx")
print("wrote test/fixtures/chunk_policy.onnx")
