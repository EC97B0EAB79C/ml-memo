#!/usr/bin/env python
import argparse

##
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", type=str, required=True, help = "Model to create data for"
)
parser.add_argument(
    "-o", "--output", type=str, default = "./modified.onnx", help = "Output file name"
)
parser.add_argument(
    "-d", "--dims", type=str, help = "Dimension JSON file"
)
args = parser.parse_args()

import json
import onnx
from onnx import helper, TensorProto, shape_inference

LAYER_DICT = {}
with open(args.dims) as f:
    LAYER_DICT = json.load(f)

model = onnx.load(args.model)

new_value_info = []
for value in LAYER_DICT:
    new_value_info.append(
        helper.make_tensor_value_info(
            value, 
            TensorProto.UINT8, 
            LAYER_DICT[value]
            )
        )

model.graph.value_info.extend(new_value_info)
model = shape_inference.infer_shapes(model)
onnx.save(model, args.output)