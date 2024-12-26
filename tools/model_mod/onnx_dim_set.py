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
parser.add_argument(
    "--force", action='store_true', help="Force set all dynamic dims to 1"
)
args = parser.parse_args()

import json
import onnx
from onnx import helper, version_converter
from onnx import shape_inference

LAYER_DICT = {}
with open(args.dims) as f:
    LAYER_DICT = json.load(f)

model = onnx.load(args.model)
print("===Before===")
for input in model.graph.input:
    print(input.name, input.type)
for output in model.graph.output:
    print(output.name, output.type)


model = shape_inference.infer_shapes(model)
graph = model.graph

for input_tensor in graph.input:
    if input_tensor.name not in LAYER_DICT:
        continue
    print(f"Modifying {input_tensor.name}")
    new_dims = LAYER_DICT[input_tensor.name]
    for idx, d in enumerate(new_dims):
        input_tensor.type.tensor_type.shape.dim[idx].dim_value = d

for output_tensor in graph.output:
    if output_tensor.name not in LAYER_DICT:
        continue
    print(f"Modifying {output_tensor.name}")
    new_dims = LAYER_DICT[output_tensor.name]
    for idx, d in enumerate(new_dims):
        output_tensor.type.tensor_type.shape.dim[idx].dim_value = d

for value_info in model.graph.value_info:
    if value_info.name not in LAYER_DICT:
        continue
    print(f"Modifying {value_info.name}")
    new_dims = LAYER_DICT[value_info.name]
    for idx, d in enumerate(new_dims):
        value_info.type.tensor_type.shape.dim[idx].dim_value = d

model = shape_inference.infer_shapes(model)
graph = model.graph

if args.force:
    for value_info in model.graph.value_info:
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.dim_param:
                dim.dim_param = ""
                dim.dim_value = 1

print("===After===")
for input in model.graph.input:
    print(input.name, input.type)
for output in model.graph.output:
    print(output.name, output.type)


onnx.save(model, args.output)