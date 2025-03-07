#!/usr/bin/env python
import argparse

##
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", type=str, required=True, help = "Model to create data for"
)
parser.add_argument(
    "-a", "--all", action='store_true', help="Print all model information"
)
args = parser.parse_args()

##
# Verify model
import onnx
import onnxruntime as ort
import numpy as np
model = onnx.load(args.model)

print("\n>> ONNX Model validation")
try:
    onnx.checker.check_model(model)
    print("Model is valid.")
except onnx.checker.ValidationError as e:
    print(f"Model validation failed: {e}")
    exit(1)


print("\n>> ONNX inference validation")
session = ort.InferenceSession(args.model)
try:
    input_data = {}
    for input in session.get_inputs():
        input_name = input.name
        input_shape = input.shape
        input_type = input.type
        input_data[input_name] = np.random.rand(
            *[dim if isinstance(dim, int) else 1 for dim in input_shape]
        ).astype(np.float32)
    outputs = session.run(None, input_data)
    print(f"Model inference succeeded.")
except Exception as e:
    print(f"Model inference failed: {e}")

print("\n>> Model Session I/O")
print("input:")
for input in session.get_inputs():
    print(input.name, input.shape, input.type)
print("output:")
for output in session.get_outputs():
    print(output.name, output.shape, output.type)

print("\n>> Model Graph I/O")
print("input:")
for input in model.graph.input:
    print(input.name, input.type)
print("output:")
for output in model.graph.output:
    print(output.name, output.type)

if args.all:
    print(onnx.helper.printable_graph(model.graph))