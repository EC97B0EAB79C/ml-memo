#!/usr/bin/env python
import argparse

##
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", type=str, required=True, help = "Model to create data for"
)
args = parser.parse_args()

##
# Verify model
import onnx
model = onnx.load(args.model)

print("\n>> Model validation")
try:
    onnx.checker.check_model(model)
    print("Model is valid.")
except onnx.checker.ValidationError as e:
    print(f"Model validation failed: {e}")

print("\n>> Model I/O")
print("input:")
for input in model.graph.input:
    print(input.name, input.type)
print("output:")
for output in model.graph.output:
    print(output.name, output.type)
