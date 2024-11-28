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
    "--op", type=int, help = "Change opset"
)
parser.add_argument(
    "--debug", action='store_true'
)
args = parser.parse_args()

##
# Modify model
import onnx
from onnx import helper, version_converter
from onnx import shape_inference

model = onnx.load(args.model)
if args.debug:
    print("===Before===")
    for input in model.graph.input:
        print(input.name, input.type)
    for output in model.graph.output:
        print(output.name, output.type)


# Remove Dropout Layers
graph = model.graph
dropout_nodes = [node for node in graph.node if node.op_type == "Dropout"]
if len(dropout_nodes):
    print("Removing `Dropout` nodes")
    for dropout_node in dropout_nodes:
        dropout_input = dropout_node.input[0]
        dropout_output = dropout_node.output[0]

        for node in graph.node:
            for i, input_name in enumerate(node.input):
                if input_name == dropout_output:
                    node.input[i] = dropout_input
        graph.node.remove(dropout_node)


# Remove unnecessary model.graph.input
original_inputs = {input.name for input in graph.input}
used_inputs = set()
for node in graph.node:
    for input_name in node.input:
        used_inputs.add(input_name)
new_inputs = [input for input in graph.input if input.name in used_inputs]

if original_inputs != used_inputs:
    print("Remove unnecessary model.graph.input")
    graph.ClearField('input')
    graph.input.extend(new_inputs)


# Change opset
if args.op:
    print("Change OP set")
    model = version_converter.convert_version(model, args.op)

onnx.save(model, args.output)


if args.debug:
    print("===After===")
    for input in model.graph.input:
        print(input.name, input.type)
    for output in model.graph.output:
        print(output.name, output.type)
