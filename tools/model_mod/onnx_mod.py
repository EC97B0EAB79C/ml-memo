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
    "--debug", action='store_true'
)
args = parser.parse_args()

##
# Modify model
import onnx
from onnx import helper
from onnx import shape_inference

model = onnx.load(args.model)
if args.debug:
    print("===Before===")
    for input in model.graph.input:
        print(input.name, input.type)
    for output in model.graph.output:
        print(output.name, output.type)

# # Remove 'batch_size'
# for input in model.graph.input:
#     input_dim = input.type.tensor_type.shape.dim
#     input_dim[0].dim_value = 1
# for output in model.graph.output:
#     output_dim = output.type.tensor_type.shape.dim
#     output_dim[0].dim_value = 1

# Remove Dropout Layers
graph = model.graph
dropout_nodes = [node for node in graph.node if node.op_type == "Dropout"]
for dropout_node in dropout_nodes:
    dropout_input = dropout_node.input[0]
    dropout_output = dropout_node.output[0]

    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name == dropout_output:
                node.input[i] = dropout_input
    graph.node.remove(dropout_node)

# Save model
onnx.save(model, args.output)

if args.debug:
    print("===After===")
    for input in model.graph.input:
        print(input.name, input.type)
    for output in model.graph.output:
        print(output.name, output.type)
