#!/usr/bin/env python
import argparse

##
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "model", type=str, help = "Model to modify"
)
args = parser.parse_args()

##
# Modify model
import onnx

def remove_initializer_from_input(model):
    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name not in name_to_input:
            model.graph.input.append(initializer)


model = onnx.load(args.model)
remove_initializer_from_input(model)
onnx.save(model, args.model)