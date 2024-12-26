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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path=args.model)

print("\n>> LiteRT inference validation")
try:
    interpreter.allocate_tensors()
    interpreter.invoke()
    print("Model inference succeeded.")
except Exception as e:
    print(f"Model inference failed: {e}")

print("\n>> Model I/O")
print("input:")
for data in interpreter.get_input_details():
    print(data["name"], data["shape"], data["dtype"])
print("output:")
for data in interpreter.get_output_details():
    print(data["name"], data["shape"], data["dtype"])
