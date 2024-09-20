#!/usr/bin/env python
import argparse
import os

##
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", type=str, required=True, help = "Model to create data for"
)
parser.add_argument(
    "-i", "--image", type=str, required=True, help = "Image file to convert to raw data"
)
parser.add_argument(
    "-o", "--output", type=str, required=True, help = "Output directory"
)
parser.add_argument(
    "-f", "--format", nargs='+', choices = ("bin", "npy"), default = "npy", help = "Output format"
)
parser.add_argument(
    "-l", "--layout", type=str, choices = ("nhwc", "nchw"), default = "nchw", help = "Tensor layout of output data"
)
parser.add_argument(
    "-d", "--datatype", type=str, choices = ("fp32", "uint8"), default = "fp32", help = "Datatype of input data"
)
parser.add_argument(
    "--scale", type = float, help = "Scale for uint8 -> fp32 conversion, required when datatype is set to fp32"
)
parser.add_argument(
    "--offset", type = float, help = "Offset for uint8 -> fp32 conversion, required when datatype is set to fp32"
)
args = parser.parse_args()
if args.datatype == "fp32" and (args.scale==None or args.offset == None):
    print("ERROR: scale or offset not set for fp32 input")
    parser.print_help()
    exit()


##
# Create raw data
import tensorflow as tf
import numpy as np
from PIL import Image
def img_2_uint(image):
    return np.array(image, dtype=np.uint8)

def img_2_fp(image, scale, offest):
    data = np.array(image, dtype=np.float32)
    return (data / scale) - offest

def hwc_2_chw(data):
    return np.transpose(data, (0, 3, 1, 2))

def export_bin(data, dir):
    with open(dir, "wb") as file:
        file.write(data.tobytes())

def export_npy(data, dir):
    with open(dir, "wb") as file:
        np.save(file, data)

interpreter = tf.lite.Interpreter(model_path=args.model)

# Data preparation
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_name = input_details[0]["name"]

image = Image.open(args.image)
image_resized = image.resize((input_shape[1], input_shape[2]))
if args.datatype == "fp32":
    input_data = img_2_fp(image_resized, args.scale, args.offset)
elif args.datatype == "uint8":
    input_data = img_2_uint(image_resized)
input_data = np.expand_dims(input_data, axis=0)

# Execute model
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Export data
if args.layout == "nchw":
    input_data = hwc_2_chw(input_data)

if "bin" in args.format:
    input_file = os.path.join(args.output, f"{input_details[0]["name"]}.bin") 
    export_bin(input_data, input_file)
if "npy" in args.format:
    input_file = os.path.join(args.output, f"{input_details[0]["name"]}.npy") 
    export_npy(input_data, input_file)

for idx, output_detail in enumerate(output_details):
    output_data = interpreter.get_tensor(output_detail['index'])
    if "bin" in args.format:
        output_file = os.path.join(args.output, f"output{idx}.bin")
        export_bin(output_data, output_file)
    if "npy" in args.format:
        output_file = os.path.join(args.output, f"output{idx}.npy")
        export_npy(output_data, output_file)