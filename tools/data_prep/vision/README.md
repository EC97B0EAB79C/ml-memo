# [Tools](/tools) > [Data Preparation](/tools/data_prep) > Vision

Create raw input and output for vision tasks.
## [TFlite](./tflite.py)
Usage:
```bash
$ ./tflite.py -h
usage: tflite.py [-h] -m MODEL -i IMAGE -o OUTPUT [-f {bin,npy} [{bin,npy} ...]] [-l {nhwc,nchw}] [-d {fp32,uint8}] [--scale SCALE] [--offset OFFSET]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model to create data for
  -i IMAGE, --image IMAGE
                        Image file to convert to raw data
  -o OUTPUT, --output OUTPUT
                        Output directory
  -f {bin,npy} [{bin,npy} ...], --format {bin,npy} [{bin,npy} ...]
                        Output format
  -l {nhwc,nchw}, --layout {nhwc,nchw}
                        Tensor layout of output data
  -d {fp32,uint8}, --datatype {fp32,uint8}
                        Datatype of input data
  --scale SCALE         Scale for uint8 -> fp32 conversion, required when datatype is set to fp32
  --offset OFFSET       Offset for uint8 -> fp32 conversion, required when datatype is set to fp32
```