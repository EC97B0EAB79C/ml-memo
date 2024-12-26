# [Tools](/tools) > Model Verifier

Verify model and prints its input/output layers.

## [ONNX](./onnx_verify.py)
Usage:
```bash
./onnx_verify.py --help
usage: onnx_verify.py [-h] -m MODEL [-a]

options:
  -h, --help         show this help message and exit
  -m, --model MODEL  Model to create data for
  -a, --all          Print all model information
```

## [LiteRT](./tflite_verify.py)
Usage:
```bash
./tflite_verify.py --help
usage: tflite_verify.py [-h] -m MODEL

options:
  -h, --help         show this help message and exit
  -m, --model MODEL  Model to create data for
```