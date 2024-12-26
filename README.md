# ml-memo
Repository for
- Practicing ML
- Memo for ML
- Tools for ML

## Tools
### Data Preparation
#### [Vision](tools/data_prep/vision)
- [TFLite](tools/data_prep/vision/tflite.py)
- [ONNX](tools/data_prep/vision/onnx.py)
- [Img](tools/data_prep/vision/img.py)

### [Quantizer](tools/quantizer)
- [FP32 to UInt8](tools/data_prep/quantizer/fp32_2_uint8.py)

## [Model Modifier](tools/model_mod)
- [ONNX Mod](tools/model_mod/onnx_mod.py): Modifies ONNX model
- [ONNX rm Node](tools/model_mod/onnx_rm_node.py): Remove node from ONNX model
- [ONNX Dim Set](tools/model_mod/onnx_dim_set.py): Set dimension of ONNX model
- [ONNX Value Info](tools/model_mod/onnx_value_info.py): Add value info to ONNX model

## [Model Verifier](tools/verify)
- [ONNX](tools/verify/onnx_verify.py)
- [LiteRT](tools/verify/tflite_verify.py)