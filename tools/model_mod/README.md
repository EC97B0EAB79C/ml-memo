# [Tools](/tools) > Model Modifier

Modify models for certain purpose.

## ONNX
### [`onnx_mod`](./onnx_mod.py)
**Features**:
- Removes unused input layers
- Removes `Dropout` nodes
- Removes `Softmax` nodes
- Change OP set.
- Reorganize `model.graph.input`

Usage:
```bash
./onnx_mod.py --help
usage: onnx_mod.py [-h] -m MODEL [-o OUTPUT] [--op OP] [--debug]

options:
  -h, --help           show this help message and exit
  -m, --model MODEL    Model to create data for
  -o, --output OUTPUT  Output file name
  --op OP              Change opset
  --debug
```

### [`onnx_rm_node`](./onnx_rm_node.py)
**Features**:
- Removes specified nodes
- Removes specified output layers

Usage:
```bash
./onnx_rm_node.py --help
usage: onnx_rm_node.py [-h] -m MODEL [-o OUTPUT] [--nodes NODES [NODES ...]] [--layer LAYER [LAYER ...]]

options:
  -h, --help            show this help message and exit
  -m, --model MODEL     Model to create data for
  -o, --output OUTPUT   Output file name
  --nodes NODES [NODES ...]
                        List of nodes to remove
  --layer LAYER [LAYER ...]
                        List of output layers to remove
```

### [`onnx_dim_set`](./onnx_dim_set.py)
**Feature**: Set node and layer dimensions.

Usage:
```bash
./onnx_dim_set.py --help
usage: onnx_dim_set.py [-h] -m MODEL [-o OUTPUT] [-d DIMS] [--force]

options:
  -h, --help           show this help message and exit
  -m, --model MODEL    Model to create data for
  -o, --output OUTPUT  Output file name
  -d, --dims DIMS      Dimension JSON file
  --force              Force set all dynamic dims to 1
```

### [`onnx_value_info`](./onnx_value_info.py)
**Feature**: Adds missing `value_info`

Usage:
```bash
./onnx_value_info.py --help
usage: onnx_value_info.py [-h] -m MODEL [-o OUTPUT] [-d DIMS]

options:
  -h, --help           show this help message and exit
  -m, --model MODEL    Model to create data for
  -o, --output OUTPUT  Output file name
  -d, --dims DIMS      Dimension JSON file
```