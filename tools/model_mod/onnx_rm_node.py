import onnx
from onnx import helper, version_converter
from onnx import shape_inference

MODEL = "yolov3-12.onnx"
RM_NODE = ["yolonms_layer_1/non_max_suppression/NonMaxSuppressionV3", "Cast"]
RM_OUTPUT = ["yolonms_layer_1/concat_2:0"]
MODIFIED_MODEL = "./modified.onnx"

model = onnx.load(MODEL)
graph = model.graph

target_node = [node for node in graph.node if node.name in RM_NODE]
for node in target_node:
    graph.node.remove(node)

new_output = [o for o in graph.output if o.name not in RM_OUTPUT]
graph.ClearField('output')
graph.output.extend(new_output)


onnx.save(model, MODIFIED_MODEL)