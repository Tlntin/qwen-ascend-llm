import os
import onnx
import onnx.helper as helper
from onnx import TensorProto
from tqdm import tqdm

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
old_onnx_dir = os.path.join(output_dir, "onnx")
if not os.path.exists(old_onnx_dir):
    os.mkdir(old_onnx_dir)
new_onnx_dir = os.path.join(output_dir, "onnx2")
if not os.path.exists(new_onnx_dir):
    os.mkdir(new_onnx_dir)

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
model_name = "qwen1.5_0.5b_chat.onnx"

src_model_path = os.path.join(old_onnx_dir, model_name)
target_model_path = os.path.join(new_onnx_dir, model_name)

model = onnx.load(src_model_path)
new_nodes = []

for node in tqdm(model.graph.node, desc="replace node..."):
    # 判断节点类型
    new_node = node
    if node.op_type == "Trilu":
        new_node = helper.make_node(
            "Trilu",
            inputs=[node.input[0]],
            outputs=node.output,
            upper=0
        )
    if node.op_type == "Cast":
        # 替换为新的算子类型
        to_attribute = next(attr for attr in node.attribute if attr.name == "to")
        if to_attribute.i == TensorProto.INT8:
            new_node = helper.make_node(
                "AscendQuant",
                inputs=node.input,
                outputs=node.output,
                offset=0.,
                scale=1.,
            )
    new_nodes.append(new_node)
print("make new graph")
new_graph = helper.make_graph(
    new_nodes,
    "new_graph",
    inputs=model.graph.input,
    outputs=model.graph.output,
    value_info=model.graph.value_info,
    initializer=model.graph.initializer
)
print("make new model")
new_model = helper.make_model(new_graph, producer_name=model.producer_name,opset_imports=model.opset_import,ir_version = model.ir_version)
# new_model.ir_version = model.ir_version
# new_model.opset_import = model.opset_import
# new_model.metadata_props = model.metadata_props
print("will save model in ", target_model_path)
onnx.save(new_model, target_model_path, save_as_external_data=True)
