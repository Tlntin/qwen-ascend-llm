import os
import subprocess
import argparse
from transformers.models.qwen2 import Qwen2Config

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
onnx_model_dir = os.path.join(output_dir, "onnx2")
if not os.path.exists(onnx_model_dir):
    os.mkdir(onnx_model_dir)
model_dir = os.path.join(output_dir, "model")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--hf_model_dir',
    type=str,
    help="model and tokenizer path, only support huggingface model",
    default=os.path.join(project_dir, "download", "Qwen2-1.5B-Instruct")
)
parser.add_argument(
    "--onnx_model_path",
    help="output onnx path",
    type=str,
    default=os.path.join(onnx_model_dir, "qwen2_1.5b_chat.onnx")
)
parser.add_argument(
    "--om_model_path",
    help=".om model path",
    type=str,
    default= os.path.join(model_dir, "qwen2_1.5b_chat")
)
parser.add_argument(
    "--kv_cache_length",
    help="kv-cache length",
    type=int,
    default=1024,
)


args = parser.parse_args()

def get_soc_version():
    """
    _summary_
    获取芯片信息，返回具体的芯片型号
    Returns:
        _type_: _description_
    """
    # 启动一个新的进程，并获取输出
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
    # print(result.stdout)
    line_list = result.stdout.split("\n")
    soc_version = None
    for line in line_list:
        for data in line.split():
            data = data.strip()
            if data.startswith("310B") or data.startswith("310P") or data.startswith("910B"):
                soc_version = data
                break
        if soc_version is not None:
            break
    assert soc_version is not None, print("soc_version", soc_version)
    print("SoC Version is ", soc_version)
    return soc_version


model_config = Qwen2Config.from_pretrained(args.hf_model_dir)
num_hidden_layers = model_config.num_hidden_layers
num_key_value_heads = model_config.num_key_value_heads
hidden_size = model_config.hidden_size
num_attention_heads = model_config.num_attention_heads
per_head_dim = hidden_size // num_attention_heads
kv_cache_length = args.kv_cache_length
batch_size = 1
seq_len = 1
all_len = seq_len + kv_cache_length
attention_mask_shape = [batch_size, all_len]
past_key_values_shape = [
    num_hidden_layers,
    2,
    1,
    num_key_value_heads,
    kv_cache_length,
    per_head_dim
]
attention_mask_shape = [str(x) for x in attention_mask_shape]
past_key_values_shape = [str(x) for x in past_key_values_shape]

command_lines = [
    "atc",
    "--framework=5",
    '--model="{}"'.format(args.onnx_model_path),
    '--output="{}"'.format(args.om_model_path),
    "--soc_version=Ascend{}".format(get_soc_version()),
    "--precision_mode=must_keep_origin_dtype",
    "--input_format=ND",
    '--input_shape="input_ids:1,1;attention_mask:{};position_ids:1,1;past_key_values:{}"'.format(
       ",".join(attention_mask_shape), ",".join(past_key_values_shape)
    )
]
print("============ run command ==============")
print(" ".join(command_lines))
print("=======================================")
subprocess.run(
    " ".join(command_lines),
    shell=True,
    check=True,
)