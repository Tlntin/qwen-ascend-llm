import os
import ctypes
import subprocess
import argparse
import math
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
    '--soc_version',
    type=str,
    default="auto",
    help="NPU full name, like Ascend310B1、Ascend310B4、Ascend310P1、Ascend910A、Ascend910B..., default is `auto`, will auto detect soc version.",
)
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
    "--max_batch",
    help="max batch",
    type=int,
    default=1,
)
parser.add_argument(
    "--max_prefill_length",
    help="max prefill length in first inference. "
        "Attention max_prefill_length + max_output_length <= kv_cache_length. "
        "the number must by 2^xx, like 1, 2, 4, 8, 16, 32, 64, 128, 256... "
        "Note! The higher this number, the longer it will take to compile.",
    type=int,
    default=8,
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
    max_len = 512
    rtsdll = ctypes.CDLL(f"libruntime.so")
    c_char_t = ctypes.create_string_buffer(b"\xff" * max_len, max_len)
    rtsdll.rtGetSocVersion.restype = ctypes.c_uint64
    rt_error = rtsdll.rtGetSocVersion(c_char_t, ctypes.c_uint32(max_len))
    if rt_error:
        print("rt_error:", rt_error)
        return ""
    soc_full_name = c_char_t.value.decode("utf-8")
    find_str = "Short_SoC_version="
    ascend_home_dir = os.environ.get("ASCEND_HOME_PATH")
    assert ascend_home_dir is not None, \
        print("ASCEND_HOME_PATH is None, you need run `source /usr/local/Ascend/ascend-toolkit/set_env.sh`")
    with open(f"{ascend_home_dir}/compiler/data/platform_config/{soc_full_name}.ini", "r") as f:
        for line in f:
            if find_str in line:
                start_index = line.find(find_str)
                soc_short_name = line[start_index + len(find_str):].strip()
                return {
                   "soc_full_name": soc_full_name,
                   "soc_short_name": soc_short_name
                }
    raise Exception("can't get you soc version")

max_batch = args.max_batch
model_config = Qwen2Config.from_pretrained(args.hf_model_dir)
num_hidden_layers = model_config.num_hidden_layers
num_key_value_heads = model_config.num_key_value_heads
hidden_size = model_config.hidden_size
num_attention_heads = model_config.num_attention_heads
per_head_dim = hidden_size // num_attention_heads
kv_cache_length = args.kv_cache_length
max_prefill_log2 = int(math.log2(args.max_prefill_length))
max_prefill_length = 2 ** max_prefill_log2 
prefill_length_range = list(range(0, max_prefill_log2 + 1))
prefill_length_range = [2 ** idx for idx in prefill_length_range]
assert (max_prefill_length < kv_cache_length), \
    print("max_input_length max be smaller than kv_cache_length, because max_input_length + max_output_length <= kv_cache")
input_ids_length_range = prefill_length_range
attention_length_range = [
    length + kv_cache_length
    for length in prefill_length_range
]
position_length_range = prefill_length_range
input_ids_shape = [
    f"1~{max_batch}" if max_batch > 1 else "1",
    "-1" if max_prefill_length > 1 else "1",
]
attention_mask_shape = [
    f"1~{max_batch}" if max_batch > 1 else "1",
    "-1" if max_prefill_length > 1 else str(1 + kv_cache_length)
]
position_ids_shape = [
    f"1~{max_batch}" if max_batch > 1 else "1",
    "-1" if max_prefill_length > 1 else "1"
]
dynamic_dims = []
for dynamic_dim in zip(
    input_ids_length_range, attention_length_range, position_length_range
):
    dynamic_dim = [str(dim) for dim in dynamic_dim]
    dynamic_dims.append(",".join(dynamic_dim))
past_key_values_shape = [
    num_hidden_layers,
    2,
    f"1~{max_batch}" if max_batch > 1 else "1",
    num_key_value_heads,
    kv_cache_length,
    per_head_dim
]
past_key_values_shape = [str(x) for x in past_key_values_shape]
if args.soc_version == "auto":
    print("[INFO] soc_version is `auto`, will auto detect soc version")
    soc_dict = get_soc_version()
    print("[INFO] {}".format(soc_dict))
    soc_version = soc_dict["soc_full_name"]
else:
    soc_version = args.soc_version
command_lines = [
    "atc",
    "--framework=5",
    '--model="{}"'.format(args.onnx_model_path),
    '--output="{}"'.format(args.om_model_path),
    "--soc_version={}".format(soc_version),
    "--precision_mode=must_keep_origin_dtype",
    "--input_format=ND",
    '--input_shape="input_ids:{};attention_mask:{};position_ids:{};past_key_values:{}"'.format(
        ",".join(input_ids_shape),
        ",".join(attention_mask_shape),
        ",".join(position_ids_shape),
        ",".join(past_key_values_shape)
    ),
]
if max_prefill_length > 1:
    command_lines.append(
        "--dynamic_dims \"{}\"".format(";".join(dynamic_dims))
    )
print("============ run command ==============")
print(" ".join(command_lines))
print("=======================================")
subprocess.run(
    " ".join(command_lines),
    shell=True,
    check=True,
)