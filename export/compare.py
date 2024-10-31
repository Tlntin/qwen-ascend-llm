import os
import time
import subprocess
import numpy as np
import onnxruntime
import argparse
from transformers.models.qwen2 import Qwen2Tokenizer, Qwen2Config


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
result_output_dir = os.path.join(project_dir, "result")
input_data_dir = os.path.join(project_dir, "output", "input_data")
if not os.path.exists(result_output_dir):
    os.mkdir(result_output_dir)
if not os.path.exists(input_data_dir):
    os.mkdir(input_data_dir)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dtype",
    type=str,
    help="float16 or float32",
    choices=["float16", "float32"],
    default="float32",
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
    default=os.path.join(project_dir, "output", "onnx", "qwen2_1.5b_chat.onnx")
)
parser.add_argument(
    "--om_model_path",
    help="mindspore model path",
    type=str,
    default= os.path.join(project_dir, "output", "model", "qwen2_1.5b_chat.om")
)
parser.add_argument(
    "--kv_cache_length",
    help="kv-cache length",
    type=int,
    default=2048,
)
parser.add_argument(
    "--max_batch",
    help="max batch",
    type=int,
    default=1,
)
parser.add_argument(
    "--cpu_thread" ,
    type=int,
    help="num of cpu thread when convert onnx to om",
    default=1,
)
parser.add_argument(
    "--max_prefill_length",
    help="max prefill length in first inference. "
        "Attention max_prefill_length + max_output_length <= kv_cache_length. "
        "the number must by 2^xx, like 1, 2, 4, 8, 16, 32, 64, 128, 256... "
        "Note! The higher this number, the longer it will take to compile.",
    type=int,
    default=8
)
args = parser.parse_args()

if args.dtype == "float16":
    np_dtype = np.float16
elif args.dtype == "float32":
    np_dtype = np.float32
else:
    raise Exception("not support dtype, only support float16/float32")


def create_kv_cache(config: Qwen2Config, kv_cache_length=args.kv_cache_length):
    return np.zeros(
        [
            1,
            kv_cache_length,
            config.num_hidden_layers * 2 * config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads
        ],
        dtype=np_dtype
    )


def get_inputs(kv_cache, seq_len: int, real_kv_size=0, input_pos=0, past_kv_size: int = args.kv_cache_length):
    """
    获取指定长度的kv_cache, 顺便生成mask和position_id
    Args:
        kv_cache
        seq_len (int): 待获取的kv-cache长度
        real_kv_size: 真实kv_size长度
        input_pos: 当前真实token所在位置
        past_kv_size

    Returns:
        List[np.ndarray]: _description_
    """

    """
    self.kv_cache shape (
        1,
        self.kv_cache_length,
        self.num_hidden_layers * 2 * self.num_key_value_heads,
        self.per_head_dim
    )
    """
    cache = kv_cache[:, :past_kv_size]
    mask = np.ones((1, past_kv_size + seq_len), dtype=np.int64)
    mask[:, real_kv_size: past_kv_size] = 0
    pos_id = np.arange(
        input_pos,
        input_pos + seq_len,
        dtype=np.int64
    ).reshape(1, -1)
    return cache, mask, pos_id


tokenizer = Qwen2Tokenizer.from_pretrained(args.hf_model_dir)
model_config = Qwen2Config.from_pretrained(args.hf_model_dir)
prompt = "你好"
system_prompt: str = "You are a helpful assistant."
history = []
if len(history) == 0:
    history = [{"role": "system", "content": system_prompt}]
history.append({"role": "user", "content": prompt})
print("history: ", history)
text = tokenizer.apply_chat_template(
    history,
    tokenize=False,
    add_generation_prompt=True
)
print("raw_text", text)
input_ids = tokenizer(
    [text], return_tensors="np"
)["input_ids"].astype(np.int64)[:, :1]
print("input_ids", input_ids)

# options = onnxruntime.SessionOptions()
# options.intra_op_num_threads = 4
# options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
# options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
# 
# llm_session = onnxruntime.InferenceSession(
#     args.onnx_model_path,
#     sess_options=options,
#     providers=[
#         "CPUExecutionProvider",
#     ],
# )

seq_len = input_ids.shape[-1]
kv_cache1 = create_kv_cache(model_config)
now_kv_cache, attn_mask, position_ids = get_inputs(kv_cache1, 1)
print("now_kv_cache shape: ", now_kv_cache.shape)
print("attention_mask shape: ", attn_mask.shape)
print("position_ids shape: ", position_ids.shape)
# save input data
# input_ids
input_ids_path = os.path.join(input_data_dir, "input_ids.npy")
np.save(input_ids_path, input_ids)
# attention_mask
attention_mask_path = os.path.join(input_data_dir, "attention_mask.npy")
np.save(attention_mask_path, attn_mask)
# position_ids
position_ids_path = os.path.join(input_data_dir, "position_ids.npy")
np.save(position_ids_path, position_ids)
# past_key_values
past_key_values_path = os.path.join(input_data_dir, "past_key_values.npy")
np.save(past_key_values_path, now_kv_cache)
input_path_list = [input_ids_path, attention_mask_path, position_ids_path, past_key_values_path]

max_batch = args.max_batch
max_prefill_length = args.max_prefill_length
kv_cache_length = args.kv_cache_length
model_config = Qwen2Config.from_pretrained(args.hf_model_dir)
num_hidden_layers = model_config.num_hidden_layers
num_key_value_heads = model_config.num_key_value_heads
hidden_size = model_config.hidden_size
num_attention_heads = model_config.num_attention_heads
per_head_dim = hidden_size // num_attention_heads

input_ids_shape = [
    str(max_batch),
    str(max_prefill_length) 
]
attention_mask_shape = [
    str(max_batch),
    str(max_prefill_length + kv_cache_length)
]
position_ids_shape = [
    str(max_batch),
    str(max_prefill_length) 
]
past_key_values_shape = [
    str(max_batch),
    str(kv_cache_length),
    str(num_hidden_layers * 2 * num_key_value_heads),
    str(per_head_dim)
]


command_lines = [
    "msit debug compare",
    "-gm {}".format(args.onnx_model_path),
    "-om {}".format(args.om_model_path),
    "-c /usr/local/Ascend/ascend-toolkit/latest",
    # '--input \"{}\"'.format(",".join(input_path_list)),
    '--input-shape \"input_ids:{};attention_mask:{};position_ids:{};past_key_values:{}\"'.format(
        ",".join(input_ids_shape),
        ",".join(attention_mask_shape),
        ",".join(position_ids_shape),
        ",".join(past_key_values_shape)
    ),
    "-o {}".format(result_output_dir),
    "--advisor"
]
print("============ run command ==============")
print(" \\\r\n    ".join(command_lines))
print("=======================================")
subprocess.run(
    " \\\n    ".join(command_lines),
    shell=True,
    check=True,
)