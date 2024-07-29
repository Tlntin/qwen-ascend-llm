import os
import numpy as np
import onnxruntime
import argparse
from transformers.models.qwen2 import Qwen2Tokenizer, Qwen2Config


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)

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
args = parser.parse_args()

if args.dtype == "float16":
    np_dtype = np.float16
elif args.dtype == "float32":
    np_dtype = np.float32
else:
    raise Exception("not support dtype, only support float16/float32")


def create_kv_cache(config: Qwen2Config, kv_cache_length=1024):
    return np.zeros(
        [
            config.num_hidden_layers,
            2,
            1,
            config.num_key_value_heads,
            kv_cache_length,
            config.hidden_size // config.num_attention_heads
        ],
        dtype=np_dtype
    )


def get_inputs(kv_cache, seq_len: int, real_kv_size=0, input_pos=0, past_kv_size: int = 1024):
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
        self.num_hidden_layers,
        2,
        1,
        self.num_key_value_heads,
        self.kv_cache_length,
        self.per_head_dim
    )
    """
    cache = kv_cache[:, :, :, :, :past_kv_size]
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
)["input_ids"].astype(np.int64)
print("input_ids", input_ids)

options = onnxruntime.SessionOptions()
llm_session = onnxruntime.InferenceSession(
    args.onnx_model_path,
    sess_options=options,
    providers=[
        "CPUExecutionProvider",
    ],
)

seq_len = input_ids.shape[-1]
kv_cache1 = create_kv_cache(model_config)
now_kv_cache, attn_mask, position_ids = get_inputs(kv_cache1, 1)
print("now_kv_cache shape: ", now_kv_cache.shape)
print("attention_mask shape: ", attn_mask.shape)
print("position_ids shape: ", position_ids.shape)
outputs = llm_session.run(None, {
    "input_ids": input_ids[:, :1],
    "attention_mask": attn_mask,
    "position_ids": position_ids,
    "past_key_values": now_kv_cache,
})
print("==== onnx runtime ====")
print("output length: ", len(outputs))
logits = outputs[0]
print("logits shape: ", logits.shape)
print("logits mean: ", logits.astype(np.float32).mean().item())
print("logits max: ", logits.astype(np.float32).max().item())
new_kv_cache = outputs[1]  # [:, :, :, :, :-1, :]
print("new_kv_cache: shape", new_kv_cache.shape)
print("new_kv_cache: mean: ", new_kv_cache.astype(np.float32).mean().item())
print("new_kv_cache: max: ", new_kv_cache.astype(np.float32).max().item())


