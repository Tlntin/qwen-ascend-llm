import os
import torch
import argparse
from modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2 import Qwen2Tokenizer, Qwen2Config

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--device_str",
    type=str,
    choices=["npu", "cuda", "cpu"],
    help="support npu, cuda, cpu",
    default="cpu",
)
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


args = parser.parse_args()
device_str = args.device_str
if device_str == "cpu" and args.dtype == "float16":
    raise Exception("CPU not support float16")
if args.dtype == "float16":
    torch_dtype = torch.float16
elif args.dtype == "float32":
    torch_dtype = torch.float32
else:
    raise Exception("not support dtype, only support float16/float32")


def create_kv_cache(config: Qwen2Config, kv_cache_length=1024):
    return torch.zeros(
        [
            config.num_hidden_layers,
            2,
            1,
            config.num_key_value_heads,
            kv_cache_length,
            config.hidden_size // config.num_attention_heads
        ],
        dtype=torch_dtype
    ).to(device_str)


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
    mask = torch.ones((1, past_kv_size + seq_len), dtype=torch.long).to(device_str)
    mask[:, real_kv_size: past_kv_size] = 0
    pos_id = torch.arange(
        input_pos,
        input_pos + seq_len,
        dtype=torch.long
    ).reshape(1, -1).to(device_str)
    return cache, mask, pos_id


tokenizer = Qwen2Tokenizer.from_pretrained(args.hf_model_dir)
model_config = Qwen2Config.from_pretrained(args.hf_model_dir)
model = Qwen2ForCausalLM.from_pretrained(
    args.hf_model_dir,
    torch_dtype=torch_dtype
).to(device_str)
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
    [text], return_tensors="pt"
)["input_ids"].to(device_str)
print("input_ids", input_ids)
kv_cache1 = create_kv_cache(model_config)
now_kv_cache, attn_mask, position_ids = get_inputs(kv_cache1, 2, )
print("now_kv_cache shape: ", now_kv_cache.shape)
print("attention_mask shape: ", attn_mask.shape)
print("position_ids shape: ", position_ids.shape)
outputs = model.forward(
    input_ids[:, :2],
    attn_mask,
    position_ids,
    now_kv_cache,
    # use_cache=True,
    # output_attentions=True,
)
print("==== pytorch runtime ====")
print("output length: ", len(outputs))
logits = outputs[0][:, :-1, :]  # 1: -0.10800
# logits = outputs[0][:, -1:, :]  # 2: -0.008756

print("logits shape: ", logits.shape)
print("logits mean: ", logits.float().mean().item())
print("logits max: ", logits.float().max().item())
new_kv_cache = outputs[1][:, :, :, :, :-1, :]  # 1: 0.0009:
# new_kv_cache = outputs[1][:, :, :, :, -1:, :]  # 2: 0.003526

print("new_kv_cache: shape:", new_kv_cache.shape)
# print("new_kv_cache: mean: ", new_kv_cache.astype(np.float32).mean().item())
print("new_kv_cache: mean: ", new_kv_cache.float().mean().item())
# print("new_kv_cache: max: ", new_kv_cache.astype(np.float32).max().item())
print("new_kv_cache: max: ", new_kv_cache.float().max().item())

