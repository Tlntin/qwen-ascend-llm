"""_summary_
qwen2 modeling_qwen2.py download: https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/qwen2/modeling_qwen2.py
"""

import os
import json
import sys
from typing import List
import torch
import shutil
# from transformers import AutoModel, Qwen2Config
from transformers.models.qwen2 import Qwen2Config
from modeling_qwen2 import Qwen2ForCausalLM

import onnx
import io
import argparse


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
onnx_model_dir = os.path.join(output_dir, "onnx")
if not os.path.exists(onnx_model_dir):
    os.mkdir(onnx_model_dir)
if len(os.listdir(onnx_model_dir)) > 0:
    print("found some file in {}, will clear it".format(onnx_model_dir))
    for temp_file in os.listdir(onnx_model_dir):
        temp_path = os.path.join(onnx_model_dir, temp_file)
        os.remove(temp_path)


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device_str",
        type=str,
        choices=["npu", "cuda", "cpu"],
        help="support npu, cuda, cpu",
        default="npu",
    )
    parser.add_argument(
        "--dtype" ,
        type=str,
        help="support float16/float32, if use CPU, only support fp32",
        choices=["float16", "float32"],
        default="float16",
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
        "--kv_cache_length",
        help="kv-cache length",
        type=int,
        default=1024,
    )
    return parser.parse_args()


def export_onnx(
    device_str,
    dtype: str,
    hf_model_dir: str,
    onnx_model_path: str,
    kv_cache_length: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    per_head_dim: int,
):
    if device_str == "npu":
        import torch_npu
    if dtype == "float16":
        assert device_str.lower() != "cpu", print("cpu not support fp16")
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise Exception("unsupport dtype")

    device = torch.device(device_str)
    model = Qwen2ForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=torch_dtype,
        # trust_remote_code=True
    ).to(device)
    quantize_cfg = {
        "query_key_value": {
            "type": "W8X8",
            "act_scale": False
        },
        "dense": {
            "type": "W8X8",
            "act_scale": False
        },
        "dense_h_to_4h": {
            "type": "W8X8",
            "act_scale": False
        },
        "dense_4h_to_h": {
            "type": "W8X8",
            "act_scale": False
        }
    }
    quantize_cfg = {}
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values"
    ]
    output_names = ["logits", "out_key_values"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length+kv_len"},
        "position_ids": {0: "batch_size", 1: "seq_length"},
        "past_key_values": {2: "batch_size", 4: "kv_len"},
    }
    batch_size = 1
    seq_len = 1
    all_len = seq_len + kv_cache_length

    input_ids = torch.zeros((batch_size, seq_len)).long().to(device)
    attention_mask = torch.zeros((batch_size, all_len)).long().to(device)
    position_ids = torch.zeros((batch_size, seq_len)).long().to(device)
    past_key_values = torch.rand(
        (
            num_hidden_layers,
            2,
            1,
            num_key_value_heads,
            kv_cache_length,
            per_head_dim
        ),
        dtype=torch_dtype
    ).to(device)
    input_args = (
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        # None,  # inputs_embeds: Optional[torch.FloatTensor] = None,
        # None,  # labels: Optional[torch.LongTensor] = None,
        # True,  # use_cache: Optional[bool] = None,
        # True,  # output_attentions: Optional[bool] = None,
        # None,  # output_hidden_states
        # False  # return_dict:
    )
    model.eval()
    with torch.no_grad():
        # from quantize import quantize
        # quantize(model, cfg=quantize_cfg)
        # print(model)
        torch.onnx.export(
            model,
            f=onnx_model_path,
            args=input_args,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            opset_version=14,
            export_params=True
        )


if __name__ == "__main__":
    args = parser_arguments()
    # model_config = Qwen2Config.from_pretrained(args.hf_model_dir)
    # copy modeling_qwen2.py to model dir
    src_file_path = os.path.join(now_dir, "modeling_qwen2.py")
    target_file_path = os.path.join(args.hf_model_dir, "modeling_qwen2.py")
    shutil.copy(src_file_path, target_file_path)
    # print(model_config)
    config_json = os.path.join(args.hf_model_dir, "config.json")
    with open(config_json, "rt", encoding="utf-8") as f:
        model_config = json.load(f)
    model_config["auto_map"] = {
        "AutoModel": "modeling_qwen2.Qwen2ForCausalLM",
        "AutoModelForCausalLM": "modeling_qwen2.Qwen2ForCausalLM",
        "AutoModelForSeq2SeqLM": "modeling_qwen2.Qwen2ForCausalLM",
        "AutoModelForSequenceClassification": "modeling_qwen2.Qwen2ForSequenceClassification"
    }
    with open(config_json, "wt", encoding="utf-8") as f:
        json.dump(model_config, f, indent=4)
    test_model_config = Qwen2Config.from_pretrained(args.hf_model_dir)
    # print(test_model_config)
    test_model_config.torch_dtype = "float16"
    test_model_config.save_pretrained(args.hf_model_dir)
    num_hidden_layers = test_model_config.num_hidden_layers
    num_attention_heads = test_model_config.num_attention_heads
    num_key_value_heads = test_model_config.num_key_value_heads
    hidden_size = test_model_config.hidden_size
    per_head_dim = hidden_size // num_attention_heads
    print("new model config save ok in ", args.hf_model_dir)
    print("begin export onnx")
    export_onnx(
        device_str=args.device_str,
        dtype=args.dtype,
        hf_model_dir=args.hf_model_dir,
        onnx_model_path=args.onnx_model_path,
        kv_cache_length=args.kv_cache_length,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        per_head_dim=per_head_dim
    )
