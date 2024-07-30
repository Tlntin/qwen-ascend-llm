import sys
import math
import argparse
from concurrent.futures import ThreadPoolExecutor
from config import InferenceConfig
from utils.inference import Inference
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hf_model_dir',
        type=str,
        help="model and tokenizer path, only support huggingface model",
        default=os.path.join(project_dir, "download", "Qwen2-1.5B-Instruct")
    )
    parser.add_argument(
        "--session_type",
        type=str,
        default="acl",
        help="acl or onnx",
        choices=["acl", "onnx"],
    )
    parser.add_argument(
        '--onnx_model_path',
        type=str,
        help="onnx_model_path",
        default=os.path.join(project_dir, "output", "onnx", "qwen2_1.5b_chat.onnx")
    )
    parser.add_argument(
        "--om_model_path",
        help="mindspore model path",
        type=str,
        default= os.path.join(project_dir, "output", "model", "qwen2_1.5b_chat.om")
    )
    parser.add_argument(
        "--max_batch",
        help="max batch",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_input_length",
        help="max input length",
        type=int,
        default=512,
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
        "--max_output_length",
        help="max output length (contain input + new token)",
        type=int,
        default=1024,
    )
    return parser.parse_args()


def inference_cli():
    infer_engine = Inference(config)
    print("\n欢迎使用Qwen聊天机器人，输入exit或者quit退出，输入clear清空历史记录")
    history = []
    while True:
        input_text = input("Input: ")
        if input_text in ["exit", "quit", "exit()", "quit()"]:
            break
        if input_text == 'clear':
            history = []
            print("Output: 已清理历史对话信息。")
            continue
        print("Output: ", end='')
        response = ""
        is_first = True
        first_token_lantency, decode_speed = 0, 0
        for (
                new_text,
                first_token_lantency,
                decode_speed,
                total_speed
            ) in infer_engine.stream_predict(input_text, history=history, do_speed_test=True):
            if is_first:
                if len(new_text.strip()) == 0:
                    continue
                is_first = False
            print(new_text, end='', flush=True)
            response += new_text
        print("")
        print(
            "[INFO] first_token_lantency: {:.4f}s,".format(first_token_lantency),
            " decode_speed: {:.2f} token/s, ".format(decode_speed),
            " total_speed(prefill+decode): {:.2f} token/s".format(total_speed),
        )
        
        history.append([input_text, response])
if __name__ == '__main__':
    args = parser_args()
    max_prefill_log2 = int(math.log2(args.max_prefill_length))
    max_prefill_length = 2 ** max_prefill_log2 
    config = InferenceConfig(
        hf_model_dir=args.hf_model_dir,
        om_model_path=args.om_model_path,
        onnx_model_path=args.onnx_model_path,
        session_type=args.session_type,
        max_batch=args.max_batch,
        max_output_length=args.max_output_length,
        max_input_length=args.max_input_length,
        kv_cache_length=args.max_output_length,
        max_prefill_length=max_prefill_length,
    )
    # main()
    inference_cli()