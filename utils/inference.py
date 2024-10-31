import numpy as np
import os
import time
import gc
from transformers import AutoTokenizer
from enum import Enum
from threading import Lock
from utils.session import Session
from config import InferenceConfig
from tqdm import trange, tqdm
import torch



class Inference:
    def __init__(self, config: InferenceConfig) -> None:
        self.max_input_length = config.max_input_length
        self.max_output_length = config.max_output_length
        # self.tokenizer=Tokenizer(config.tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_dir, trust_remote_code=True
        )
        self.sampling_method = config.sampling_method
        self.sampling_value = config.sampling_value
        self.temperature = config.temperature
        self.session = Session.fromConfig(config)
        self.session_type = config.session_type
        if config.device_str == "cpu":
            self.torch_device = torch.device("cpu")
        elif config.device_str == "cuda":
            self.torch_device = torch.device("cuda")
        elif config.device_str == "npu":
            self.torch_device = torch.device("npu")
        else:
            raise Exception(f"unsport device {config.device_str}")
        # self.prompt=config.prompt
        self.kv_cache_length = config.kv_cache_length
        self.state: dict = {"code":200,"isEnd":False,"message":""}
        self.reset()
        self.lock = Lock()
        self.first = True
        # self.stop_mp = {"[|Human|]":6,"[|AI|]":5,"<|assistant|>":6,"<|user|>":5}
        print("[INFO] init success")


    def generate_cache(self, prompt: str):
        """
        生成kv-cache
        Args:
            prompt (str): 提示词

        Returns:
            返回下一个token与logits
        """
        if len(prompt) == 0 :
            return
        self.first = False
        input_ids = np.asarray(
            self.tokenizer.encode(prompt), dtype=np.int64
        ).reshape(1,-1)
        logits = self.session.run(input_ids)[0]
        next_token = self.sample_logits(
            logits[0][-1:],
            self.sampling_method,
            self.sampling_value,
            self.temperature
        ) 
        return next_token, logits

    def sample_logits(
        self,
        logits: np.ndarray,
        sampling_method: str = "greedy",
        sampling_value: float = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        对logits做采样，得到下一个token
        Args:
            logits (np.ndarray): 
            sampling_method (str, optional):  采样方法，默认是"greedy"，支持top_p, top_k
            sampling_value (float, optional): _description_. Defaults to None.
            temperature (float, optional): _description_. Defaults to 1.0.

        Raises:
            Exception: _description_

        Returns:
            np.ndarray: _description_
        """
        if temperature == 0 or sampling_method == "greedy":
            next_token = np.argmax(logits, axis=-1).astype(np.int64)

        elif sampling_method == "top_k" or sampling_method == "top_p":
            assert sampling_value is not None
            logits = logits.astype(np.float32)
            logits /= temperature
            probs = np.exp(logits) / np.sum(np.exp(logits))
            sorted_probs = np.sort(probs)[:, ::-1]
            sorted_indices = np.argsort(probs)[:, ::-1]

            if sampling_method == "top_k":
                index_of_interest = int(sampling_value)
            elif sampling_method == "top_p":
                p = sampling_value
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                for index_of_interest, cumulative_prob in enumerate(
                    cumulative_probs[0]
                ):
                    if cumulative_prob > p:
                        break

            probs_of_interest = sorted_probs[:, : index_of_interest + 1]
            indices_of_interest = sorted_indices[:, : index_of_interest + 1]
            probs_of_interest /= np.sum(probs_of_interest)
            next_token = np.array(
                [np.random.choice(indices_of_interest[0], p=probs_of_interest[0])]
            )
        else:
            raise Exception(f"Unknown sampling method {sampling_method}")

        return next_token
    
    def stream_predict(
        self,
        prompt,
        history=None,
        sampling_config: dict = {},
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 1024,
        do_speed_test: bool = False,
        show_progress: bool = False,
    ):
        if history is None:
            history = [] 
        sampling_value = sampling_config.get("sampling_value", self.sampling_value)
        temperature = sampling_config.get("temperature", self.temperature)
        messages = [{"role": "system", "content": system_prompt}]
        # print("prompt: ", prompt)
        with self.lock:
            self.state['isEnd'],self.state['message'] = False,""   
        if prompt == "":
            return
        for (use_msg, bot_msg) in history:
            messages.append({"role": "user", "content": use_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": prompt})
        # print("history: ", history)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if self.session_type in ["onnx", "acl"]:
            input_ids = self.tokenizer(
                [text], return_tensors="np"
            )["input_ids"].astype(np.int64).reshape(1, -1)
        elif self.session_type == "pytorch":
            input_ids = self.tokenizer(
                [text], return_tensors="pt"
            )["input_ids"].to(torch.long).reshape(1, -1).to(self.torch_device)
        else:
            raise Exception(f"unknown session_type {self.session_type}")
        input_ids = input_ids[:, -self.max_input_length:]
        # print("input_ids shape: ", input_ids.shape)
        self.first = False
        ids_list = []
        text_length = 0
        input_length = input_ids.shape[1]
        if do_speed_test:
            first_token_start = time.time()
            first_token_latency = 0
            decode_speed = 0
        max_output_len = self.max_output_length - input_length
        max_output_len = min(max_output_len, max_new_tokens)
        if show_progress:
            temp_list = trange(max_output_len, desc="decode")
        else:
            temp_list = range(max_output_len)
        prefill_show_progress = False
        decode_speed, totol_speed = 0.0, 0.0
        for i in temp_list:
            if i == 0:
                if show_progress:
                    prefill_show_progress = True
                # reset counter
                self.session.reset()
            else:
                prefill_show_progress = False
            logits = self.session.run(
                input_ids,
                show_progress=prefill_show_progress,
            )
            input_ids = self.sample_logits(
                logits[0][-1:],
                self.sampling_method,
                sampling_value,
                temperature
            )
            input_ids = input_ids.reshape(1, -1)
            if do_speed_test and i == 0:
                decode_token_start = time.time()
                first_token_latency = decode_token_start - first_token_start
            with self.lock:
                # early stop
                if input_ids[0] == self.tokenizer.eos_token_id:
                    self.state['message'],self.state['isEnd'] = self.tokenizer.decode(ids_list),True
                    break
                ids_list.append(input_ids[0].item())
                text_out = self.tokenizer.decode(ids_list)
                # stop_word = is_stop_word_or_prefix(text_out, ["[|Human|]", "[|AI|]"])
                self.state['message'] = text_out
                new_text = text_out[text_length: ]
                if do_speed_test and i > 0:
                    now_time = time.time()
                    decode_duration = now_time - decode_token_start
                    total_duration = now_time - first_token_start
                    decode_speed = (len(ids_list) - 1) / decode_duration
                    totol_speed = (input_length + len(ids_list)) / total_duration
                if b"\xef\xbf\xbd" in new_text.encode():
                    continue
                if len(new_text) > 0:
                    if do_speed_test:
                        yield new_text, first_token_latency, decode_speed, totol_speed
                    else:
                        yield new_text
                    text_length = len(text_out)
        with self.lock:
            self.state['isEnd'] = True
    
    def predict(
        self,
        prompt,
        history=None,
        sampling_config: dict = {},
        system_prompt: str="You are a helpful assistant.",
        max_new_tokens: int = 1024,
        show_progress: bool = False,
    ):
        if history is None:
            history = []
        sampling_value = sampling_config.get("sampling_value", self.sampling_value)
        temperature = sampling_config.get("temperature", self.temperature)
        messages = [{"role": "system", "content": system_prompt}]
        # print("prompt: ", prompt)
        with self.lock:
            self.state['isEnd'], self.state['message'] = False,""
        if prompt == "":
            return    
        for (use_msg, bot_msg) in history:
            messages.append({"role": "user", "content": use_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": prompt})
        # print("history: ", history)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if self.session_type in ["onnx", "acl"]:
            input_ids = self.tokenizer(
                [text], return_tensors="np"
            )["input_ids"].astype(np.int64).reshape(1, -1)
        elif self.session_type == "pytorch":
            input_ids = self.tokenizer(
                [text], return_tensors="pt"
            )["input_ids"].to(torch.long).reshape(1, -1).to(self.torch_device)
        else:
            raise Exception(f"unknown session_type {self.session_type}")
        input_ids = input_ids[:, -self.max_input_length:]
        self.first = False
        ids_list = []
        # text_length = 0
        input_length = input_ids.shape[1]
        # start = time.time()
        # first_token_latency = 0
        # decode_speed = 0
        max_output_len = self.max_output_length - input_length
        max_output_len = min(max_output_len, max_new_tokens)
        if show_progress:
            temp_list = trange(max_output_len, desc="decode")
        else:
            temp_list = range(max_output_len)
        prefill_show_progress = False
        for i in temp_list:
            if i == 0:
                if show_progress:
                    prefill_show_progress = True
                # reset counter
                self.session.reset()
            else:
                prefill_show_progress = False
            logits = self.session.run(
                input_ids,
                show_progress=prefill_show_progress
            )
            input_ids = self.sample_logits(
                logits[0][-1:],
                self.sampling_method,
                sampling_value,
                temperature
            )
            input_ids = input_ids.reshape(1, -1)
            # if i == 0:
            #     first_token_latency = time.time() - start
            with self.lock:
                # early stop
                if input_ids[0] == self.tokenizer.eos_token_id:
                    self.state['message'],self.state['isEnd'] = self.tokenizer.decode(ids_list),True
                    break
                ids_list.append(input_ids[0].item())
                # text_out = self.tokenizer.decode(ids_list)
                # stop_word = is_stop_word_or_prefix(text_out, ["[|Human|]", "[|AI|]"])
                # self.state['message'] = text_out
                # decode_speed =
        with self.lock:
            self.state['isEnd'] = True
        text_out = self.tokenizer.decode(ids_list)
        return text_out
    
    def generate(
        self,
        input_ids,
        sampling_config: dict = {},
        max_new_tokens: int = 1024,
        show_progress: bool = False,
    ):
        sampling_value = sampling_config.get("sampling_value", self.sampling_value)
        temperature = sampling_config.get("temperature", self.temperature)
        self.first = False
        ids_list = []
        input_ids = input_ids[:, -self.max_input_length:]
        input_length = input_ids.shape[1]
        max_output_len = self.max_output_length - input_length
        max_output_len = min(max_output_len, max_new_tokens)
        if show_progress:
            temp_list = trange(max_output_len, desc="decode")
        else:
            temp_list = range(max_output_len)
        prefill_show_progress = False
        for i in temp_list:
            if i == 0:
                if show_progress:
                    prefill_show_progress = True
                # reset counter
                self.session.reset()
            else:
                prefill_show_progress = False
            logits = self.session.run(
                input_ids,
                show_progress=prefill_show_progress
            )
            input_ids = self.sample_logits(
                logits[0][-1:],
                self.sampling_method,
                sampling_value,
                temperature
            )
            input_ids = input_ids.reshape(1, -1)
            with self.lock:
                # early stop
                if input_ids[0] == self.tokenizer.eos_token_id:
                    self.state['message'],self.state['isEnd'] = self.tokenizer.decode(ids_list),True
                    break
                ids_list.append(input_ids[0].item())
                text_out = self.tokenizer.decode(ids_list)
                # print("Debug: ", text_out)
                # stop_word = is_stop_word_or_prefix(text_out, ["[|Human|]", "[|AI|]"])
                self.state['message'] = text_out
        with self.lock:
            self.state['isEnd'] = True 
        text_out = self.tokenizer.decode(ids_list)
        return text_out

    def reset(self):
        self.first = True
        self.session.run_times = 0
        self.session.reset()
        # self.generate_cache(self.prompt)


    def getState(self):
        with self.lock:
            return self.state.copy()

# def preprocess(text:str) -> str:
#     # 将输入转换为指定格式
#     return f"<|user|>\n{text}</s>\n<|assistant|>"
#     
# 
# def is_stop_word_or_prefix(s: str, stop_words: list) -> int:
#     for stop_word in stop_words:
#         if s.endswith(stop_word):
#             return stop_word
#     return ""
# 