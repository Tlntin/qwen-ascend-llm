import numpy as np
import os
import time
import gc
from transformers import AutoTokenizer
from enum import Enum
from threading import Lock
from utils.session import Session
from config import InferenceConfig



class Inference:
    def __init__(self, config:InferenceConfig) -> None:
        self.max_input_length = config.max_input_length
        self.max_output_length = config.max_output_length
        # self.tokenizer=Tokenizer(config.tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_dir, trust_remote_code=True
        )
        self.sampling_method = config.sampling_method
        self.sampling_value = config.sampling_value
        self.temperature = config.temperature
        self.session=Session.fromConfig(config)
        # self.prompt=config.prompt
        self.kv_cache_length = config.kv_cache_length
        self.state: dict = {"code":200,"isEnd":False,"message":""}
        self.reset()
        self.lock = Lock()
        self.first = True
        # self.stop_mp = {"[|Human|]":6,"[|AI|]":5,"<|assistant|>":6,"<|user|>":5}
        print("init success")


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
        system_prompt: str="You are a helpful assistant.",
    ):
        if history is None:
            history = []    
        if len(history) == 0:
            history = [{"role": "system", "content": system_prompt}]
        # print("prompt: ", prompt)
        with self.lock:
            self.state['isEnd'],self.state['message'] = False,""   
        if prompt == "":
            return    
        history.append({"role": "user", "content": prompt})
        # print("history: ", history)
        text = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )
        input_ids = self.tokenizer(
            [text], return_tensors="np"
        )["input_ids"].astype(np.int64).reshape(1, -1)
        self.first = False
        ids_list = []
        text_length = 0
        input_length = input_ids.shape[1]
        start = time.time()
        first_token_latency = 0
        decode_speed = 0
        max_output_len = self.max_output_length - input_length
        for i in range(max_output_len):
            logits = self.session.run(input_ids)[0]
            input_ids = self.sample_logits(
                logits[0][-1:],
                self.sampling_method,
                self.sampling_value,
                self.temperature
            )
            input_ids = input_ids.reshape(1, -1)
            if i == 0:
                first_token_latency = time.time() - start
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
                # decode_speed =
                duration = time.time() - start
                decode_speed = len(ids_list) / duration
                totol_speed = (input_length + len(ids_list)) / duration
                if b"\xef\xbf\xbd" in new_text.encode():
                    continue
                if len(new_text) > 0:
                    yield new_text, first_token_latency, decode_speed, totol_speed
                    text_length = len(text_out)
        with self.lock:
            self.state['isEnd'] = True 

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