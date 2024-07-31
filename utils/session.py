from config import InferenceConfig
from utils.kvcache import create_kv_cache
import numpy as np
from typing import List
import math
import time
import sys
from utils.engine import ACLModel, init_resource, destroy_resource
import onnxruntime as ort
from tqdm import tqdm, trange


class Session:
    def __init__(self, config: InferenceConfig) -> None:
        self.kv_cache = create_kv_cache(config)
        self.run_times = 0

    def run(self,input_ids:np.ndarray, show_progress: bool = False):
        pass
    
    @staticmethod
    def fromConfig(config:InferenceConfig) -> 'Session':
        if config.session_type == "onnx":
            return OnnxSession(config)
        elif config.session_type=='acl':
            return AclSession(config)
        else:
            return None
    
    def reset(self):
        if self.run_times == 0:
            self.kv_cache.reset(0)
        else:
            self.kv_cache.reset()

    def rollback(self,seq_len):
        self.kv_cache.rollback(seq_len)

class OnnxSession(Session):
    def __init__(self,config:InferenceConfig)->None:
        super().__init__(config)
        import onnxruntime
        options = onnxruntime.SessionOptions()
        self.llm_session = onnxruntime.InferenceSession(
            config.onnx_model_path,
            sess_options=options,
            providers=[
                "CPUExecutionProvider",
            ],
        )

    def run(self, input_ids:np.ndarray, show_progress=False):
        seq_len=input_ids.shape[-1]
        cache, mask, pos_ids = self.kv_cache.get_inputs(seq_len)
        result = self.llm_session.run(None,{
            "input_ids": input_ids,
            "attention_mask":mask,
            "past_key_values": cache,
            "position_ids": pos_ids,
        })
        self.kv_cache.update(seq_len,result[1])
        return result
    
# onnxruntime-cann is preview, not work now
"""
class CANNOnnxSession(Session):
    def __init__(self,config:InferenceConfig)->None:
        super().__init__(config)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        self.llm_session = ort.InferenceSession(
            config.onnx_model_path,
            sess_options=options,
            providers=[
                (
                    "CANNExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "npu_mem_limit": 20 * 1024 * 1024 * 1024,
                        "op_select_impl_mode": "high_performance",
                        "optypelist_for_implmode": "Gelu",
                        "enable_cann_graph": True
                    },
                ),
                "CPUExecutionProvider",
            ]
        )

    def run(self, input_ids:np.ndarray):
        seq_len=input_ids.shape[-1]
        past_key_values, attention_mask, position_ids = self.kv_cache.get_inputs(seq_len)
        input_ids_cann = ort.OrtValue.ortvalue_from_numpy(input_ids, device_type="cann", device_id=0)
        attention_mask_cann = ort.OrtValue.ortvalue_from_numpy(attention_mask, device_type="cann", device_id=0)
        position_ids_cann = ort.OrtValue.ortvalue_from_numpy(position_ids, device_type="cann", device_id=0)
        past_key_values_cann = ort.OrtValue.ortvalue_from_numpy(past_key_values, device_type="cann", device_id=0)
        io_binding = self.llm_session.io_binding()
        io_binding.bind_ortvalue_input(name="input_ids", ortvalue=input_ids_cann)
        io_binding.bind_ortvalue_input(name="attention_mask", ortvalue=attention_mask_cann)
        io_binding.bind_ortvalue_input(name="position_ids", ortvalue=position_ids_cann)
        io_binding.bind_ortvalue_input(name="past_key_values", ortvalue=past_key_values_cann)
        io_binding.bind_output("logits", device_type="cann", device_id=0)
        io_binding.bind_output("out_key_values", device_type="cann", device_id=0)
        self.llm_session.run_with_iobinding(io_binding)
        logitsts = io_binding.get_outputs()[0].numpy()
        new_kv_cache = io_binding.get_outputs()[1].numpy()
        self.kv_cache.update(seq_len, new_kv_cache)
        return (logitsts, new_kv_cache)
"""

class AclSession(Session):
    context = None
    def __init__(self, config:InferenceConfig):
        super().__init__(config)
        self.device_id = config.device_id
        self.context = init_resource(self.device_id)
        self.model = ACLModel(config, self.context)
        self.max_batch = config.max_batch
        self.input_ids = np.zeros((1,16),dtype=np.int64)
        self.kv_cache.kv_cache = self.model.kv_cache
        self.max_prefill_length = config.max_prefill_length
        self.prefill_log2_number = int(math.log2(self.max_prefill_length))
        self.prefill_log2_list = list(range(self.prefill_log2_number, -1, -1))
        self.prefill_log2_list = [2**index for index in self.prefill_log2_list]
        
    
    def __del__(self):
        destroy_resource(self.device_id, self.context)
    
    def decompose_number(self, n, start_index=0):
        """
        将数字n分解成若干个2的指数的和，并返回这些2的指数构成的列表。
        参数:
        n -- 要分解的数字
        返回:
        分解后的列表，例如 [8, 4]
        """
        if n == 0:
            return []
    
        for i in range(start_index, self.prefill_log2_number + 1):
            power = self.prefill_log2_list[i]
            if power <= n:
                return [power] + self.decompose_number(n - power, i)
        return []
    
    def run(self, input_ids: np.ndarray, show_progress:bool=False):
        seq_len = input_ids.shape[-1]
        logits = None
        is_dynamic = bool(self.max_prefill_length > 1)
        # dynamic inference
        if is_dynamic:
            seq_list = self.decompose_number(seq_len)
            if show_progress:
               seq_list = tqdm(seq_list, desc="prefill") 
            start_i = 0
            for seq in seq_list:
                end_i = start_i + seq
                logits = self.run_some(
                    input_ids[:, start_i: end_i],
                    seq,
                    is_dynamic,
                )
                start_i += seq
                # if show_progress:
                #     seq_list.update(seq)
        # static inference
        else:
            if show_progress:
                idx_list = trange(seq_len, desc="prefill")
            else:
                idx_list = range(seq_len)
            for i in idx_list:
                logits = self.run_some(input_ids[:,i])
        return [logits]
    
    def run_some(
        self,
        input_ids: np.ndarray,
        seq_length: int = 1,
        is_dynamic: bool = False
    ):
        # print(
        #     "self.run_times: ", self.run_times,
        #     "real kv size: ", self.kv_cache.real_kv_size
        # )
        self.run_times += seq_length 
        cache, mask, pos_ids = self.kv_cache.get_inputs(seq_length)
        result:List[np.ndarray] = self.model.inference(
                [input_ids, mask, pos_ids, cache], seq_length, is_dynamic
            )
        # if self.run_times <= 20:
        #     print(" === Debug === ")
        #     print("run times: ", self.run_times) 
        #     logits = result[0]
        #     new_kv_cache = result[1]
        #     print("logits shape: ", logits.shape)
        #     print("logits mean: ", logits.astype(np.float32).mean().item())
        #     print("logits max: ", logits.astype(np.float32).max().item())
        #     print("new_kv_cache: shape", new_kv_cache.shape)
        #     print("new_kv_cache: mean: ", new_kv_cache.astype(np.float32).mean().item())
        #     print("new_kv_cache: max: ", new_kv_cache.astype(np.float32).max().item())
        self.kv_cache.update(seq_length, result[1])
        return result[0].reshape(self.max_batch, seq_length,-1)

    def run_all_logits(self, input_ids: np.ndarray):
        seq_len, i = input_ids.shape[-1], 0
        logits = []
        while i < seq_len:
            end = i + 16 if i+16 < seq_len else seq_len
            cache,mask,pos_ids = self.kv_cache.get_inputs(16)
            self.input_ids[0:end-i] = input_ids[i:end]
            result:List[np.ndarray] = self.model.inference([self.input_ids, mask, pos_ids, cache])
            self.kv_cache.update(end-i,result[1])
            logits.append(result[0][0:end-i].reshape(1,-1))
        return [np.concatenate(logits).reshape(1,1,-1)]