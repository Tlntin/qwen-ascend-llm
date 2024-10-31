import torch

from config import InferenceConfig
from utils.kvcache import create_kv_cache
import numpy as np
from typing import List
import math
import time
import sys
import onnxruntime as ort
from tqdm import tqdm, trange


class Session:
    def __init__(self, config: InferenceConfig) -> None:
        self.run_times = 0

    def run(self,input_ids:np.ndarray, show_progress: bool = False):
        pass
    
    @staticmethod
    def fromConfig(config:InferenceConfig) -> 'Session':
        if config.session_type == "onnx":
            return OnnxSession(config)
        elif config.session_type=='acl':
            return AclSession(config)
        elif config.session_type == "pytorch":
            return PyTorchSession(config)
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
        self.kv_cache = create_kv_cache(config)
        import onnxruntime
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = config.cpu_thread
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
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
        self.kv_cache.update(seq_len, result[1])
        return result[0]


class PyTorchSession(Session):
    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)
        self.kv_cache = create_kv_cache(config)
        from export.modeling_qwen2 import Qwen2ForCausalLM
        self.device_str = config.device_str
        self.model = Qwen2ForCausalLM.from_pretrained(
            config.hf_model_dir,
            torch_dtype=config.torch_dtype
        ).to(config.device_str)

    def run(self, input_ids: np.ndarray, show_progress=False):
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids).long().to(self.device_str)
        seq_len = input_ids.shape[-1]
        cache, mask, pos_ids = self.kv_cache.get_inputs(seq_len)
        result = self.model(input_ids, mask, pos_ids, cache)
        self.kv_cache.update(seq_len, result[1])
        return result[0].cpu().detach().numpy()

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
        from utils.engine import ACLModel, init_resource
        self.device_id = config.device_id
        self.context = init_resource(self.device_id)
        self.model = ACLModel(config, self.context)
        self.max_batch = config.max_batch
        self.input_ids = np.zeros((1,16),dtype=np.int64)
        # self.kv_cache = create_kv_cache(config)
        # self.kv_cache.kv_cache = self.model.kv_cache
        self.max_prefill_length = config.max_prefill_length
        self.prefill_log2_number = int(math.log2(self.max_prefill_length))
        self.prefill_log2_list = list(range(self.prefill_log2_number, -1, -1))
        self.prefill_log2_list = [2**index for index in self.prefill_log2_list]
        
    def reset(self):
        self.model.reset()
    
    def __del__(self):
        from utils.engine import destroy_resource
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
    
    def run(self, input_ids: np.ndarray, show_progress: bool = False):
        seq_len = input_ids.shape[-1]
        logits = None
        is_prefill = True
        is_dynamic = bool(self.max_prefill_length > 1)
        # dynamic inference
        if is_dynamic:
            seq_list = self.decompose_number(seq_len)
            if show_progress:
               seq_list = tqdm(seq_list, desc="prefill") 
            start_i = 0
            for (ii, seq) in enumerate(seq_list):
                end_i = start_i + seq
                if (ii == len(seq_list) - 1):
                    is_prefill = False
                logits = self.run_some(
                    input_ids[:, start_i: end_i],
                    seq,
                    is_dynamic,
                    is_prefill=is_prefill
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
                if (i == len(idx_list) - 1):
                    is_prefill = False
                logits = self.run_some(input_ids[:,i], is_prefill=is_prefill)
        return logits
    
    def run_some(
        self,
        input_ids: np.ndarray,
        seq_length: int = 1,
        is_dynamic: bool = False,
        is_prefill: bool = False,
    ):
        self.run_times += seq_length 
        mask, pos_ids = self.model.get_inputs(seq_length)
        # print("=========================")
        # print("input_ids: ", input_ids)
        # print("attention_mask: ", mask)
        # print("position_ids: ", pos_ids)
        # print("=========================")
        logits = self.model.inference(
            [input_ids, mask, pos_ids], seq_length, is_dynamic, is_prefill=is_prefill
        )
        if not is_prefill:
            return logits.reshape(self.max_batch, seq_length,-1)
        else:
            return None
