from config import InferenceConfig
from utils.kvcache import create_kv_cache
import numpy as np
from typing import List
import time
import sys
from utils.engine import ACLModel, init_resource, destroy_resource
import onnxruntime as ort

class Session:
    def __init__(self, config: InferenceConfig) -> None:
        self.kv_cache = create_kv_cache(config)
        self.run_times = 0
    def run(self,input_ids:np.ndarray):
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

    def run(self, input_ids:np.ndarray):
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
        self.input_ids = np.zeros((1,16),dtype=np.int64)
        self.kv_cache.kv_cache = self.model.kv_cache
    
    def __del__(self):
        destroy_resource(self.device_id, self.context)
    def run(self, input_ids: np.ndarray):
        seq_len = input_ids.shape[-1]
        logits = None
        for i in range(seq_len):
            logits = self.run_one(input_ids[:,i])
        return [logits]
    
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
    
    def run_one(self, input_ids: np.ndarray):
        self.run_times += 1     
        cache, mask, pos_ids = self.kv_cache.get_inputs(1)
        result:List[np.ndarray] = self.model.inference(
                [input_ids, mask, pos_ids, cache]
            )
        # if self.run_times <= 2:
        #     print(" == Debug == ")
        #     logits = result[0]
        #     new_kv_cache = result[1]
        #     print("logits shape: ", logits.shape)
        #     print("logits mean: ", logits.astype(np.float32).mean().item())
        #     print("logits max: ", logits.astype(np.float32).max().item())
        #     print("new_kv_cache: shape", new_kv_cache.shape)
        #     print("new_kv_cache: mean: ", new_kv_cache.astype(np.float32).mean().item())
        #     print("new_kv_cache: max: ", new_kv_cache.astype(np.float32).max().item())
        self.kv_cache.update(1,result[1])
        return result[0].reshape(1,1,-1)
