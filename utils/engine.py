import time
from typing import Dict, List
import acl
import numpy as np
import os
from functools import reduce
from operator import mul
import ctypes
from config import InferenceConfig
from ctypes import c_void_p, c_int, c_size_t, c_ulong, c_int64,POINTER


ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_NORMAL_ONLY = 2
NPY_FLOAT32 = 11

libc = ctypes.CDLL("libc.so.6")
# mmap函数原型
mmap_func = libc.mmap
mmap_func.argtypes = [c_void_p, c_size_t, c_int, c_int, c_int, c_int64]
mmap_func.restype = c_void_p

# munmap函数原型
munmap_func = libc.munmap
munmap_func.argtypes = [c_void_p, c_size_t]
munmap_func.restype = c_int

def mmap_file(file_path):  
    # 打开文件并获取文件描述符  
    file_descriptor = os.open(file_path, os.O_RDONLY)  
    file_size = os.lseek(file_descriptor, 0, os.SEEK_END)  
    os.lseek(file_descriptor, 0, os.SEEK_SET)  
    # 调用mmap映射文件到内存  
    # PROT_READ和MAP_PRIVATE的值可能因系统而异，这里假设为1和2  
    protection_flags = 1  # PROT_READ  
    visibility_flags = 2  # MAP_PRIVATE  
    mapped_memory = mmap_func(None, file_size, protection_flags, visibility_flags, file_descriptor, 0)    
    if mapped_memory == -1:  
        raise Exception("Error mapping the file.")  

    # 关闭文件描述符，映射区域仍然有效  
    os.close(file_descriptor)  
    
    # 返回映射区域的地址  
    return mapped_memory,file_size

def check_ret(str,ret):
    if ret != 0:
        print(f"return code is {ret}, detail: {str}",flush=True) 

def init_resource(device_id: int):
    ret = acl.init()
    check_ret("init", ret)
    ret = acl.rt.set_device(device_id)
    check_ret("set_device", ret)
    context,ret = acl.rt.create_context(device_id)
    check_ret("create_context", ret)
    return context

def destroy_resource(device_id: int, context):
    ret = acl.rt.reset_device(device_id)
    ret = acl.finalize()
    ret = acl.rt.destroy_context(context)

dtype2NpType = {0:np.float32,1:np.float16,2:np.int8,3:np.int32,9:np.int64}

class ACLModel:
    def __init__(self, config: InferenceConfig, context=None,callback=None):
        self.context = context
        self.model_id = None
        self.model_desc = None
        self.callback_func = callback 
        self.tid = None
        self.stream = None
        self.callback_interval = 1
        self.exit_flag = False
        self.kv_cache = None
        self.max_batch = config.max_batch
        self.kv_cache_length = config.kv_cache_length
        self.input_dataset, self.output_dataset = None, None
        self.inputs:List[Dict[str,]] = []
        self.outputs:List[Dict[str,]] =  []
        self.config = config
        self.load_model(config.om_model_path)
        self.allocate_memory()
        if not callback:
            return
        self.stream, ret = acl.rt.create_stream()
        self.tid, ret = acl.util.start_thread(self._process_callback,
                                         [self.context, 50])
        check_ret("acl.util.start_thread", ret)
        ret = acl.rt.subscribe_report(self.tid, self.stream)
        check_ret("acl.rt.subscribe_report", ret)
    
    def unload(self):
        if self.callback_func:
            ret = acl.rt.synchronize_stream(self.stream)
            # 2.7 取消线程注册，Stream上的回调函数不再由指定线程处理。
            ret = acl.rt.unsubscribe_report(self.tid, self.stream)
            self.exit_flag = True
            ret = acl.util.stop_thread(self.tid)
            ret = acl.rt.destroy_stream(self.stream)
        self.free_memory()
        self.unload_model()


    def load_model(self, model_path):
        """
        加载模型
        Args:
            model_path (_type_): _description_
        """
        model_add, model_size = mmap_file(model_path)
        self.model_id, ret = acl.mdl.load_from_mem(model_add, model_size)
        
        #self.model_id, ret = acl.mdl.load_from_file(model_path)
        check_ret("load model",ret)
        munmap_func(model_add, model_size)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("get model desc",ret)
    
    def unload_model(self):
        """
        卸载模型
        """
        ret = acl.mdl.unload(self.model_id)
        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

    def allocate_memory(self):
        """
        分配内存
        """
        self.input_dataset = acl.mdl.create_dataset()
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        self.inputs = []
        for i in range(input_size):
            buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            if i == 3:
                buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
                self.kv_cache = acl.util.ptr_to_numpy(
                    buffer, self.config.past_key_value_shape, 23 # 23：NPY_HALF，NPY_FLOAT16
                )
                data = acl.create_data_buffer(buffer, buffer_size)
                _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, data)
                check_ret("add_dataset_buffer",ret)
                self.inputs.append({"buffer": buffer, "size": buffer_size})
            else:
                buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
                check_ret("alloc input memory",ret)
                data = acl.create_data_buffer(buffer, buffer_size)
                _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, data)
                check_ret("add_dataset_buffer",ret)
                self.inputs.append({"buffer": buffer, "size": buffer_size})

        self.output_dataset = acl.mdl.create_dataset()
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self.outputs = []
        for i in range(output_size):
            buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            data_type = acl.mdl.get_output_data_type(self.model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("alloc output memory",ret)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, data)
            check_ret("add_dataset_buffer",ret)
            buffer_host, ret = acl.rt.malloc_host(buffer_size)
            check_ret("alloc output host memory",ret)
            self.outputs.append(
                {
                    "buffer": buffer,
                    "size": buffer_size,
                    "buffer_host":buffer_host,
                    'dtype':dtype2NpType[data_type]
                }
            )

    def free_memory(self):
        """
        释放内存
        """
        for item in self.input_data:
            ret = acl.rt.free(item["buffer"])
        ret = acl.mdl.destroy_dataset(self.input_dataset)
        for item in self.output_data:
            ret = acl.rt.free(item["buffer"])
            ret = acl.rt.free_host(item["buffer_host"])
        ret = acl.mdl.destroy_dataset(self.output_dataset)

    def inference(self, input_data_list: List[np.ndarray], seq_length=1, is_dynamic=False) -> List[np.ndarray]:
        """
        执行推理，同步方式
        Args:
            input_data_list (_type_): _description_
            seq_length: 推理长度

        Returns:
            List[np.ndarray]: _description_
        """
        start = time.time()
        acl.rt.set_context(self.context)
        for i in range(len(input_data_list)):
            if i == 3:
                continue
            else:
                input_data = input_data_list[i]
                input_size = input_data.size
                input_itemsize = input_data.itemsize
                bytes_data = input_data.tobytes()
                np_ptr = acl.util.bytes_to_ptr(bytes_data)
                if is_dynamic:
                    input_copy_size = input_size * input_itemsize
                else:
                    input_copy_size = self.inputs[i]["size"]
                ret = acl.rt.memcpy(
                    self.inputs[i]["buffer"],
                    self.inputs[i]["size"],
                    np_ptr,
                    input_copy_size,
                    ACL_MEMCPY_HOST_TO_DEVICE
                )
                check_ret("memcpy input", ret)
        output_sizes = []
        if is_dynamic:
            # link https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC1/apiref/appdevgapi/aclpythondevg_01_0159.html
            index, ret = acl.mdl.get_input_index_by_name(
                self.model_desc, "ascend_mbatch_shape_data"
            )
            check_ret("get_input_index_by_name", ret)
            dynamic_dims = [
                # input_ids
                self.max_batch,
                seq_length, 
                # attention_mask
                self.max_batch,
                seq_length + self.kv_cache_length, 
                # position_ids
                self.max_batch,
                seq_length  
            ]
            dynamic_dims += self.config.past_key_value_shape
            # will set dynamic input shape
            ret = acl.mdl.set_input_dynamic_dims(
                self.model_id,
                self.input_dataset,
                index,
                {
                    'dimCount': len(dynamic_dims),
                    'name': '',
                    'dims': dynamic_dims
                }
            )
            check_ret("set_iniput_dynamic_dims", ret)
            output_itemsize1 = np.dtype(self.outputs[0]["dtype"]).itemsize
            output_itemsize2 = np.dtype(self.outputs[1]["dtype"]).itemsize
            logits_size = self.max_batch * seq_length * self.config.vocab_size
            logits_itemsize = logits_size * output_itemsize1
            new_kv_cache_size = (
                self.config.num_hidden_layers \
                * 2 \
                * self.max_batch \
                * self.config.num_key_value_heads \
                * seq_length \
                * self.config.per_head_dim \
            )
            new_kv_cache_itemsize = new_kv_cache_size * output_itemsize2
            output_sizes = [logits_size, new_kv_cache_size]
            output_itemsizes = [logits_itemsize, new_kv_cache_itemsize]
        logits_shape = [self.max_batch, seq_length, self.config.vocab_size]
        new_kv_cache_shape = [
            self.config.num_hidden_layers,
            2,
            self.max_batch,
            self.config.num_key_value_heads,
            seq_length,
            self.config.per_head_dim
        ]
        output_shapes = [logits_shape, new_kv_cache_shape]

        ret = acl.mdl.execute(
            self.model_id,
            self.input_dataset,
            self.output_dataset
        )
        check_ret("model_execute", ret)
        inference_result = []

        for output_idx, out in enumerate(self.outputs):
            if is_dynamic:
                output_itemsize = output_itemsizes[output_idx]
                output_size = output_sizes[output_idx]
            else:
                output_itemsize = out["size"]
                output_size = output_itemsize // np.dtype(out["dtype"]).itemsize
            ret = acl.rt.memcpy(
                out['buffer_host'],
                out["size"],
                out["buffer"],
                output_itemsize,
                ACL_MEMCPY_DEVICE_TO_HOST
            )
            check_ret("memcpy output", ret)
            bytes_out = acl.util.ptr_to_bytes(out['buffer_host'], out["size"])
            out_data = np.frombuffer(
                bytes_out,
                dtype=out['dtype'],
                count=output_size,
            ).reshape(output_shapes[output_idx])
            inference_result.append(out_data)
        return inference_result
    
    def inference_async(self, data, other_args) -> List[np.ndarray]:
        """
        执行推理，异步方式
        Args:
            data (_type_): _description_
            other_args (_type_): _description_

        Returns:
            List[np.ndarray]: _description_
        """
        acl.rt.set_context(self.context)
        # print(f"wait lock {other_args[1]}",flush=True)
        # self.lock.acquire()
        # print(f"get lock {other_args[1]}",flush=True)
        for i in range(len(data)):
            bytes_data = data[i].tobytes()
            np_ptr = acl.util.bytes_to_ptr(bytes_data)
            ret = acl.rt.memcpy(
                self.inputs[i]["buffer"],
                self.inputs[i]["size"],
                np_ptr,
                self.inputs[i]["size"],
                ACL_MEMCPY_HOST_TO_DEVICE
            )
            check_ret("memcpy", ret)
        ret = acl.mdl.execute_async(
            self.model_id,
            self.input_dataset,
            self.output_dataset,
            self.stream
        )
        check_ret("exec_async", ret)
        print(f"submit exec task {other_args[1]}")
        ret = acl.rt.launch_callback(
            self.call_post_process, other_args, 1, self.stream
        )
        check_ret("launch callback", ret)

    def _process_callback(self, args_list):
        context, timeout = args_list
        acl.rt.set_context(context)
        while self.callback_interval:
            acl.rt.process_report(timeout)
            if self.exit_flag:
                print("[Callback] exit acl.rt.process_report")
                break

    def call_post_process(self,other_args):
        print("start callback",flush=True)
        time1 = time.time()
        inference_result = []
        for out in self.outputs:
            ret = acl.rt.memcpy(
                out['buffer_host'],
                out["size"],
                out["buffer"],
                out["size"],
                ACL_MEMCPY_DEVICE_TO_HOST
            )
            bytes_out = acl.util.ptr_to_bytes(out['buffer_host'], out["size"])
            data = np.frombuffer(bytes_out, dtype=out['dtype'])
            inference_result.append(data)
        # self.lock.release()
        # print(f"free lock {other_args[1]}",flush=True)
        if not self.callback_func:
            return
        self.callback_func(inference_result,other_args)
        print(f"end callback, use time: {time.time()-time1}")
