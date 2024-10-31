import time
from typing import Dict, List
import acl
import numpy as np
import os
import gc
from functools import reduce
from operator import mul
import ctypes
from config import InferenceConfig
from ctypes import c_void_p, c_int, c_size_t, c_ulong, c_int64,POINTER


ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
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
    print("[INFO] acl init")
    ret = acl.init()
    check_ret("init", ret)
    print(f"[INFO] acl set device, device_id: {device_id}")
    ret = acl.rt.set_device(device_id)
    check_ret("set_device", ret)
    print(f"[INFO] acl create context")
    context, ret = acl.rt.create_context(device_id)
    check_ret("create_context", ret)
    return context

def destroy_resource(device_id: int, context):
    print("[INFO] acl reset device")
    ret = acl.rt.reset_device(device_id)
    check_ret("reset device", ret)
    print("[INFO] acl finalize")
    ret = acl.finalize()
    check_ret("finalize", ret)
    print("[INFO] destory context")
    ret = acl.rt.destroy_context(context)
    check_ret("destory context", ret)

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
        self.max_batch = config.max_batch
        self.kv_cache_length = config.kv_cache_length
        self.max_prefill_length = config.max_prefill_length
        # kv_cache的长度和max_output_length的长度一样
        self.past_kv_size=self.kv_cache_length
        self.input_pos = 0
        self.real_kv_size = 0
        # self.kv_cache = np.zeros(config.past_key_value_shape, dtype=np.float16)
        self.input_dataset, self.output_dataset = None, None
        self.inputs:List[Dict[str,]] = []
        self.outputs:List[Dict[str,]] =  []
        self.config = config
        self.past_key_value_shape = config.past_key_value_shape
        self.half_past_key_value_shape = list(config.past_key_value_shape)
        self.half_past_key_value_shape[1] = self.half_past_key_value_shape[1] // 2
        self.load_model(config.om_model_path)
        self.allocate_memory()
        if not callback:
            return
        self.stream, ret = acl.rt.create_stream()
        check_ret("create stream", ret)
        self.tid, ret = acl.util.start_thread(self._process_callback,
                                         [self.context, 50])
        check_ret("start thread", ret)
        check_ret("acl.util.start_thread", ret)
        ret = acl.rt.subscribe_report(self.tid, self.stream)
        check_ret("acl.rt.subscribe_report", ret)

    def get_inputs(self, seq_len: int) -> List[np.ndarray]:
        """
        获取指定长度的kv_cache, 顺便生成mask和position_id
        Args:
            seq_len (int): 待获取的kv-cache长度

        Returns:
            List[np.ndarray]: _description_
        """

        """
        self.kv_cache shape (
            1,
            self.kv_cache_length,
            self.num_hidden_layers * 2 * self.num_key_value_heads,
            self.per_head_dim
        )
        """
        temp_seq_len = self.real_kv_size + seq_len
        if self.max_prefill_length > 1 and temp_seq_len <= self.kv_cache_length // 2:
            temp_kv_size = self.kv_cache_length // 2
        else:
            temp_kv_size = self.kv_cache_length
            
        mask = np.ones((1, temp_kv_size + seq_len), dtype=np.int64)
        mask[:, self.real_kv_size: temp_kv_size] = 0
        pos_id =np.arange(
            self.input_pos, 
            self.input_pos + seq_len,
            dtype=np.int64
        ).reshape(1,-1)
        return mask, pos_id

    def reset(self):
        # 重置kv-cache
        self.input_pos=0
        self.real_kv_size=0
        ret = acl.rt.memset(
            self.inputs[3]["buffer"], # 内存的起始地址。
            self.inputs[3]["size"], # 内存的最大长度，单位Byte。
            0,
            self.inputs[3]["size"] # 需要设置为指定值的内存长度，单位Byte。
        )
        check_ret("reset device kv-cache", ret)
    
    def update_kv_cache(self, seq_len):
        self.input_pos = self.real_kv_size + seq_len
        if seq_len + self.real_kv_size > self.kv_cache_length:
            seq_len = self.kv_cache_length - self.real_kv_size
        if seq_len <= 0:
            return
        # 用device memory完成下面的操作
        # self.kv_cache[:, self.real_kv_size: self.real_kv_size + seq_len] = new_kv_cache[:, 0: seq_len]
        # kv-cache shape
        """
        new_kv_cache_shape = [
            self.max_batch,
            seq_length,
            self.config.num_hidden_layers * 2 * self.config.num_key_value_heads,
            self.config.per_head_dim
        ]
        """
        base_size = self.config.num_hidden_layers * 2 * self.config.num_key_value_heads * self.config.per_head_dim
        # print("base_size: ", base_size)
        # 默认是void指针，想要往前切片，需要将数据个数 * 2（代表float16)偏移
        ret = acl.rt.memcpy(
            self.inputs[3]["buffer"] + (base_size * self.real_kv_size * self.max_batch) * 2, # 目的内存地址指针地址。
            base_size * (self.kv_cache_length - self.real_kv_size) * 2, # 目的内存地址的最大内存长度，单位Byte。
            self.outputs[1]["buffer"],
            base_size * seq_len * 2,
            ACL_MEMCPY_DEVICE_TO_DEVICE
        )
        check_ret("update device cache", ret)
        self.real_kv_size += seq_len
    
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
        # 方法1：通过map的方式加载,大概24秒
        # model_add, model_size = mmap_file(model_path)
        # self.model_id, ret = acl.mdl.load_from_mem(model_add, model_size)
        # check_ret("load model",ret)
        # munmap_func(model_add, model_size)
        # 方法2：直接加载model，用时34秒
        # self.model_id, ret = acl.mdl.load_from_file(model_path)
        # check_ret("load model",ret)
        # 方法3：将模型加载到device内存中 
        # 先获取模型大小
        model_buffer_size = os.path.getsize(model_path)
        # 分配模型buffer到device内存中
        model_buffer, ret = acl.rt.malloc(model_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
        p_model_buffer = model_buffer
        check_ret("alloc model buffer",ret)
        # 分块读取模型文件，然后将其拷贝到device model中
        # 块大小（例如 50MB）
        chunk_size = 50 * 1024 * 1024
        have_load_size = 0
        with open(model_path, 'rb') as file:
            while True:
                # 读取一块数据
                chunk = file.read(chunk_size)
                chunk_bytes = len(chunk)
                # 如果读取的数据为空，说明已经读取完毕
                if not chunk:
                    break
                # 获取这块数据的内存地址
                writable_buffer = ctypes.create_string_buffer(chunk)
                chunk_address = ctypes.addressof(writable_buffer)
                ret = acl.rt.memcpy(
                    p_model_buffer,
                    model_buffer_size - have_load_size,
                    chunk_address,
                    chunk_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE
                )
                del writable_buffer
                check_ret("memcpy input", ret)
                progress = have_load_size * 100 / model_buffer_size
                print(f"\r[INFO] load model buffer {progress:.2f}%", end="")
                have_load_size += chunk_bytes
                p_model_buffer += chunk_bytes
        print("\r[INFO] load model buffer 100.00%")
        gc.collect()
        st = time.time()
        print("[INFO] load model from memory, please wait a monment...")
        self.model_id, ret = acl.mdl.load_from_mem(model_buffer, model_buffer_size)
        check_ret("load model",ret)
        et = time.time()
        # 模型加载完后，model_buffer实测可以清理掉了，节省大量空间
        ret = acl.rt.free(model_buffer)
        check_ret(f"free model buffer device memory", ret)
        print("[INFO] load model duration: ", et - st)
        print("[INFO] get model desc")
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
        # 给输入分配Device内存
        for i in range(input_size):
            buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            # print(f"input[{i}], buffer size = {buffer_size}")
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("alloc input memory",ret)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, data)
            check_ret("add_dataset_buffer",ret)
            self.inputs.append({"buffer": buffer, "size": buffer_size})

        self.output_dataset = acl.mdl.create_dataset()
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self.outputs = []
        # 给输出分配device和host内存
        for i in range(output_size):
            buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            # print(f"output[{i}], buffer size = {buffer_size}")
            data_type = acl.mdl.get_output_data_type(self.model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("alloc output memory",ret)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, data)
            check_ret("add_dataset_buffer",ret)
            if i == 0:
                buffer_host, ret = acl.rt.malloc_host(buffer_size)
                check_ret("alloc output host memory",ret)
            # 对于new_kv_cache，不需要分配host内存，后面直接在device内存进行更新，节省内存
            else:
                buffer_host = None
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
        print("[INFO] free input and output buffer")
        for i, item in enumerate(self.input_data):
            ret = acl.rt.free(item["buffer"])
            check_ret(f"free input[{i}] device memory",ret)
        ret = acl.mdl.destroy_dataset(self.input_dataset)
        for i, item in enumerate(self.output_data):
            ret = acl.rt.free(item["buffer"])
            check_ret("free output device memory",ret)
            # 分配结果只分配了logitst的CPU内存，所以释放的时候也只释放logists的
            if i == 0: 
                ret = acl.rt.free_host(item["buffer_host"])
        ret = acl.mdl.destroy_dataset(self.output_dataset)

    def inference(self, input_data_list: List[np.ndarray], seq_length=1, is_dynamic=False, is_prefill=False) -> List[np.ndarray]:
        """
        执行推理，同步方式
        Args:
            input_data_list (_type_): _description_
            seq_length: 推理长度
            is_dynamic: 是否动态推理
            is_prefill: 是否是prefill阶段

        Returns:
            List[np.ndarray]: _description_
        """
        start = time.time()
        acl.rt.set_context(self.context)
        for i in range(len(input_data_list)):
            # 内存拷贝，忽略kv_cache，待会直接在device侧更新
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
            # 新逻辑，将kv_cache_length切成两片
            if (self.real_kv_size + seq_length) > self.kv_cache_length // 2:
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
                dynamic_dims += self.past_key_value_shape
            else:
                dynamic_dims = [
                    # input_ids
                    self.max_batch,
                    seq_length, 
                    # attention_mask
                    self.max_batch,
                    seq_length + self.kv_cache_length // 2, 
                    # position_ids
                    self.max_batch,
                    seq_length  
                ]
                dynamic_dims += self.half_past_key_value_shape
                
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

        ret = acl.mdl.execute(
            self.model_id,
            self.input_dataset,
            self.output_dataset
        )
        check_ret("model_execute", ret)

        """
        获取输出结果, 从GPU拷贝输出数据到CPU
        # 输出结果1：logits
        # 输出结果2：new_kv_cache
        prefill结果可以跳过logits的拷贝
        """
        # == update device kv cache ==
        self.update_kv_cache(seq_len=seq_length)
        # 非prefill阶段才拷贝logits作为输出    
        if not is_prefill:
            # === update logits === 
            if is_dynamic:
                output_itemsize = output_itemsizes[0]
                output_size = output_sizes[0]
            else:
                output_itemsize = self.outputs[0]["size"]
                output_size = output_itemsize // np.dtype(self.outputs[0]["dtype"]).itemsize
            logits_shape = [self.max_batch, seq_length, self.config.vocab_size]
            ret = acl.rt.memcpy(
                self.outputs[0]['buffer_host'],
                self.outputs[0]["size"],
                self.outputs[0]["buffer"],
                output_itemsize,
                ACL_MEMCPY_DEVICE_TO_HOST
            )
            check_ret("memcpy output", ret)
            bytes_out = acl.util.ptr_to_bytes(self.outputs[0]['buffer_host'], self.outputs[0]["size"])
            logits = np.frombuffer(
                bytes_out,
                dtype=self.outputs[0]['dtype'],
                count=output_size,
            ).reshape(logits_shape)
            return logits
        else:
            return None
    
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
        
