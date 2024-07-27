import numpy as np
from typing import Optional,Tuple,List
from config import InferenceConfig
# 对KV缓存和输出输出格式进行管理
class KVCacheManger:
    def __init__(self, config: InferenceConfig) -> None:
        self.num_key_value_heads = config.num_key_value_heads # head len
        self.kv_cache_length = config.kv_cache_length # max_size
        self.input_pos = 0
        self.past_kv_size = 0
        self.num_hidden_layers = config.num_hidden_layers  # n_layer
        self.cache_format = config.cache_format
        self.num_key_value_heads = config.num_key_value_heads  # head_num
        self.per_head_dim = config.per_head_dim # head_dim
        self.past_key_value_shape = config.past_key_value_shape
        self.real_kv_size = 0  # 真正的kv_cache长度
        if config.dtype == "float16":
            self.dtype=np.float16
        elif config.dtype=="float32":
            self.dtype=np.float32
        else:
            raise Exception("only support float16 and float32, not ", np.dtype)
        # self.kv_cache = None
        self.kv_cache = np.zeros(self.past_key_value_shape, dtype=self.dtype)

    def create_empty_cache(self):
        """
        创建空的kv_cache
        """
        if self.cache_format == "huggingface-tensor":
            self.kv_cache = np.zeros(self.past_key_value_shape, dtype=self.dtype)

    def update(
        self,
        seq_len: int,
        new_kv_cache: Tuple[List[np.ndarray],List[np.ndarray]],
        scores: Optional[np.ndarray]=None
    )->None:
        """
        更新kv_cache，暂未实现，等子类实现
        Args:
            seq_len (int): _description_
            newKV (Tuple[List[np.ndarray],List[np.ndarray]]): _description_
            scores (Optional[np.ndarray], optional): _description_. Defaults to None.
        """
        pass

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
            self.num_hidden_layers,
            2,
            1,
            self.num_key_value_heads,
            self.kv_cache_length,
            self.per_head_dim
        )
        """ 
        cache = self.kv_cache[:, :, :, :, :self.past_kv_size]
        mask = np.ones((1,self.past_kv_size + seq_len),dtype=np.int64)
        mask[:, self.real_kv_size: self.past_kv_size] = 0
        pos_id =np.arange(
            self.input_pos, 
            self.input_pos + seq_len,
            dtype=np.int64
        ).reshape(1,-1)
        return cache, mask, pos_id
    
    def reset(self,num=1):
        self.input_pos=0
        self.real_kv_size=0
        if num != 0:
            self.create_empty_cache()

    def rollback(self,seq_len):
        self.real_kv_size -=seq_len
        
    

class BasicKVCache(KVCacheManger):
    def __init__(self, cfg: InferenceConfig) -> None:
        super().__init__(cfg)
        
    def update(
        self,
        seq_len: int,
        new_kv_cache: Tuple[List[np.ndarray]],
        scores: Optional[np.ndarray] = None
    ) -> None:
        """
        更新kv_cache
        Args:
            seq_len (int): 新kv-cache的长度
            new_kv_cache (Tuple[List[np.ndarray]]): 新的kv-cache
            scores (Optional[np.ndarray], optional): _description_. Defaults to None.

        Raises:
            RuntimeError: _description_
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
        if seq_len + self.past_kv_size > self.kv_cache_length:
            raise RuntimeError("超出KV缓存长度限制")
        if self.format=="huggingface-tensor":
            temp_shape = list(self.past_key_value_shape)
            temp_shape[-2] = -1
            new_kv_cache = new_kv_cache.reshape(temp_shape)
            self.kv_cache[:, :, :, :, self.past_kv_size:self.past_kv_size + seq_len] = \
                new_kv_cache[:, :, :, :, 0:seq_len]
        self.past_kv_size += seq_len
        self.input_pos += seq_len
        self.real_kv_size += seq_len
    
    def reset(self):
        self.past_kv_size=0
        return super().reset()

class FixSizeKVCache(KVCacheManger):
    def __init__(self, cfg: InferenceConfig) -> None:
        super().__init__(cfg)
        # kv_cache的长度和max_output_length的长度一样
        self.past_kv_size=self.kv_cache_length
        
    def update(
        self,
        seq_len: int,
        new_kv_cache: Tuple[List[np.ndarray]],
        scores: Optional[np.ndarray] = None
    ) -> None:
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
        self.input_pos = self.real_kv_size + seq_len
        if seq_len + self.real_kv_size > self.kv_cache_length:
            seq_len = self.kv_cache_length - self.real_kv_size
        if seq_len <= 0:
            return
        if self.cache_format=="huggingface-tensor":
            temp_shape = list(self.past_key_value_shape)
            temp_shape[-2] = -1
            new_kv_cache = new_kv_cache.reshape(temp_shape)
            self.kv_cache[:, :, :, :, self.real_kv_size: self.real_kv_size + seq_len] = \
                new_kv_cache[:, :, :, :, 0: seq_len]
        self.real_kv_size += seq_len

class FixSizeStreamLLM(KVCacheManger):
    def __init__(self, cfg:InferenceConfig) -> None:
        super().__init__(cfg)
        self.past_len = 0
        self.past_kv_size=self.kv_cache_length

    def update(
        self,
        seq_len:int,
        new_kv_cache: Tuple[List[np.ndarray],List[np.ndarray]],
        score:Optional[np.ndarray] = None
    ):
        self.input_pos+=seq_len
        while self.past_len+ seq_len  > self.kv_cache_length:
            self.update_part(new_kv_cache, self.past_len, self.kv_cache_length - self.past_len)
            seq_len -= (self.kv_cache_length-self.past_len)
            self.past_len= self.head_len
        self.update_part(new_kv_cache, self.past_len, seq_len)
        self.past_len+= seq_len
        self.real_kv_size = max(self.past_len, self.real_kv_size)

    def update_part(
        self,
        new_kv_cache:Tuple[List[np.ndarray],List[np.ndarray]],
        begin:int,
        len:int
    ):
        """
        局部更新kv-cache
        Args:
            new_kv_cache (Tuple[List[np.ndarray],List[np.ndarray]]): 待更新的新的kv-chace
            begin (int): 更新起始位置
            len (int): 更新长度
        """

        if self.cache_format == 'huggingface-tensor': #[n_layer,2,batch_size,head_num,len,head_dim]
            self.kv_cache[:, :, :, :, self.past_len: self.past_len + len] = \
                new_kv_cache[:, :, :, :, begin: begin + len]	
        if self.cache_format =='seq_nhead_headdim': # [batch, n_layers, seq_len, n_heads, head_dim]
            self.kv_cache[0][:, :, self.past_len: self.past_len + len] = \
                new_kv_cache[0][:, :, begin : begin+len]
            self.kv_cache[1][:, :, self.past_len: self.past_len + len] = \
                new_kv_cache[1][:, :, begin: begin + len]
        elif self.cache_format == 'nhead_seq_headdim':    # [batch, n_layers, n_heads, seq_len, head_dim]
            self.kv_cache[0][:, :, :, self.past_len: self.past_len + len] = \
                new_kv_cache[0][:, :, :, begin :begin + len]
            self.kv_cache[1][:,:,:,self.past_len: self.past_len + len] = \
                new_kv_cache[1][:, :, :, begin: begin + len]
        elif self.format=='huggingface-list': # (n_layer,2) * [batch_size,head_num,len,head_dim]
            for i in range(self.num_hidden_layers):
                self.kv_cache[i][0][:, :, self.past_len: self.past_len + len,:] = \
                    new_kv_cache[i][0][:, :, begin: begin + len,:]	
                self.kv_cache[i][1][:, :, self.past_len: self.past_len + len,:] = \
                    new_kv_cache[i][1][:, :, begin:begin + len,:]	
    
    def reset(self):
        self.past_len = 0
        self.real_kv_size = 0
        return super().reset()

# 未完成
# TODO：
class FixSizeH2O(KVCacheManger):
    def __init__(self,cfg:InferenceConfig) -> None:
        super().__init__(cfg)
        self.scores = np.zeros((self.n_layer,1,self.head_num,self.past_kv_size),dtype=self.dtype)
    
    def update(
        self,
        new_kv_cache: Tuple[List[np.ndarray],List[np.ndarray]],
        score: Optional[np.ndarray] = None
    ):
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
        # score [n_layer,batch,nheader,input_len,all_len]
        seq_len = new_kv_cache[0][0].shape[-2]
        if self.real_kv_size + seq_len <  self.past_kv_size:
            self.kv_cache[:, :, :, :, self.real_kv_size: self.real_kv_size + seq_len,:] = new_kv_cache
            self.real_kv_size += seq_len
            self.scores[:, :, :, :self.real_kv_size] = \
                self.scores[:,:,:,:self.real_kv_size] * 0.5 + score[:, :, :, :self.real_kv_size]
        score = score.sum(-1)
        if self.format == 'huggingface-tensor': #[n_layer,2,batch_size,head_num,len,head_dim]
            # self.kv_cache[:,:,:,:,self.p:self.p+len,:] = new_kv_cache[:,:,:,:,begin:begin+len,:]
            for i in range(self.n_layer):
                idx = np.argpartition(score[i],-seq_len)
                self.kv_cache[i,:,idx,:] = new_kv_cache[i,:,idx,:]
                self.scores[i,idx] = score[i,idx]
                
    def update_one(
        self,
        new_kv_cache:Tuple[List[np.ndarray],List[np.ndarray]], 
        score:Optional[np.ndarray],
    ):
        if self.real_kv_size <  self.past_kv_size:
            self.kv_cache[:, :, :, :, self.real_kv_size, :] = new_kv_cache
            self.real_kv_size += 1
            self.scores[:, :, :, :self.real_kv_size] = \
                self.scores[:, :, :, :self.real_kv_size] * 0.5 + score[:, :, :, :self.real_kv_size]


def create_kv_cache(config: InferenceConfig) -> KVCacheManger:
    if config.kvcache_method == "basic":
        return BasicKVCache(config)
    elif config.kvcache_method == "fixsize":
        return FixSizeKVCache(config)
    elif config.kvcache_method == 'streamllm':
        return FixSizeStreamLLM(config)
    elif config.kvcache_method == 'H2O':
        return FixSizeH2O(config)
    else:
        return None
