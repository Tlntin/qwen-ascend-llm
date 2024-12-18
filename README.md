### 说明
- 本项目参考了[ascend-llm](https://gitee.com/yinghuo302/ascend-llm)项目。
- 仅在昇腾310B1上面做了测试，理论上也兼容其他昇腾芯片。
- 可以用纯CPU运行pytorch或者onnx
- 仅测试了qwen1.5-0.5b-chat与qwen2-1.5b-instruct模型，理论上支持qwen1.5/qwen2系列所有chat/instruct模型。
- CANN环境安装可以参考[该教程](https://www.hiascend.com/forum/thread-0286155882998311250-1-1.html)，建议安装CANN 8.0RC2或者更高版本。
- 如果你没有昇腾NPU设备，但是也想要体验一下试试，可以试试下面的免费云平台，注册即送50积分，可以体验25小时的昇腾910。
```text
您的好友正在邀请您加入OpenI启智AI协作平台，畅享充沛的普惠算力资源(GPU/NPU/GCU/GPGPU/DCU/MLU)。
注册地址：https://openi.pcl.ac.cn/user/sign_up?sharedUser=Tlntin
推荐人：Tlntin
```

### 准备工作
1. 下载本项目
  ```bash
  git clone https://github.com/Tlntin/qwen-ascend-llm.git
  ```

2. 下载qwen1.5/qwen2的模型，选择chat模型或者instruct模型，将其放到download文件夹，仅支持huggingface下载的模型，网络不好的可以用镜像站：https://hf-mirror.com/Qwen


### Docker运行相关
- （可选）构建部署用的docker。需要先参考[该教程](https://www.hiascend.com/forum/thread-0286157793000580492-1-1.html)登录并拉取镜像（建议跑通下面的所有步骤，得到.om文件后再编译docker）。
  ```bash
  docker build . -t qwen_ascend_llm
  ```

- （可选）构建开发用的docker。如果你想用docker来编译运行自定义芯片和自定义模型，可以运行下面的命令来构建镜像。同样的，需要先参考[该教程](https://www.hiascend.com/forum/thread-0286157793000580492-1-1.html)登录并拉取镜像
  ```bash
  docker build -f Dockerfile_dev . -t qwen_ascend_llm_dev
  ```

- 拉取编译好的镜像（仅适配昇腾310B1,例如香橙派AIPro 20T版）镜像内置了一个Qwen2-1.5B-Instruct模型以及对应的.om文件。
  ```bash
  docker pull registry.cn-guangzhou.aliyuncs.com/tlntin/qwen_ascend_llm:v0.0.1_310B1_arm64
  docker tag registry.cn-guangzhou.aliyuncs.com/tlntin/qwen_ascend_llm:v0.0.1_310B1_arm64 qwen_ascend_llm
  ```

- 启动部署用的容器（如果是开发用的容器，可以参考该脚本稍微修改，比如最底下的`python api.py`命令可以换成`sleep 8640000`让100天内不会关闭，然后加上-v 参数挂载一下download/output目录）。
  ```bash
  ./run_container.sh
  ```

- 查看容器日志，出现`INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`则代表启动成功。
  ```bash
  docker logs qwen_ascend_llm
  ```

- 调用容器提供的api接口。进入本项目的client目录，可以运行里面的文件请求服务端。
  ```bash
  # openai_stream_client.py 流式请求，类似打字机效果，发送请求后立刻得到响应
  # openai_normal_client.py 非流式请求，需要等模型推理完再返回
  # openai_function_call.py 测试function_call，该功能启用时建议增加max_input_length和kv_cache_length的长度。
  ```

### （可选）验证模型结构
- 在完成pytorch模型结构魔改后，需要验证一下模型是否正常。 
- 验证pytorch CPU环境下，对话是否正常，该步骤主要是验证模型整体结构是否ok，可以多试几个demo，推荐试试`背诵《出师表》`
  ```bash
  python3 ./cli_chat.py \
      --session_type="pytorch" \
      --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
      --device_str="cpu" \
      --dtype="float32" \
      --torch_dtype="float32" \
      --max_input_length=1024 \
      --max_output_length=2048
  ```

### 详细运行步骤
##### 步骤1：编译模型（以Qwen2-1.5B-Instruct）为例。
1. 除了上面说的CANN环境安装外，还需额外安装一些python模块（当然，你也可以使用docker构建开发环境，但是注意你的芯片和对应的得是310B系列，如果不是，需要参考官方镜像文档做一些修改）。
  ```bash
  cd qwen-ascend-llm
  pip install -r ./requirements.txt
  ```
2. 导出onnx，当前我设置的kv-cache长度为2048，可以根据自己的内存、显存来设置更大参数，最大则不建议超过`max_position_embeddings`这个数，可以去模型里面的config.json文件里面看，qwen2-1.5B-Instruct里面，这个数值为`32768`
  ```bash
  python3 export/export_onnx.py \
    --device_str=npu \
    --dtype=float16 \
    --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
    --onnx_model_path="./output/onnx/qwen2_1.5b_chat.onnx" \
    --kv_cache_length=2048
  ```

3. 验证onnx，返回项目根目录，运行cli_chat.py，测试一下onnx对话是否正常（注意：由于是cpu运行，所以速度较慢，请耐心等待）。
  - `--max_input_length`为单次最大可以输入是数据量，该数值必须小于编译onnx的时候指定的`--kv_cache_length` 
  - `--max_output_length`则必须和之前转onnx的时候指定的`--kv_cache_length`保持一致，否则onnx输出将会异常。
  - 注：最大可以生成token数=`max_output_length`-min(max_input_length, 实际输入的token数)
  - npu转出的onnx，dtype取float16，cpu转出来的onnx，dtype取float32
  - `--cpu_thread`根据你的cpu线程数设置，默认取4
  ```bash
  python3 ./cli_chat.py \
    --session_type=onnx \
    --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
    --onnx_model_path="./output/onnx/qwen2_1.5b_chat.onnx" \
    --dtype="float16" \
    --cpu_thread=4 \
    --max_input_length=1024 \
    --max_output_length=2048
  ```

4. 改变onnx结构，目前导出的Trilu算子有些问题，atc命令无法识别，需要改一下结构。
  ```bash
  python3 export/change_node.py \
    --input_model_path="./output/onnx/qwen2_1.5b_chat.onnx" \
    --output_model_path="./output/onnx2/qwen2_1.5b_chat.onnx"
  ```

5. 转onnx为om模型, 将修改后的onnx利用atc命令导出到onnx，**注意此处的om_model_path不带`.om`后缀**。
  - 运行过程可能会有一些警告，或者子图融合报错，只要结果是提示`success`就说明没啥问题。
  - kv_cache_length长度和第一步导出onnx时的长度保持一致。
  - `--max_prefill_length`为prefill阶段，单次能处理的最大长度，该数值越长则越能降低首字延迟，但是相应的onnx转om的时间也会变长。设置该数值时，一般为2的指数，例如2、4、8、16等等，推理时会利用递归自动匹配合适的prefill长度，例如输入12，会匹配[8, 4]。当前默认数值为4，如果设置为1，则不会开启动态shape推理功能。**注意：开启动态shape后，模型体积会有50%-100%的增长，并且推理时占用的内存也会相应增长，如果对内存比较敏感，则建议关闭动态shape。**
  - 该脚本会自动检测你的NPU类型，如果你想手动指定，可以加上`--soc_version=xxxx`来指定，例如`--soc_version=Ascend310B1`
  - `--kv_cache_length`的数值必须前面转onnx的时候指定的`--kv_cache_length`保持一致，否则大概率会转换失败。
  - `--cpu_thread`为转onnx为om时，开启的cpu线程数，默认为1个线程并行编译，如果内存很多（每个线程单独占用一份内存，所以很费内存），可以调高一些。
  ```bash
  python3 export/onnx2om.py \
    --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
    --onnx_model_path="./output/onnx2/qwen2_1.5b_chat.onnx" \
    --om_model_path="./output/model/qwen2_1.5b_chat" \
    --kv_cache_length=2048 \
    --cpu_thread=1 \
    --max_prefill_length=4
  ```


##### 步骤2：在终端运行模型进行对话
- 使用下面的命令直接运行模型
  - `--max_prefill_length`需要和上面编译om模型时使用的数值相同。
  - `--max_input_length`为单次最大可以输入是数据量，该数值必须小于编译onnx的时候指定的`--kv_cache_length` 
  - `--max_output_length`则必须和之前转onnx的时候指定的`--kv_cache_length`保持一致，否则onnx输出将会异常。
  - 注：最大可以生成token数=`max_output_length`-min(max_input_length, 实际输入的token数)
  ```bash
  python3 ./cli_chat.py \
    --session_type="acl" \
    --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
    --om_model_path="./output/model/qwen2_1.5b_chat.om" \
    --max_input_length=1024 \
    --max_output_length=2048 \
    --max_prefill_length=4
  ```

- demo展示1（演示模型，qwen1.5-0.5b-chat，未开启动态shape推理）
![](./image/qwen1.5_0.5b_chat.gif)

- demo展示2（演示模型，qwen2-1.5b-instruct，开启动态shape推理, max_prefill_length=8）
![](./image/qwen2-1.5b-instruct.gif)

- demo展示3（演示模型，qwen2-1.5b-instruct，onnx cpu推理，CPU: i9-10900k 10核20线程）
![](./image/qwen2_1.5b_onnx_chat_cpu.png)


##### 步骤3：部署兼容OpenAI的api
- 使用下面的命令直接运行api，`--max_prefill_length`需要和上面编译的时候使用的数值相同。
  ```bash
  python3 ./api.py \
    --hf_model_dir="./download/Qwen2-1.5B-Instruct" \
    --om_model_path="./output/model/qwen2_1.5b_chat.om" \
    --max_input_length=1024 \
    --max_output_length=2048 \
    --max_prefill_length=4
  ```

- 进入client目录，可以运行里面的文件请求服务端。
  ```bash
  # openai_stream_client.py 流式请求，类似打字机效果，发送请求后立刻得到响应
  # openai_normal_client.py 非流式请求，需要等模型推理完再返回
  # openai_function_call.py 测试function_call，该功能启用时建议增加max_input_length和kv_cache_length的长度。
  ```

- functional_call demo展示(使用qwen2-1.5b-instruct)![](./image/qwen2-1.5b-instruct-functional-call.jpg)

### （可选）对比onnx和om网络层结果
- 假设编译好的om文件推理输出异常（比如origin或者fp32精度正常，fp16异常），而onnx输出正常，我们需要找到异常的网络层结构，我们需要使用工具来导出onnx和om每一层的输入输出结果，看看是哪一层开始溢出或者结果差异较大。
- 这里我们可以采用昇腾官方提供的msit工具，下面是msit的开源主页：[链接](https://gitee.com/ascend/msit)
- 我们需要安装msit工具，安装方法参考官方网站：[链接](https://gitee.com/ascend/msit/tree/master/msit/docs/install)
  ```bash
  pip install msit
  msit install compare
  ```
- 安装完成后，开始做文件对比，对比的时候建议使用om静态图做对比，即转onnx为om时，设置max_prefill_length=1。
- 对比的时候，模型越小越好，建议可以用Qwen-0.5B-Instruct模型，这样可以节省时间，也方便分析。
- 对比方法参考官方网站：[链接](https://gitee.com/ascend/msit/tree/master/msit/docs/debug/compare#/ascend/msit/blob/master/msit/docs/install/README.md)，目前我已经将其封装成了一个python代码，下面是一个示例：
  ```bash
  python3 export/compare.py \
	  --hf_model_dir="./download/Qwen2-0.5B-Instruct" \
    --onnx_model_path="./output/onnx2/qwen2_0.5b_chat.onnx" \
    --om_model_path="./output/model/qwen2_0.5b_chat.om" \
    --kv_cache_length=2048 \
    --cpu_thread=1 \
    --dtype="float16" \
    --max_prefill_length=1
  ```
- 对比结果，参考官网网站说明：[链接](https://gitee.com/ascend/msit/blob/master/msit/examples/cli/debug/compare/result_analyse/README.md)

### 当前功能
- [x] 导出onnx, om模型
- [x] 模型推理，支持onnx推理（仅支持CPU）。
- [x] 模型推理，支持CANN推理。
- [x] CANN推理时使用动态shape推理以降低首字延迟。
- [x] 流式传输
- [x] 兼容OpenAI的api搭建
- [x] 支持functional call
- [ ] 支持模型量化，如weight only, smooth quant等
- [x] 支持Docker快速部署
