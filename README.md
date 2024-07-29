### 说明
- 本项目参考了[ascend-llm](https://gitee.com/yinghuo302/ascend-llm)项目。
- 仅在昇腾310B1做了测试，理论上也兼容其他昇腾芯片。
- 仅测试了qwen1.5-0.5b-chat模型，理论上支持qwen1.5/qwen2系列所有chat/instruct模型。

### 准备工作
1. 下载本项目
2. 下载qwen1.5/qwen2的模型，选择chat模型或者instruct模型，将其放到download文件夹。


### 快速运行


### 分步骤运行
##### 步骤1：编译模型
1. 进入export文件夹, 导出onnx。
  ```bash
  cd export
  python3 export_onnx.py --hf_model_dir="download/[你下载的模型路径]"
  cd..
  ```

2. 验证onnx，返回项目根目录，运行cli_chat.py，测试一下onnx对话是否正常。
  ```bash
  python3 ./cli_chat.py --session_type=onnx 
  ```

3. 进入export文件夹，改变onnx结构，目前导出的Trilu算子和Cast算子有些问题，atc命令无法识别，需要改一下结构。
  ```bash
  cd export
  python3 change_node.py
  cd ..
  ```

4. 转onnx为om模型
  ```bash
  cd export
  python3 onnx2om.py --hf_model_dir="download/[你下载的模型路径]"
  cd ..
  ```


##### 步骤2：运行模型
- 使用下面的命令直接运行模型
  ```bash
  python3 ./cli_chat.py --hf_model_dir="download/[你下载的模型路径]"
  ```

- demo展示（演示模型，qwen1.5-0.5chat)
![](./image/qwen1.5_0.5b_chat.gif)


### 当前功能
- [x] 导出onnx, om模型
- [x] 模型推理，支持onnx推理（仅支持CPU）。
- [x] 模型推理，支持acl推理。
- [x] 流式传输
- [ ] 兼容OpenAI的api搭建
- [ ] 支持functional call
- [ ] 支持模型量化，如weight only, smooth quant等
- [ ] 支持Docker快速部署