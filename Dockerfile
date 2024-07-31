FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/ascend-infer-310b:24.0.RC1-dev-arm

# 修改时区
# ENV TZ=Asia/Shanghai
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# python换源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装CANN依赖的python模块
RUN pip install "numpy==1.26.3" jinja2 "attrs==23.2.0" \
   "decorator==5.1.1" psutil cloudpickle "scipy==1.12.0" \
   "tornado==6.4" synr==0.5.0 absl-py sympy ml-dtypes \
   scipy tornado --no-cache-dir

# 安装torch 2.1.0
RUN pip install torch==2.1.0 --no-cache-dir

# 创建一个qwen_ascend_llm目录,以及output目录
RUN mkdir /home/AscendWork/qwen_ascend_llm/ && \
  mkdir /home/AscendWork/qwen_ascend_llm/output
WORKDIR /home/AscendWork/qwen_ascend_llm/



# 安装torch_npu(本容器是3.9，需要下载3.9的torch_npu2.1.0)
RUN curl -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3" -L https://gitee.com/ascend/pytorch/releases/download/v6.0.rc2-pytorch2.1.0/torch_npu-2.1.0.post6-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
  -o torch_npu-2.1.0.post6-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
RUN pip install ./torch_npu-2.1.0.post6-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --no-cache-dir

# 拷贝python依赖文件
COPY ./requirements.txt .

# 安装本项目需要的python模块
RUN pip install -r requirements.txt --no-cache-dir

# 拷贝模型文件
COPY download ./download
COPY output/model ./output/model

# 拷贝代码路径
COPY client ./client
COPY export ./export
COPY image ./image
COPY utils ./utils
COPY ./api.py .
COPY ./cli_chat.py .
COPY ./config.py .
COPY ./README.md .

# 清理下载的torch_npu文件
RUN rm ./*.whl

# 暴露默认的8000端口用于api
EXPOSE 8000

# 切换root账号改变文件权限(以防万一)
USER root
RUN chown -R HwHiAiUser:HwHiAiUser ./*

USER HwHiAiUser

# 启动程序, 默认启动api
CMD ["bash", "-c", "/home/AscendWork/run.sh && python3 api.py"]