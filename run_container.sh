docker run -d \
    -p 8000:8000 \
		--device=/dev/upgrade:/dev/upgrade \
		--device=/dev/davinci0:/dev/davinci0 \
		--device=/dev/davinci_manager \
		--device=/dev/vdec:/dev/vdec \
		--device=/dev/vpc:/dev/vpc \
		--device=/dev/pngd:/dev/pngd \
		--device=/dev/venc:/dev/venc \
		--device=/dev/sys:/dev/sys \
		--device=/dev/svm0 \
		--device=/dev/ts_aisle:/dev/ts_aisle \
		--device=/dev/dvpp_cmdlist:/dev/dvpp_cmdlist \
		-v /etc/sys_version.conf:/etc/sys_version.conf:ro \
		-v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg:ro \
		-v /usr/lib64/libaicpu_processer.so:/usr/lib64/libaicpu_processer.so:ro \
		-v /usr/lib64/libaicpu_prof.so:/usr/lib64/libaicpu_prof.so:ro \
		-v /usr/lib64/libaicpu_sharder.so:/usr/lib64/libaicpu_sharder.so:ro \
		-v /usr/lib64/libadump.so:/usr/lib64/libadump.so:ro \
		-v /usr/lib64/libtsd_eventclient.so:/usr/lib64/libtsd_eventclient.so:ro \
		-v /usr/lib64/libaicpu_scheduler.so:/usr/lib64/libaicpu_scheduler.so:ro \
		-v /usr/lib/aarch64-linux-gnu/libcrypto.so.1.1:/usr/lib/aarch64-linux-gnu/libcrypto.so.1.1:ro \
		-v /usr/lib/aarch64-linux-gnu/libyaml-0.so.2:/usr/lib/aarch64-linux-gnu/libyaml-0.so.2:ro \
		-v /usr/lib64/libdcmi.so:/usr/lib64/libdcmi.so:ro \
		-v /usr/lib64/libmpi_dvpp_adapter.so:/usr/lib64/libmpi_dvpp_adapter.so:ro \
		-v /usr/lib64/aicpu_kernels/:/usr/lib64/aicpu_kernels/:ro \
		-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
		-v /usr/lib64/libstackcore.so:/usr/lib64/libstackcore.so:ro \
		-v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
		-v /var/slogd:/var/slogd:ro \
		-v /var/dmp_daemon:/var/dmp_daemon:ro \
		-v /etc/slog.conf:/etc/slog.conf:ro \
		--name qwen_ascend_llm \
		qwen_ascend_llm \
		bash -c "/home/AscendWork/run.sh && python3 api.py"