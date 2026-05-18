# NPU Commands
# 昇腾 NPU 卡信息查询命令

# 查看 NPU 卡基本信息
npu-smi info

# 查看 NPU 卡详细信息（driver版本、固件版本）
npu-smi info -v

# 查看 NPU 占用情况
npu-smi top

# 查看指定 NPU 卡的详细信息
npu-smi info -i 0

# 查看 NPU driver版本
npu-smi list
