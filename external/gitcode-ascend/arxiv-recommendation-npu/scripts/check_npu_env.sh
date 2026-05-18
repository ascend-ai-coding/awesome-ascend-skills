#!/bin/bash
# NPU 环境检查脚本

echo "========================================"
echo "NPU 环境检查"
echo "========================================"

# 检查 torch_npu
echo -n "torch_npu: "
python -c "import torch_npu; print('OK')" 2>/dev/null || echo "Not installed"

# 检查 CANN
echo -n "CANN: "
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "$ASCEND_TOOLKIT_HOME"
else
    echo "Not found"
fi

# 检查 NPU 设备
echo "NPU 设备:"
npu-smi info -l 2>/dev/null || echo "npu-smi not available"

# 检查 PyTorch
echo -n "PyTorch: "
python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed"

# 检查 deepxiv
echo -n "deepxiv: "
which deepxiv 2>/dev/null || echo "Not installed"

echo "========================================"
echo "检查完成"
echo "========================================"