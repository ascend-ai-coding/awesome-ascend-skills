#!/bin/bash
# ProteinBERT NPU 一键部署脚本
# 用法: bash setup.sh
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ENV_NAME=proteinbert_npu
CANN_PATH=${CANN_PATH:-/home/Ascend/ascend-toolkit}
WEIGHTS_DIR=~/proteinbert_models

echo "============================================================"
echo "ProteinBERT NPU 部署工具"
echo "============================================================"

# 1. CANN 环境
echo ""
echo "[1/5] 初始化 CANN 环境..."
if [ -f "${CANN_PATH}/set_env.sh" ]; then
    source "${CANN_PATH}/set_env.sh"
    echo "      CANN: $(cat ${CANN_PATH}/latest/version.cfg 2>/dev/null | grep toolkit_running | head -1)"
else
    echo "      警告: 未找到 ${CANN_PATH}/set_env.sh"
    echo "      请设置 CANN_PATH 环境变量指向 ascend-toolkit 路径"
fi

# 2. Conda 环境
echo ""
echo "[2/5] 创建 Conda 环境 ${ENV_NAME}..."
if conda env list | grep -q "${ENV_NAME}"; then
    echo "      环境已存在，跳过创建"
else
    conda create -n ${ENV_NAME} python=3.11 -y
fi

# 3. 安装依赖
echo ""
echo "[3/5] 安装依赖..."
conda run -n ${ENV_NAME} pip install -r "${SCRIPT_DIR}/requirements.txt" \
    -i https://repo.huaweicloud.com/repository/pypi/simple/

# 4. 验证 NPU
echo ""
echo "[4/5] 验证 NPU..."
conda run -n ${ENV_NAME} python -c "
import torch, torch_npu
a = torch.randn(2,3).npu()
print('NPU 验证通过:', a.device)
"

# 5. 权重转换
echo ""
echo "[5/5] 检查权重文件..."
mkdir -p ${WEIGHTS_DIR}
TF_PKL="${WEIGHTS_DIR}/epoch_92400_sample_23500000.pkl"
PT_FILE="${WEIGHTS_DIR}/proteinbert_pytorch.pt"

if [ -f "${PT_FILE}" ]; then
    echo "      PyTorch 权重已存在: ${PT_FILE}"
elif [ -f "${TF_PKL}" ]; then
    echo "      找到 TF 权重，开始转换..."
    conda run -n ${ENV_NAME} python "${SCRIPT_DIR}/convert_weights.py" \
        --input "${TF_PKL}" --output "${PT_FILE}"
else
    echo "      警告: 未找到权重文件"
    echo "      请下载 TF 权重到: ${TF_PKL}"
    echo "      下载地址: https://zenodo.org/records/10371965"
    echo "      然后运行:"
    echo "        conda run -n ${ENV_NAME} python ${SCRIPT_DIR}/convert_weights.py -i ${TF_PKL} -o ${PT_FILE}"
fi

echo ""
echo "============================================================"
echo "部署完成！使用方法:"
echo "============================================================"
echo ""
echo "推理:"
echo "  conda run -n ${ENV_NAME} python ${SCRIPT_DIR}/inference_npu.py \\"
echo "      --weights ${PT_FILE} \\"
echo "      --seqs MKTVRQERLKSIVRILERSKEPVSGAQ ACDEFGHIKLMNPQRSTUVWXY"
echo ""
echo "微调:"
echo "  conda run -n ${ENV_NAME} python ${SCRIPT_DIR}/finetune_npu.py \\"
echo "      --weights ${PT_FILE} \\"
echo "      --train-csv /path/to/train.csv \\"
echo "      --test-csv /path/to/test.csv \\"
echo "      --task binary"
echo ""
