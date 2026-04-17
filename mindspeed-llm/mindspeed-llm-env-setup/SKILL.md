---
name: mindspeed-llm-env-setup
description: MindSpeed-LLM 环境搭建指南，用于华为昇腾 NPU。覆盖 CANN 环境激活、PyTorch + torch_npu 安装、MindSpeed 加速库安装、Megatron-LM 核心模块集成、MindSpeed-LLM 安装及环境验证。当用户需要在昇腾 NPU 上搭建 MindSpeed-LLM 训练环境时使用。
keywords:
    - mindspeed
    - mindspeed-llm
    - environment
    - 环境搭建
    - installation
    - 安装
    - cann
    - torch_npu
    - megatron
    - ascend npu
---

# MindSpeed-LLM 昇腾 NPU 环境搭建

本 Skill 指导用户在华为昇腾 NPU 上搭建 MindSpeed-LLM 分布式训练环境。

## 组件关系

```
Megatron-LM (NVIDIA)     ← 分布式训练核心 (TP/PP)，使用 core_v0.12.1 分支
    ↑
MindSpeed (Huawei)       ← 昇腾适配层，猴子补丁优化 Megatron 内核
    ↑
MindSpeed-LLM (Huawei)   ← 应用层：训练脚本、数据预处理、权重转换
```

## 快速开始

6 步完成环境搭建：

```bash
# 1. 激活 CANN 环境（CANN 8.5.0+ 路径；旧版用 /usr/local/Ascend/ascend-toolkit/set_env.sh）
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 2. 安装 PyTorch + torch_npu
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.7.1rc1
pip install numpy pyyaml scipy attrs decorator psutil

# 3. 克隆并安装 MindSpeed
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed && pip install -r requirements.txt && pip install -e . && cd ..

# 4. 克隆 Megatron-LM 并复制核心模块
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && git checkout core_v0.12.1 && cd ..
cp -r Megatron-LM/megatron MindSpeed-LLM/

# 5. 安装 MindSpeed-LLM
cd MindSpeed-LLM && pip install -r requirements.txt

# 6. 验证环境
python -c "
import torch
import torch_npu
print(f'NPUs available: {torch_npu.npu.device_count()}')
print(f'NPU ready: {torch.npu.is_available()}')
"
```

## 版本兼容矩阵

| CANN | PyTorch | torch_npu | Python | Megatron-LM |
|------|---------|-----------|--------|-------------|
| 8.5.0 | 2.7.1 | 2.7.1rc1 | 3.10 | core_v0.12.1 |

> 以上为当前 master 分支版本。历史版本请查阅 [官方安装文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/install_guide.md)。
> 检查 CANN 版本：`cat /usr/local/Ascend/cann/latest/aarch64-linux/ascend_toolkit_install.info`（CANN 8.5.0+）或 `cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info`（旧版）。

## Docker 容器创建

```bash
docker run -it --name mindspeed-llm \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /home/workspace:/home/workspace \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10 \
    bash
```

> 注意：CANN 容器镜像不含 ML 依赖，进入容器后需手动安装 PyTorch、torch_npu 等。

## 目录结构

安装完成后，工作区结构应如下：

```
workspace/
├── MindSpeed/                    # 加速库
├── MindSpeed-LLM/                # 主项目
│   ├── megatron/                 # 从 Megatron-LM 复制的核心模块
│   ├── examples/mcore/           # 各模型的示例脚本
│   ├── preprocess_data.py        # 数据预处理入口
│   ├── convert_ckpt.py           # 权重转换 v1
│   ├── convert_ckpt_v2.py        # 权重转换 v2
│   ├── pretrain_gpt.py           # 预训练入口
│   └── posttrain_gpt.py          # 微调/后训练入口
├── Megatron-LM/                  # NVIDIA 原始仓库（仅用于复制 megatron/）
├── model_from_hf/                # HuggingFace 模型权重
└── dataset/                      # 训练数据集
```

## 环境验证清单

| 检查项 | 命令 | 预期结果 |
|--------|------|----------|
| CANN 环境 | `npu-smi info` | 显示 NPU 设备信息 |
| PyTorch | `python -c "import torch; print(torch.__version__)"` | `2.7.1` |
| torch_npu | `python -c "import torch_npu; print(torch_npu.__version__)"` | `2.7.1rc1` |
| NPU 数量 | `python -c "import torch_npu; print(torch_npu.npu.device_count())"` | ≥1（与硬件一致） |
| MindSpeed | `pip show mindspeed` | 显示包信息 |
| megatron 模块 | `ls MindSpeed-LLM/megatron/` | 目录存在且非空 |

## 常见问题

**Q: `ModuleNotFoundError: No module named 'yaml'` 导入 torch_npu 时**

CANN Docker 镜像缺少基础 Python 包：

```bash
pip install numpy pyyaml scipy attrs decorator psutil
```

**Q: 网络问题导致 pip install 或 git clone 失败**

使用代理：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

**Q: Megatron-LM checkout 分支找不到**

确认使用正确的分支名（注意 `core_v` 前缀）：

```bash
cd Megatron-LM
git branch -a | grep core
git checkout core_v0.12.1
```

## 使用顺序

环境搭建完成后，按以下顺序进行训练部署：

1. **数据预处理** → 使用 [mindspeed-llm-data-prep](../mindspeed-llm-data-prep/SKILL.md)
2. **权重转换** → 使用 [mindspeed-llm-weight-prep](../mindspeed-llm-weight-prep/SKILL.md)
3. **训练启动** → 使用 [mindspeed-llm-training](../mindspeed-llm-training/SKILL.md)

## 参考资源

- [详细安装指南](references/installation.md) - 完整安装步骤和多版本配置
- [MindSpeed-LLM 仓库](https://gitcode.com/ascend/MindSpeed-LLM)
- [MindSpeed 仓库](https://gitcode.com/ascend/MindSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [昇腾文档](https://www.hiascend.com/document)
