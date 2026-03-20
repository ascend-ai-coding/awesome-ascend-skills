# MindSpeed-LLM 详细安装指南

## 多版本 CANN 环境

不同 CANN 版本的环境激活路径：

```bash
# CANN 8.5.0+（新路径）
source /usr/local/Ascend/cann/set_env.sh

# CANN 8.0.x - 8.3.x（旧路径）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ATB 加速库
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## 完整版本兼容矩阵

| CANN | PyTorch | torch_npu | Python | Megatron-LM | MindSpeed |
|------|---------|-----------|--------|-------------|-----------|
| 8.5.0 | 2.7.1 | 2.7.1rc1 | 3.10 | core_v0.12.1 | master |

> 以上为当前 master 分支版本。历史版本请查阅 [官方安装文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/install_guide.md)。
> 查看 CANN 版本：`cat /usr/local/Ascend/cann/latest/aarch64-linux/ascend_toolkit_install.info`（CANN 8.5.0+）或 `cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info`（旧版）。

## 安装步骤详解

### 1. PyTorch CPU 版安装

MindSpeed-LLM 在 NPU 上训练，不需要 CUDA 版 PyTorch：

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

### 2. torch_npu 安装

```bash
pip install torch_npu==2.7.1rc1
```

常见缺失依赖（CANN 容器可能不含）：

```bash
pip install numpy pyyaml scipy attrs decorator psutil
```

### 3. MindSpeed 安装

```bash
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
pip install -r requirements.txt
pip install -e .
cd ..
```

### 4. Megatron-LM 核心模块

**不需要安装整个 Megatron-LM**，只需复制 `megatron/` 目录：

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cd ..
cp -r Megatron-LM/megatron MindSpeed-LLM/
```

### 5. MindSpeed-LLM 依赖

```bash
cd MindSpeed-LLM
pip install -r requirements.txt
```

## Docker 镜像选择

| 镜像 | 说明 |
|------|------|
| `ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10` | Ascend 910B，CANN 8.5.0 |
| `ascendhub/cann:8.3.RC1-910b-ubuntu22.04-py3.10` | Ascend 910B，CANN 8.3.RC1 |

镜像仓库前缀：`swr.cn-south-1.myhuaweicloud.com/`

## 多机环境额外配置

### SSH 免密登录

```bash
ssh-keygen -t rsa -N ""
ssh-copy-id root@node1
ssh-copy-id root@node2
```

### HCCL 网络配置

```bash
# 检查 NPU 间通信
hccl_test -b 8 -e 256M -d float16 -o allreduce
```

## 故障排查

### torch_npu 导入失败

```bash
# 检查 CANN 环境
echo $ASCEND_HOME_PATH
ls /usr/local/Ascend/ascend-toolkit/latest/

# 重新激活环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### pip 安装超时

```bash
pip install --timeout 300 -i https://pypi.tuna.tsinghua.edu.cn/simple torch_npu==2.7.1rc1
```

### Megatron-LM 分支找不到

```bash
cd Megatron-LM
git fetch --all
git branch -a | grep core_v
```

## 官方参考

- [MindSpeed-LLM 安装文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/install_guide.md)
- [torch_npu 版本发布](https://gitcode.com/Ascend/pytorch/releases)
- [CANN 下载](https://www.hiascend.com/developer/download)
