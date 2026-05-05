---
name: external-gitcode-ascend-modelscope-cli
description: ModelScope CLI 模型与数据集下载工具。当用户需要从 ModelScope 下载模型或数据集、批量下载模型、校验文件完整性、统计模型参数量、或进行网络诊断时使用。
original-name: modelscope-cli
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
license: UNKNOWN
---

# 概述

ModelScope CLI Skill 提供 ModelScope 平台模型与数据集的批量下载、完整性校验、参数量统计等功能，支持 Ascend NPU 等多种部署场景。

**核心价值：**
- 批量下载 ModelScope 模型和数据集
- 自动环境检测与前置验证
- 网络诊断与故障排查
- SHA256 完整性校验
- 模型参数量智能统计
- 循环重试机制应对网络不稳定

**推荐：** Ascend NPU 部署建议优先从 [Eco-Tech 组织](https://modelscope.cn/organization/Eco-Tech) 下载已优化的量化模型，详见 [reference/ASCEND_MODELS.md](reference/ASCEND_MODELS.md)。

---

# 前置条件

## 必需环境

| 依赖 | 版本要求 | 检查命令 |
|-----|---------|---------|
| Python | >= 3.7 | `python3 --version` |
| ModelScope CLI | 最新版 | `modelscope --version` |
| 磁盘空间 | 100GB+ (推荐) | `df -h` |

## 安装 ModelScope

```bash
# 推荐：清华镜像源
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 网络要求

- 外网访问能力（或配置代理）
- 内网环境需配置代理和 SSL 证书，详见 [代理配置](#代理配置)

---

# 快速开始

## 步骤 1：环境检查

```bash
# 执行前置检查
bash scripts/run_preflight_check.sh

# 如检查失败，运行诊断
bash scripts/run_network_diagnose.sh
```

## 步骤 2：下载模型

编辑 `scripts/run_ms_model_download.sh` 配置模型列表，然后执行：

```bash
bash scripts/run_ms_model_download.sh
```

## 步骤 3：校验完整性

```bash
bash scripts/run_check_sha.sh ./models/Qwen-2B
```

---

# 使用方法

## 模型下载

### 配置下载列表

编辑 `scripts/run_ms_model_download.sh`：

```bash
# 模型列表
MODELS=(
  Qwen/Qwen3.5-2B-Base
  Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp
)

# 下载目录（请根据实际环境修改）
DIR="./models"

# 排除文件
EXCLUDE="*.onnx *.onnx_data"
```

### 执行下载

```bash
# 基本下载
bash scripts/run_ms_model_download.sh

# 循环重试（网络不稳定时推荐）
bash scripts/ms_loop.sh scripts/run_ms_model_download.sh

# 跳过前置检查
SKIP_PREFLIGHT=1 bash scripts/run_ms_model_download.sh
```

### 直接使用 CLI

```bash
# 下载模型
modelscope download --model 'Qwen/Qwen3.5-2B-Base' --local_dir ./models

# 下载数据集
modelscope download --dataset 'WorldVQA/WorldVQA' --local_dir ./datasets

# 排除文件
modelscope download --model 'Qwen/Qwen3.5-2B-Base' --exclude '*.onnx,*.onnx_data'
```

更多 CLI 用法详见 [reference/wiki.md](reference/wiki.md)。

---

## 数据集下载

编辑 `scripts/run_ms_datasets_download.sh`：

```bash
DATASETS=(
  WorldVQA/WorldVQA
)
DIR="./datasets"
```

执行：

```bash
bash scripts/run_ms_datasets_download.sh
```

---

## 完整性校验

```bash
# 校验单个模型
bash scripts/run_check_sha.sh ./models/Qwen-2B

# 校验会自动生成 .sha256sum 文件并验证
```

---

## 参数量统计

```bash
# 统计单个模型
bash scripts/run_report_param.sh ./models/Qwen-2B

# 统计目录下所有模型
bash scripts/run_report_param.sh ./models
```

**输出示例：**
```
模型: Qwen-2B
========================================
权重文件数量: 2
模型总大小: 4.00 GB
数据精度: BF16/FP16 (每参数 2.0 字节)
推测参数量: 2.00 B (1-7B)
```

参数量计算原理详见 [reference/wiki.md - 参数量计算](reference/wiki.md)。

---

## 网络诊断

```bash
# 完整诊断
bash scripts/run_network_diagnose.sh
```

**诊断项目：**
- 环境变量（代理配置）
- DNS 解析
- Ping 连通性
- HTTP 连接测试
- SSL 证书信息
- Python SSL 测试

诊断结果的详细排查方法见 [reference/wiki.md - 故障排查](reference/wiki.md)。

---

## 代理配置

```bash
# 交互式代理配置
bash scripts/setup_proxy.sh
```

该脚本支持：
- 检测已有代理配置
- 交互式设置 HTTP/HTTPS 代理
- 持久化到 `~/.bashrc`
- 可选配置 pip 镜像源

也可手动设置：

```bash
export HTTP_PROXY=http://proxy-host:port
export HTTPS_PROXY=http://proxy-host:port
```

---

## 循环重试

网络不稳定时使用循环重试：

```bash
# 基本用法
bash scripts/ms_loop.sh scripts/run_ms_model_download.sh

# 自定义重试间隔（秒）
bash scripts/ms_loop.sh scripts/run_ms_model_download.sh 10

# 记录日志
bash scripts/ms_loop.sh scripts/run_ms_model_download.sh 5 2>&1 | tee download.log
```

---

# 脚本说明

| 脚本 | 功能 | 用法 | 参考文档 |
|-----|------|------|---------|
| `run_preflight_check.sh` | 环境前置检查 | `bash scripts/run_preflight_check.sh` | - |
| `run_network_diagnose.sh` | 网络诊断 | `bash scripts/run_network_diagnose.sh` | [wiki.md - 故障排查](reference/wiki.md) |
| `run_ms_model_download.sh` | 批量下载模型 | `bash scripts/run_ms_model_download.sh` | [wiki.md - CLI 命令](reference/wiki.md) |
| `run_ms_datasets_download.sh` | 批量下载数据集 | `bash scripts/run_ms_datasets_download.sh` | [wiki.md - CLI 命令](reference/wiki.md) |
| `run_check_sha.sh` | SHA256 校验 | `bash scripts/run_check_sha.sh <目录>` | - |
| `run_report_param.sh` | 参数量统计 | `bash scripts/run_report_param.sh <目录>` | [wiki.md - 参数量计算](reference/wiki.md) |
| `ms_loop.sh` | 循环重试 | `bash scripts/ms_loop.sh <脚本> [间隔]` | - |
| `setup_proxy.sh` | 代理配置 | `bash scripts/setup_proxy.sh` | [wiki.md - 环境配置](reference/wiki.md) |

---

# 常见问题

## 1. 前置检查失败

**现象：** `run_preflight_check.sh` 报告检查失败

**解决：**
```bash
# 运行诊断
bash scripts/run_network_diagnose.sh

# 根据诊断结果：
# - 代理未配置 → 运行 bash scripts/setup_proxy.sh 或手动设置环境变量
# - SSL 证书失败 → 安装证书或禁用验证
# - 网络不通 → 检查防火墙/代理
```

---

## 2. SSL 证书验证失败

**原因：** 内网自签名证书

**解决方法 1 - 安装证书（推荐）：**
```bash
# 将自签名 CA 证书添加到系统信任库
# CentOS/RHEL:
sudo cp your-ca.crt /etc/pki/ca-trust/source/anchors/
sudo update-ca-trust

# Ubuntu/Debian:
sudo cp your-ca.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

**解决方法 2 - 禁用验证：**
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

---

## 3. 下载中断

**解决：**
```bash
# 使用循环重试
bash scripts/ms_loop.sh scripts/run_ms_model_download.sh

# 断点续传（重新执行会自动跳过已下载文件）
bash scripts/run_ms_model_download.sh
```

---

## 4. 磁盘空间不足

**解决：**
```bash
# 检查空间
df -h

# 清理缓存
rm -rf ~/.cache/modelscope/hub/

# 修改下载目录
# 编辑 run_ms_model_download.sh 中的 DIR 变量
```

---

## 5. 模型 ID 找不到

**解决：**
- 确认格式：`组织/模型名`
- 访问 https://modelscope.cn/models 搜索
- 检查大小写是否正确

---

# 相关资源

## 官方文档

- [ModelScope 官方文档](https://modelscope.cn/docs)
- [模型下载指南](https://modelscope.cn/docs/models/download)
- [数据集下载指南](https://modelscope.cn/docs/datasets/dataset)

## 推荐模型组织

| 组织 | 链接 | 说明 |
|-----|------|------|
| Eco-Tech | [链接](https://modelscope.cn/organization/Eco-Tech) | Ascend 优化量化模型 |
| vllm-ascend | [链接](https://modelscope.cn/organization/vllm-ascend) | vLLM-Ascend 基准模型 |
| Qwen | [链接](https://modelscope.cn/organization/Qwen) | 通义千问系列 |
| ZhipuAI | [链接](https://modelscope.cn/organization/ZhipuAI) | 智谱 GLM 系列 |

更多模型推荐见 [reference/ASCEND_MODELS.md](reference/ASCEND_MODELS.md)。

## 参考文档

- [reference/wiki.md](reference/wiki.md) - CLI 详细用法、Python SDK、参数量计算、故障排查
- [reference/ASCEND_MODELS.md](reference/ASCEND_MODELS.md) - Ascend 推荐模型与量化说明
