# ModelScope CLI 参考资料

本文档提供 ModelScope CLI 的参考资料、使用技巧、Python SDK 示例和故障排查指南。

脚本使用中遇到问题时，可结合以下脚本进行排查：
- 环境问题 → `bash scripts/run_preflight_check.sh`
- 网络问题 → `bash scripts/run_network_diagnose.sh`
- 代理配置 → `bash scripts/setup_proxy.sh`
- 下载校验 → `bash scripts/run_check_sha.sh <目录>`
- 参数估算 → `bash scripts/run_report_param.sh <目录>`

---

## 官方文档

- [ModelScope 官方文档](https://modelscope.cn/docs/) - ModelScope 完整官方文档
- [模型下载指南](https://modelscope.cn/docs/models/download) - CLI 模型下载命令详解
- [数据集下载指南](https://modelscope.cn/docs/datasets/dataset) - CLI 数据集下载命令详解
- [Python SDK 文档](https://modelscope.cn/docs/v1.0.0/en/sdk_reference/index.html) - Python API 参考

---

## 模型仓库

| 平台 | 链接 | 说明 |
|-----|------|------|
| ModelScope Hub | [modelscope.cn](https://modelscope.cn/) | 官方模型中心 |
| 数据集中心 | [modelscope.cn/datasets](https://modelscope.cn/datasets) | 官方数据集中心 |

---

## 常用模型

### 通用大模型

| 系列 | 链接 | 说明 |
|-----|------|------|
| Qwen | [搜索](https://modelscope.cn/models?name=Qwen) | 通义千问系列 |
| ZhiPu | [搜索](https://modelscope.cn/models?name=GLM) | 智谱 GLM 系列 |
| Minimax | [搜索](https://modelscope.cn/models?name=Minimax) | Minimax 系列 |
| KIMI | [搜索](https://modelscope.cn/models?name=Kimi) | KIMI 系列 |
| DeepSeek | [搜索](https://modelscope.cn/models?name=DeepSeek) | DeepSeek 系列 |

### Ascend 优化模型

| 组织 | 链接 | 说明 |
|-----|------|------|
| Eco-Tech | [链接](https://modelscope.cn/organization/Eco-Tech) | Ascend 量化模型 (W8A8Z, W4A8) |
| vllm-ascend | [链接](https://modelscope.cn/organization/vllm-ascend) | vLLM-Ascend 基准模型 |

完整推荐列表见 [ASCEND_MODELS.md](ASCEND_MODELS.md)。

---

## CLI 命令详解

### 模型搜索

```bash
# 按名称搜索
modelscope search --model 'qwen'

# 搜索特定组织
modelscope search --model 'Qwen/*'

# 限制结果数
modelscope search --model 'Qwen/*' --limit 10
```

### 模型下载

```bash
# 下载到默认目录
modelscope download --model 'Qwen/Qwen3.5-2B-Base'

# 下载到指定目录
modelscope download --model 'Qwen/Qwen3.5-2B-Base' --local_dir ./models

# 下载特定版本
modelscope download --model 'Qwen/Qwen3.5-2B-Base' --revision v1.0.0

# 包含特定文件
modelscope download --model 'Qwen/Qwen3.5-2B-Base' --include '*.safetensors'

# 排除特定文件
modelscope download --model 'Qwen/Qwen3.5-2B-Base' --exclude '*.onnx,*.onnx_data'
```

### 数据集下载

```bash
# 下载数据集
modelscope download --dataset 'PAI/OmniThought' --local_dir ./datasets
```

> 💡 批量下载建议使用脚本：`bash scripts/run_ms_model_download.sh` 或 `bash scripts/run_ms_datasets_download.sh`

---

## Python SDK

### 基本使用

```python
from modelscope import snapshot_download

# 下载模型
model_dir = snapshot_download('Qwen/Qwen3.5-2B-Base')

# 下载到指定目录
model_dir = snapshot_download(
    'Qwen/Qwen3.5-2B-Base',
    cache_dir='/path/to/cache'
)

# 下载数据集
dataset_dir = snapshot_download('PAI/OmniThought', dataset=True)
```

### 文件过滤

```python
# 只下载特定文件
model_dir = snapshot_download(
    'Qwen/Qwen3.5-2B-Base',
    allow_patterns=['*.safetensors', 'config.json']
)

# 排除特定文件
model_dir = snapshot_download(
    'Qwen/Qwen3.5-2B-Base',
    ignore_patterns=['*.onnx', '*.onnx_data']
)
```

### 批量下载

```python
from modelscope import snapshot_download

models = [
    'Qwen/Qwen3.5-2B-Base',
    'ZhipuAI/GLM-4-9B-Chat',
]

for model_id in models:
    try:
        snapshot_download(model_id)
        print(f"✅ {model_id}")
    except Exception as e:
        print(f"❌ {model_id}: {e}")
```

---

## 环境配置

### 代理配置

```bash
# 手动设置
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# 或使用交互式配置脚本
bash scripts/setup_proxy.sh

# Python 中设置
import os
os.environ['HTTP_PROXY'] = 'http://proxy.example.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxy.example.com:8080'
```

> 💡 `setup_proxy.sh` 会自动持久化配置到 `~/.bashrc`，并支持 pip 镜像源配置。

### SSL 证书问题

```python
# 禁用 SSL 验证（内网环境）
import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
```

> 💡 如需安装自签名 CA 证书，参考 SKILL.md 常见问题章节。

### 下载优化

```bash
# 增加下载线程
export MODELSCOPE_DOWNLOAD_THREAD_NUM=8

# 设置缓存目录
export MODELSCOPE_CACHE=/path/to/cache
```

---

## 参数量计算

使用 `run_report_param.sh` 可自动完成参数量估算，以下为计算原理：

### 计算公式

```
参数量(B) = 模型文件总大小(GB) / 每参数字节数
```

### 精度对照表

| 精度 | 字节/参数 | 文件标识 | 说明 |
|-----|----------|---------|------|
| FP32 | 4.0 | `*FP32*` | 全精度 |
| BF16/FP16 | 2.0 | `*BF16*`, `*FP16*` | 半精度 |
| W8A8Z/W8A8 | 1.0 | `*W8A8*` | 8位量化 |
| W4A8/Q4 | 0.5 | `*W4A8*`, `*Q4*` | 4位量化 |

> 💡 自动统计: `bash scripts/run_report_param.sh <模型目录>` 会根据文件名自动识别精度并计算参数量。

---

## 故障排查

> 💡 遇到网络问题时，先运行 `bash scripts/run_network_diagnose.sh` 获取诊断信息。

### 下载速度慢

**检查项：**
1. 网络带宽
2. 下载线程数（默认4，可增加）
3. 是否使用镜像源
4. 磁盘 I/O

**解决：**
```bash
# 增加线程
export MODELSCOPE_DOWNLOAD_THREAD_NUM=8

# 使用循环重试
bash scripts/ms_loop.sh scripts/run_ms_model_download.sh
```

### 下载中断

```bash
# 断点续传（直接重新运行）
bash scripts/run_ms_model_download.sh

# 查看日志
grep -i "error\|fail" download.log
```

### 权限错误

```bash
# 修改目录权限
sudo chown -R $USER:$USER /path/to/model_dir
```

### 磁盘空间不足

```bash
# 检查空间
df -h

# 清理缓存
rm -rf ~/.cache/modelscope/hub/

# 修改下载目录
# 编辑脚本中的 DIR 变量
```

### 模型 ID 找不到

- 确认格式：`组织/模型名`
- 访问 modelscope.cn 搜索确认
- 检查大小写

---

## 常见使用场景

### 场景 1: 下载到共享存储

```bash
# 挂载共享存储
sudo mount -t nfs <server>:/path /mnt/nfs

# 修改脚本 DIR 变量
DIR=/mnt/nfs/models
```

### 场景 2: 只下载权重文件

```bash
# 修改 EXCLUDE 变量
EXCLUDE="*.onnx *.onnx_data *.bin *.msgpack"
```

### 场景 3: 下载后自动校验

```bash
# 下载
bash scripts/run_ms_model_download.sh

# 校验
bash scripts/run_check_sha.sh ./models/Qwen-2B
```

### 场景 4: Docker 中使用

```dockerfile
FROM python:3.10-slim

RUN pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV MODELSCOPE_CACHE=/cache/modelscope
RUN mkdir -p /cache/modelscope

RUN modelscope download --model 'Qwen/Qwen3.5-2B-Base' --local_dir /models

WORKDIR /models
```
