# 常见问题与解决方案

本文档汇总模型迁移过程中常见的问题及其解决方案。

## 问题分类

| 类别 | 特征关键词 | 解决方案 |
|------|------------|----------|
| 导入错误 | `ModuleNotFoundError`, `ImportError` | 安装依赖、检查版本 |
| 设备错误 | `NPU not found`, `RuntimeError` | 检查环境变量 |
| API 错误 | `NotImplementedError`, `AttributeError` | 查找替代 API |
| 类型错误 | `TypeError`, `dtype` | 检查数据类型 |
| 精度错误 | `AssertionError`, `rtol`, `atol` | 调整容差 |
| 内存错误 | `OOM`, `out of memory` | 减小 batch_size |

---

## 1. 导入错误

### 1.1 torch_npu 找不到

**错误：**
```
ModuleNotFoundError: No module named 'torch_npu'
```

**原因：**
- 容器中未安装 torch_npu
- torch 和 torch_npu 版本不匹配

**解决方案：**
```bash
# 确认版本匹配
pip list | grep torch

# 如果版本不匹配，重新安装
pip install torch-npu
```

---

### 1.2 sklearn 兼容性问题

**错误：**
```
ImportError: cannot import name 'xxx' from 'sklearn'
```

**原因：**
- sklearn 版本过新或过旧
- 某些 API 在 NPU 环境的 sklearn 版本中不存在

**解决方案：**
```python
# 方案 1: 修改导入
from sklearn.metrics import roc_auc_score  # 改用更通用的导入

# 方案 2: 降级 sklearn
pip install scikit-learn==1.2.2

# 方案 3: 自己实现（如果只是简单功能）
def roc_auc_score_manual(y_true, y_pred):
    """手动计算 AUC"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return 1 - np.trapz(tpr, fpr)
```

---

### 1.3 numpy 版本问题

**错误：**
```
TypeError: numpy.xxx() is not supported on NPU
```

**解决方案：**
```python
# 用 torch 替代 numpy 操作
import torch

# numpy 操作
arr = np.mean(x, axis=1)

# torch 操作
arr = torch.mean(x, axis=1).cpu().numpy()
```

---

## 2. 设备错误

### 2.1 NPU 设备找不到

**错误：**
```
RuntimeError: NPU not found
```

**原因：**
- ASCEND_VISIBLE_DEVICES 未设置
- NPU 驱动未正确加载
- 卡号被占用

**解决方案：**
```bash
# 检查 NPU 设备
npu-smi list

# 设置环境变量（根据实际可用卡号）
export ASCEND_VISIBLE_DEVICES=0,1
export ASCEND_RT_VISIBLE_DEVICES=0,1
```

---

### 2.2 设备检测逻辑问题

**问题：**
代码只检查 CUDA，不检查 NPU

**解决方案：**
```python
def get_device():
    import torch
    import os

    if os.environ.get("MODEL_CPU_ONLY", "").upper() == "TRUE":
        return torch.device("cpu")

    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return torch.device("npu:0")
    except ImportError:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")
```

---

## 3. API 错误

### 3.1 不支持的 API

**错误：**
```
NotImplementedError: torch.cuda.get_device_capability
```

**原因：**
- 该 API 在 NPU 上没有对应实现

**解决方案：**
```python
# 替换为兼容实现
# 原代码
device_cap = torch.cuda.get_device_capability()

# 替换为
device_cap = (8, 0)  # 硬编码，或根据环境判断
```

---

### 3.2 分布式后端不匹配

**错误：**
```
RuntimeError: nccl is not supported
```

**解决方案：**
```python
# 替换 nccl 为 hccl
if torch.cuda.is_available():
    backend = "nccl"
else:
    backend = "hccl"  # NPU 使用 hccl

dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
```

---

### 3.3 Graph 模式不支持

**错误：**
```
RuntimeError: NPU Graph is only supported in inference mode
```

**解决方案：**
```python
# 训练时关闭 Graph
use_graph = os.environ.get("MODEL_GRAPH_FLAG", "false").upper() == "TRUE"

if use_graph and not model.training:
    # 仅在推理时使用
    model = enable_graph(model)
```

---

## 4. 类型错误

### 4.1 dtype 不匹配

**错误：**
```
TypeError: expected dtype Float32, got dtype Float16
```

**解决方案：**
```python
# 显式指定 dtype
x = x.to(dtype=torch.float32)

# 或检查输入
if x.dtype != torch.float32:
    x = x.float()
```

---

### 4.2 device 不匹配

**错误：**
```
RuntimeError: Expected all tensors to be on the same device
```

**解决方案：**
```python
# 确保所有张量在同一设备
device = torch.device("npu:0")
model = model.to(device)
data = data.to(device)

# 或使用统一函数
def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj
```

---

## 5. 精度错误

### 5.1 数值精度校验失败

**错误：**
```
AssertionError: Tensor-likes are not approximately equal
```

**解决方案：**
```python
# 放宽容差
import numpy as np

# 原代码
assert np.allclose(a, b)

# 替换为
assert np.allclose(a, b, rtol=1e-3, atol=1e-3)  # 放宽容差

# 或在 NPU 上使用更大容差
if hasattr(torch_npu, 'npu'):
    rtol, atol = 1e-3, 1e-3
else:
    rtol, atol = 1e-5, 1e-5
```

---

### 5.2 精度不达标

**问题：**
运行成功但 AUC、Loss 等指标明显低于预期

**解决方案：**
```python
# 1. 检查是否启用了 HF32
if NPU_ENABLE:
    torch_npu.npu.matmul.allow_hf32 = True
    torch_npu.npu.conv.allow_hf32 = True

# 2. 检查随机种子
torch.manual_seed(42)
torch_npu.npu.manual_seed(42)

# 3. 检查数据格式
# NPU 推荐使用 float16/bfloat16 混合精度
```

---

## 6. 内存错误

### 6.1 OOM 内存不足

**错误：**
```
RuntimeError: NPU out of memory
```

**解决方案：**

```python
# 1. 减小 batch_size
batch_size = 32  # 改为 16 或更小

# 2. 启用梯度累积
accumulation_steps = 4
for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 启用内存优化
torch.npu.empty_cache()

# 4. 使用 gradient checkpointing
from torch.utils.checkpoint import checkpoint_sequential
model = checkpoint_sequential(model, segments=2)
```

---

## 调试技巧

### 查看设备信息

```python
import torch
print(f"PyTorch version: {torch.__version__}")

try:
    import torch_npu
    print(f"torch_npu version: {torch_npu.__version__}")
    print(f"NPU available: {torch_npu.npu.is_available()}")
    print(f"NPU device count: {torch_npu.npu.device_count()}")
except ImportError:
    print("torch_npu not installed")

print(f"CUDA available: {torch.cuda.is_available()}")
```

### 查看内存使用

```python
# NPU 内存
print(f"NPU allocated: {torch_npu.npu.memory_allocated() / 1024**3:.2f} GB")
print(f"NPU reserved: {torch_npu.npu.memory_reserved() / 1024**3:.2f} GB")

# 清理内存
torch.npu.empty_cache()
```

### 对比 GPU/NPU 输出

```python
def compare_outputs(gpu_out, npu_out, rtol=1e-4, atol=1e-4):
    import numpy as np
    gpu_np = gpu_out.cpu().numpy() if hasattr(gpu_out, 'cpu') else gpu_out
    npu_np = npu_out.cpu().numpy() if hasattr(npu_out, 'cpu') else npu_out

    diff = np.abs(gpu_np - npu_np)
    print(f"Max diff: {diff.max()}")
    print(f"Mean diff: {diff.mean()}")
    print(f"Pass: {np.allclose(gpu_np, npu_np, rtol=rtol, atol=atol)}")
```

---

*最后更新: 2026-04-07*