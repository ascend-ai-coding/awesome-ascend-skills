# Domain 2: Device Layer Scan

## Overview

扫描代码库中所有 CUDA 设备层 API 调用，提供逐项替换方案。

## Scan Commands

```bash
# torch.cuda.* API calls
rg "torch\.cuda\." --include="*.py" -n

# Hardcoded "cuda" device strings
rg '"cuda"' --include="*.py" -n
rg "'cuda'" --include="*.py" -n
rg "device=\"cuda\"" --include="*.py" -n

# NCCL backend
rg "backend.*nccl" --include="*.py" -n

# autocast with cuda
rg "autocast.*cuda" --include="*.py" -n
rg "autocast\('cuda'" --include="*.py" -n
rg "autocast\(device_type.*cuda" --include="*.py" -n

# torch.compile usage
rg "@torch\.compile" --include="*.py" -n
rg "torch\.compile\(" --include="*.py" -n
rg "torch\.compiler\." --include="*.py" -n
rg "torch\._dynamo" --include="*.py" -n

# init_device_mesh with cuda
rg "init_device_mesh" --include="*.py" -n

# CUDA Stream/Event
rg "torch\.cuda\.Stream" --include="*.py" -n
rg "torch\.cuda\.Event" --include="*.py" -n

# Device capability checks
rg "get_device_capability" --include="*.py" -n
rg "is_available" --include="*.py" -n | grep cuda
```

## Replacement Solutions

### 2.1 One-Line Auto-Redirect

在入口脚本最顶部添加：

```python
import torch
import torch_npu

torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False

from torch_npu.contrib import transfer_to_npu
```

此行 monkey-patch **自动覆盖**：

| Auto-Covered API | Redirected To |
|------------------|---------------|
| `torch.cuda.empty_cache()` | `torch.npu.empty_cache()` |
| `torch.cuda.ipc_collect()` | `torch.npu.ipc_collect()` |
| `torch.cuda.set_device(x)` | `torch.npu.set_device(x)` |
| `torch.cuda.device_count()` | `torch.npu.device_count()` |
| `torch.cuda.is_available()` | `torch.npu.is_available()` |
| `tensor.to("cuda")` | `tensor.to("npu")` |
| `tensor.cuda()` | `tensor.npu()` |
| `torch.Generator(device="cuda")` | `torch.Generator(device="npu")` |

### 2.2 Manual Replacements Required

以下 API **不会被 transfer_to_npu 自动覆盖**，需手动替换：

#### NCCL → HCCL

```python
# Original
dist.init_process_group(backend="nccl", ...)

# Ascend
dist.init_process_group(backend="hccl", ...)
```

#### autocast device_type

```python
# Original
with amp.autocast(device_type='cuda', dtype=torch.float32):

# Ascend
with amp.autocast(device_type='npu', dtype=torch.float32):
```

#### init_device_mesh

```python
# Original
init_device_mesh("cuda", (dp_size, cp_size), ...)

# Ascend
init_device_mesh("npu", (dp_size, cp_size), ...)
```

#### torch.cuda.Stream

```python
# Original
stream = torch.cuda.Stream()

# Ascend
stream = torch.npu.Stream()
```

#### Hardcoded self.device = "cuda"

```python
# Ascend
self.device = "npu"
```

#### torch.cuda.get_device_capability()

```python
# Original (hardware feature check)
if torch.cuda.get_device_capability()[0] >= 9: ...

# Ascend: 移除此检查，或替换为 NPU 能力查询
# 通常此检查用于决定是否启用某个优化（如 TMA），
# 在 NPU 上应使用 NPU 专有的替代方案
```

### 2.3 torch.compile Handling

| Pattern | Action |
|---------|--------|
| `@torch.compile` decorator | 移除装饰器 |
| `model = torch.compile(model)` | 移除 compile 调用，直接使用 model |
| `@torch.compiler.disable` | 移除装饰器 |
| `torch._dynamo.config.*` | 移除配置 |
| `--enable_compile` flag | 移除参数，或改为 no-op |

**理由**: torch_npu 对 torch.compile 的支持有限且不稳定。禁用编译的性能损失由 NPU 原生算子补偿。

### 2.4 Environment Variables

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
```

## Output Template

```markdown
### Device Layer Replacement Checklist

| # | File | Line | Original | Replacement | Auto/Manual |
|---|------|------|----------|-------------|-------------|
| 1 | run_xxx.py | 42 | `backend="nccl"` | `backend="hccl"` | Manual |
| 2 | model.py | 81 | `self.device = "cuda"` | `self.device = "npu"` | Manual |
| 3 | model.py | 82 | `autocast('cuda')` | `autocast('npu')` | Manual |
| 4 | train.py | 22 | `torch.cuda.empty_cache()` | (auto-redirected) | Auto |
| ... | | | | | |

**Manual replacements: X 处**
**Auto-covered by transfer_to_npu: Y 处**
**Estimated effort: Z 天**
```
