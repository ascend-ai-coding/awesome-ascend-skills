# 参考：代码级迁移常见模式（按需阅读）

> 无主流程 §。与 [part-04-code-migration.md](part-04-code-migration.md) §5.2～§5.3 配合；命令见 [part-07-commands.md](part-07-commands.md)。

---

## 1. PyTorch：统一 device 抽象（推荐）

避免散落 `.cuda()` / `.npu()`，在工程内集中：

```python
import torch

def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401  # side effect: registers torch.npu backend
        return torch.npu.is_available()
    except ImportError:
        return False

def get_device(prefer: str = "auto") -> torch.device:
    """prefer: cpu | cuda | npu | auto（默认 auto：npu → cuda → cpu）"""
    prefer = prefer.lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "npu":
        return torch.device("npu:0") if _npu_available() else torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if prefer != "auto":
        raise ValueError(f"unknown prefer={prefer!r}; use cpu | cuda | npu | auto")
    if _npu_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

device = get_device("auto")   # NPU 迁移默认
# device = get_device("cuda")  # 仅跑 GPU baseline 时用，不会误选 NPU
model.to(device)
tensor = tensor.to(device, non_blocking=False)
```

| `prefer` | 行为 |
|----------|------|
| `"cpu"` | 固定 CPU |
| `"npu"` | 优先 NPU，不可用则回退 CPU（**不会**选 CUDA） |
| `"cuda"` | 优先 CUDA，不可用则回退 CPU（**不会**选 NPU；适合补 GPU baseline） |
| `"auto"`（默认） | NPU → CUDA → CPU，与 NPU 迁移主路径一致 |

**落盘**：在 `Mig_report` §5.1 注明是否新增 `device_utils` 及调用点。

---

## 2. PyTorch：CUDA → NPU 对照表

| CUDA 写法 | NPU 常见改法 | 备注 |
|-----------|--------------|------|
| `tensor.cuda()` | `tensor.npu()` 或 `.to("npu:0")` | 须已 `import torch_npu` |
| `torch.cuda.device(i)` | `torch.npu.device(i)` | |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` | 性能 profiling 口径一致 |
| `torch.cuda.amp.autocast` | `torch.npu.amp.autocast` | 以当前 torch_npu 文档为准 |
| `GradScaler()` | `torch.npu.amp.GradScaler()` | 核对是否启用 |
| `pin_memory=True` | 通常 `False` | DataLoader |
| `backend="nccl"` | `backend="hccl"` | 分布式 |
| `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` | 多卡可见性 |

**检索命令（改前扫描）**：

```text
rg -n "\.cuda\(|cuda:|torch\.cuda|CUDA_VISIBLE|nccl" --glob "*.py"
```

---

## 3. PyTorch：单卡训练 loop 最小改动

```python
import torch
import torch_npu  # noqa: F401  # side effect: must import before creating npu tensors

device = torch.device("npu:0")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())

scaler = torch.npu.amp.GradScaler(enabled=use_amp)
for batch in loader:
    inputs = batch["image"].to(device, non_blocking=False)
    labels = batch["label"].to(device, non_blocking=False)
    optimizer.zero_grad(set_to_none=True)
    with torch.npu.amp.autocast(enabled=use_amp):
        loss = model(inputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**smoke**：跑 1 个 batch，检查 `loss.item()` 有限且无 NaN → 记入 `Mig_report` §6。

---

## 4. PyTorch：HCCL 分布式初始化（示意）

```python
import os
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # side effect: registers torch.npu backend

def init_dist():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    return rank, local_rank
```

启动命令见 [part-07-commands.md](part-07-commands.md)「多卡 HCCL」。

---

## 5. 自定义算子 / CUDA 扩展：处置顺序

1. 查项目与 CANN 是否已有 Ascend 算子或 `torch_npu` 融合 API  
2. 小算子改 **CPU 回退**（在 forward 内 `x.cpu()` 算完再 `.to(device)`，注明性能）  
3. 替换为等价 `torch` 算子组合  
4. 仍不可行 → `Mig_report` §7 记录，回流 part-02 / part-06  

```python
def safe_op(x):
    if x.device.type == "npu":
        return legacy_cpu_impl(x.cpu()).to(x.device)
    return legacy_cpu_impl(x)
```

---

## 6. MindSpore：上下文与入口

```python
import mindspore as ms
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)

def train_step(data, label):
    loss = network(data, label)
    return loss
```

**动图调试**：可临时 `PYNATIVE_MODE` smoke，再切回 `GRAPH_MODE` 做性能评测；变更写入 `Mig_report` §5.4。

---

## 7. 预处理与 Golden 对齐

Golden 对比前必须一致：

- resize / crop / normalize 参数与 **基线同一实现**（勿 NPU 侧改顺序）  
- `NCHW` vs `NHWC`、mean/std 数值  
- 固定 `torch.manual_seed` / `numpy` seed  
- **容差按目标精度设定**（默认 FP16：`rtol` 1e-2~1e-3、`atol` ~1e-3）；勿照抄 FP32 级 `atol 1e-5` → 见 `Compare` §3.1、[part-07](part-07-commands.md)

输出对比记录：`shape`、max abs diff、mean abs diff、所用 rtol/atol → `Compare` §3.1、`Mig_report` §6。

---

## 关联索引

- **主清单**：[part-04-code-migration.md](part-04-code-migration.md)  
- **命令**：[part-07-commands.md](part-07-commands.md)  
- **排障**：[part-09-examples-troubleshooting.md](part-09-examples-troubleshooting.md)
