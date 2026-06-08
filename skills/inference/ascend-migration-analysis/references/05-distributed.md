# Domain 5: Distributed Communication Analysis

## Overview

扫描项目中的分布式通信代码，评估 HCCL 替代可行性。

## Scan Commands

```bash
# Distributed initialization
rg "init_process_group" --include="*.py" -n
rg "backend.*=.*['\"]" --include="*.py" -n | grep -i nccl

# Device mesh
rg "init_device_mesh|DeviceMesh" --include="*.py" -n

# Collective communication
rg "dist\.all_reduce|dist\.all_gather|dist\.all_to_all|dist\.broadcast|dist\.scatter|dist\.reduce" --include="*.py" -n
rg "all_to_all_single" --include="*.py" -n

# Point-to-point
rg "dist\.send|dist\.recv|dist\.isend|dist\.irecv" --include="*.py" -n
rg "P2POp|batch_isend_irecv" --include="*.py" -n

# Barrier / sync
rg "dist\.barrier" --include="*.py" -n
rg "synchronize" --include="*.py" -n

# Sequence/Context parallel
rg "ulysses|ring_attention|sequence_parallel|context_parallel" --include="*.py" -n

# Process group management
rg "new_group|get_process_group_ranks|get_world_size|get_rank" --include="*.py" -n
```

## Replacement Solutions

### 5.1 Backend: NCCL → HCCL

```python
# Original
dist.init_process_group(backend="nccl", ...)

# Ascend
import torch_npu
dist.init_process_group(backend="hccl", ...)
torch_npu.npu.set_device(local_rank)
```

HCCL 支持 NCCL 的全部标准集合通信操作。

### 5.2 Collective Operations Compatibility

| Operation | HCCL Support | Notes |
|-----------|-------------|-------|
| `dist.all_reduce` | Yes | 直接替换后端即可 |
| `dist.all_gather` | Yes | 直接替换后端即可 |
| `dist.all_to_all_single` | Yes | 需验证大 tensor 场景 |
| `dist.broadcast` | Yes | 直接替换后端即可 |
| `dist.scatter` | Yes | 直接替换后端即可 |
| `dist.reduce` | Yes | 直接替换后端即可 |
| `dist.barrier` | Yes | 直接替换后端即可 |
| `dist.isend` / `dist.irecv` | Yes | P2P 通信 |
| `dist.batch_isend_irecv` | **需验证** | 批量 P2P，需测试 HCCL 兼容性 |
| `dist.P2POp` | **需验证** | 同上 |

### 5.3 DeviceMesh

```python
# Original
mesh = init_device_mesh("cuda", (dp_size, sp_size), mesh_dim_names=("dp", "sp"))

# Ascend
mesh = init_device_mesh("npu", (dp_size, sp_size), mesh_dim_names=("dp", "sp"))
```

**风险**: `torch_npu` 对 `DeviceMesh` 的支持需要验证。如不支持，需退化为手动 `dist.new_group()` 管理 process group。

### 5.4 Sequence Parallel Frameworks

| Original Framework | Ascend Alternative | Notes |
|-------------------|-------------------|-------|
| 手写 Ulysses all-to-all | `yunchang` 框架 `xFuserLongContextAttention` | 成熟方案 |
| Ring Attention | `yunchang` 框架内置 Ring 支持 | 成熟方案 |
| DeepSpeed ZeRO | FSDP on NPU | 需验证 |
| 手写 Context Parallel | 适配为 `yunchang` 的 SP 模式 | 需重构 |

### 5.5 Dual-Channel Communication Pattern

来自 Wan 迁移经验：为每个 process group 创建 device group (HCCL) + CPU group (Gloo)：

```python
device_group = dist.new_group(ranks, backend="hccl")  # Tensor 通信
cpu_group = dist.new_group(ranks, backend="gloo")      # 元数据通信
```

## Output Template

```markdown
### Distributed Communication Analysis

| # | Operation | File | Line | HCCL Support | Notes |
|---|-----------|------|------|-------------|-------|
| 1 | init_process_group(nccl) | run_xxx.py | 42 | Yes | → hccl |
| 2 | all_to_all_single | sp_util.py | 51 | Yes | 验证大 tensor |
| 3 | batch_isend_irecv | comm.py | 29 | Verify | 需测试 |
| 4 | init_device_mesh("cuda") | cp_util.py | 26 | Verify | → "npu" |
```
