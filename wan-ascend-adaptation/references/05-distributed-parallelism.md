# Domain 5: Distributed Parallelism Refactoring

## Overview

The original codebase supports simple FSDP + Ulysses sequence parallelism. The Ascend version introduces a comprehensive 4-dimensional parallel framework: **TP (Tensor Parallel) × SP (Sequence Parallel) × CFG (Classifier-Free Guidance Parallel)**, managed through a unified parallel manager.

## 5.1 Parallel Architecture Comparison

```
Original:
  world_size = sp_size (Ulysses only)
  Single NCCL process group

Ascend:
  world_size = tp_size × sp_size × cfg_size
  sp_size = ulysses_size × ring_size
  Multiple HCCL process groups (TP, SP, CFG, World)
  Each group has device_group (HCCL) + cpu_group (Gloo)
```

## 5.2 ParallelConfig

```python
@dataclass
class ParallelConfig:
    tp_degree: int = 1        # Tensor parallel degree
    sp_degree: int = 1        # Sequence parallel degree (ulysses × ring)
    ulysses_degree: int = 1   # Ulysses sub-parallelism within SP
    ring_degree: int = 1      # Ring attention sub-parallelism within SP
    use_cfg_parallel: bool = False  # CFG parallel (2× if enabled)
    world_size: int = 1

    # Auto-derived: cfg_degree = 2 if use_cfg_parallel else 1
    # Constraint: tp_degree * sp_degree * cfg_degree == world_size
```

## 5.3 RankGenerator — Orthogonal Group Assignment

The `RankGenerator` creates non-overlapping process groups for each parallelism dimension:

```python
class RankGenerator:
    def __init__(self, tp, sp, cfg, order="tp-sp-cfg", rank_offset=0):
        # order defines dimension nesting: "tp-sp-cfg" means
        # TP groups are innermost, CFG groups are outermost
        ...

    def get_ranks(self, token):
        # token: "tp", "sp", or "cfg"
        # Returns: list of rank groups for the specified dimension
        ...
```

**Example:** 8 GPUs with tp=2, sp=2, cfg=2, order="tp-sp-cfg":
```
TP groups:  [[0,1], [2,3], [4,5], [6,7]]
SP groups:  [[0,2], [1,3], [4,6], [5,7]]
CFG groups: [[0,4], [1,5], [2,6], [3,7]]
```

## 5.4 GroupCoordinator — Communication Abstraction

```python
class GroupCoordinator:
    """Manages a single process group with dual-channel communication."""

    def __init__(self, group_ranks, local_rank, torch_distributed_backend, ...):
        # Create device group (HCCL for NPU) for tensor communication
        self.device_group = dist.new_group(ranks, backend=torch_distributed_backend)
        # Create CPU group (Gloo) for metadata communication
        self.cpu_group = dist.new_group(ranks, backend="gloo")

    # Rich communication primitives
    def all_reduce(self, input_): ...
    def all_gather(self, input_, dim=-1, separate_tensors=False): ...
    def broadcast(self, input_, src=0): ...
    def send(self, tensor, dst): ...
    def recv(self, size, dtype, src): ...
    def barrier(self): ...
    def broadcast_tensor_dict(self, tensor_dict, src=0): ...
```

**Dual-channel design:**
- Device group (HCCL): High-bandwidth tensor transfers
- CPU group (Gloo): Low-latency metadata, shapes, dtypes

**NPU device detection:**
```python
if torch.npu.is_available():
    self.device = torch.device(f"npu:{local_rank}")
```

## 5.5 Initialization Flow

```python
def init_parallel_env(parallel_config: ParallelConfig):
    # Step 1: Initialize distributed environment with HCCL
    init_distributed_environment(backend='hccl')

    # Step 2: Create process groups for each parallelism dimension
    initialize_model_parallel(
        classifier_free_guidance_degree=cfg_degree,
        sequence_parallel_degree=sp_degree,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        tensor_parallel_degree=tp_degree,
    )
```

Inside `initialize_model_parallel()`:
1. Create `RankGenerator` with order `"tp-sp-cfg"`
2. Initialize **CFG group** from `rank_generator.get_ranks("cfg")`
3. Initialize **SP group** with Ulysses/Ring sub-groups via `yunchang.set_seq_parallel_pg()`
4. Initialize **TP group** from `rank_generator.get_ranks("tp")`

## 5.6 CFG Parallel — Halving Forward Passes

The most impactful inference optimization. In standard CFG, each denoising step requires **2 forward passes** (conditional + unconditional):

```python
# Original: Sequential CFG
noise_pred_cond = model(x, t=t, **arg_c)[0]      # Forward 1
noise_pred_uncond = model(x, t=t, **arg_null)[0]  # Forward 2
noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

# Ascend: Parallel CFG (when cfg_world_size == 2)
if get_classifier_free_guidance_world_size() == 2:
    # Each rank computes either cond or uncond
    arg_all = {
        'context': context if get_classifier_free_guidance_rank() == 0 else context_null,
        'seq_len': seq_len
    }
    noise_pred = model(x, t=t, **arg_all, t_idx=t_idx)[0]  # Single forward
    noise_pred_cond, noise_pred_uncond = get_cfg_group().all_gather(
        noise_pred, separate_tensors=True)  # All-gather results
```

**Result:** Each rank only computes one forward pass per step, then exchanges results via all_gather. This effectively halves the compute per rank.

## 5.7 Tensor Parallel (TensorParallelApplicator)

Automatic model sharding for DiT models:

```python
from wan.distributed.tp_applicator import TensorParallelApplicator

applicator = TensorParallelApplicator(tp_size=2, device_map="cpu")
applicator.apply_to_model(transformer)
```

### Sharding Strategy

**Self-Attention:**
- Q, K, V projections → `ColumnParallelLinear` (split output dim by heads)
- O projection → `RowParallelLinear` (split input dim, all-reduce output)
- `num_heads` and `dim` divided by `tp_size`

**FFN:**
- First linear → `ColumnParallelLinear` (split hidden dim)
- Second linear → `RowParallelLinear` (split input dim, all-reduce output)

**RMSNorm → TensorParallelRMSNorm:**
- Weight split by `tp_size`
- Variance computed locally, then `all_reduce` + divide by `tp_size`

### RowParallelLinear All-Reduce Modes

```python
class RowParallelLinear(nn.Linear):
    # Three modes for matmul + all_reduce fusion:
    # 1. "torch":     F.linear() + dist.all_reduce()  (standard)
    # 2. "atb":       atb_ops.matmul_allreduce()       (ATB fused)
    # 3. "torch_npu": torch_npu.npu_mm_all_reduce_base() (NPU fused)
```

## 5.8 Optimized All-to-All Communication (`comm.py`)

```python
def all_to_all_4D(input_, scatter_idx=2, gather_idx=1, group=None):
    """
    Optimized 4D tensor all-to-all using reshape+transpose
    instead of chunk+list+cat.

    Uses dist.all_to_all_single for contiguous memory efficiency.
    """
```

Compared to the original `all_to_all()` which uses `dist.all_to_all(list, list)`, this version:
- Avoids creating intermediate Python lists
- Uses zero-copy reshape/transpose for memory layout
- Calls `dist.all_to_all_single` which is more efficient for contiguous data

## 5.9 Sequence Parallel Changes

**Original SP forward:**
```python
from .util import get_rank, get_world_size
from .ulysses import distributed_attention
# Uses Ulysses-only distributed attention
```

**Ascend SP forward:**
```python
from .parallel_mgr import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group
from ..modules.attn_layer import xFuserLongContextAttention
# Uses xFuser long-context attention (Ulysses + Ring hybrid)
```

Key changes in `sp_attn_forward()`:
- `distributed_attention()` → `xFuserLongContextAttention()()`
- Passes `rainfusion_config`, `t_idx`, `b_idx` for sparse attention
- Context parallel gather: `gather_forward()` → `get_sp_group().all_gather()`

## Pitfalls

1. **World size constraint**: `tp_size × sp_size × cfg_size` must exactly equal `world_size`.
2. **CFG parallel requires even split**: CFG degree is either 1 (disabled) or 2 (enabled). No other values.
3. **TP + FSDP interaction**: When using both TP and FSDP, TP is applied first, then FSDP wraps the TP-sharded model. The FSDP sharding is per-block (`WanAttentionBlock`).
4. **All-to-All 4D strict input**: `all_to_all_4D` requires exactly 4D input tensors. Use standard `all_to_all` for other ranks.
