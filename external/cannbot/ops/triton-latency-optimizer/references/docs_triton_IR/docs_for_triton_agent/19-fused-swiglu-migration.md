# Fused SwiGLU 迁移实例：GPU → NPU

## 触发条件

当 Agent 迁移 Fused SwiGLU 类算子（融合 SiLU 门控线性单元）到 NPU 时，参考本文档。典型场景：

- 将 GPU 版 FusedSwiGLU kernel 迁移到 Ascend 910_95
- 迁移包含前向（fwd）、反向 bias（bwd_b）、反向输入（bwd_x）、反向权重（bwd_w）四个子 kernel 的融合算子
- 遇到 SwiGLU 相关的 autotune 配置、编译参数、libdevice API 等迁移问题

---

## 核心知识：迁移 Diff 分析

### 源文件对照

| 版本 | 文件路径 |
|------|---------|
| GPU | `fused_swiglu.py` |
| NPU | `fused_swiglu_npu.py` |

### Diff 总览

迁移涉及以下关键改动类别：

| 改动类别 | GPU 版本 | NPU 版本 | 影响范围 |
|---------|---------|---------|---------|
| 导入与设备 | `import torch` + `from maybe_triton_jit import maybe_triton_jit` + `from triton.language.extra import libdevice` | `import torch` + `import torch_npu` + 注释掉 `maybe_triton_jit` 和 `libdevice`（改用 `tl.fdiv`/`tl.exp`） | 全局 |
| Autotune 装饰器 | `@maybe_triton_jit(configs=..., key=...)` | `@triton.autotune(configs=..., key=...)` | 4 个 kernel |
| Autotune 配置 | `else` 分支返回单个固定 Config | `else` 分支返回列表推导式多 Config 搜索空间 | 4 个 autotune 函数 |
| 数学函数 | `libdevice.fast_dividef` / `libdevice.fast_expf` | `tl.fdiv` / `tl.exp` | `fast_sigmoid`、`fast_silu` |
| 指针算术 | 循环外预计算 offset，循环内指针递增 | 循环内每次重新计算 offset 和 mask | fwd、bwd_x、bwd_w kernel |
| tl.sum 轴 | `tl.sum(sum_b_g, 1)`（沿 axis=1 归约） | `tl.sum(sum_b_g, 0)`（沿 axis=0 归约） | bwd_b kernel |
| bwd_w 循环方向 | `for k in range(K, 0, -BLOCK_SIZE_K * SPLIT_K)` 反向遍历 | `for k in range(0, tl.cdiv(K, BLOCK_SIZE_K))` 正向遍历 | bwd_w kernel |
| SPLIT_K 原子操作 | `if SPLIT_K == 1: tl.store(...) else: atomic_store(...)` | 直接 `tl.store(...)`，注释掉 SPLIT_K 分支 | bwd_w kernel |
| 编译参数 | 无 | `enable_auto_bind_sub_block`、`enable_flatten`、`multibuffer`、`sync_solver` 等 | 4 个 kernel 启动调用 |
| 测试代码 | `device="cuda"` + 随机数据 | `device="npu"` + `torch_npu.profiler` + 加载 GPU dump 数据对比 | `__main__` |

---

## 代码模式：GPU → NPU 代码对比

### 1. 导入与设备 API

**GPU 版本：**
```python
import torch
import triton
import triton.language as tl
from maybe_triton_jit import maybe_triton_jit
from triton.language.extra import libdevice
from .utils import is_hopper, is_ampere
```

**NPU 版本：**
```python
import torch
import torch_npu
import triton
import triton.language as tl
# from maybe_triton_jit import maybe_triton_jit
# from triton.language.extra import libdevice
from utils import is_hopper, is_ampere
```

**迁移要点：**
- 必须添加 `import torch_npu`
- `maybe_triton_jit` 被注释掉，改用标准 `@triton.autotune`
- `libdevice` 被注释掉，改用 `tl.fdiv` / `tl.exp` 替代 `libdevice.fast_dividef` / `libdevice.fast_expf`
- 导入路径从相对导入 `.utils` 改为 `utils`（NPU 版本为独立脚本）

### 2. fast_silu / fast_sigmoid 数学函数替换

**GPU 版本：**
```python
@triton.jit
def fast_sigmoid(x):
    return libdevice.fast_dividef(1.0, 1.0 + libdevice.fast_expf(-x))

@triton.jit
def fast_silu(x):
    dtype = x.type.element_ty
    x = x.to(tl.float32)
    return libdevice.fast_dividef(x, 1.0 + libdevice.fast_expf(-x)).to(dtype)
```

**NPU 版本：**
```python
@triton.jit
def fast_sigmoid(x):
    return tl.fdiv(1.0, 1.0 + tl.exp(-x))

@triton.jit
def fast_silu(x):
    dtype = x.type.element_ty
    x = x.to(tl.float32)
    return tl.fdiv(x, 1.0 + tl.exp(-x)).to(dtype)
```

**迁移要点：**
- `libdevice.fast_dividef(a, b)` → `tl.fdiv(a, b)`
- `libdevice.fast_expf(x)` → `tl.exp(x)`
- NPU 上 `tl.fdiv` 和 `tl.exp` 由 CANN libdevice 提供，路径为 `triton.language.extra.cann.libdevice`
- 但此处直接使用 `tl.fdiv` / `tl.exp`（Triton 内建函数），无需额外导入

### 3. Autotune 装饰器替换

**GPU 版本：**
```python
@maybe_triton_jit(
    configs=fwd_autotune_config(),
    key=["N", "K", "IS_TRAINING"],
)
@triton.jit
def fused_swiglu_fwd_kernel(...):
```

**NPU 版本：**
```python
@triton.autotune(
    configs=fwd_autotune_config(),
    key=["N", "K", "IS_TRAINING"],
)
@triton.jit
def fused_swiglu_fwd_kernel(...):
```

**迁移要点：**
- `@maybe_triton_jit` 是 GPU 版本特有的条件 JIT 装饰器，NPU 上替换为标准 `@triton.autotune`
- 装饰器顺序保持不变：`@triton.autotune` 在 `@triton.jit` 上方
- `key` 参数保持一致，无需修改

### 4. Autotune 配置变化（核心差异）

**GPU 版本 — `else` 分支返回单个固定 Config：**
```python
def fwd_autotune_config():
    if is_hopper():
        return [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4)]
    elif is_ampere():
        return [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4)]
    else:
        return [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4)]
```

**NPU 版本 — `else` 分支返回列表推导式多 Config 搜索空间：**
```python
def fwd_autotune_config():
    if is_hopper():
        return [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4)]
    elif is_ampere():
        return [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4)]
    else:
        return [
            triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=4,
        )
        for bm in [128, 256, 512]
        for bn in [64, 128, 256, 512]
        for bk in [32, 64, 128, 512]
    ]
```

**迁移要点：**

| 维度 | GPU | NPU |
|------|-----|-----|
| `else` 分支策略 | 单个固定 Config | 列表推导式生成搜索空间 |
| 搜索空间大小 | 1 | 3 x 4 x 4 = 48（fwd） |
| `num_stages` | 保留（GPU 有效） | 保留但无效（NPU 忽略） |
| `num_warps` | 保留（GPU 有效） | 保留但无效（NPU 忽略） |

**四个 kernel 的 NPU 搜索空间对比：**

| Kernel | 搜索维度 | 候选值 | 组合数 |
|--------|---------|--------|--------|
| fwd | BM x BN x BK | [128,256,512] x [64,128,256,512] x [32,64,128,512] | 48 |
| bwd_b | BM x BN | range(16,129,16) x range(16,65,16) | 56 |
| bwd_x | BM x BN x BK | [128,256,512] x [64,128,256,512] x [32,64,128,512] | 48 |
| bwd_w | BM x BN x BK | [128,256,512] x [64,128,256,512] x [32,64,128,512] | 48 |

> **注意**：NPU 版本当前仍保留了 `num_stages` 和 `num_warps` 参数，这是遗留写法。建议在正式迁移时移除这两个参数，因为 NPU 上它们无效。

### 5. 前向 Kernel 指针算术改动

**GPU 版本 — 循环外预计算 offset，循环内指针递增：**
```python
offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offset_k = tl.arange(0, BLOCK_SIZE_K)
x_ptrs = x_ptr + (offset_xm[:, None] * K + offset_k[None, :])
w_g_ptrs = w_g_ptr + (offset_k[:, None] * N + offset_wn[None, :])
w_fc_ptrs = w_fc_ptr + (offset_k[:, None] * N + offset_wn[None, :])
...
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x = tl.load(x_ptrs, mask=offset_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    w_g = tl.load(w_g_ptrs, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    w_fc = tl.load(w_fc_ptrs, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    accumulator_g = tl.dot(x, w_g, accumulator_g)
    accumulator_fc = tl.dot(x, w_fc, accumulator_fc)
    x_ptrs += BLOCK_SIZE_K
    w_g_ptrs += BLOCK_SIZE_K * N
    w_fc_ptrs += BLOCK_SIZE_K * N
```

**NPU 版本 — 循环内每次重新计算 offset 和 mask：**
```python
offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
...
for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    offset_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offset_xm[:, None] * K + offset_k[None, :])
    x_mask = (offset_xm[:, None] < M) & (offset_k[None, :] < K)
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)

    g_fc_offs = (offset_k[:, None] * N + offset_wn[None, :])
    g_fc_mask = (offset_k[:, None] < K) & (offset_wn[None, :] < N)

    w_g = tl.load(w_g_ptr + g_fc_offs, mask=g_fc_mask, other=0.0)
    w_fc = tl.load(w_fc_ptr + g_fc_offs, mask=g_fc_mask, other=0.0)
    accumulator_g = tl.dot(x, w_g, accumulator_g)
    accumulator_fc = tl.dot(x, w_fc, accumulator_fc)
```

**迁移要点：**
- GPU 版本使用 `% M` 取模防止越界，NPU 版本移除取模，改用完整 mask 条件
- GPU 版本使用指针递增（`x_ptrs += BLOCK_SIZE_K`），NPU 版本改为每次迭代重新计算地址
- NPU 版本的 mask 更严格：对 x 使用 `(offset_xm[:, None] < M) & (offset_k[None, :] < K)`，对 w_g/w_fc 使用 `(offset_k[:, None] < K) & (offset_wn[None, :] < N)`
- 这种改动是为了适配 NPU 编译器的地址计算方式，避免指针递增在 NPU 编译后产生不可预期的行为

### 6. 反向 Bias Kernel — tl.sum 轴变化

**GPU 版本：**
```python
sum_b_g = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
...
tl.store(db_g_ptr + col_off, tl.sum(sum_b_g, 1), mask=col_off < N)
```

**NPU 版本：**
```python
sum_b_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
...
tl.store(db_g_ptr + col_off, tl.sum(sum_b_g, 0), mask=col_off < N)
```

**迁移要点：**
- GPU 版本 accumulator 形状为 `(BLOCK_SIZE_N, BLOCK_SIZE_M)`，沿 axis=1 归约
- NPU 版本 accumulator 形状为 `(BLOCK_SIZE_M, BLOCK_SIZE_N)`，沿 axis=0 归约
- 同时指针布局也做了调整：GPU 版 `dy_ptrs = dy_ptr + (row_off[None, :] * N + col_off[:, None])`，NPU 版 `dy_ptrs = dy_ptr + (row_off[:, None] * N + col_off[None, :])`
- 这是为了适配 NPU 编译器对矩阵布局的偏好，确保内存访问连续性

### 7. 反向权重 Kernel — 循环方向与 SPLIT_K 处理

**GPU 版本 — 反向遍历 + SPLIT_K 原子操作：**
```python
for k in range(K, 0, -BLOCK_SIZE_K * SPLIT_K):
    x = tl.load(x_ptrs, mask=offset_k[None, :] < k, other=0.0)
    dg = tl.load(dg_ptrs, mask=offset_k[:, None] < k, other=0.0)
    ...
    x_ptrs += BLOCK_SIZE_K * SPLIT_K * M
    dg_ptrs += BLOCK_SIZE_K * SPLIT_K * N
    dfc_ptrs += BLOCK_SIZE_K * SPLIT_K * N
...
if SPLIT_K == 1:
    tl.store(dw_g_ptrs, dw_g, mask=dw_mask)
    tl.store(dw_fc_ptrs, dw_fc, mask=dw_mask)
else:
    atomic_store(dw_g_ptrs, dw_g, dw_mask, LOCK_G, SPLIT_K)
    atomic_store(dw_fc_ptrs, dw_fc, dw_mask, LOCK_FC, SPLIT_K)
```

**NPU 版本 — 正向遍历 + 直接 store：**
```python
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    offset_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offset_xm[:, None] + offset_k[None, :] * M)
    ...
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)
    dg = tl.load(dg_ptrs, mask=dg_fc_mask, other=0.0)
    ...
tl.store(dw_g_ptrs, dw_g, mask=dw_mask)
tl.store(dw_fc_ptrs, dw_fc, mask=dw_mask)
# if SPLIT_K == 1:
#     tl.store(dw_g_ptrs, dw_g, mask=dw_mask)
#     tl.store(dw_fc_ptrs, dw_fc, mask=dw_mask)
# else:
#     atomic_store(dw_g_ptrs, dw_g, dw_mask, LOCK_G, SPLIT_K)
#     atomic_store(dw_fc_ptrs, dw_fc, dw_mask, LOCK_FC, SPLIT_K)
```

**迁移要点：**
- GPU 版本使用反向循环 `range(K, 0, -BLOCK_SIZE_K * SPLIT_K)` 配合指针递增
- NPU 版本改为正向循环 `range(0, tl.cdiv(K, BLOCK_SIZE_K))`，每次重新计算 offset
- NPU 版本注释掉了 SPLIT_K 分支和 `atomic_store`，直接使用 `tl.store`
- 当前 NPU 版本 SPLIT_K 固定为 1，不支持多 SPLIT_K 并行归约
- 如果需要 SPLIT_K > 1，需要重新实现原子操作逻辑（NPU 的 `tl.atomic_cas` / `tl.atomic_xchg` 行为可能与 GPU 不同）

### 8. 编译参数添加（NPU 特有）

**GPU 版本 — 无编译参数：**
```python
fused_swiglu_fwd_kernel[grid](
    x, w_g, w_fc, b_g, b_fc, y, g, fc,
    total_len, out_dim, in_dim,
    IS_TRAINING=is_training and not is_recompute,
)
```

**NPU 版本 — 添加编译参数：**
```python
fused_swiglu_fwd_kernel[grid](
    x, w_g, w_fc, b_g, b_fc, y, g, fc,
    total_len, out_dim, in_dim,
    IS_TRAINING=is_training and not is_recompute,
    enable_auto_bind_sub_block=True,
    enable_flatten=False,
    set_workspace_multibuffer=2,
    sync_solver=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    multibuffer=True,
)
```

**四个 kernel 的编译参数配置：**

| Kernel | 类型 | enable_auto_bind_sub_block | enable_flatten | multibuffer | 其他参数 |
|--------|------|---------------------------|----------------|-------------|---------|
| fwd | CV 融合（含 tl.dot） | True | False | True | set_workspace_multibuffer=2, sync_solver=True, limit_auto_multi_buffer_of_local_buffer="no-limit" |
| bwd_b | 纯 Vector | False | True | True | - |
| bwd_x | CV 融合（含 tl.dot） | True | False | True | set_workspace_multibuffer=2, sync_solver=True, limit_auto_multi_buffer_of_local_buffer="no-limit" |
| bwd_w | CV 融合（含 tl.dot） | True | False | True | set_workspace_multibuffer=2, sync_solver=True, limit_auto_multi_buffer_of_local_buffer="no-limit" |

**参数选择逻辑：**
- 含 `tl.dot` 的 kernel（fwd、bwd_x、bwd_w）属于 CV 融合算子，需要 Cube-Vector 协同参数
- 纯 Vector 的 kernel（bwd_b）只需 `enable_flatten=True` 和 `multibuffer=True`
- `enable_flatten=False` 用于 CV 融合算子，因为 Cube 和 Vector 需要各自独立的循环结构

### 9. 测试代码改动

**GPU 版本：**
```python
x = torch.randn((220000, 512), dtype=dtype, device="cuda").requires_grad_()
...
y = FusedSwiglu.apply(x, w_g, w_fc, b_g, b_fc, True, False)
y.backward(dy)
...
if torch.allclose(y, ref, atol=atol, rtol=rtol):
    print("✅ [Fwd]Triton and Torch match")
```

**NPU 版本：**
```python
DEVICE = torch.device("npu")
torch.manual_seed(1024)
x = torch.load("/home/tsz/Code/dump-swiglu-200k/x.pt", map_location=torch.device('cpu')).detach().requires_grad_(True)
...
x = x.to(DEVICE).to(dtype).detach().requires_grad_(True)
...
y = FusedSwiglu.apply(x, w_g, w_fc, b_g, b_fc, True, False)
y.backward(dy)
...
# 三方对比：NPU vs GPU vs CPU FP32 参考值
third_part_cmp(y, y_gpu, ref, "y")
...
# NPU Profiler
with torch_npu.profiler.profile(...) as prof:
    for i in range(30):
        y = FusedSwiglu.apply(x, w_g, w_fc, b_g, b_fc, True, False)
        torch.npu.synchronize()
        prof.step()
...
print(f"fused_swiglu_fwd_kernel Best config: {fused_swiglu_fwd_kernel.best_config}")
```

**迁移要点：**
- `device="cuda"` → `torch.device("npu")`
- 添加 `torch.manual_seed(1024)` 确保可复现
- 使用 `torch.load` 加载 GPU dump 数据进行三方对比（NPU vs GPU vs CPU FP32）
- 添加 `torch_npu.profiler` 进行性能剖析
- 使用 `torch.npu.synchronize()` 替代 `torch.cuda.synchronize()`
- 打印 `best_config` 查看 autotune 选择的最优配置

---

## 910_95 特别注意

| 特性 | 说明 |
|------|------|
| multibuffer 默认值差异 | 910_95 默认 `False`（910B 默认 True），必须显式传入 `multibuffer=True` |
| UB 容量与 Tiling | 910_95 UB 为 256KB，允许更大 BLOCK_SIZE；910B 应限制搜索空间 |
| 搜索空间过大 | 48 个配置首次 autotune 耗时长，建议控制在 20 个以内 |
| SPLIT_K 支持 | 当前仅支持 SPLIT_K=1，如需 >1 需重新实现原子操作 |

> 完整 910_95 硬件规格见 [00-hardware-quick-ref.md](00-hardware-quick-ref.md)。

---

## 相关文档链接

- [01-migration-overview.md](../docs_for_triton_agent/01-migration-overview.md) — GPU → NPU 迁移概览
- [02-api-differences.md](../docs_for_triton_agent/02-api-differences.md) — API 差异对照
- [07-compile-params.md](../docs_for_triton_agent/07-compile-params.md) — NPU 编译参数速查
- [10-autotune-on-npu.md](../docs_for_triton_agent/10-autotune-on-npu.md) — NPU Autotune 配置指南
- [08-data-type-precision.md](../docs_for_triton_agent/08-data-type-precision.md) — 数据类型与精度处理
- `fused_swiglu.py` — GPU 版源码
- `fused_swiglu_npu.py` — NPU 版源码
