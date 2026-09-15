# Device-side Gather 连续化

## 问题描述

**问题：** 一个 Triton kernel 在循环内部反复按动态/离散索引从 GM gather 输入数据，每次迭代都执行一次随机访存：

```python
for blk_start in range(0, N, BLOCK_K):
    idx = tl.load(idx_ptr + idx_base + blk_offs)          # 动态索引
    x_tile = tl.load(
        src_ptr + src_base + idx_clamped[:, None] * D + d_offs[None, :],
        mask=idx_valid[:, None] & d_valid[None, :], other=0.0)
    ... = f(x_tile)                                         # 基于 x_tile 的计算
```

这类代码的典型问题：
- `idx` 来自运行时加载，编译器无法静态分析访存模式，**离散 load** 难以被 MTE2 合并或 L2 预取。
- 为了在 UB 内同时容纳 gathered tile、累加器、中间结果，通常被迫使用较小的 `BLOCK_D/BLOCK_DV`，无法使用 full-feature tile。
- 当循环迭代次数多（如 `topK / BLOCK_K` 较大）时，重复 gather 成为主要瓶颈。

## 优化方案

**原理：** 把“循环内每次迭代离散 gather”改为“device 端一个独立 kernel 预先 gather”。把按索引收集到的数据写到连续的 GM workspace，后续计算 kernel 从连续 buffer 读取，从而把随机访存转成连续访存，并允许使用更大的计算 tile。

### 关键判据：为什么不直接在原 kernel 里 fused gather+compute

Fused 方案需要同时保留：
1. gathered 输入 tile（`BLOCK_K × D`）
2. 计算输出累加器（`BLOCK_G × D`）
3. 中间结果 tile（score、P、partial sum 等）

在 Ascend 有限的 UB（如 192KB）下，三者叠加经常溢出。与其勉强 fused 并使用极小的 tile，不如**拆成 gather kernel + compute kernel**，用 GM workspace 桥接，让两个 kernel 各自按自己的 UB 约束取最优 tile。

### Step 1：Device-side gather kernel（通用模板）

```python
@triton.jit
def _gather_kernel(
    src_ptr, idx_ptr, gathered_ptr, valid_ptr,
    M, N,
    BATCH: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_valid = n_offs < N

    # 读取索引并 clamp
    idx = tl.load(idx_ptr + pid_b * N + n_offs, mask=n_valid, other=-1)
    idx_valid = n_valid & (idx != -1) & (idx < M)
    idx_clamped = tl.where(idx_valid, idx, 0)

    # gather src[BATCH, M, D] -> gathered[BATCH, N, D]
    d_offs = tl.arange(0, BLOCK_D)
    d_valid = d_offs < D
    src_base = pid_b * M * D
    x_tile = tl.load(
        src_ptr + src_base + idx_clamped[:, None] * D + d_offs[None, :],
        mask=idx_valid[:, None] & d_valid[None, :], other=0.0,
        care_padding=False)
    gathered_base = pid_b * N * D
    tl.store(
        gathered_ptr + gathered_base + n_offs[:, None] * D + d_offs[None, :],
        x_tile, mask=n_valid[:, None] & d_valid[None, :])

    # 可选：记录 validity mask，供下游 kernel 使用
    tl.store(valid_ptr + pid_b * N + n_offs,
             idx_valid.to(valid_ptr.dtype.element_ty), mask=n_valid)
```

若有多个源张量需要按同一份索引 gather（如 SFA 中的 `K` 和 `KR`），可在同一个 kernel 内增加额外的 `src2_ptr / gathered2_ptr`，或启动多个 gather kernel，视 UB 与并行度而定。

### Step 2：从连续 buffer 读取的计算 kernel

```python
@triton.autotune(configs=[
    triton.Config({"BLOCK_G": 8, "BLOCK_K": 128, "BLOCK_D": 512}),
    ...
], key=["BATCH", "N", "D"])
@triton.jit
def _compute_kernel(
    q_ptr, gathered_ptr, valid_ptr, out_ptr,
    N, D,
    BATCH: tl.constexpr,
    BLOCK_G: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 与原始 kernel 类似，但 x_tile 从 gathered_ptr 连续读取
    # 因此 BLOCK_D 可以放大到 full-D，dot/reduce 调用次数显著减少
    ...
```

### Step 3：Host 侧 workspace 复用

```python
_GATHER_WS_CACHE = {}

def _get_gather_ws(batch, n, d, dtype, device):
    key = (batch, n, d, dtype, device)
    if key not in _GATHER_WS_CACHE:
        _GATHER_WS_CACHE[key] = (
            ms.mint.empty((batch, n, d), dtype=dtype),   # gathered
            ms.mint.empty((batch, n), dtype=ms.int8),      # valid mask
        )
    return _GATHER_WS_CACHE[key]
```

关键点：
- 使用 `ms.mint.empty` 而非 `ms.mint.zeros`，避免首次调用时大规模 zero-fill。
- workspace 在 module 级缓存，避免每次调用 allocate/free 的内存池抖动。
- workspace 是内部量，不进算子接口返回值。

## 案例：Sparse Flash Attention Forward（SFA）

SFA 是本优化点的典型应用：每个 `(b, s1)` 按 `sparse_idx` 从巨大的 KV cache 中 gather 出 `topK` 行，然后做 attention 计算。

### 原始实现

```python
@triton.jit
def _sfa_kernel(...):
    for blk_start in range(0, topK, BLOCK_K):
        tok = tl.load(sparse_ptr + sp_base + blk_offs)
        k_tile = tl.load(k_ptr + k_base + tok_clamped[:, None] * D + ...)
        kr_tile = tl.load(kr_ptr + kr_base + tok_clamped[:, None] * D_ROPE + ...)
        scores = tl.dot(q_tile, tl.trans(k_tile))
        scores += tl.dot(qr_tile, tl.trans(kr_tile))
        ...
```

### Device Gather + Gathered Attention

```python
def _sfa_core(...):
    if topK <= 128:
        # 小 topK：单 block 即可覆盖，走原 kernel 避免 workspace 开销
        _sfa_kernel[grid](...)
    else:
        gk, gkr, gvalid = _get_gather_ws(B * S1, topK, D, D_ROPE, q_bsnd.dtype, device)
        _sfa_gather_kernel[gather_grid](
            k_bsnd, kr_bsnd, sparse_idx, gvalid,
            gk, gkr, ...
        )
        _sfa_kernel_gathered[grid](
            q_bsnd, qr_bsnd, gk, gkr, gvalid,
            out, sm_max, sm_sum, ...
        )
```

### 性能对比（SFA Forward）

| 指标 | 原始 inline gather | Device gather + workspace cache |
|------|--------------------|---------------------------------|
| attention kernel 时间 | `~11.7ms` | `~3.8ms` |
| gather kernel 时间 | — | `~2.7ms` |
| kernel 总时间 | `~11.7ms` | `~6.5ms` |
| 端到端（首次 / 缓存后） | `24.53ms` | `69ms` / `21.75ms` |
| workspace 申请开销 | 无 | 首次 1.2GB allocate+zero；缓存后消除 |
| 精度 | 通过 | 通过（over=0.00%） |

**收益：**
- attention 本身提速 `~3x`
- 端到端在 workspace 缓存后提速 `~11%`

## 优势分析

### 1. 把随机 gather 转成连续 load

```python
# 原始：每次 chunk 按动态索引跳读
x_tile = tl.load(src_ptr + idx_clamped[:, None] * D + ...)

# 优化：compute kernel 从连续 buffer 读
x_offs = blk_start + tl.arange(0, BLOCK_K)
x_tile = tl.load(gathered_ptr + x_offs[:, None] * D + d_offs[None, :])
```

连续 load 更容易命中 L2，且可触发更大的 burst。

### 2. 允许使用更大的计算 tile

原始 kernel 因 UB 放不下 gathered tile + 输出累加器 + 中间 tile，只能使用小 `BLOCK_D`。Device gather 后，compute kernel 不需要在 UB 里同时保留 gather 源和完整累加器，可以把 `BLOCK_D/BLOCK_DV` 放大到 full-feature 尺寸，显著减少 `tl.dot` / `tl.reduce` 调用次数。

### 3. Gather 与 Compute 可独立调优

```python
# gather kernel：关注 gather 并行度，BLOCK_N 取 64，BLOCK_D 取 128
# compute kernel：关注 dot/reduce 效率，BLOCK_K 取 128，BLOCK_D 取 512
```

两个 kernel 的 UB 约束独立，不再被迫互相妥协。

### 4. Workspace cache 消除内存池抖动

首次调用如果使用 `ms.mint.zeros` + `empty_cache()`，大 workspace 的 allocate/zero/free 会占主导。Module 级 cache 把该开销平摊到多次调用上。

## 关键技术障碍与绕过

### 障碍 1：Fused gather+compute UB 溢出

```python
# 尝试：在 compute kernel 内把每个 chunk 的输入 gather 到 UB，再 full-D 计算
@triton.autotune(configs=[
    triton.Config({"BLOCK_G": 8, "BLOCK_K": 64, "BLOCK_D": 512}),
    ...
])
def _fused_kernel(...):
    for blk_start in range(0, N, BLOCK_K):
        x_tile = tl.load(..., care_padding=False)   # [BLOCK_K, BLOCK_D]
        ... = f(x_tile)
        acc += g(x_tile)
```

**结果：** `BLOCK_K=64` 时 UB 需要数 MB > 192KB，编译失败；`BLOCK_K=32` 能编译但 chunk 数翻倍、性能更差。

**绕过：** 必须拆成两个 kernel，用 GM workspace 桥接。

### 障碍 2：Workspace 首次分配/清零开销巨大

```python
# ❌ 首次使用 zeros + empty_cache
gathered = ms.mint.zeros((BATCH, N, D), dtype=src.dtype)
# 端到端可能退化数倍

# ✅ 使用 empty + module 级 cache
_GATHER_WS_CACHE = {}

def _get_gather_ws(...):
    if key not in _GATHER_WS_CACHE:
        _GATHER_WS_CACHE[key] = ms.mint.empty(...)
    return _GATHER_WS_CACHE[key]
```

### 障碍 3：Gather kernel 自身的 tile 选择

```python
# ❌ BLOCK_N=128, BLOCK_D=512
# gather 一个 tile 就是 128×512×2 = 128KB UB，加上索引 tile 后溢出

# ✅ BLOCK_N=64, BLOCK_D=128
# 分多轮覆盖 D=512，每个 tile 64KB，UB 安全
```

### 障碍 4：Validity mask 的 dtype 与存储

Gather kernel 需要把 `idx_valid` 存下来，供 compute kernel 判断哪些 gathered 行有效。使用 `int8` 或 `bool` 类型，占用小；compute kernel 中读回后转成 bool 使用。

## 适用条件

| 条件 | 说明 |
|------|------|
| ✅ 适用 | Kernel 内部存在按随机/离散索引的 gather，且该 gather 在循环中被多次重复执行 |
| ✅ 适用 | 离散 gather 导致无法使用大 tile，compute 部分成为 latency-bound 或 MTE2-bound |
| ✅ 适用 | Gather 数据总量可控，能放进一次性 workspace（如 `BATCH × N × D` 在 device 内存预算内） |
| ✅ 适用 | 同一输入在多次调用中 shape 相同，可用 module 级 cache 摊薄 allocate 开销 |
| ⚠️ 注意 | 通常需要拆成独立 gather kernel + compute kernel；fused 方案 UB 一般不够 |
| ⚠️ 注意 | Workspace 需 `empty` 而非 `zeros`，并做缓存，否则首次调用会退化 |
| ⚠️ 注意 | Gather kernel 的 tile 需按自身 UB 约束单独设计，不能简单复用 compute kernel 的 BLOCK_K |
| ❌ 不适用 | 索引范围很小，gather 开销低于 workspace 管理开销 |
| ❌ 不适用 | Gather 数据量过大，device 内存放不下 workspace |
| ❌ 不适用 | 输入 shape 每次调用都变，cache key 失效，缓存收益为负 |

## 常见错误

### 错误 1：尝试 host 侧 `ops.gather` 预 gather

```python
# ❌ 在 host 侧用 ops.gather 把输入 gather 好再传进 kernel
# 结果：host 端离散 gather + 数据传输主导，端到端严重退化

# ✅ device-side gather kernel
_gather_kernel[gather_grid](...)
```

### 错误 2：fused gather+compute 强行使用大 BLOCK_K

```python
# ❌ BLOCK_K=64/128，UB 溢出
# ub overflow, requires X bits while Y bits available

# ✅ 拆 kernel，gather 用 BLOCK_N=64，compute 用 BLOCK_K=128
```

### 错误 3：workspace 未缓存导致首次调用退化

```python
# ❌ 每次调用都 zeros + empty_cache
# 端到端大幅退化

# ✅ module 级 cache + ms.mint.empty
# 端到端恢复最优
```

### 错误 4：gather kernel 和 compute kernel 共用同一套 BLOCK 参数

```python
# ❌ 直接把 compute kernel 的 BLOCK_D=512 套到 gather kernel
# gather kernel 需要同时保留多个源 tile，D=512 会爆 UB

# ✅ 为 gather kernel 单独设计更小的 BLOCK_D（如 128），分多次覆盖 D
```

## 总结

| 优化 | 方法 | 收益来源 |
|------|------|---------|
| Device-side Gather 连续化 | 用独立 device kernel 把离散 gather 结果写到连续 workspace | 随机访存 → 连续访存；可使用更大计算 tile；gather 与 compute 可独立调优 |

**核心：**
- 当 compute kernel 内部存在重复离散 gather、且大 tile 被 UB 限制时，可在 device 上先 gather 到连续 workspace。
- 通常必须拆成 gather kernel + compute kernel；fused in-kernel gather 通常 UB 不够。
- Workspace 用 `ms.mint.empty` + module 级 cache，避免首次 allocate/zero/free 开销。
- Gather kernel 的 tile 按自身 UB 约束单独设计；compute kernel 因此获得使用更大 tile 的能力。
- 仅当 gather 数据量适中、workspace 可缓存时净收益为正；小范围 gather 直接走原 kernel。

---

## 来自 SKILL.md 的原始描述（优化点：Device-side Gather 连续化）

**适用条件**：算子内部存在按随机/离散索引重复 gather 输入数据，且该离散 gather 限制了后续 compute kernel 使用更大 tile，导致 kernel 处于 latency-bound 或 MTE2-bound；同时 gather 数据总量可控、可在 device 内存中物化为连续 workspace。

**典型代码特征**：
```python
# 问题代码：compute kernel 内每 chunk 按动态索引离散 gather 输入
for blk_start in range(0, N, BLOCK_K):
    idx = tl.load(idx_ptr + idx_base + blk_offs)
    x_tile = tl.load(src_ptr + src_base + idx_clamped[:, None] * D + ...)
    ... = f(x_tile)
```

**判断逻辑**：
- 检查 kernel 是否存在循环内按随机索引 `tl.load` gather 大块数据（如 `[BLOCK_K, D]` 的 tile）。
- 检查离散 gather 是否导致 `BLOCK_D/BLOCK_DV` 等关键维度无法放大到 full-feature 尺寸。
- 检查 gather 数据总量 `BATCH × N × D` 是否在 device 内存预算内。
- 检查调用方 shape 是否稳定，能否通过 module 级 cache 复用 workspace。
- 尝试 fused gather+compute：若 `BLOCK_K` 稍大即 UB 溢出，则必须拆 kernel。
- 若命中 → 新增 device gather kernel 把输入/valid_mask 写到连续 workspace，compute kernel 从 workspace 读；否则跳过。

**命中条件**：算子存在重复离散 gather，gather 后数据可连续复用，fused 方案 UB 不足，且 workspace 可被缓存或调用频次足够摊薄 allocate 开销。

**参考文档**：`references/device-side-gather.md`（本文档）

---
