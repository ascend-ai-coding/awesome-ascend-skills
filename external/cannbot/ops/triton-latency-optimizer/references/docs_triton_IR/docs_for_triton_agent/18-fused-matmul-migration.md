# Fused Matmul 迁移实例文档

> 触发条件：Agent 迁移 Fused Matmul 类算子时

---

## 核心知识：迁移 diff 分析

本文档基于 GPU 版本 `fused_matmul.py` 与 NPU 版本 `fused_matmul_npu_v3.py` 的逐行对比，提取所有关键 diff 点，为后续同类算子迁移提供可复用的代码模式。

---

## Diff 1：Autotune 配置变化（移除 num_stages/num_warps）

### 问题

GPU 的 `triton.Config` 使用 `num_stages` 控制软件流水线深度、`num_warps` 控制 warp 并行度。NPU 不使用这两个参数，它们在 NPU 上无意义甚至可能导致编译问题。

### GPU 代码

```python
def fwd_autotune_config():
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [64, 128]
        for BN in [64, 128]
        for BK in [32, 64]
        for s in [3, 4]
        for w in [4, 8]
    ]
    return configs
```

### NPU 代码

```python
def fwd_autotune_config():
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [128]
        for BN in [128]
        for BK in [128]
        for s in [3, 4]
        for w in [4, 8]
    ]
    return configs
```

### 分析

| 维度 | GPU | NPU | 说明 |
|------|-----|-----|------|
| `num_stages` | 必填，控制软件流水线深度 | **应移除**，NPU 硬件自动管理流水线 | 当前 NPU 版本仍保留，属于遗留代码 |
| `num_warps` | 必填，控制 warp 数量 | **应移除**，NPU SIMD 模式下无意义 | 当前 NPU 版本仍保留，属于遗留代码 |
| BLOCK 候选值 | 多值搜索 `[64, 128]` | 收窄为单值 `[128]` | NPU 上 128 是 Cube 单元的高效分块大小 |
| BLOCK_SIZE_K 候选值 | `[32, 64]` | `[128]` | NPU Cube 对 K=128 的矩阵乘效率更高 |

**迁移要点**：
1. 从 `triton.Config` 中移除 `num_stages` 和 `num_warps` 参数
2. BLOCK 候选值收窄，优先使用 128（Cube 单元最优分块）
3. `bwd_w_autotune_config` 中 `num_warps` 已被注释掉，说明实际迁移中应逐步移除

### bwd_w 特殊情况

GPU 版本的 `bwd_w_autotune_config` 同时包含 `num_stages` 和 `num_warps`：

```python
# GPU
for s in [3, 4]
for w in [4, 8]
```

NPU 版本注释掉了 `num_warps`：

```python
# NPU
for s in [3, 4]
# for w in [4, 8]
```

同时 BLOCK_SIZE_M 使用了非 2 的幂次值 `86`：

```python
for BM in [86, 128]
```

> **注意**：`86` 不是 2 的幂次，在 NPU 上可能存在对齐问题，建议优先使用 2 的幂次值（如 64、128）。

---

## Diff 2：Bias 融合优化（提前加载 bias + tl.broadcast_to 显式广播）

### 问题

NPU 采用 CV（Cube-Vector）分离架构。如果 bias 在 matmul 循环之后加载并使用隐式广播 `b[None, :]`，编译器无法将 bias 累加识别为 Cube 流水线的一部分，导致 bias 累加在 Vector 核上执行，产生额外的 Cube->UB 数据搬运和 CV 间同步开销。

### GPU 代码

```python
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x = tl.load(x_ptrs, mask=offset_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    w = tl.load(w_ptrs, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    accumulator = tl.dot(x, w, accumulator)
    x_ptrs += BLOCK_SIZE_K
    w_ptrs += BLOCK_SIZE_K * N
if HAS_BIAS:
    b_ptrs = b_ptr + offset_wn
    b = tl.load(b_ptrs, mask=offset_wn < N, other=0.0)
    accumulator += b[None, :]
```

### NPU 代码

```python
if HAS_BIAS:
    offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    b_ptrs = b_ptr + offset_wn
    b = tl.load(b_ptrs, mask=offset_wn < N, other=0.0)
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x_ptrs = a_ptrs_base + k * BLOCK_SIZE_K
    w_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * N
    x = tl.load(
        x_ptrs,
        mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    w = tl.load(
        w_ptrs,
        mask=msk_n[None, :] and (offs_k[:, None] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    accumulator = tl.dot(x, w, accumulator)

if HAS_BIAS:
    accumulator += tl.broadcast_to(b[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
```

### 关键改动点

| 改动 | GPU | NPU | 原因 |
|------|-----|-----|------|
| bias 加载时机 | matmul 循环之后 | **matmul 循环之前** | 使 AscendNPU-IR 在编译期识别 Cube 累加模式 |
| 广播方式 | `b[None, :]` 隐式广播 | `tl.broadcast_to(b[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))` 显式广播 | 显式广播帮助编译器正确生成 Cube 指令 |
| bias 累加位置 | Vector 核执行 | **Cube 流水线内完成** | 消除跨核数据搬运和 CV 间同步 |

### 性能收益

| 方面 | 优化前 | 优化后 |
|------|--------|--------|
| bias 累加执行位置 | Vector 核，需 Cube->UB 搬运 | Cube 流水线内，零额外搬运 |
| CV 间同步 | 需要同步等待 | 无跨核同步 |

---

## Diff 3：指针计算优化（避免 +=，用基地址+偏移量）

### 问题

GPU 版本在循环内使用 `ptrs += offset` 的方式推进指针，这种写法在 NPU 编译器中可能导致指针依赖链过长、不利于指令调度优化。NPU 版本改用"基地址+偏移量"的方式，每次循环从基地址出发计算当前指针，消除循环间的指针依赖。

### GPU 代码

```python
offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offset_k = tl.arange(0, BLOCK_SIZE_K)
x_ptrs = x_ptr + (offset_xm[:, None] * K + offset_k[None, :])
w_ptrs = w_ptr + (offset_k[:, None] * N + offset_wn[None, :])

accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x = tl.load(x_ptrs, mask=offset_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    w = tl.load(w_ptrs, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    accumulator = tl.dot(x, w, accumulator)
    x_ptrs += BLOCK_SIZE_K
    w_ptrs += BLOCK_SIZE_K * N
```

### NPU 代码

```python
offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs_base = x_ptr + (offs_am[:, None] * K + offs_k[None, :])
b_ptrs_base = w_ptr + (offs_k[:, None] * N + offs_bn[None, :])

accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x_ptrs = a_ptrs_base + k * BLOCK_SIZE_K
    w_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * N
    x = tl.load(
        x_ptrs,
        mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    w = tl.load(
        w_ptrs,
        mask=msk_n[None, :] and (offs_k[:, None] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    accumulator = tl.dot(x, w, accumulator)
```

### 关键改动点

| 改动 | GPU | NPU | 原因 |
|------|-----|-----|------|
| 指针命名 | `x_ptrs` / `w_ptrs` | `a_ptrs_base` / `b_ptrs_base`（基地址） | 语义更清晰，表明是基地址 |
| 循环内指针推进 | `x_ptrs += BLOCK_SIZE_K` | `x_ptrs = a_ptrs_base + k * BLOCK_SIZE_K` | 消除循环间指针依赖，利于编译器指令调度 |
| offset 变量命名 | `offset_xm` / `offset_wn` | `offs_am` / `offs_bn` | 纯命名风格变化 |

### 迁移模板

```python
# 1. 在循环外计算基地址
ptrs_base = base_ptr + (row_offs[:, None] * stride + col_offs[None, :])

# 2. 在循环内用基地址+偏移量计算当前指针
for k in range(0, tl.cdiv(K, BLOCK_K)):
    cur_ptrs = ptrs_base + k * BLOCK_K
    data = tl.load(cur_ptrs, mask=..., other=0.0)
```

---

## Diff 4：Mask 计算优化（移除 % 取余，用独立 mask 变量）

### 问题

GPU 版本在计算 offset 时使用 `% M` / `% N` 取余来处理边界，同时在 `tl.load` 的 mask 中使用内联表达式。NPU 版本将 mask 拆分为独立的变量，并移除了 `%` 取余操作。

### GPU 代码

```python
offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offset_k = tl.arange(0, BLOCK_SIZE_K)
x_ptrs = x_ptr + (offset_xm[:, None] * K + offset_k[None, :])
w_ptrs = w_ptr + (offset_k[:, None] * N + offset_wn[None, :])

for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x = tl.load(x_ptrs, mask=offset_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    w = tl.load(w_ptrs, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
```

### NPU 代码

```python
offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs_base = x_ptr + (offs_am[:, None] * K + offs_k[None, :])
b_ptrs_base = w_ptr + (offs_k[:, None] * N + offs_bn[None, :])
msk_m = offs_am < M
msk_n = offs_bn < N

for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x_ptrs = a_ptrs_base + k * BLOCK_SIZE_K
    w_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * N
    x = tl.load(
        x_ptrs,
        mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    w = tl.load(
        w_ptrs,
        mask=msk_n[None, :] and (offs_k[:, None] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
```

### 关键改动点

| 改动 | GPU | NPU | 原因 |
|------|-----|-----|------|
| offset 计算 | `(pid_m * BLOCK_SIZE_M + tl.arange(...)) % M` | `pid_m * BLOCK_SIZE_M + tl.arange(...)`（无取余） | 取余操作在 NPU 上产生额外计算开销；配合 early return 已保证 pid_m * BLOCK_SIZE_M < M |
| mask 拆分 | 内联 mask 表达式 | 独立 `msk_m` / `msk_n` 变量 | 减少重复计算，编译器可更好地优化 mask 复用 |
| mask 组合 | `offset_k[None, :] < K - k * BLOCK_SIZE_K` | `msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_SIZE_K)` | 行维度 mask 和 K 维度 mask 独立计算后组合 |

### 前置条件保证

NPU 版本在循环前有 early return 检查：

```python
if (pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N):
    return
```

这保证了进入循环时 `pid_m * BLOCK_SIZE_M < M`，因此 `offs_am` 不会越界，无需 `% M` 取余。

---

## Diff 5：编译参数配置

### 问题

NPU 使用一组专门的编译参数来控制 Cube-Vector 协同、多缓冲流水线和编译路径，与 GPU 的 `num_stages`/`num_warps` 完全不同。

### GPU 代码

```python
fused_matmul_fwd_kernel[grid](
    x, w, b, y,
    total_len, out_dim, in_dim,
    HAS_BIAS=has_bias,
)
```

### NPU 代码

```python
fused_matmul_fwd_kernel[grid](
    x, w, b, y,
    total_len, out_dim, in_dim,
    HAS_BIAS=has_bias,
    enable_auto_bind_sub_block=True,
    set_workspace_multibuffer=2,
    sync_solver=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    multibuffer=True,
    enable_flatten=True,
)
```

### 各 kernel 的编译参数对比

| Kernel | GPU 参数 | NPU 参数 |
|--------|---------|---------|
| `fwd_kernel` | 无 | `enable_auto_bind_sub_block=True`, `set_workspace_multibuffer=2`, `sync_solver=True`, `limit_auto_multi_buffer_of_local_buffer="no-limit"`, `multibuffer=True`, `enable_flatten=True` |
| `bwd_b_kernel` | 无 | `enable_auto_bind_sub_block=False` |
| `bwd_x_kernel` | 无 | `enable_auto_bind_sub_block=False` |
| `bwd_w_kernel` | 无 | `enable_auto_bind_sub_block=False` |

### 参数选择逻辑

```
kernel 是否包含 tl.dot()？
├── 否 → 纯 Vector 算子（bwd_b）
│   └── enable_auto_bind_sub_block=False
│
└── 是 → 是否有 Cube 后的 Vector 处理（如 bias 累加）？
    ├── 是 → CV 融合算子（fwd_kernel）
    │   └── enable_auto_bind_sub_block=True, enable_flatten=True,
    │       set_workspace_multibuffer=2, sync_solver=True,
    │       limit_auto_multi_buffer_of_local_buffer="no-limit",
    │       multibuffer=True
    │
    └── 否 → 纯 Cube 算子（bwd_x, bwd_w）
        └── enable_auto_bind_sub_block=False
```

**fwd_kernel 使用 `enable_flatten=True` 的原因**：虽然 fwd_kernel 包含 bias 累加（CV 融合），但 bias 累加通过提前加载 + `tl.broadcast_to` 已被融合到 Cube 流水线中（见 Diff 2），Vector 后处理部分（类型转换 + store）可以展平优化。

**bwd_x/bwd_w 使用 `enable_auto_bind_sub_block=False` 的原因**：这两个 kernel 虽然包含 `tl.dot`，但当前实现中不需要 Cube-Vector 协同调度。

---

## Diff 6：bwd_x / bwd_w kernel 的指针计算优化

### 6.1 bwd_x kernel

### GPU 代码

```python
offset_dym = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offset_k = tl.arange(0, BLOCK_SIZE_K)
dy_ptrs = dy_ptr + (offset_dym[:, None] * K + offset_k[None, :])
w_ptrs = w_ptr + (offset_k[:, None] + offset_wn[None, :] * K)

accumulator_dx = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    k_remaining = K - k * BLOCK_SIZE_K
    dy = tl.load(dy_ptrs, mask=offset_k[None, :] < k_remaining, other=0.0)
    w = tl.load(w_ptrs, mask=offset_k[:, None] < k_remaining, other=0.0)
    accumulator_dx = tl.dot(dy, w, accumulator_dx)
    dy_ptrs += BLOCK_SIZE_K
    w_ptrs += BLOCK_SIZE_K
```

### NPU 代码

```python
offset_dym = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
accumulator_dx = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    offset_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    dy_ptrs = dy_ptr + (offset_dym[:, None] * K + offset_k[None, :])
    dy_mask = (offset_dym[:, None] < M) & (offset_k[None, :] < K)

    w_ptrs = w_ptr + (offset_k[:, None] + offset_wn[None, :] * K)
    w_mask = (offset_k[:, None] < K) & (offset_wn[None, :] < N)

    dy = tl.load(dy_ptrs, mask=dy_mask, other=0.0)
    w = tl.load(w_ptrs, mask=w_mask, other=0.0)
    accumulator_dx = tl.dot(dy, w, accumulator_dx)
```

### bwd_x 关键改动点

| 改动 | GPU | NPU | 原因 |
|------|-----|-----|------|
| offset_k 位置 | 循环外定义 `offset_k = tl.arange(0, BLOCK_SIZE_K)` | **循环内定义** `offset_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)` | 配合基地址+偏移量模式，offset_k 包含了 k 的偏移 |
| 指针推进 | `dy_ptrs += BLOCK_SIZE_K` / `w_ptrs += BLOCK_SIZE_K` | 循环内重新计算指针 | 消除循环间指针依赖 |
| mask 计算 | `offset_k[None, :] < k_remaining` | `(offset_dym[:, None] < M) & (offset_k[None, :] < K)` | 独立 mask 变量，行维度和 K 维度分别计算 |
| 移除 % 取余 | `(pid_m * ...) % M` | `pid_m * ...`（无取余） | 同 fwd_kernel，early return 已保证不越界 |

### 6.2 bwd_w kernel

### GPU 代码

```python
offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offset_dyn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offset_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
x_ptrs = x_ptr + (offset_xm[:, None] + offset_k[None, :] * M)
dy_ptrs = dy_ptr + (offset_k[:, None] * N + offset_dyn[None, :])

accumulator_dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(K, 0, -BLOCK_SIZE_K * SPLIT_K):
    x = tl.load(x_ptrs, mask=offset_k[None, :] < k, other=0.0)
    dy = tl.load(dy_ptrs, mask=offset_k[:, None] < k, other=0.0)
    accumulator_dw = tl.dot(x, dy, accumulator_dw)
    x_ptrs += BLOCK_SIZE_K * SPLIT_K * M
    dy_ptrs += BLOCK_SIZE_K * SPLIT_K * N
```

### NPU 代码

```python
offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
offset_dyn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
accumulator_dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

for pid_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    offset_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offset_xm[:, None] + offset_k[None, :] * M)
    dy_ptrs = dy_ptr + (offset_k[:, None] * N + offset_dyn[None, :])

    x_mask = (offset_xm[:, None] < M) & (offset_k[None, :] < K)
    dy_mask = (offset_k[:, None] < K) & (offset_dyn[None, :] < N)

    x = tl.load(x_ptrs, mask=x_mask, other=0.0)
    dy = tl.load(dy_ptrs, mask=dy_mask, other=0.0)
    accumulator_dw = tl.dot(x, dy, accumulator_dw)
```

### bwd_w 关键改动点

| 改动 | GPU | NPU | 原因 |
|------|-----|-----|------|
| 循环方向 | `range(K, 0, -BLOCK_SIZE_K * SPLIT_K)` 逆序 | `range(0, tl.cdiv(K, BLOCK_SIZE_K))` 正序 | NPU 版本简化循环，不再使用逆序步进 |
| SPLIT_K 步进 | 循环步长为 `BLOCK_SIZE_K * SPLIT_K` | 循环步长为 `BLOCK_SIZE_K`（无 SPLIT_K 步进） | SPLIT_K 的并行化改为通过 grid 的第二维度实现，不再在循环内步进 |
| 指针推进 | `x_ptrs += BLOCK_SIZE_K * SPLIT_K * M` | 循环内重新计算指针 | 消除循环间指针依赖 |
| 移除 % 取余 | `% M` / `% N` | 无取余 | early return 保证不越界 |
| mask 拆分 | 内联 `offset_k[None, :] < k` | 独立 `x_mask` / `dy_mask` 变量 | 减少重复计算 |

---

## Diff 7：SPLIT_K 相关变化

### 问题

GPU 版本的 `bwd_w_kernel` 使用 SPLIT_K 实现沿 K 维度的并行归约，通过 `pid_k = tl.program_id(axis=1)` 获取第二个 grid 维度的 program ID，并在循环内使用 `BLOCK_SIZE_K * SPLIT_K` 作为步进步长。NPU 版本对 SPLIT_K 的实现方式进行了简化。

### GPU 代码

```python
# kernel 参数包含 pid_k
pid_k = tl.program_id(axis=1)

# 循环使用逆序 + SPLIT_K 步进
for k in range(K, 0, -BLOCK_SIZE_K * SPLIT_K):
    x = tl.load(x_ptrs, mask=offset_k[None, :] < k, other=0.0)
    dy = tl.load(dy_ptrs, mask=offset_k[:, None] < k, other=0.0)
    accumulator_dw = tl.dot(x, dy, accumulator_dw)
    x_ptrs += BLOCK_SIZE_K * SPLIT_K * M
    dy_ptrs += BLOCK_SIZE_K * SPLIT_K * N

# SPLIT_K > 1 时使用 atomic_store
if SPLIT_K == 1:
    tl.store(dw_ptrs, dw, mask=dw_mask)
else:
    atomic_store(dw_ptrs, dw, dw_mask, LOCK_W, SPLIT_K)
```

```python
# grid 包含 SPLIT_K 维度
grid = lambda META: (
    triton.cdiv(in_dim, META["BLOCK_SIZE_M"])
    * triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),
    META["SPLIT_K"],
)
```

### NPU 代码

```python
# 无 pid_k，循环使用正序
for pid_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    offset_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offset_xm[:, None] + offset_k[None, :] * M)
    dy_ptrs = dy_ptr + (offset_k[:, None] * N + offset_dyn[None, :])
    x_mask = (offset_xm[:, None] < M) & (offset_k[None, :] < K)
    dy_mask = (offset_k[:, None] < K) & (offset_dyn[None, :] < N)
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)
    dy = tl.load(dy_ptrs, mask=dy_mask, other=0.0)
    accumulator_dw = tl.dot(x, dy, accumulator_dw)

# 直接 store，无 atomic_store 分支
tl.store(dw_ptrs, dw, mask=dw_mask)
```

```python
# grid 不包含 SPLIT_K 维度
grid = lambda META: (
    triton.cdiv(in_dim, META["BLOCK_SIZE_M"])
    * triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),
)
```

### 关键改动点

| 改动 | GPU | NPU | 原因 |
|------|-----|-----|------|
| `pid_k` 来源 | `tl.program_id(axis=1)` | 移除，循环变量 `pid_k` 代替 | NPU 版本将 K 维度归约放在单 program 内循环完成 |
| 循环方向 | 逆序 `range(K, 0, -BLOCK_SIZE_K * SPLIT_K)` | 正序 `range(0, tl.cdiv(K, BLOCK_SIZE_K))` | 简化循环逻辑 |
| SPLIT_K 步进 | 循环步长为 `BLOCK_SIZE_K * SPLIT_K` | 循环步长为 `BLOCK_SIZE_K` | 不再在循环内跳步 |
| 结果写入 | `atomic_store`（SPLIT_K > 1） | 直接 `tl.store` | NPU 版本将 K 归约放在单 program 内，无需原子操作 |
| grid 维度 | 2D grid `(num_blocks, SPLIT_K)` | 1D grid `(num_blocks,)` | SPLIT_K 不再作为 grid 维度 |
| early return 条件 | 包含 `pid_k * BLOCK_SIZE_K >= K` 检查 | 移除 `pid_k` 相关检查 | 不再使用 `program_id(axis=1)` |

> **注意**：NPU 版本当前移除了 SPLIT_K 的并行归约机制，将 K 维度归约完全放在单 program 内循环完成。这意味着 SPLIT_K 参数虽然在 autotune config 中仍存在，但实际 kernel 逻辑中不再使用它来控制并行度。如果 K 维度很大导致性能问题，可能需要重新引入 SPLIT_K 并行归约。

---

## Diff 8：设备 API 变化（torch.cuda -> torch.npu）

### GPU 代码

```python
import torch
# 无需额外导入

x = torch.randn((256, 512), dtype=dtype, device="cuda").requires_grad_()
w = torch.randn((512, 1024), dtype=dtype, device="cuda").requires_grad_()
b = torch.randn((1024), dtype=dtype, device="cuda").requires_grad_()
dy = torch.randn((256, 1024), dtype=dtype, device="cuda")
```

### NPU 代码

```python
import torch
import torch_npu

x = torch.randn((256, 512), dtype=dtype, device="npu").requires_grad_()
w = torch.randn((512, 1024), dtype=dtype, device="npu").requires_grad_()
b = torch.randn((1024), dtype=dtype, device="npu").requires_grad_()
dy = torch.randn((256, 1024), dtype=dtype, device="npu")
```

### 关键改动点

| 改动 | GPU | NPU |
|------|-----|-----|
| 导入 | `import torch` | `import torch` + `import torch_npu` |
| 设备字符串 | `"cuda"` | `"npu"` |
| 同步 API | `torch.cuda.synchronize()` | `torch_npu.npu.synchronize()` |
| Profiler | `torch.cuda.profiler.*` | `torch_npu.profiler.*` |

### NPU 特有的 Profiler 代码

NPU 版本新增了性能 profiling 代码：

```python
experimental_config = torch_npu.profiler._ExperimentalConfig(
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
)
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    with_stack=False,
    record_shapes=False,
    profile_memory=False,
    schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=30, repeat=10, skip_first=1),
    experimental_config=experimental_config,
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
) as prof:
    for i in range(30):
        y = FusedMatmul.apply(x, w, b)
        y.backward(dy)
        torch_npu.npu.synchronize()
        prof.step()
    prof.stop()
```

---

## Diff 9：装饰器变化（maybe_triton_jit -> triton.autotune）

### GPU 代码

```python
@maybe_triton_jit(
    configs=fwd_autotune_config(),
    key=["N", "K"],
)
@triton.jit
def fused_matmul_fwd_kernel(...):
```

### NPU 代码

```python
@triton.autotune(
    configs=fwd_autotune_config(),
    key=["N", "K"],
)
@triton.jit
def fused_matmul_fwd_kernel(...):
```

### 关键改动点

| 改动 | GPU | NPU | 原因 |
|------|-----|-----|------|
| 装饰器 | `@maybe_triton_jit` | `@triton.autotune` | NPU 使用标准 `triton.autotune`，不需要 `maybe_triton_jit` 兼容层 |
| 导入 | `from maybe_triton_jit import maybe_triton_jit` | 移除 | 不再需要 |

---

## Diff 10：NPU 版本新增的精度评估工具

NPU 版本新增了 `benchmark_compare_close` 函数和 `eval_standard` 字典，用于更严格的精度评估：

```python
eval_standard = {
    torch.float32: {"rtol": 1e-6, "small_value": 1e-6, "small_value_atol": 1e-9, "etol": 1e-4},
    torch.float16: {"rtol": 1e-3, "small_value": 1e-3, "small_value_atol": 1e-5, "etol": 1e-3},
    torch.bfloat16: {"rtol": 4e-3, "small_value": 1e-3, "small_value_atol": 1e-5, "etol": 1e-3},
}
```

该函数评估四个维度：
1. 相对误差最大值（actual vs standard 的 10 倍限制）
2. 相对误差均值（actual vs standard 的 2 倍限制）
3. 小值域 error 占比（actual vs standard 的 2 倍限制）
4. 均方根误差（actual vs standard 的 2 倍限制）

---

## 910_95 特别注意

| 特性 | 说明 |
|------|------|
| multibuffer 需显式开启 | 910_95 默认 `False`，必须显式设置 `multibuffer=True` |
| UB 空间更大 | 256KB（A2/A3 为 192KB），允许更大的 BLOCK_SIZE |
| FixPipe 直通 UB | 支持 L0C -> UB 直通，详见 [11-fixpipe-and-bias-fusion.md](11-fixpipe-and-bias-fusion.md) |
| BLOCK_SIZE 对齐 | Tiling 值须满足 32B 对齐，使用 2 的幂次值天然满足 |

> 完整 910_95 硬件规格见 [00-hardware-quick-ref.md](00-hardware-quick-ref.md)。

---

## 迁移检查清单

迁移 Fused Matmul 类算子时，按以下清单逐项检查：

| 序号 | 检查项 | 状态 |
|------|--------|------|
| 1 | Autotune Config 移除 `num_stages` 和 `num_warps` | |
| 2 | BLOCK 候选值收窄为 NPU 高效值（优先 128） | |
| 3 | Bias 提前加载到 matmul 循环之前 | |
| 4 | Bias 广播改为 `tl.broadcast_to(b[None, :], (BLOCK_M, BLOCK_N))` | |
| 5 | 指针推进改为基地址+偏移量模式，移除 `+=` | |
| 6 | Offset 计算移除 `%` 取余 | |
| 7 | Mask 拆分为独立变量 `msk_m` / `msk_n` | |
| 8 | fwd_kernel 添加编译参数（CV 融合参数集） | |
| 9 | bwd_kernel 添加 `enable_auto_bind_sub_block=False` | |
| 10 | bwd_x/bwd_w 指针计算改为循环内重新计算 | |
| 11 | bwd_w SPLIT_K 逻辑简化（移除逆序循环和 atomic_store） | |
| 12 | 设备 API 从 `torch.cuda` 改为 `torch.npu` / `torch_npu` | |
| 13 | 装饰器从 `maybe_triton_jit` 改为 `triton.autotune` | |
| 14 | 910_95 上显式设置 `multibuffer=True` | |
| 15 | BLOCK_SIZE 使用 2 的幂次值，避免非对齐值 | |

---

## 相关文档链接

- `fused_matmul.py` (GPU 版本) - GPU 原始实现
- `fused_matmul_npu_v3.py` (NPU 版本) - NPU 迁移后实现
- [07-compile-params.md](../docs_for_triton_agent/07-compile-params.md) - NPU 编译参数速查
- [10-autotune-on-npu.md](../docs_for_triton_agent/10-autotune-on-npu.md) - NPU Autotune 配置指南
- [11-fixpipe-and-bias-fusion.md](../docs_for_triton_agent/11-fixpipe-and-bias-fusion.md) - FixPipe 随路操作与 Bias 融合优化
- [04-memory-access-patterns.md](../docs_for_triton_agent/04-memory-access-patterns.md) - 内存访问模式优化
- [00-hardware-quick-ref.md](../docs_for_triton_agent/00-hardware-quick-ref.md) - 910_95 硬件规格速查
