# Workspace 物化解耦优化

## 问题描述

**问题：** 一个 kernel 需要同时产出多个梯度/统计量，但它们**要求相互冲突的循环遍历顺序**——一个输出需要 `A-outer / B-inner`，另一个需要 `B-outer / A-inner`。由于 UB 容量（Ascend910 = 192KB）放不下"全 A 常驻"或"全 B 常驻"的累加器，两者无法在同一循环顺序下干净合并。被迫用**多 pass 重复遍历**，每个 pass 各自 gather 输入、重算共享中间量。

```python
# 问题代码：3-pass，每 pass 都 gather k_cat + 重算 scores/P/dS
# Pass A: dq 需 hc-outer（acc_dq 跨 blk 常驻，per-hc 36KB；blk-outer 需全 hc 144KB 装不下）
for hc in range(HC_LOOP):
    q_cat = tl.load(...); do_pad = tl.load(...); o_pad = tl.load(...)
    for blk_start in range(0, topK, BLOCK_K):
        k_cat = tl.load(kcat_ptr + tok_clamped[:, None] * D_TOT + ...)  # gather #1
        scores = tl.dot(q_cat, tl.trans(k_cat)) * scale                 # 重算
        P = tl.exp(scores - sm_max[:, None]) * inv_sm_sum[:, None]      # 重算 exp
        dPv = tl.dot(do_pad, tl.trans(k_cat))
        dS = P * (dPv - delta[:, None]) * scale                          # 重算
        acc_dqcat += tl.dot(dS.to(k_cat.dtype), k_cat)

# Pass B1: dk 需 blk-outer（k_cat 跨 hc 共享，atomic 最少）
for blk_start in range(0, topK, BLOCK_K):
    k_cat = tl.load(kcat_ptr + tok_clamped[:, None] * D_TOT + ...)      # gather #2（重复！）
    for hc in range(HC_LOOP):
        q_cat = tl.load(...); do_pad = tl.load(...); o_pad = tl.load(...)
        scores = tl.dot(q_cat, tl.trans(k_cat)) * scale                 # 重算（重复！）
        P = tl.exp(scores - sm_max[:, None]) * inv_sm_sum[:, None]      # 重算 exp（重复！）
        dPv = tl.dot(do_pad, tl.trans(k_cat))
        dS = P * (dPv - delta[:, None]) * scale                          # 重算（重复！）
        dkcat_acc += tl.dot(tl.trans(dS).to(q_cat.dtype), q_cat)
    tl.atomic_add(dkcat_ptr + ..., dkcat_acc, mask=tok_valid[:, None])

# Pass B2: dv 同样 blk-outer，又一遍 gather + 重算 scores/P
for blk_start in range(0, topK, BLOCK_K):
    k_cat = tl.load(kcat_ptr + ...)                                     # gather #3（重复！）
    for hc in range(HC_LOOP):
        ...scores/P 重算...
        dv_acc += tl.dot(tl.trans(P).to(do_tile.dtype), do_tile)
    tl.atomic_add(dv_ptr + ..., dv_acc, mask=tok_valid[:, None])
```

**开销分析（以 SFA Grad, topK=2048, BLOCK_K=32, HC=4 为例）：**
- gather `k_cat` 次数：3 pass × (topK/BLOCK_K) = 3 × 64 = 192 次/program（离散 gather，最贵）
- `tl.dot` 调用：Pass A 3 次/blk × 64 + Pass B1 4 次/blk × 64 + Pass B2 2 次/blk × 64 = 576 次/program
- `tl.exp` 调用：3 pass × 64 = 192 次/program
- `o_pad` load：每 hc 4× 重复加载（仅用于算 delta）
- profiling 特征：**issue-bound**，cube 利用率 0.14%（闲置），scalar 47% 且 97% 时间在等数据

## 优化方案

**原理：** 冲突循环顺序的根因是"共享中间量（dS/P）被两个输出以不同顺序消费"。把共享中间量在 Pass A（按 A-outer 顺序）**物化到 GM workspace**，Pass B（按 B-outer 顺序）从 workspace **读回**，即可在 B-outer 顺序下直接算出第二个输出，**无需重新 gather k_cat、无需重算 scores/P/dS**。循环顺序冲突被 workspace 解耦。

### 关键判据：为什么不能直接合并 pass

合并多 pass 的前提是"同一循环顺序下 UB 能容纳所有累加器"。当满足以下任一条件时，合并不可行，应转用 workspace 解耦：

1. **累加器跨循环维度常驻，UB 装不下**：如 `acc_dq` 需跨 `blk` 维常驻（per-hc 36KB），若改成 blk-outer 则需全 hc 144KB > 192KB。
2. **atomic 太贵，不能代替常驻累加器**：把 blk-outer 改成 hc-outer 会让 dk/dv 的 scatter-add 变成每 blk 一次 atomic（atomic 数爆炸），实测远比重算贵。
3. **两输出循环顺序 genuinely 冲突**：A 输出需 `X-outer/Y-inner`，B 输出需 `Y-outer/X-inner`，无单一顺序能同时满足 1、2。

### Pass A：物化中间量到 GM workspace

```python
# Pass A（hc-outer，不变）：算 scores/P/dPv/dS + 累积 dq，额外存 dS/P 到 workspace
for hc in range(HC_LOOP):
    q_cat = tl.load(...); do_pad = tl.load(...)
    delta = tl.load(delta_ptr + ...)              # delta 也可 host 预计算（见优势 4）
    acc_dqcat = tl.zeros([BLOCK_G, D_TOT], dtype=tl.float32)
    for blk_start in range(0, topK, BLOCK_K_A):
        k_cat = tl.load(kcat_ptr + tok_clamped[:, None] * D_TOT + ...)  # gather（仅此 pass）
        scores = tl.dot(q_cat, tl.trans(k_cat)) * scale
        P = tl.exp(scores - sm_max[:, None]) * inv_sm_sum[:, None]
        dPv = tl.dot(do_pad, tl.trans(k_cat))
        dS = P * (dPv - delta[:, None]) * scale
        acc_dqcat += tl.dot(dS.to(k_cat.dtype), k_cat)

        # ★ 物化 dS/P 到 GM workspace（bf16，精度与下方 downcast dot 等价）
        ws_offs = ws_base + g_offs_s[:, None] * topK + blk_offs[None, :]
        tl.store(ds_ptr + ws_offs, dS.to(ds_ptr.dtype.element_ty), mask=ws_mask)
        tl.store(p_ptr  + ws_offs, P.to(p_ptr.dtype.element_ty),   mask=ws_mask)
    tl.store(dqcat_ptr + ..., acc_dqcat.to(...), mask=...)
```

### Pass B：从 workspace 读回，无 gather 无重算

```python
# Pass B1（blk-outer）：dk = dS^T · q_cat，从 workspace 读 dS
for blk_start in range(0, topK, BLOCK_K_B):
    tok = tl.load(sparse_ptr + sp_base + blk_offs)               # 仅 token 索引，无 k_cat gather
    dkcat_acc = tl.zeros([BLOCK_K_B, D_TOT], dtype=tl.float32)
    for hc in range(HC_LOOP):
        q_cat = tl.load(qcat_ptr + ...)
        dS = tl.load(ds_ptr + ws_offs, mask=ws_mask, other=0.0)  # 读回，无重算
        dkcat_acc += tl.dot(tl.trans(dS).to(q_cat.dtype), q_cat).to(tl.float32)
    tl.atomic_add(dkcat_ptr + ..., dkcat_acc, mask=tok_valid[:, None])

# Pass B2（blk-outer）：dv = P^T · do，从 workspace 读 P
for blk_start in range(0, topK, BLOCK_K_B):
    tok = tl.load(sparse_ptr + sp_base + blk_offs)
    dv_acc = tl.zeros([BLOCK_K_B, D], dtype=tl.float32)
    for hc in range(HC_LOOP):
        do_tile = tl.load(do_ptr + ...)
        P = tl.load(p_ptr + ws_offs, mask=ws_mask, other=0.0)    # 读回，无重算
        dv_acc += tl.dot(tl.trans(P).to(do_tile.dtype), do_tile).to(tl.float32)
    tl.atomic_add(dv_ptr + ..., dv_acc, mask=tok_valid[:, None])
```

## 案例：Sparse Flash Attention Grad（SFA Backward）

### 循环顺序冲突分析

SFA backward 需输出 dq/dk/dv（+ dqr/dkr）。核心冲突：

| 输出 | 理想循环顺序 | 原因 |
|------|-------------|------|
| dq | **hc-outer / blk-inner** | `acc_dq` 需跨 blk 常驻（per-hc 36KB）；blk-outer 需全 hc 144KB > UB |
| dk, dv | **blk-outer / hc-inner** | `k_cat` 跨 hc 共享（gather 一次复用），scatter-add 的 atomic 数最少 |

hc-outer 与 blk-outer **不可同时满足** → 无法干净合并 → 原始实现被迫 3-pass 重复 gather + 重算。

### 原始实现（3-pass 重算）

```python
@triton.jit
def _sfa_grad_kernel(..., o_ptr, ...):    # o_ptr 用于算 delta
    # Pass A: dq, hc-outer —— gather k_cat + 算 scores/P/dS
    for hc in range(HC_LOOP):
        q_cat = tl.load(...); do_pad = tl.load(...); o_pad = tl.load(o_ptr + ...)
        delta = tl.sum(do_pad.to(tl.float32) * o_pad.to(tl.float32), axis=1)  # 每 hc 重算 delta
        for blk_start in range(0, topK, BLOCK_K):
            k_cat = tl.load(kcat_ptr + tok_clamped[:, None] * D_TOT + ...)
            scores = tl.dot(q_cat, tl.trans(k_cat)).to(tl.float32) * scale_value
            P = tl.exp(scores - sm_max[:, None]) * inv_sm_sum[:, None]
            dPv = tl.dot(do_pad, tl.trans(k_cat)).to(tl.float32)
            dS = P * (dPv - delta[:, None]) * scale_value
            acc_dqcat += tl.dot(dS.to(k_cat.dtype), k_cat).to(tl.float32)
    # Pass B1: dk, blk-outer —— 又一遍 gather + 重算 scores/P/dS
    # Pass B2: dv, blk-outer —— 第三遍 gather + 重算 scores/P
    ...
```

### Workspace 解耦后

```python
@triton.jit
def _sfa_grad_kernel(
    ..., delta_ptr,                  # delta=rowsum(dO*O)，host 预计算，移除 o_pad
    ds_ptr, p_ptr,                   # workspace (bf16): dS/P [B*S1,N1,topK]
    ...,
    BLOCK_K_A: tl.constexpr, BLOCK_K_B: tl.constexpr,   # A/B 各自独立 tile
):
    ws_base = pid_bs1 * N1 * topK    # 本 program 的 workspace 行
    # Pass A: hc-outer，物化 dS/P
    for hc in range(HC_LOOP):
        q_cat = tl.load(...); do_pad = tl.load(...)
        delta = tl.load(delta_ptr + pid_bs1 * N1 + g_offs_s, ...)   # 读预计算 delta
        for blk_start in range(0, topK, BLOCK_K_A):
            k_cat = tl.load(kcat_ptr + tok_clamped[:, None] * D_TOT + ...)
            scores = tl.dot(q_cat, tl.trans(k_cat)).to(tl.float32) * scale_value
            P = tl.exp(scores - sm_max[:, None]) * inv_sm_sum[:, None]
            dPv = tl.dot(do_pad, tl.trans(k_cat)).to(tl.float32)
            dS = P * (dPv - delta[:, None]) * scale_value
            acc_dqcat += tl.dot(dS.to(k_cat.dtype), k_cat).to(tl.float32)
            ws_offs = ws_base + g_offs_s[:, None] * topK + blk_offs[None, :]
            tl.store(ds_ptr + ws_offs, dS.to(ds_ptr.dtype.element_ty), mask=ws_mask)
            tl.store(p_ptr  + ws_offs, P.to(p_ptr.dtype.element_ty),   mask=ws_mask)
        tl.store(dqcat_ptr + ..., acc_dqcat.to(...), mask=...)
    # Pass B1: blk-outer，读 dS 算 dk（无 gather、无重算）
    # Pass B2: blk-outer，读 P 算 dv（无 gather、无重算）
    ...
```

### 性能对比

| 指标 | 原始（3-pass 重算） | 优化（workspace 解耦） | 收益 |
|------|---------------------|------------------------|------|
| `k_cat` gather 次数/program | 192（3 pass × 64） | 64（仅 Pass A） | **减至 1/3** |
| `tl.dot` 调用/program | 576 | 320（Pass A 192 + B1 64 + B2 64） | **减至 5/8** |
| `tl.exp` 调用/program | 192 | 64（仅 Pass A） | **减至 1/3** |
| `o_pad` load | 每 hc 4× 重复 | 0（delta host 预计算） | 消除 |
| UB stall | 1002us/block（o_pad 挤占 UB） | 0 | 消除 |
| Task Duration | 109.5ms | 62.5ms | **-42.9%** |

**实测性能（Ascend910, shape `(1,512,4096,64,2048)`，bf16）：**
- 原始 3-pass 重算：109.5ms
- workspace 解耦：62.5ms
- **加速比：1.75x**，精度两 shape 全过（over=0.00%）

## 优势分析

### 1. 消除重复 gather（最大收益）

```python
# 原始：3 pass 各自按 tok 索引 gather k_cat（离散访存，最贵）
for blk_start: k_cat = tl.load(kcat_ptr + tok_clamped[:, None] * D_TOT + ...)  # ×3

# 优化：仅 Pass A gather 一次，B1/B2 从连续 workspace 读
for blk_start: dS = tl.load(ds_ptr + ws_offs, ...)   # 连续读，L2 命中
```

离散 gather（按 `tok` 索引跳读）远比连续 load 贵。workspace 写入虽是 strided，但**读回走连续偏移**，且 Pass A 写 → Pass B 读同一地址，L2 命中率 99%。

### 2. 消除 scores/P/dS 重算

```python
# 原始：每 pass 都重算 scores（dot）、P（exp）、dS
scores = tl.dot(q_cat, tl.trans(k_cat)) * scale    # ×3
P = tl.exp(scores - sm_max[:, None]) * inv_sm_sum  # ×3

# 优化：仅 Pass A 算一次，物化后 B1/B2 直接读
dS = tl.load(ds_ptr + ws_offs, ...)   # B1 读 dS
P  = tl.load(p_ptr  + ws_offs, ...)   # B2 读 P
```

`tl.exp` 是向量单元指令，issue-bound kernel 下 exp 数减至 1/3 直接降延迟。

### 3. 解耦循环顺序，各 pass 独立 tile

```python
# workspace 解耦后，A/B 可各自选最优 tile（UB 独立核算）
BLOCK_K_A: tl.constexpr, BLOCK_K_B: tl.constexpr
# Pass A tile 受 trans(k_cat) [D_TOT, BK] 占用约束
# Pass B tile 受 dk_acc/dv_acc [BK, D_TOT]/[BK, D] 累加器约束
```

原 3-pass 共用一个 BLOCK_K，被迫取各方 UB 下限。解耦后 A/B 独立 constexpr，可分别调优。

### 4. 配套：host 预计算标量中间量

```python
# 原始：kernel 内每 hc 重算 delta = rowsum(do * o)，需加载 o_pad（18KB/hc）
delta = tl.sum(do_pad.to(tl.float32) * o_pad.to(tl.float32), axis=1)

# 优化：host 预计算 delta 传入，移除 o_pad 加载
delta_flat = (do_flat.to(ms.float32) * o_flat.to(ms.float32)).sum(axis=-1).reshape(B*S1*N1)
delta = tl.load(delta_ptr + pid_bs1 * N1 + g_offs_s, ...)
```

delta 是逐 (b,s1,head) 标量，host 算一次即可。移除 o_pad 既省 mte2 又释放 UB（实测 UB stall 1002us → 0）。**仅当中间量是低维标量、可 host 低成本预算时适用**。

## 关键技术障碍与绕过（triton-ascend 限制）

### 障碍 1：workspace 存储成为新地板

物化 dS/P 引入 `tl.store` 到 GM。每 program 约 `HC × (topK/BLOCK_K_A)` 条 strided store，每条 ~1.5us issue 延迟，成为新的 mte3 瓶颈（实测占每 block 25%）。

**绕过尝试与结论：**
- **加大 BLOCK_K_A 减半 store 数**：`BLOCK_K_A=64` 时 `tl.trans(k_cat)` tile 达 `[D_TOT, 64]`=72KB，UB 溢出（217KB > 192KB）。否决。
- **转置加载 k_cat 避免 trans**：破坏按 `tok` 索引的 gather 连续性，离散度上升 576×，更差。否决。
- **`tl.join` 融合 dS/P 为单条 store**（`[BG,BK,2]` 3D tile）：triton-ascend 3D tile store 触发运行时 UB 地址越界。否决。
- **结论**：workspace store 是该结构的硬地板，需在"重算成本"与"store 成本"间取平衡点。当重算 pass 数 ≥ 2 且共享中间量维度适中时，workspace 净收益为正。

### 障碍 2：B1+B2 合并 UB 溢出

自然想进一步把 B1（dk）和 B2（dv）合并成单 blk-outer pass，halve workspace 读。但 `dkcat_acc [BK, D_TOT]` 与 `dv_acc [BK, D]` 同存，加各自 dot 中间量，UB 需 244KB > 192KB。

**绕过：保持 B1/B2 分离**。合并省的 workspace 读（连续 L2 命中，廉价）抵不上 UB 溢出。分离 pass 各自 UB 独立、安全。

### 障碍 3：workspace dtype 选择

```python
# ❌ fp32 workspace：精度最高，但 [B*S1,N1,topK] fp32 = 512MB（×2），store 带宽翻倍
ds_buf = ms.mint.zeros((B*S1, N1, topK), dtype=ms.float32)

# ✅ bf16 workspace：与下游 dot 的 downcast 精度等价（dot 前本就要 .to(bf16)），带宽减半
ds_buf = ms.mint.zeros((B*S1, N1, topK), dtype=q_bsnd.dtype)
```

下游 `tl.dot(tl.trans(dS).to(q_cat.dtype), q_cat)` 本就在 dot 前把 dS 降到 bf16，故 workspace 存 bf16 与存 fp32 再降 bf16 **精度等价**，但 store 带宽减半。仅当下游需要 fp32 累加输入时才需 fp32 workspace。

### 障碍 4：workspace buffer 必须在 device 上

```python
# ❌ ms.mint.zeros 默认在 CPU，triton 拒绝该指针
ds_buf = ms.mint.zeros((B*S1, N1, topK), dtype=q_bsnd.dtype)

# ✅ 显式 .to('Ascend')
ds_buf = ms.mint.zeros((B*S1, N1, topK), dtype=q_bsnd.dtype).to('Ascend')
```

且 workspace 是内部量，**不进 `_ms_pyfunc` 的 infer/core 返回值**（接口语义不变），仅作为 kernel 入参透传。

## 适用条件

| 条件 | 说明 |
|------|------|
| ✅ 适用 | 多输出 kernel，输出间循环遍历顺序 genuinely 冲突（A-outer vs B-outer） |
| ✅ 适用 | 冲突根因是 UB 放不下跨维度常驻累加器，且 atomic 太贵不能代替常驻 |
| ✅ 适用 | 存在两输出共享的中间量（如 dS/P），物化后可被多个 pass 复用 |
| ✅ 适用 | workspace 读回走连续偏移，且写→读同址可命中 L2（非随机索引） |
| ⚠️ 注意 | workspace store 是新地板，需重算 pass 数 ≥ 2 且共享中间量维度适中才净收益为正 |
| ⚠️ 注意 | workspace dtype 选 bf16（与下游 downcast dot 等价）以减半带宽，除非下游需 fp32 累加 |
| ⚠️ 注意 | A/B tile 拆为独立 constexpr，分别按各自 UB 约束调优 |
| ❌ 不适用 | 单输出 kernel，或输出间循环顺序一致（直接 pass 合并即可，见 pass-merge.md） |
| ❌ 不适用 | 共享中间量维度过大，workspace store 带宽超过重算成本 |
| ❌ 不适用 | workspace 读回需随机索引（无 L2 局部性），离散 load 抵消解耦收益 |

## 常见错误

### 错误 1：单 pass 强行合并（atomic 爆炸）

```python
# ❌ 错误：为消除重算，把 dk/dv 并入 hc-outer pass，dq 常驻 + dk/dv 每 blk atomic
for hc in range(HC_LOOP):
    for blk_start in range(0, topK, BLOCK_K):
        ...算 dS/P...
        acc_dqcat += tl.dot(dS, k_cat)              # 常驻 OK
        tl.atomic_add(dkcat_ptr + ..., dot(trans(dS), q_cat))  # 每 hc×blk 一次 atomic！
        tl.atomic_add(dv_ptr + ..., dot(trans(P), do_tile))    # atomic 数爆炸

# 实测：231ms，比 3-pass 重算还慢。atomic 远比重算贵。
```

### 错误 2：加大 BLOCK_K 减 store 却忽略 trans tile UB

```python
# ❌ 错误：BLOCK_K_A=64 想减半 store 数，但 trans(k_cat) [D_TOT, 64] 占 72KB
# ub overflow, requires 1777920 bits while 1572864 bits available!

# ✅ 正确：A/B tile 拆分独立 constexpr，BK_A 受 trans(k_cat) 约束取 32，BK_B 受累加器约束取 32
```

### 错误 3：用 tl.join 融合存储（3D tile 运行时越界）

```python
# ❌ 错误：tl.join(dS, P) 成 [BG,BK,2] 单条 store 减半 store 数
ws_tile = tl.join(dS.to(...), P.to(...))   # [BLOCK_G, BLOCK_K, 2]
tl.store(ws_ptr + ws_offs_3d, ws_tile, mask=ws_mask_3d)
# VEC instruction error: the ub address out of bounds（triton-ascend 3D tile store 不稳）

# ✅ 正确：保持 2D 分离 store（dS、P 各一条），接受 store 数地板
tl.store(ds_ptr + ws_offs, dS.to(...), mask=ws_mask)
tl.store(p_ptr  + ws_offs, P.to(...),   mask=ws_mask)
```

### 错误 4：workspace 用 fp32（带宽翻倍无精度收益）

```python
# ❌ 错误：fp32 workspace，store 带宽翻倍，但下游 dot 前仍 .to(bf16)，精度无提升
ds_buf = ms.mint.zeros((B*S1, N1, topK), dtype=ms.float32).to('Ascend')

# ✅ 正确：bf16 workspace，与下游 downcast dot 精度等价
ds_buf = ms.mint.zeros((B*S1, N1, topK), dtype=q_bsnd.dtype).to('Ascend')
```

### 错误 5：g_valid elision 用 tl.full 反而变慢

```python
# ❌ 错误：N1%BLOCK_G==0 时用 tl.full 编译期常量替代 i32 LT 比较以避标量降级
if NEED_G_VALID:
    g_valid = g_offs < N1
else:
    g_valid = tl.full((BLOCK_G,), 1, dtype=tl.int1)   # 分支阻碍编译器优化
# 实测：62ms → 71ms（变慢！）

# ✅ 正确：保留 g_valid = g_offs < N1，让编译器自行处理；标量降级在该场景非主瓶颈
```

## 其他案例

### 通用：多输出规约冲突

```python
# 原始：out1 需按行规约（row-outer），out2 需按列规约（col-outer），3-pass 重算共享中间
for r: for c: shared = f(data[r,c]); out1[r] += g(shared)   # pass 1
for c: for r: shared = f(data[r,c]); out2[c] += h(shared)   # pass 2（重算 shared）
for r: for c: shared = f(data[r,c]); out3[r] += k(shared)   # pass 3（重算 shared）

# 优化：pass 1 物化 shared 到 workspace，pass 2/3 读回
for r: for c: shared = f(data[r,c]); ws[r,c] = shared; out1[r] += g(shared)
for c: for r: shared = ws[r,c]; out2[c] += h(shared)        # 读回，无重算
for r: for c: shared = ws[r,c]; out3[r] += k(shared)        # 读回，无重算
```

适用前提同 SFA：行/列常驻累加器 UB 装不下、atomic 不能代替、shared 维度适中、workspace 连续可 L2 命中。

## 总结

| 优化 | 方法 | 收益来源 |
|------|------|---------|
| Workspace 物化解耦 | 物化共享中间量到 GM，解耦冲突循环顺序 | 消除重复 gather + 消除重算 + 各 pass 独立 tile |

**核心：**
- 当多输出循环顺序 genuine 冲突（UB 放不下常驻累加器 + atomic 太贵）时，物化共享中间量到 GM workspace 解耦
- Pass A 按 A-outer 顺序算并存共享中间量，Pass B 按 B-outer 顺序读回算第二个输出，无 gather 无重算
- workspace dtype 选 bf16（与下游 downcast dot 等价），buffer 在 device 上、不进接口返回值
- A/B tile 拆为独立 constexpr 分别按 UB 调优；workspace store 是新地板，重算 pass 数 ≥ 2 才净收益为正
- 配套可 host 预计算低维标量中间量（如 delta），移除 kernel 内重复 load 并释放 UB
- 不可：单 pass 强行合并（atomic 爆炸）、盲目加大 BLOCK_K（trans tile UB 溢出）、tl.join 3D 融合 store（运行时越界）、g_valid elision 用 tl.full（分支阻碍优化）

---

## 来自 SKILL.md 的原始描述（优化点：Workspace 物化解耦优化）

**适用条件**：多输出 kernel 的输出间循环遍历顺序 genuine 冲突（一需 X-outer、一需 Y-outer），冲突根因是 UB 放不下跨维度常驻累加器且 atomic 太贵不能代替常驻，存在可物化复用的共享中间量。

**典型代码特征**：
```python
# 问题代码：3-pass，每 pass 各自 gather + 重算共享中间量 scores/P/dS
# Pass A (hc-outer): gather k_cat, 算 scores/P/dS, 累积 dq
# Pass B1 (blk-outer): 又一遍 gather k_cat, 重算 scores/P/dS, 算 dk
# Pass B2 (blk-outer): 第三遍 gather k_cat, 重算 scores/P, 算 dv
```

**判断逻辑**：
- 检查 kernel 是否多输出，且输出间循环遍历顺序冲突（A-outer vs B-outer）
- 检查冲突根因：UB 放不下跨维度常驻累加器？atomic 太贵不能代替常驻？两者皆有？
- 检查是否存在两输出共享的中间量（如 dS/P），物化后可被多 pass 复用
- 检查 workspace 读回是否走连续偏移、可命中 L2（非随机索引）
- 评估 workspace store 新成本 vs 重算成本：重算 pass 数 ≥ 2 且共享中间量维度适中 → 净收益为正
- 若命中且 UB/带宽允许 → Pass A 物化中间量 + Pass B 读回解耦 + A/B 独立 tile
- 若输出间循环顺序一致 → 不涉及，转 pass-merge.md

**命中条件**：多输出 kernel 存在循环顺序冲突（UB/atomic 约束导致无法合并），且有可物化复用的共享中间量、workspace 读回具 L2 局部性、重算 pass 数 ≥ 2。

**参考文档**：`references/workspace-decoupling.md`（本文档）

---
