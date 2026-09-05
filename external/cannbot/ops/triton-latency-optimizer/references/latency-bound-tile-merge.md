# Latency-Bound 循环维度 Tile 合并优化

## 问题描述

**问题：** kernel 处于 **latency-bound**（亦称 overhead-bound）状态——算力利用率极低、带宽也未饱和，时延由每个 `tl.dot` 的**固定指令发射（issue）开销 + cube↔vector 同步等待**主导，而非由算力或带宽决定。此时若存在一个**外层循环**（如 head-chunk 循环），每个迭代发起一组 dot，且这些 dot 的某个维度（如 M 维）**远小于 cube 微块尺寸**，会导致：① dot 调用次数被外层循环放大，每条 dot 付满额 issue/同步开销；② 每个 dot 的 cube 微块半填，单 dot 效率低。两者叠加使固定开销成为时延主导。

> 术语说明：**issue** 是体系结构标准术语，指指令从调度队列被"发射"到执行单元执行的动作，每条指令有固定 issue 开销，与处理的数据量弱相关。latency-bound / overhead-bound 是性能分析中"算力与带宽均未饱和、时延被固定开销与依赖等待主导"的标准瓶颈分类（Roofline 模型中归为 non-compute-bound）。本优化点针对此类状态。

```python
# 问题代码：head-chunk 外层循环，每迭代 dot 的 M=BLOCK_G=8
# Ascend cube 微块为 16×16，M=8 半填；dot 的 issue/同步开销主导时延
for hc in range(NUM_HC):                 # 8 次外层迭代
    g_offs = hc * BLOCK_G + tl.arange(0, BLOCK_G)
    q_cat = tl.load(qcat_ptr + g_offs[:, None] * D_TOT + ...)   # [8, 576]
    do_pad = tl.load(do_ptr + g_offs[:, None] * D + ...)        # [8, 576]
    acc_dqcat = tl.zeros([BLOCK_G, D_TOT], dtype=tl.float32)    # [8, 576]
    for blk_start in range(0, topK, BLOCK_K_A):                 # 64 次内层
        k_cat = tl.load(kcat_ptr + tok_clamped[:, None] * D_TOT + ...)
        scores = tl.dot(q_cat, tl.trans(k_cat)).to(tl.float32) * scale   # M=8
        dPv = tl.dot(do_pad, tl.trans(k_cat)).to(tl.float32)             # M=8
        P = tl.exp(scores - sm_max[:, None]) * inv_sm_sum[:, None]
        dS = P * (dPv - delta[:, None]) * scale
        acc_dqcat = tl.dot(dS.to(k_cat.dtype), k_cat, acc=acc_dqcat)     # M=8
        ...
```

**开销分析（以 SFA Grad Stage A, NUM_HC=8, BLOCK_G=8, topK=2048, BLOCK_K_A=32 为例）：**
- `tl.dot` 调用：每 (hc, blk) 3 次 × 8 hc × 64 blk = **1536 次/program**
- 每个 dot 的 M=8，cube 微块(16×16)半填 → cube 利用率极低
- profiling 特征：**latency-bound**，cube_ratio **0.14%**（算力几乎闲置），vec_ratio 0.4%；cube wait_id0=2059us、vector wait_id1=1742us（cube↔vector 同步等待主导）
- 关键：dot 不是 compute-bound 也不是 memory-bound，而是 **latency-bound**（固定 issue/同步开销主导）→ 减 dot 调用次数比放大单 dot 算力更有效

## 优化方案

**原理：** 当 kernel 是 latency-bound 且存在"放大 dot 维度可减外层迭代数"的外层循环时，把**多个外层迭代合并进 dot 的某个维度（通常是 M 维）**，单次 dot 处理原本多个迭代的数据。两个叠加收益：① dot 调用次数按合并倍数下降，固定 issue/同步开销总量下降；② dot 维度放大后填满 cube 微块，单 dot 效率提升。因 dot 的 issue 开销与 dot 大小弱相关，放大 M 维不增加 issue 开销却能多处理数据。

### 关键判据：何时用循环维度合并 vs 何时不能用

适用本优化的前提是 **latency-bound**（dot 的固定 issue/同步开销主导，算力/带宽闲置）。判断依据：

1. **profiling 显示算力利用率极低**（如 cube_ratio <5%）且带宽未饱和，但 dot 调用频繁 → latency-bound，减 dot 数有效。
2. **外层循环每次迭代发起一组 dot，且 dot 的某维可放大**：如 head-chunk 循环，每个 hc 的 dot M=BLOCK_G，合并 N 个 hc → M=N×BLOCK_G。
3. **放大后的 tile 仍在 UB 内**：合并后 `acc/q/do` 等张量按 M 维线性增长，需核算 UB（Ascend910 = 192KB）。

**不适用情形（重要）：**
- **若放大维度后 dot 的 cube 内部累加溢出 CC（L0C = 128KB）**：cube 硬件固定 fp32 内部累加，输出 `[M_or_N, D_TOT]` 的累加 = `dim × D_TOT × 4` bytes，超 128KB 即编译失败。此时合并方向受 CC 锁死（见障碍 1）。
- **若该阶段瓶颈不是 dot 而是 atomic**：合并 dot 无收益（见案例 Stage B 中性）。
- **若 dot 维度已填满 cube 微块**（如 M≥16）：再放大单 dot 收益边际递减，且 UB 风险上升。

### 合并实现：连续单 tile，而非独立张量堆叠

```python
# 优化：GROUP_HC 个 head chunk 合并进 dot 的 M 维，MG = GROUP_HC * BLOCK_G
if NUM_HC >= 2:
    GROUP_HC: tl.constexpr = 2
    N_GROUPS: tl.constexpr = NUM_HC // 2
else:
    GROUP_HC: tl.constexpr = 1
    N_GROUPS: tl.constexpr = 2      # 保留 range(1) crash workaround
MG: tl.constexpr = GROUP_HC * BLOCK_G
for hg in range(N_GROUPS):           # 外层迭代数减半：8 → 4
    g_offs = hg * MG + tl.arange(0, MG)                      # 一次取 16 个 head
    g_valid = g_offs < N1
    g_offs_s = tl.where(g_valid, g_offs, 0) if NEED_CLAMP else g_offs
    q_cat = tl.load(qcat_ptr + g_offs_s[:, None] * D_TOT + ..., mask=g_valid[:, None])  # [16,576]
    do_pad = tl.load(do_ptr + g_offs_s[:, None] * D + ..., mask=g_valid[:, None] & dt_lt_D[None, :])
    acc_dqcat = tl.zeros([MG, D_TOT], dtype=tl.float32)       # [16,576]
    for blk_start in range(0, topK, BLOCK_K_A):
        k_cat = tl.load(kcat_ptr + tok_clamped[:, None] * D_TOT + ..., mask=tok_valid[:, None])
        scores = tl.dot(q_cat, tl.trans(k_cat)).to(tl.float32) * scale_value   # M=16
        dPv = tl.dot(do_pad, tl.trans(k_cat)).to(tl.float32)                   # M=16
        P = tl.exp(scores - sm_max[:, None]) * inv_sm_sum[:, None]
        dS = P * (dPv - delta[:, None]) * scale_value
        acc_dqcat = tl.dot(dS.to(k_cat.dtype), k_cat, acc=acc_dqcat)           # M=16
```

**关键：用连续的 `[MG, D_TOT]` 单 tile 合并**，而不是用 N 组独立的 `[BLOCK_G, D_TOT]` 张量堆叠。单 tile 让 cube 一次填满微块；独立张量堆叠会引入额外寄存器/UB 占用且 dot 仍逐个发起（见障碍 2）。

## 案例：Sparse Flash Attention Grad Stage A（SFA Backward）

### latency-bound 诊断

SFA Grad 经 Workspace 物化解耦优化后为 62.5ms。profiling 显示 cube_ratio 仅 0.14%、vec_ratio 0.4%——算力与带宽均闲置，**latency-bound**。进一步探针拆解：
- A-only（清空 B 阶段）= 22.96ms，full = 62.5ms（注：此处为 hc-merge 前的基线口径）
- 跳过 dPv dot 省 11.7ms → **dot 及其 cube↔vector 同步是 A 阶段主导成本**
- dot 调用数 = 1536/program，每 dot M=8 半填 cube 微块

### 合并前后

```python
# 原始（hc-outer，M=BLOCK_G=8）：NUM_HC=8 次外层，1536 dots/program
for hc in range(HC_LOOP):
    g_offs = hc * BLOCK_G + tl.arange(0, BLOCK_G)
    acc_dqcat = tl.zeros([BLOCK_G, D_TOT], dtype=tl.float32)   # [8,576]
    for blk_start in range(0, topK, BLOCK_K_A):
        ...scores/dPv/acc_dqcat dots, M=8...

# 优化（hg-outer，MG=2*BLOCK_G=16）：N_GROUPS=4 次外层，768 dots/program
GROUP_HC = 2; N_GROUPS = NUM_HC // 2; MG = GROUP_HC * BLOCK_G
for hg in range(N_GROUPS):
    g_offs = hg * MG + tl.arange(0, MG)
    acc_dqcat = tl.zeros([MG, D_TOT], dtype=tl.float32)        # [16,576]
    for blk_start in range(0, topK, BLOCK_K_A):
        ...scores/dPv/acc_dqcat dots, M=16...
```

### 性能对比

| 指标 | 原始（M=8） | 优化（M=16） | 收益 |
|------|------------|-------------|------|
| 外层迭代数/program | 8 | 4 | 减半 |
| `tl.dot` 调用/program | 1536 | 768 | **减半** |
| cube 微块填充率 | 50%（M=8/16） | 100%（M=16/16） | 填满 |
| Stage A 耗时 | ~45.6ms | ~26ms | **-43%** |
| 全 kernel Task Duration | 62.5ms | 44.5ms | **-28.8%** |

**实测性能（Ascend910, shape `(1,512,4096,64,2048)`，bf16）：**
- hc-merge 前（Workspace 解耦后基线）：62.5ms
- hc-merge（MG=16）：44.5ms
- **加速比：1.41x**（相对该基线）；相对最初 3-pass 重算 109.5ms 为 2.46x
- 精度两 shape 全过（10/10 ok，over=0.00%）

## 优势分析

### 1. 减少 dot 调用次数（latency-bound 下最大收益）

latency-bound kernel 下，每个 `tl.dot` 付出固定的 issue + cube↔vector 同步开销，与 dot 大小弱相关。合并 N 个外层迭代 → dot 数降为 1/N，固定 issue/同步开销总量降为 1/N。

```python
# 原始：8 hc × 64 blk × 3 dot = 1536 dots，每个付满额 issue+sync
for hc in range(8): for blk in range(64): scores=dot(...); dPv=dot(...); acc=dot(...)

# 优化：4 hg × 64 blk × 3 dot = 768 dots，issue+sync 总开销减半
for hg in range(4): for blk in range(64): scores=dot(M=16); dPv=dot(M=16); acc=dot(M=16)
```

### 2. 填满 cube 微块，提升单 dot 效率

Ascend cube 微块为 16×16。M=8 时微块半填，cube 空转；M=16 时正好填满。这与减 dot 数叠加，使 Stage A 耗时近减半。

### 3. 共享 k_cat gather 天然成立

合并的多个 head chunk 在同一 `(hg, blk)` 迭代内共享同一个 `k_cat`（按 token 索引 gather，与 head 无关）。M 维合并后 `scores`、`dPv` 两个 dot 共享 `k_cat`，gather 次数不变但被更多 dot 复用，gather 摊销下降。

## 关键技术障碍与绕过（triton-ascend 限制）

### 障碍 1：MG 受 UB 与 CC 双重锁死，放大有上限

合并使 `acc_dqcat / q_cat / do_pad` 按 M 维线性增长，且 dot 的 cube 内部累加也按输出维增长：

- **UB 约束**：`acc_dqcat[MG, D_TOT] fp32 + q_cat[MG, D_TOT] bf16 + do_pad[MG, D_TOT] bf16 + k_cat[BK, D_TOT] + trans(k_cat)`。MG=16 时 ~184KB（含 cube 内部 bf16→fp32 上转换副本）刚好容下 192KB；**MG=32 → ~368KB，UB 溢出**。
- **CC（L0C=128KB）约束**：cube 内部固定 fp32 累加，`acc_dqcat[MG, D_TOT]` 累加 = `MG × D_TOT × 4` bytes。MG=32 时 32×576×4 = 72KB 单看可容，但与 dot operand 副本并发即超。

**结论**：MG=2×BLOCK_G=16 是该结构的天花板，不能再放大。`tl.dot(acc=)` 强制 fp32 累加器，无法用 bf16/fp16 acc 绕过（见障碍 3）。

### 障碍 2：独立张量堆叠 ≠ 维度合并（UB 溢出且无收益）

```python
# ❌ 错误：用 2 组独立 [BLOCK_G, D_TOT] 张量"堆叠"模拟合并
q_cat_0 = tl.load(...); q_cat_1 = tl.load(...)   # 2 个 [8,576]
do_pad_0 = tl.load(...); do_pad_1 = tl.load(...) # 2 个 [8,576]
acc_0 = tl.zeros([8,576]); acc_1 = tl.zeros([8,576])
# dots 仍逐个发起（M=8 未变），且 2q+2do+2acc 独立驻留 → UB 溢出
```

独立张量堆叠既不减 dot 调用次数（dot 仍 M=8 逐个发），又因多张量并发驻留溢出 UB。**必须用连续 `[MG, D_TOT]` 单 tile**，让 dot 一次以 M=16 发起。

### 障碍 3：bf16/fp16 acc 绕过 CC 失败

自然想用低精度累加器放宽 CC 约束以放大 MG：

```python
# ❌ tl.dot(acc=bf16) 被 triton-ascend 拒绝
acc = tl.zeros([MG, D_TOT], dtype=tl.bfloat16)
acc = tl.dot(dS, k_cat, acc=acc)   # AssertionError: acc must be fp32

# ❌ out_dtype=float16 手动累加，cube 内部仍 fp32 累加
acc = tl.zeros([MG, D_TOT], dtype=tl.float16)
acc = acc + tl.dot(dS, k_cat, out_dtype=tl.float16).to(tl.float16)
# cc overflow: cube 内部 fp32 累加 [MG,D_TOT] 仍占 CC，未绕过
# （out_dtype 仅控输出 dtype，不控内部累加；out_dtype=bf16 不被支持）
```

**结论**：cube 硬件固定 fp32 内部累加，CC 约束不可绕过，MG 上限被锁死。

### 障碍 4：盲目对非 dot 瓶颈阶段套用 → 中性

Stage B（dk/dv）的 dot 是 `trans(dS[8,32])·q_cat[8,576] → [32,576]`，**M=32 已填满 cube 微块**，且 B 阶段瓶颈是 atomic_add（实测 no-atomic 探针：B 全部开销 ≈ atomic 21.6ms，dot+load 近零）。对 B 套用同款 hc-merge：

```python
# Stage B hc-merge：实测 44.5ms → 44.7ms（中性，噪声范围）
for hg in range(N_GROUPS):   # MG=16 合并
    ...dkcat_acc = tl.dot(trans(dS).to(q_cat.dtype), q_cat, acc=dkcat_acc)...
```

B 的 dot M=32 已满，合并不减单 dot 效率；B 的 dot 数本就少，减次数收益可忽略；瓶颈 atomic 不受影响。**已回退**。判据：合并只对"dot 维度未满 + dot 数被外层放大 + 阶段是 dot/latency-bound"的阶段有效。

## 适用条件

| 条件 | 说明 |
|------|------|
| ✅ 适用 | kernel 处于 latency-bound（算力利用率极低，dot 的固定 issue/同步开销主导） |
| ✅ 适用 | 存在外层循环，每迭代发起一组 dot，且 dot 的某维（常 M）可放大 |
| ✅ 适用 | 放大维度后填满 cube 微块（M 从 <16 提到 ≥16） |
| ✅ 适用 | 放大后 tile 仍在 UB 内（核算 acc/q/do 等张量按维线性增长） |
| ⚠️ 注意 | 合并用连续单 tile，禁止独立张量堆叠（不减 dot 数且溢出 UB） |
| ⚠️ 注意 | MG 上限受 UB + CC（cube fp32 累加）双重锁死，放大有天花板 |
| ⚠️ 注意 | 仅对 dot/latency-bound 阶段套用；atomic-bound 阶段套用为中性（见障碍 4） |
| ❌ 不适用 | dot 维度已填满 cube 微块（M≥16）且阶段非 dot-bound |
| ❌ 不适用 | 放大维度致 cube 内部 fp32 累加溢出 CC（128KB），且无法用低精度 acc 绕过 |
| ❌ 不适用 | 阶段瓶颈是 atomic/store/load 而非 dot（合并不触及瓶颈） |

## 常见错误

### 错误 1：独立张量堆叠冒充维度合并

```python
# ❌ 错误：2 组独立 [BLOCK_G, D_TOT] 张量，dot 仍 M=8 逐个发，2q+2do+2acc 溢出 UB
q0 = tl.load(...); q1 = tl.load(...); acc0 = zeros([8,576]); acc1 = zeros([8,576])
for blk: acc0 = dot(dS0, k_cat, acc=acc0); acc1 = dot(dS1, k_cat, acc=acc1)
# ub overflow（多张量并发驻留）+ dot 数未减

# ✅ 正确：连续 [MG, D_TOT] 单 tile，dot 一次 M=16 发起
q_cat = tl.load(... [MG, D_TOT]); acc = zeros([MG, D_TOT])
for blk: acc = dot(dS, k_cat, acc=acc)   # M=16，dot 数减半
```

### 错误 2：盲目加大 MG 致 UB/CC 溢出

```python
# ❌ 错误：MG=32（4×合并）想进一步减 dot 数
MG = 4 * BLOCK_G   # 32
acc_dqcat = tl.zeros([MG, D_TOT], dtype=tl.float32)  # [32,576] + q/do + k_cat → UB ~368KB
# ub overflow, requires ~368KB while 192KB available

# ✅ 正确：MG=2*BLOCK_G=16 是 UB 天花板；先核算 acc+q+do+operand 副本总占用
```

### 错误 3：用低精度 acc 绕过 CC

```python
# ❌ 错误：bf16 acc 想放宽 CC 以放大 MG
acc = tl.zeros([MG, D_TOT], dtype=tl.bfloat16)
acc = tl.dot(dS, k_cat, acc=acc)   # AssertionError: acc must be fp32
# 或 out_dtype=float16 手动累加 → cube 内部仍 fp32 累加，cc overflow

# ✅ 正确：接受 cube fp32 累加的 CC 约束，MG 取 CC 允许上限；不试图绕过
```

### 错误 4：对 atomic-bound 阶段套用

```python
# ❌ 错误：Stage B（dot M=32 已满，瓶颈是 atomic）套用 hc-merge
for hg in range(N_GROUPS): dkcat_acc = dot(trans(dS), q_cat, acc=dkcat_acc)
# 实测中性（44.5→44.7ms）：B 非 dot-bound，合并不触及 atomic 瓶颈

# ✅ 正确：先用 no-atomic/skip-dot 探针定位阶段瓶颈；仅对 dot/latency-bound 阶段合并
```

## 其他案例

### 通用：latency-bound 下外层循环维度并入 dot

```python
# 原始：group-outer 循环，每迭代 dot 的 M=GROUP_SIZE（< cube 微块 16），latency-bound
for g in range(NUM_GROUPS):
    a = tl.load(... [GROUP_SIZE, K]); b = tl.load(... [K, N])
    acc = tl.zeros([GROUP_SIZE, N], dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(a_tile, b_tile, acc=acc)   # M=GROUP_SIZE 半填
# profiling: cube_ratio <5%, dot 调用 = NUM_GROUPS × (K/BK)

# 优化：MERGE 个 group 并入 M 维，MG=MERGE*GROUP_SIZE 填满微块
MG = MERGE * GROUP_SIZE
for gm in range(NUM_GROUPS // MERGE):
    a = tl.load(... [MG, K]); acc = tl.zeros([MG, N], dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(a_tile, b_tile, acc=acc)   # M=MG 填满，dot 数 /MERGE
```

适用前提同 SFA Stage A：latency-bound（算力利用率低）、M 维可放大且放大后填满微块、UB/CC 容得下连续单 tile。

## 总结

| 优化 | 方法 | 收益来源 |
|------|------|---------|
| Latency-Bound 循环维度 Tile 合并 | 外层循环多迭代并入 dot 的 M 维（连续单 tile） | 减 dot 调用次数 + 填满 cube 微块 |

**核心：**
- 当 kernel 是 latency-bound（算力利用率极低，dot 的固定 issue/同步开销主导）且存在"放大 dot 维度可减外层迭代数"的循环时，把多个外层迭代合并进 dot 的 M 维
- 合并用**连续 `[MG, D]` 单 tile**，一次 dot 以放大后的 M 发起；禁止独立张量堆叠（不减 dot 数且溢出 UB）
- MG 上限受 **UB（192KB）+ CC（128KB，cube 固定 fp32 累加）** 双重锁死，放大有天花板；bf16/fp16 acc 无法绕过 CC
- 仅对 **dot/latency-bound 阶段**套用；atomic-bound 阶段（瓶颈在 scatter atomic）套用为中性，须先用探针定位阶段瓶颈
- 不可：独立张量堆叠、盲目加大 MG、低精度 acc 绕 CC、对非 dot-bound 阶段套用

---

## 来自 SKILL.md 的原始描述（优化点：Latency-Bound 循环维度 Tile 合并）

**适用条件**：kernel 处于 latency-bound（算力利用率极低，dot 的固定 issue/同步开销主导），存在外层循环每迭代发起一组 dot，且 dot 的某维（常 M 维）小于 cube 微块尺寸可放大。

**典型代码特征**：
```python
# 问题代码：head-chunk 外层循环，每迭代 dot 的 M=BLOCK_G=8（< cube 微块 16），latency-bound
for hc in range(NUM_HC):
    acc = tl.zeros([BLOCK_G, D_TOT], dtype=tl.float32)
    for blk_start in range(0, topK, BLOCK_K_A):
        scores = tl.dot(q_cat, tl.trans(k_cat))   # M=8 半填，dot 数 = NUM_HC × (topK/BK) × 3
        ...
```

**判断逻辑**：
- 检查 profiling：算力利用率极低（<5%）且带宽未饱和，但 dot 调用频繁 → latency-bound
- 检查是否存在外层循环，每迭代发起一组 dot，且 dot 某维（M）小于 cube 微块（16）可放大
- 用 no-atomic/skip-dot 探针确认阶段是 dot/latency-bound（而非 atomic/store-bound）
- 核算放大后 tile 是否在 UB 内（acc/q/do 按维线性增长）+ cube 内部 fp32 累加是否溢出 CC（128KB）
- 若 latency-bound 且 UB/CC 容得下连续单 tile → 外层迭代并入 dot M 维，MG 取 UB/CC 允许上限
- 若阶段是 atomic-bound → 不适用（合并中性），转 atomic 相关优化
- 若 dot 维度已填满微块 → 不适用

**命中条件**：kernel latency-bound，外层循环放大 dot 维度可减迭代数，放大后连续单 tile 在 UB/CC 内，且阶段为 dot/latency-bound。

**参考文档**：`references/latency-bound-tile-merge.md`（本文档）

---
