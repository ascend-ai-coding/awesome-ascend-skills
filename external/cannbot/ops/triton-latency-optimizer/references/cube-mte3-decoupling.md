# Cube/MTE3 分阶段批量解耦优化

## 问题描述

**问题：** 一个 kernel 在**同一循环体**内既要做 Cube 计算（跨循环累加的输出，如 `dq`），又要做 scatter-add 写回（`atomic_add` 到 `dk/dv/dkr`）。`atomic_add` 是带锁/仲裁的 **read-modify-write** 三步：读旧值 → UB 加法 → 写回。把 Cube 计算与 MTE3 atomic RMW **交替串行**在同一热循环里，MTE3 管线被带锁 RMW 占满，Cube 每次迭代后必须等 MTE3 完成才能继续 → **Cube 流水断流**。

更糟的是：当某个归约维（如 head 维 `N1`）被切成多个 program 并行处理时，该维的归约只能靠**多 program 对同一输出地址的 atomic 竞争**完成——`num_g_groups` 个 program 各自发 atomic_add 打到同一 token 位置，RMW 被锁串行仲裁，atomic 次数爆炸。

```python
# 问题代码：单 pass，Cube(dq) 与 MTE3 atomic(dk/dv/dkr) 在同一循环体交替串行
# Grid: (B*S1 * num_g_groups,) —— head 维被切成 num_g_groups 个 program 并行
for blk_start in range(0, topK, BLOCK_K):
    k_full  = tl.load(k_ptr  + tok_clamped[:, None] * D          + d_offs[None, :])
    kr_full = tl.load(kr_ptr + tok_clamped[:, None] * D_ROPE     + dr_offs[None, :])
    scores  = (tl.dot(q_nope, tl.trans(k_full))
             + tl.dot(q_rope, tl.trans(kr_full))) * scale_value
    P   = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]
    dS  = P * (tl.dot(do_tile, tl.trans(k_full)) - delta[:, None]) * scale_value
    acc_dq += tl.dot(dS.to(k_full.dtype), k_full)                  # Cube

    # 3× atomic_add RMW 夹在 Cube 之间 → MTE3 占满，Cube 等 MTE3
    tl.atomic_add(dk_ptr  + ..., tl.dot(tl.trans(dS), q_nope))     # RMW #1
    tl.atomic_add(dv_ptr  + ..., tl.dot(tl.trans(P),  do_tile))    # RMW #2
    tl.atomic_add(dkr_ptr + ..., tl.dot(tl.trans(dS), q_rope))     # RMW #3
tl.store(dq_ptr + ..., acc_dq)   # dq 循环末尾一次性写
```

**开销分析（以 SFA Grad, N1=64, BLOCK_G=16, num_g_groups=4, topK=2048, BLOCK_K=32 → 64 次 blk 迭代为例）：**
- `atomic_add` 次数/program：3 × 64 = 192
- `atomic_add` 次数/(row,blk) **含竞争**：`num_g_groups` 个 program 对同一 token 位置各发 3 次 → 12 次 RMW 被锁串行仲裁
- MTE3 管线：被带锁 RMW 占满 → Cube 每次迭代后空等 MTE3 → Cube 流水断流
- profiling 特征：**MTE3-bound**，Cube 利用率被 atomic 阻塞压低；atomic 吞吐成为硬地板（scatter-add 路径的固有瓶颈）

## 优化方案

**原理：** 把 Cube 计算与 MTE3 写回从"同一循环体交替串行"重构为"**分阶段批量**"：

1. **Stage A（纯 Cube 阶段）**：热循环只累加 `acc_dq/acc_dqr`，**零 MTE3 写** → Cube 不被 Vector/MTE3 阻塞，流水打满；循环结束一次性连续 `tl.store`。配合 grid 改为"一行一 program"，dq 写入位置在 head 维由同一 program 独占、**无竞争** → 普通 store，**无需 atomic**。
2. **Stage B（批量 atomic 阶段）**：把原本跨 program 的 head 维归约（v1 用 atomic 竞争实现）收进**单 program 的 UB 累加器**（`dk_acc/dkr_acc/dv_acc` 跨 `hc` 在 UB 内做 dot+add，零 HBM 往返），每个 blk 只发**一次** `atomic_add`。RMW 从"零散竞争"变为"批量最少次"。

### 关键判据：何时该分阶段而非合并

满足以下条件时，应把 Cube 计算与 scatter-add 写回拆成分阶段批量，而非塞进同一循环：

1. **Cube 累加输出与 atomic scatter 输出在同一循环体交替**：MTE3 atomic RMW 阻塞 Cube 流水，profiling 呈 MTE3-bound、Cube 闲置。
2. **某归约维被切成多 program 并行，导致该维归约靠 atomic 竞争完成**：atomic 数随 program 数线性爆炸，且竞争加剧锁串行。
3. **该归约维可被单 program 串行遍历**：UB 放得下 per-blk 累加器（如 `dk_acc[BLOCK_K, D]`），即可把 atomic 竞争换成 UB 内累加 → atomic 次数与 program 数解耦。

### Stage A：纯 Cube，热循环零 MTE3

```python
# Grid: (B*S1,) —— 一行一 program，head 维 hc 串行遍历（不再切 program）
for hc in range(HC_LOOP):
    g_offs = hc * BLOCK_G + tl.arange(0, BLOCK_G)
    q_nope = tl.load(...); q_rope = tl.load(...)
    do_tile = tl.load(...); o_tile = tl.load(...)
    delta = tl.sum(do_tile * o_tile, axis=1)
    acc_dq = tl.zeros([BLOCK_G, D], dtype=tl.float32)
    acc_dqr = tl.zeros([BLOCK_G, D_ROPE], dtype=tl.float32)

    for blk_start in range(0, topK, BLOCK_K):
        k_full = tl.load(...); kr_full = tl.load(...)
        scores = (tl.dot(q_nope, tl.trans(k_full))
                + tl.dot(q_rope, tl.trans(kr_full))) * scale_value
        P  = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]
        dS = P * (tl.dot(do_tile, tl.trans(k_full)) - delta[:, None]) * scale_value
        acc_dq  += tl.dot(dS.to(k_full.dtype),  k_full)              # Cube，无 MTE3 写
        acc_dqr += tl.dot(dS.to(kr_full.dtype), kr_full)             # Cube，无 MTE3 写
    # 循环结束一次性 plain store（一行一 program → head 维无竞争 → 非 atomic）
    tl.store(dq_ptr  + ..., acc_dq.to(dq_ptr.dtype.element_ty),  mask=...)
    tl.store(dqr_ptr + ..., acc_dqr.to(dqr_ptr.dtype.element_ty), mask=...)
```

### Stage B1：跨 head chunk UB 累加 dk/dkr，每 blk 单次 atomic

```python
# blk-outer / hc-inner：dk_acc/dkr_acc 跨 hc 在 UB 内归约，再单次 atomic_add
for blk_start in range(0, topK, BLOCK_K):
    tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)
    tok_valid = ...; tok_clamped = tl.where(tok_valid, tok, 0)
    k_full  = tl.load(k_ptr  + tok_clamped[:, None] * D      + d_offs[None, :])   # 每 blk load 一次
    kr_full = tl.load(kr_ptr + tok_clamped[:, None] * D_ROPE + dr_offs[None, :])
    dk_acc  = tl.zeros([BLOCK_K, D],      dtype=tl.float32)
    dkr_acc = tl.zeros([BLOCK_K, D_ROPE], dtype=tl.float32)

    for hc in range(HC_LOOP):
        q_nope = tl.load(...); q_rope = tl.load(...); do_tile = tl.load(...); o_tile = tl.load(...)
        delta = tl.sum(do_tile * o_tile, axis=1)
        scores = (tl.dot(q_nope, tl.trans(k_full)) + tl.dot(q_rope, tl.trans(kr_full))) * scale_value
        P  = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]
        dS = P * (tl.dot(do_tile, tl.trans(k_full)) - delta[:, None]) * scale_value
        dk_acc  += tl.dot(tl.trans(dS).to(q_nope.dtype),  q_nope)   # UB 内累加，零 atomic
        dkr_acc += tl.dot(tl.trans(dS).to(q_rope.dtype), q_rope)    # UB 内累加，零 atomic

    # 跨 hc 归约完毕，每 blk 仅 1 次 atomic_add（head 维竞争已消除）
    tl.atomic_add(dk_ptr  + ..., dk_acc,  mask=tok_valid[:, None])
    tl.atomic_add(dkr_ptr + ..., dkr_acc, mask=tok_valid[:, None])
```

### Stage B2：跨 head chunk UB 累加 dv，独立 pass

```python
# 与 B1 分离：dv_acc 不与 dk_acc 同存，避免 UB 溢出
for blk_start in range(0, topK, BLOCK_K):
    tok = tl.load(...); tok_valid = ...; tok_clamped = ...
    k_full = tl.load(...); kr_full = tl.load(...)
    dv_acc = tl.zeros([BLOCK_K, D], dtype=tl.float32)
    for hc in range(HC_LOOP):
        q_nope = tl.load(...); q_rope = tl.load(...); do_tile = tl.load(...)
        scores = (tl.dot(q_nope, tl.trans(k_full)) + tl.dot(q_rope, tl.trans(kr_full))) * scale_value
        P  = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]
        dv_acc += tl.dot(tl.trans(P).to(do_tile.dtype), do_tile)    # UB 内累加
    tl.atomic_add(dv_ptr + ..., dv_acc, mask=tok_valid[:, None])    # 每 blk 1 次 atomic
```

## 案例：Sparse Flash Attention Grad（SFA Backward）

### Cube/MTE3 交替阻塞 + head 维 atomic 竞争分析

SFA backward 需输出 dq/dqr（Cube 累加）+ dk/dkr/dv（scatter-add）。核心冲突：

| 输出 | 性质 | v1 处理 | v3 处理 |
|------|------|---------|---------|
| dq / dqr | 跨 blk Cube 累加 | 热循环内夹 3× atomic → Cube 被 MTE3 阻塞 | Stage A 纯 Cube 累加，热循环零 MTE3，结束 plain store |
| dk / dkr / dv | scatter-add (atomic) | 每 blk 3× atomic；head 维靠 `num_g_groups` 个 program atomic 竞争归约 | Stage B1/B2 跨 hc UB 累加；每 blk 1× atomic，head 维竞争消除 |

**Grid 变化**：`(B*S1 * num_g_groups,)` → `(B*S1,)`——一行一 program，head 维从"多 program 并行 + atomic 竞争"改为"单 program 串行 + UB 累加"。

### 原始实现（v1：单 pass 交替串行）

```python
@triton.jit
def _sfa_grad_kernel(..., BLOCK_G: tl.constexpr, BLOCK_K: tl.constexpr, ...):
    pid = tl.program_id(0)
    num_g_groups = (N1 + BLOCK_G - 1) // BLOCK_G
    pid_bs1 = pid // num_g_groups          # head 维切 program
    pid_g   = pid %  num_g_groups
    g_offs  = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    ...
    for blk_start in range(0, topK, BLOCK_K):
        k_full = tl.load(...); kr_full = tl.load(...)
        scores = (tl.dot(q_nope, tl.trans(k_full)) + tl.dot(q_rope, tl.trans(kr_full))) * scale_value
        P  = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]
        dS = P * (tl.dot(do_tile, tl.trans(k_full)) - delta[:, None]) * scale_value
        acc_dq += tl.dot(dS.to(k_full.dtype), k_full)                 # Cube
        # 3× atomic RMW 夹在 Cube 之间，且 num_g_groups 个 program 竞争同一 token
        tl.atomic_add(dk_ptr  + ..., tl.dot(tl.trans(dS), q_nope))
        tl.atomic_add(dv_ptr  + ..., tl.dot(tl.trans(P),  do_tile))
        tl.atomic_add(dkr_ptr + ..., tl.dot(tl.trans(dS), q_rope))
    tl.store(dq_ptr + ..., acc_dq.to(...), mask=...)
```

### 分阶段批量解耦后（v3）

```python
@triton.jit
def _sfa_grad_kernel_v3(..., NUM_HC: tl.constexpr, NEED_CLAMP: tl.constexpr, ...):
    pid = tl.program_id(0)                 # 一行一 program
    b = pid // S1; s1 = pid % S1
    ...
    # Stage A：纯 Cube，热循环零 MTE3，结束 plain store（非 atomic）
    for hc in range(HC_LOOP):
        ...acc_dq/acc_dqr 累加...
        for blk_start in range(0, topK, BLOCK_K):
            ...scores/P/dS...
            acc_dq  += tl.dot(dS.to(k_full.dtype),  k_full)
            acc_dqr += tl.dot(dS.to(kr_full.dtype), kr_full)
        tl.store(dq_ptr + ..., acc_dq.to(...),  mask=...)
        tl.store(dqr_ptr + ..., acc_dqr.to(...), mask=...)

    # Stage B1：blk-outer/hc-inner，跨 hc UB 累加 dk/dkr，每 blk 1× atomic
    for blk_start in range(0, topK, BLOCK_K):
        ...k_full/kr_full load 一次...
        dk_acc = tl.zeros([BLOCK_K, D], ...); dkr_acc = tl.zeros([BLOCK_K, D_ROPE], ...)
        for hc in range(HC_LOOP):
            ...scores/P/dS 重算...
            dk_acc  += tl.dot(tl.trans(dS).to(q_nope.dtype),  q_nope)
            dkr_acc += tl.dot(tl.trans(dS).to(q_rope.dtype), q_rope)
        tl.atomic_add(dk_ptr + ..., dk_acc,  mask=tok_valid[:, None])
        tl.atomic_add(dkr_ptr + ..., dkr_acc, mask=tok_valid[:, None])

    # Stage B2：blk-outer/hc-inner，跨 hc UB 累加 dv，每 blk 1× atomic（与 B1 分离避 UB 溢出）
    for blk_start in range(0, topK, BLOCK_K):
        ...k_full/kr_full load 一次...
        dv_acc = tl.zeros([BLOCK_K, D], ...)
        for hc in range(HC_LOOP):
            ...scores/P 重算...
            dv_acc += tl.dot(tl.trans(P).to(do_tile.dtype), do_tile)
        tl.atomic_add(dv_ptr + ..., dv_acc, mask=tok_valid[:, None])
```

### 性能对比

> 下表为代码结构可推导的指标；实测 latency 见表后。

| 指标 | v1（单 pass 交替串行） | v3（分阶段批量） | 收益 |
|------|------------------------|-------------------|------|
| `atomic_add`/(row,blk) | 12（`num_g_groups`=4 program × 3，竞争） | 3（head 维 UB 内归约） | **-75%** |
| dq 热循环 MTE3 写 | 有（3× atomic 夹在 Cube 间） | 无 | **Cube 解阻塞** |
| dq 写回方式 | plain store（但 row 分 program，head 维跨 program） | plain store（一行一 program，head 维无竞争） | 去竞争 |
| head 维归约 | 多 program atomic 竞争 | 单 program UB 累加 | **消除竞争** |
| Cube/MTE3 流水 | 交替串行，Cube 等 MTE3 | 分阶段批量，Stage A 纯 Cube | **解阻塞** |
| scores/P/dS 计算 | 1 次/blk（单 pass） | 2 次/blk（Stage A + Stage B 各算一遍） | 重算代价 ↑（权衡） |
| k/kr gather 次数/program | 1×/blk | 2×/blk（Stage A + Stage B 各 load） | gather 代价 ↑（权衡） |

**实测性能（Ascend910，SFA Grad，bf16）：**
- v1 单 pass 交替串行：238ms
- v3 分阶段批量：147ms
- **加速比：1.62x**（238ms → 147ms）

**权衡核心**：v3 用"Stage B 重算 scores/P/dS + 重复 gather k/kr"换取"Cube 解阻塞 + atomic 减 75%"。当 **重算/gather 成本 < atomic + Cube-stall 成本**（SFA Grad 中 atomic 是硬吞吐地板，常态成立）时净收益为正。若重算成为新瓶颈，转 `references/workspace-decoupling.md`——物化 dS/P 到 GM 消除 Stage B 重算（见"与其他优化点关系"）。

## 优势分析

### 1. 解除 Cube/MTE3 交替阻塞（最大收益）

```python
# v1：Cube 与 atomic RMW 交替串行，Cube 每次迭代后空等 MTE3
for blk_start:
    acc_dq += tl.dot(dS, k_full)           # Cube
    tl.atomic_add(dk_ptr + ..., ...)       # MTE3 RMW（带锁）→ Cube 阻塞

# v3 Stage A：纯 Cube，热循环零 MTE3，流水打满
for blk_start:
    acc_dq += tl.dot(dS, k_full)           # Cube 连续发射，无 MTE3 干扰
tl.store(dq_ptr + ..., acc_dq)             # 循环外一次性写
```

Stage A 把 dq 整条 K 循环跑完才 store，Cube 连续发射 `topK/BLOCK_K` 次矩阵乘，中间零 MTE3；store 是累加完毕后的单次连续写（非 atomic），走纯 write 路径无需 read 回读。

### 2. atomic 批量化（head 维 UB 归约）

```python
# v1：num_g_groups 个 program 各自发 atomic 打到同一 token → 锁串行仲裁
# 每个 (row,blk) 实际 12 次 RMW 竞争
tl.atomic_add(dk_ptr + tok_offs, dot(trans(dS), q_nope))   # × num_g_groups program

# v3：单 program 内跨 hc 在 UB 加法归约，每 blk 仅 1 次 atomic
for hc in range(HC_LOOP):
    dk_acc += tl.dot(tl.trans(dS), q_nope)                  # UB 内，零 HBM 往返
tl.atomic_add(dk_ptr + tok_offs, dk_acc)                    # ×1
```

head 维归约从"跨 program atomic 竞争"移入"单 program UB 累加"，atomic 次数与 program 数解耦，RMW 从零散竞争变为批量最少次。

### 3. dq 去 atomic（一行一 program 的红利）

grid 改为"一行一 program"后，dq 写入位置 `[b,s1,h,d]` 在 head 维由同一 program 串行独占、**无竞争** → 普通 `tl.store`，无需 atomic。dk/dv 的写入位置 `[b, tok, d]` 中 `tok` 来自 sparse_indices，多行 program 仍可能指向同一 token（scatter 语义），故 dk/dv 保留 atomic——**能去 atomic 的去 atomic，不能去的把次数压到最低**。

### 4. 各 Stage 独立 tile

```python
# Stage A tile 受 acc_dq/acc_dqr + trans(k_full) 占用约束
# Stage B tile 受 dk_acc/dv_acc [BLOCK_K, D] 累加器约束
# 分阶段后各自 UB 独立核算，可分别调优
```

## 关键技术障碍与绕过（triton-ascend 限制）

### 障碍 1：Stage B 重算 scores/P/dS 成为新成本

v3 的 Stage B1/B2 各自重算 scores/P/dS 并重复 gather k/kr（v1 只算一次）。这是"用重算换解耦"的固有代价。

**绕过策略：**
- **接受重算**：当 atomic/Cube-stall 是硬地板（SFA Grad 常态），重算成本 < 解除阻塞收益 → 净正，采用 v3。
- **转 workspace 物化**：若重算/gather 成为新瓶颈，加载 `references/workspace-decoupling.md`，在 Stage A 物化 dS/P 到 GM，Stage B 读回消除重算。两者可叠加：先用本优化解 Cube 阻塞，再叠 workspace 消 Stage B 重算。

### 障碍 2：B1+B2 合并 UB 溢出

自然想把 B1（dk/dkr）和 B2（dv）合并成单 blk-outer pass，halve 重算。但 `dk_acc[BLOCK_K,D] + dkr_acc[BLOCK_K,D_ROPE] + dv_acc[BLOCK_K,D]` + 各自 dot 中间量同存，UB 需 >192KB。

**绕过：保持 B1/B2 分离**。合并省的重算（dot/exp）抵不上 UB 溢出。分离 pass 各自 UB 独立、安全。

### 障碍 3：NUM_HC==1 触发 scf.For 编译 crash

```python
# ❌ for hc in range(1) 触发 ttir_to_linalg 的 scf.For assertion failure
for hc in range(NUM_HC):   # NUM_HC==1 时编译器 crash

# ✅ NUM_HC==1 时强制 HC_LOOP=2，多出的 hc=1 迭代 g_valid 全 False 空转，不影响结果
if NUM_HC == 1:
    HC_LOOP: tl.constexpr = 2
else:
    HC_LOOP: tl.constexpr = NUM_HC
for hc in range(HC_LOOP):
    g_offs = hc * BLOCK_G + tl.arange(0, BLOCK_G)
    g_valid = g_offs < N1   # hc=1 时全 False → 空转
```

### 障碍 4：g_offs clamp 的 select 指令开销

```python
# ❌ 无条件 tl.where clamp，每 load 地址多一条 select
g_offs_s = tl.where(g_valid, g_offs, 0)

# ✅ NEED_CLAMP constexpr 分支：N1%BLOCK_G==0 时直接用 g_offs，跳过 select
if NEED_CLAMP:
    g_offs_s = tl.where(g_valid, g_offs, 0)
else:
    g_offs_s = g_offs
```

N1 取 2 的幂、BLOCK_G=16 时几乎总满足 `N1%BLOCK_G==0`，编译期消除 select。

### 障碍 5：dq 并行度下降

grid 从"一行 × num_g_groups program"改为"一行一 program"，dq 路径并行度降 `num_g_groups` 倍。

**判据**：当 **topK 主导计算量**（每 program 工作量大，足以掩盖并行度损失）时可接受；当 N1 极大且 topK 小（每 program 工作量小，并行度损失凸显）时需评估，必要时保留 head 维多 program 但用其他方式解 Cube 阻塞。

## 适用条件

| 条件 | 说明 |
|------|------|
| ✅ 适用 | Cube 累加输出与 atomic scatter 输出在同一循环体交替，MTE3 阻塞 Cube（profiling MTE3-bound、Cube 闲置） |
| ✅ 适用 | 某归约维被切成多 program 并行，该维归约靠 atomic 竞争完成，atomic 数随 program 数爆炸 |
| ✅ 适用 | 该归约维可被单 program 串行遍历，UB 放得下 per-blk 累加器（atomic 竞争可换 UB 累加） |
| ✅ 适用 | 重算/gather 成本 < atomic + Cube-stall 成本（atomic 是硬吞吐地板时常态成立） |
| ⚠️ 注意 | Stage B 重算成新瓶颈时，转 workspace-decoupling.md 物化中间量消除重算（可叠加） |
| ⚠️ 注意 | B1/B2 需分离避免 UB 溢出；各 Stage tile 拆独立 constexpr 分别按 UB 调优 |
| ⚠️ 注意 | "一行一 program"降低 dq 并行度，需 topK 主导计算量才划算 |
| ❌ 不适用 | 输出间循环顺序一致、无 atomic 阻塞 → 直接 pass 合并，见 pass-merge.md |
| ❌ 不适用 | 重算/gather 成本远高于 atomic 成本（atomic 非瓶颈）→ 本优化净负，转 workspace-decoupling.md |
| ❌ 不适用 | 单输出 kernel 无 scatter-add，无 Cube/MTE3 交替问题 |

## 常见错误

### 错误 1：只加 UB 累加但不改 grid（atomic 仍竞争）

```python
# ❌ 错误：head 维仍切多 program，UB 累加只在本 program 内归约，
#         跨 program 对同一 token 的 atomic 竞争仍在 → atomic 数未降
# Grid: (B*S1 * num_g_groups,)  ← 未改
for blk_start:
    dk_acc = tl.zeros(...)
    for hc in range(1):   # 每个 program 只 1 个 hc，UB 累加无效
        dk_acc += tl.dot(trans(dS), q_nope)
    tl.atomic_add(dk_ptr + ..., dk_acc)   # num_g_groups 个 program 仍竞争

# ✅ 正确：grid 改一行一 program，head 维全部 hc 在单 program 内 UB 归约
# Grid: (B_S1,)
for hc in range(HC_LOOP):                 # 跨全部 hc
    dk_acc += tl.dot(trans(dS), q_nope)
tl.atomic_add(dk_ptr + ..., dk_acc)       # 仅 1 次，无 head 维竞争
```

### 错误 2：Stage A 仍保留 atomic（未真正解耦）

```python
# ❌ 错误：dq 累加循环里仍夹 atomic_add → Stage A 不再"零 MTE3"，Cube 仍被阻塞
for blk_start:
    acc_dq += tl.dot(dS, k_full)
    tl.atomic_add(dk_ptr + ..., ...)      # MTE3 又混进热循环

# ✅ 正确：Stage A 严格零 MTE3，所有 scatter-add 移到 Stage B
for blk_start:
    acc_dq += tl.dot(dS, k_full)          # 纯 Cube
tl.store(dq_ptr + ..., acc_dq)            # 循环外写
```

### 错误 3：B1+B2 合并致 UB 溢出

```python
# ❌ 错误：单 blk-outer pass 同时累加 dk_acc + dkr_acc + dv_acc
# ub overflow, requires ... while 1572864 bits available!
dk_acc  = tl.zeros([BLOCK_K, D],      ...)
dkr_acc = tl.zeros([BLOCK_K, D_ROPE], ...)
dv_acc  = tl.zeros([BLOCK_K, D],      ...)   # 三者同存溢出

# ✅ 正确：B1（dk/dkr）与 B2（dv）分离为两 pass，UB 独立
```

### 错误 4：NUM_HC==1 未绕过 scf.For crash

```python
# ❌ 错误：NUM_HC==1 时 for hc in range(1) 触发 ttir_to_linalg scf.For assertion failure
for hc in range(NUM_HC):   # crash

# ✅ 正确：HC_LOOP 强制 ≥2，hc=1 迭代 g_valid 全 False 空转
```

### 错误 5：累加器降 bf16 致 atomic 精度损失

```python
# ⚠️ 注意：dk_acc/dv_acc 用 fp32 是必要的（多次 atomic 累加需 fp32 精度）；
#         不要为省 UB 把累加器降 bf16 → 多次 atomic 累加精度损失。
```

## 与其他优化点关系

### vs Workspace 物化解耦（references/workspace-decoupling.md）

**同**：都针对多输出 kernel 的循环冲突问题（Cube 累加输出 vs scatter-add 输出）。

**异**：策略相反。
| 维度 | 本优化（分阶段批量） | Workspace 物化解耦 |
|------|----------------------|---------------------|
| 解的是 | Cube/MTE3 交替阻塞 + atomic 竞争 | 输出间循环顺序 genuine 冲突（UB 放不下常驻累加器） |
| 对重算 | **接受重算**（Stage B 重算 scores/P/dS） | **消除重算**（物化 dS/P 到 GM 读回） |
| 对 atomic | 批量化（head 维 UB 归约，-75%） | 各 pass 末单次 atomic（本就最少） |
| 净收益条件 | 重算 < atomic+Cube-stall | 重算 pass ≥2 且 workspace store 可承受 |

**可叠加**：先用本优化把 Cube 解阻塞、atomic 批量化；若 Stage B 重算成为新瓶颈，再叠 workspace 物化 dS/P 消除重算。SFA Grad 的 v3 即本优化；workspace-decoupling.md 描述的是其上的进一步优化。

### vs Pass 消除合并（references/pass-merge.md）

pass-merge 针对输出间循环顺序一致、可直接合并的场景；本优化针对 Cube/MTE3 交替阻塞，循环顺序未必冲突（v1 本就是单 pass，问题是"交替"而非"多 pass"）。

## 其他案例

### 通用：Cube 累加 + scatter-add 共存

```python
# 原始：acc（Cube 跨循环累加）与 scatter（atomic）在同一循环交替
for i in range(N):
    x = tl.load(...)
    acc += tl.dot(x, w)                  # Cube
    tl.atomic_add(out_ptr + idx[i], contrib)   # MTE3 RMW 阻塞 Cube

# 优化：Stage A 纯 Cube 累加 acc，Stage B 批量 scatter
for i in range(N):
    x = tl.load(...); acc += tl.dot(x, w)   # 纯 Cube，零 MTE3
tl.store(acc_ptr + ..., acc)
# idx 维归约若被切多 program → 收进单 program UB 累加再单次 atomic
```

适用前提同 SFA：Cube/MTE3 交替阻塞、归约维 atomic 竞争、UB 放得下 per-blk 累加器、重算成本可接受。

## 总结

| 优化 | 方法 | 收益来源 |
|------|------|---------|
| Cube/MTE3 分阶段批量解耦 | Cube 计算与 MTE3 写回拆分阶段，head 维 UB 归约批量 atomic | 解除 Cube/MTE3 交替阻塞 + atomic 次数 -75% + dq 去 atomic |

**核心：**
- 当 Cube 累加输出与 atomic scatter 输出在同一循环体交替（MTE3 阻塞 Cube），且某归约维靠多 program atomic 竞争归约时，把计算与写回拆成分阶段批量
- Stage A 纯 Cube 累加（热循环零 MTE3），配合"一行一 program"使 dq 写无竞争 → plain store 非 atomic
- Stage B 把归约维收进单 program UB 累加器，每 blk 单次 atomic_add；atomic 次数与 program 数解耦，-75%
- 权衡：Stage B 重算 scores/P/dS + 重复 gather，换 Cube 解阻塞 + atomic 减量；重算 < atomic+stall 时净正
- B1/B2 分离避 UB 溢出，各 Stage 独立 tile；NUM_HC==1 需 HC_LOOP=2 绕 scf.For crash；NEED_CLAMP 编译期消除 select
- 不可：只加 UB 累加不改 grid（atomic 仍竞争）、Stage A 仍留 atomic（未解耦）、B1+B2 合并（UB 溢出）
- Stage B 重算成新瓶颈时，叠加 workspace-decoupling.md 物化中间量消除重算

---

## 来自 SKILL.md 的原始描述（优化点：Cube/MTE3 分阶段批量解耦优化）

**适用条件**：kernel 在同一循环体内交替进行 Cube 累加计算与 `atomic_add` scatter 写回，MTE3 带锁 RMW 阻塞 Cube 流水；且某归约维被切成多 program 并行，靠 atomic 竞争完成归约，atomic 数随 program 数爆炸。

**典型代码特征**：
```python
# 问题代码：单 pass，Cube(dq) 与 MTE3 atomic(dk/dv/dkr) 在同一循环体交替
# Grid: (B*S1 * num_g_groups,) —— head 维切多 program，atomic 竞争归约
for blk_start in range(0, topK, BLOCK_K):
    acc_dq += tl.dot(dS, k_full)                       # Cube
    tl.atomic_add(dk_ptr  + ..., dot(trans(dS), q_nope))   # RMW 夹在 Cube 间
    tl.atomic_add(dv_ptr  + ..., dot(trans(P),  do_tile))
    tl.atomic_add(dkr_ptr + ..., dot(trans(dS), q_rope))
```

**判断逻辑**：
- 检查 kernel 是否同时有 Cube 累加输出与 atomic scatter 输出，且两者在同一循环体交替
- 检查 atomic 是否为带锁 RMW、是否阻塞 Cube（profiling MTE3-bound、Cube 闲置）
- 检查某归约维是否被切多 program，靠 atomic 竞争归约（atomic 数随 program 数爆炸）
- 检查该归约维能否被单 program 串行遍历、UB 放得下 per-blk 累加器（可换 UB 累加）
- 评估 Stage B 重算/gather 成本 vs atomic+Cube-stall 成本：重算 < atomic+stall → 净收益为正
- 若命中且 UB 允许 → Stage A 纯 Cube（零 MTE3，一行一 program 使 dq 非 atomic）+ Stage B UB 归约批量 atomic
- 若重算成为新瓶颈 → 转 workspace-decoupling.md 物化中间量消除重算（可叠加）
- 若输出间循环顺序一致、无 atomic 阻塞 → 不涉及，转 pass-merge.md

**命中条件**：Cube 累加输出与 atomic scatter 输出在同一循环体交替阻塞 Cube，且某归约维靠多 program atomic 竞争归约（atomic 爆炸），该维可单 program UB 累加，重算成本可接受。

**参考文档**：`references/cube-mte3-decoupling.md`（本文档）

---
