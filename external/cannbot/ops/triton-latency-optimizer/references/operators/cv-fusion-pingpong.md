# Ascend CV 融合算子生成方法论 — Batch 流水线篇

> 基于 Sparse Flash Attention (SFA) 等 CV 融合算子优化实践总结
> 适用范围：所有需要跨 step 隐藏 Load/Compute 延迟的 Triton-Ascend CV 融合算子

---

## 1. 通用工作流抽象

CV 融合算子的本质是把计算拆成交替执行的 **Vector 阶段** 和 **Cube 阶段**，阶段之间通过 On-Chip buffer 握手。典型形态可写为：

```
V1 -> C1 -> V2 -> C2 -> V3 -> ... -> Ck -> V(k+1)
```

- **V_i**：Vector 侧阶段，负责 Load、格式转换、后处理、累加、写回 GM 等
- **C_i**：Cube 侧阶段，负责矩阵/向量密集计算
- **箭头**：阶段间握手 buffer
  - `V_i -> C_i`：数据由 Vector 产生、Cube 消费，通常放在 Cube 侧内存（如 L1）
  - `C_i -> V_{i+1}`：数据由 Cube 产生、Vector 消费，通常放在 Vector 侧内存（如 UB）

阶段数量由具体算法决定。两段式 CV 融合常见为 `V1 C1 V2 C2 V3`；更复杂的算子可能有更多段。

| 阶段 | 执行侧 | 职责 | 输出位置 |
|------|--------|------|----------|
| V_i（非最后阶段） | Vector | 为下一阶段 Cube 准备数据 | Cube 侧 On-Chip buffer |
| C_i | Cube | 计算并把结果交给下一阶段 Vector | Vector 侧 On-Chip buffer |
| V_last | Vector | 最终处理并写回 GM | GM |

## 2. 为什么需要 Batch 流水线

baseline 每个 step 按 `V1 -> C1 -> V2 -> C2 -> ... -> V_last` 串行执行，Vector 的 Load/格式转换和 Cube 的 Compute 无法重叠。当 `num_steps >= 2` 且 On-Chip 内存足够时，可以通过 **Batch 流水线** 把同类操作集中发射，让相邻 step 的各阶段重叠执行。

与无流水线的 2-scope 基线相比，Batch 流水线把 Vector 和 Cube 各自的工作拆成独立批次，消除了“当前 step 的 Accum 阻塞下一步 Load”的依赖，从而实现更紧密的 V/C 重叠。

> ⚠️ **常见误区：ping-pong buffer + 串行程序序 = 负优化**
>
> 只把 buffer 翻倍、但保持每 step `V1->C1->V2->C2->V3` 串行程序序，实测比 2-scope 基线**慢约 7%**（FA forward，N=4096）。
> 根因是依赖链没有改变：串行结构里 `C2(i)`（PV）必须 wait `p_l1(i) ready`，它挡在程序序前方，`C1(i+1)`（QK）无法提前发射，`qk_ub` 双 buffer 形同虚设；而同步指令从约 6 条/step 涨到约 12 条/step，纯增开销。
> **收益来自第 4 节的 Batch 发射顺序**（同类操作集中发射），ping-pong buffer 只是使能条件。不要分"先 ping-pong、再 Batch"两步落地，中间态就是负优化。

## 3. Buffer 分配策略

对每条握手链路都建立 Ping/Pong buffer：

```
# V_i -> C_i 握手 buffer（放在 Cube 侧内存，如 L1）
buf_v1c1_ping, buf_v1c1_pong
buf_v2c2_ping, buf_v2c2_pong
...

# C_i -> V_{i+1} 握手 buffer（放在 Vector 侧内存，如 UB）
buf_c1v2_ping, buf_c1v2_pong
buf_c2v3_ping, buf_c2v3_pong
...
```

### 3.1 共享 buffer 的优化

如果两个相邻阶段消费的是同一份物理数据（例如 K 和 V 来自同一个 `kv` tensor），不要分别为它们分配独立 buffer，而是让两个 Cube 计算阶段共用同一个 buffer：

```
# ❌ 浪费 L1
k_l1_ping/pong  +  v_l1_ping/pong  +  p_l1_ping/pong

# ✅ 共享后节省约 1/3 L1
kv_l1_ping/pong  +  p_l1_ping/pong
```

Batch 流水线本身会让 On-Chip 占用接近翻倍，共享 buffer 是控制内存压力的关键手段。

## 4. Batch 流水线实现（PIPE_STAGES > 1）

把 `num_steps` 按 `PIPE_STAGES` 分成若干 batch，每个 batch 内把同类操作集中执行：

```
Vector:  Load batch -> Process batch -> Accum batch
Cube:    QK batch   -> PV batch
```

### 4.1 代码骨架

```python
PIPE_STAGES: tl.constexpr = 2

with al.scope(core_mode="cube"):
    # 预释放 p_l1，让 vector 可以先写（次数 = PIPE_STAGES）
    prefree_p_l1()

    for batch_start in range(0, num_steps, PIPE_STAGES):
        batch_size = min(PIPE_STAGES, num_steps - batch_start)

        # Phase 1：把本 batch 的 QK 全部算完
        for i0 in range(batch_size):
            wait kv ready + qk free
            qk = tl.dot(q, trans(kv))
            fixpipe -> qk_ub_ping/pong
            set qk ready

        # Phase 2：把本 batch 的 PV 全部算完
        for i0 in range(batch_size):
            wait p_l1 ready + pv free
            pv = tl.dot(p, v)
            set p_l1 free
            fixpipe -> pv_ub_ping/pong
            set pv ready

    postwait qk free / pv free（各 PIPE_STAGES 次）

with al.scope(core_mode="vector"):
    # 预释放 qk_ub / pv_ub，让 cube 可以先写（次数 = PIPE_STAGES）
    prefree_qk_pv_ub()

    for batch_start in range(0, num_steps, PIPE_STAGES):
        batch_size = min(PIPE_STAGES, num_steps - batch_start)

        # Phase 1：批量 Load KV
        for i0 in range(batch_size):
            load KV_i -> kv_l1_ping/pong
            set kv ready

        # Phase 2：批量 Softmax + 写 P_l1
        for i0 in range(batch_size):
            wait qk ready
            softmax -> p_sub
            set qk free
            wait p_l1 free
            copy p -> p_l1_ping/pong
            set p_l1 ready
            # alpha 按 parity 落盘，供 Accum batch 取回（loop-carried，最易写错）
            if i0 % 2 == 0: alpha_ping = alpha
            else:           alpha_pong = alpha

        # Phase 3：批量 Accum
        for i0 in range(batch_size):
            wait pv ready
            acc = acc * alpha_ping/pong + pv   # 按 i0 parity 取回对应的 alpha
            set pv free

    postwait p_l1 free（PIPE_STAGES 次）
```

### 4.2 优点

- 同类操作连续发射，流水线更饱满
- Vector 的 Load batch 可与 Cube 的 QK batch 重叠
- Vector 的 Process batch 可与 Cube 的 PV batch 重叠
- 进一步隐藏 Load 到 Compute 的延迟

### 4.3 缺点

- 代码复杂度显著增加
- 需要更精细的 free/ready 信号管理
- 对编译器调度和内存容量的要求更高
- 同步指令数翻倍（约 12 条 sync/step，基线约 6 条）；重叠窗口小（step 数少、单步计算量小）时可能净负收益

### 4.4 实验验证（SFA 前向，PIPE_STAGES=2）

- 实现：沿用 ping/pong buffer 与 ready/free 信号，将 Vector 的 Load/Softmax/Accum 和 Cube 的 QK/PV 分别改为 batch 循环。
- 精度：5 个典型 shape 前反向全部通过。
- 性能：相对 2-scope 基线提升 **1.11x ~ 1.17x**。

结论：当 `num_steps >= 2` 且 2-scope 基线已经收敛后，Batch 流水线是当前最有效的进一步压榨手段。**该结论主要针对 4-phase 链（如 FA forward）；对于 5-phase V-headed 链（如 SFA backward），见 §4.5。**

## 4.5 当 Batch 流水线不适用：T0–T5 逐拍交错

Batch 流水线并非万能。当算子的 phase 链是 **V-headed 5-phase**（首尾都是 Vector phase）时，Batch 流水线会在 batch 边界形成硬同步，反而回退。

### 4.5.1 失效根因：BWD 与 FA forward 的 phase 结构不同

```
FA forward:  C(QK) → V(Softmax) → C(PV) → V(Accum)              # 2C + 2V
SFA BWD:     V(Load) → C(S/dP) → V(Process) → C(dQdVdK) → V(Accum)  # 2C + 3V
```

BWD 多出 1 个 Vector phase，且第一个 phase 就是 Load。这带来两个致命问题：

1. **batch 边界硬同步**：下一 batch 的 `C1(k+1)` 必须等 `sig0`（K ready），而 `sig0` 由 `V1(k+1)` 设置；`V1(k+1)` 在 Batch 结构里排在 `V3(k)`（Accum）之后，`V3(k)` 又必须等 `C2(k)` 完成。依赖链绕成一圈：
   ```
   C1(k+1) ← V1(k+1) ← [程序序] V3(k) ← [sig8] C2(k)
   ```
   每个 batch 边界两侧都被迫汇合，Load 窗口暴露，cube 空转。

2. **K 被 C1、C2 两次消费**：`k_l1` 既给 `S/dP` dot 用，又给 `dQ=dS·K` 用，free 信号只能放在 `C2` 之后。这进一步把下一步的 Load 锁死在当前 batch 最后一个 cube phase 后面。

因此 BWD 的 Batch 流水线实现常见结果：精度正确，但性能相对已开 `multibuffer=True` 的基线**净负**。

### 4.5.2 T0–T5 调度模板（5-phase V-headed 链）

把"一个 step 内五 phase 串行"改成"vector 领先 cube 一拍的 6 拍循环"：

```
T0: V1_0
T1: V1_1, C1_0
T2: C1_1, V2_0
T3: V2_1, C2_0
T4: C2_1, V3_0
T5: V3_1
```

各符号含义（以 SFA BWD 为例）：

| 符号 | 执行侧 | 职责 | 输出位置 |
|---|---|---|---|
| `V1_i` | Vector | Load K tile (gather) → L1 | `k_l1_ping/pong` |
| `C1_i` | Cube | `S=Q·Kᵀ`, `dP=dO·Kᵀ` → fixpipe | `s_ub_ping/pong`, `dp_ub_ping/pong` |
| `V2_i` | Vector | softmax/dS：`P=softmax(S)`, `dS=P·(dP−delta)` → copy | `p_l1_ping/pong`, `ds_l1_ping/pong` |
| `C2_i` | Cube | `dQ=dS·K`, `dV=Pᵀ·dO`, `dK=dSᵀ·Q` → fixpipe | `dq_ub_ping/pong`, `dv_ub_ping/pong`, `dk_ub_ping/pong` |
| `V3_i` | Vector | 累加 `dQ`，`atomic_add(dV+dK)` | GM |
| `_0/_1` | — | ping / pong buffer parity | — |

T0/T5 是 ramp-up/ramp-down，只有单侧工作；**T1–T4 每拍 cube 和 vector 同时有活干**。

### 4.5.3 信号语义（与 Batch 版复用同一套信号号）

```
0 : V → C   K block ready in L1
1 : C → V   S/dP ready in UB
2 : V → C   S/dP UB free
4 : V → C   P/dS ready in L1
6 : C → V   L1 (k_l1/p_l1/ds_l1) free
8 : C → V   dq/dv/dk UB ready
10: V → C   dq/dv/dk UB free
```

与 Batch 版的本质区别：**set/wait 点不再按 batch 集中，而是按时间步 T0–T5 分散**。例如：

- `T0 V1_0` 完成即 `set 0`，供 `T1 C1_0` 使用；
- `T1 C1_0` 完成即 `set 1`，供 `T2 V2_0` 使用；
- `T2 V2_0` 完成即 `set 4`，供 `T3 C2_0` 使用；
- `T3 C2_0` 完成即 `set 8`，供 `T4 V3_0` 使用；
- `T4 V3_0` 完成即 `set 10`，供下一轮 `C2` 复用 UB。

同样需要预释放（prefree）和收尾 wait（post-wait）来平衡首尾的 ping/pong buffer。

### 4.5.4 代码骨架（Vector scope）

```python
for batch_start in range(0, NUM_BLOCKS, PIPE_STAGES):
    has_pong = (batch_start + 1) < NUM_BLOCKS

    # T0: V1_0 - Load K block batch_start into ping L1
    _load_k_tile_bwd(..., k_l1_ping, ...)
    sync_set(0)  # V->C K ready

    # T1: V1_1 - Load K block batch_start+1 into pong L1
    if has_pong:
        _load_k_tile_bwd(..., k_l1_pong, ...)
        sync_set(0)

    # T2: V2_0 - Softmax/dS for block batch_start using ping UB
    sync_wait(1)  # C->V S/dP ready
    _softmax_ds_bwd(s_ub_ping, dp_ub_ping, p_l1_ping, ds_l1_ping, ...)
    sync_set(2)   # V->C S/dP UB free
    sync_wait(6)  # C->V L1 free
    sync_set(4)   # V->C P/dS ready

    # T3: V2_1 - Softmax/dS for block batch_start+1 using pong UB
    if has_pong:
        sync_wait(1)
        _softmax_ds_bwd(s_ub_pong, dp_ub_pong, p_l1_pong, ds_l1_pong, ...)
        sync_set(2)
        sync_wait(6)
        sync_set(4)

    # T4: V3_0 - Accumulate for block batch_start using ping UB
    sync_wait(8)  # C->V dq/dv/dk ready
    _accumulate_dq_dv_dk_bwd(dq_ub_ping, dv_ub_ping, dk_ub_ping, ...)
    sync_set(10)  # V->C dq/dv/dk UB free

    # T5: V3_1 - Accumulate for block batch_start+1 using pong UB
    if has_pong:
        sync_wait(8)
        _accumulate_dq_dv_dk_bwd(dq_ub_pong, dv_ub_pong, dk_ub_pong, ...)
        sync_set(10)

# post-wait: 等 cube 释放 k_l1/p_l1/ds_l1
sync_wait(6); sync_wait(6)
```

Cube scope 与 Vector scope 对称，按 T0–T5 的 cube 侧时间表（`T1 C1_0`, `T2 C1_1`, `T3 C2_0`, `T4 C2_1`）放置 `sync_wait(0)`、`sync_wait(4)+sync_wait(10)` 和 `fixpipe`/`sync_set`。

### 4.5.5 选择 Batch 流水线还是 T0–T5？

| 算子 phase 结构 | 推荐模式 | 理由 |
|---|---|---|
| 4-phase `C→V→C→V`（如 FA forward） | **Batch 流水线** | 2C+2V 对齐，cube 领先 vector 软边界 |
| 5-phase `V→C→V→C→V`（如 SFA BWD） | **T0–T5 逐拍交错** | vector 领先一拍，拆掉硬边界 |
| 基线已开 `multibuffer=True` | 优先 **T0–T5** 或先做对照实验 | compiler 已覆盖迭代间重叠，Batch 可能只增同步 |
| d 很大导致 C2 三个 dot 远长于 V phases | 可能仍回退 2-scope 基线 | 短边等长边，空转占比高；d=512 常见回退 |

### 4.5.6 实验验证（SFA BWD，T0–T5 vs Batch 流水线）

- 实现：T0–T5 BWD 参考实现代码
- d=256 典型 shape：T0–T5 相对 2-scope 基线 **2.34x ~ 2.61x**；同目录 Batch 流水线版本约等于基线。
- d=128/d=512：T0–T5 收益收窄或回退，建议 d>256 直接回退 2-scope 基线。
- 精度：与 PyTorch golden 对比，阈值 1e-1 下全过。

结论：**BWD 算子不要套用 FA forward 的 Batch 流水线模板，应直接使用 T0–T5；是否值得启用取决于 head_dim 和 phase 时长匹配度。**

## 5. 同步信号设计

需要保证：**同一 buffer 的写必须发生在读之前，读完成后才能被下一次写覆盖**。

### 5.1 方案 1：独立信号组（逻辑清晰）

为 Ping/Pong 各维护一组信号：

```
sig_V1C1_ping, sig_C1V2_ping, sig_V2C2_ping, sig_C2V3_ping, ...
sig_V1C1_pong, sig_C1V2_pong, sig_V2C2_pong, sig_C2V3_pong, ...
```

优点：信号语义简单，不易跨 step 误触发。  
缺点：信号数量随 buffer 数量线性增长，可能触及硬件信号上限。

### 5.2 方案 2：Ready + Free 信号（更省信号）

除了“数据已准备好”的 ready 信号，再增加“buffer 已被消费完、可以覆盖”的 free 信号：

```
# Vector 产生数据后 set ready
sig0: V1 done  (kv_l1 ready)
sig4: V2 done  (p_l1 ready)

# Cube 消费完后 set free
sig2: C1 done  (qk_ub free)
sig6: C2 done  (p_l1 free)

# Cube 产生数据后 set ready
sig1: C1 done  (qk_ub ready)
sig8: C2 done  (pv_ub ready)

# Vector 消费完后 set free
sig10: V3 done (pv_ub free)
```

优点：
- 相同物理含义的信号可以复用，不必为 Ping/Pong 各配一套
- Producer 收到 free 后即可提前开始下一轮写入，进一步重叠

缺点：
- 需要额外处理首轮的“预设 free”和末轮的“收尾 wait”
- 一旦 free/ready 配对写错，容易出现读写冲突或死锁

## 6. 单 step Fallback

分两种情况，区别对待：

- **`num_steps` 为编译期常量（如非 causal）且 == 1**：必须单独实现无 ping-pong 的静态分支。原因不只是"Batch 流水线没有意义、增加内存和同步开销"——实测静态单迭代循环 + 死 pong 分支会直接触发 bishengir 编译器崩溃（`dyn_cast on a non-existent value`），fallback 是编译通过的必需品，不是性能选项。分支条件用 constexpr 表达式，例如：

  ```python
  if (STAGE != 1) and ((N_CTX + BLOCK_N - 1) // BLOCK_N == 1):
      # 直接走 V1 -> C1 -> V2 -> C2 -> V3，不使用 Pong buffer
  else:
      # 走 Batch 流水线
  ```

- **`num_steps` 为运行时值（如 causal 首块）**：无需 device 侧分支，直接走 Batch 路径即可。预释放/post-wait 各 PIPE_STAGES 次时，每条信号的 set 数 = wait 数 = PIPE_STAGES + num_steps，对 `num_steps == 1` 天然平衡，不会死锁也不会多等。

## 7. Batch 流水线下的 Tiling 再评估

引入 Batch 流水线后，必须重新评估 tiling（详见 `references/operators/cv-fusion-tiling.md`）：

- **内存占用接近翻倍**：按 tiling 文档第 4 节估算后乘以 ~2，确认仍低于 L1/UB 上限。
- **可尝试更大的 `BLOCK_N`**：循环次数减少后，即使单块数据量变大，也可能整体更快。
- **head padding 会增加内存和循环**：必要时让 `BLOCK_H` 对齐实际维度。
- **Batch 流水线不改变 tiling 结论**：原本因内存溢出无法使用的 tile 仍不能使用。
- **编译器选项**：某些平台需配合 `vf_merge_level=1`，但可能带来轻微性能波动，需实测。

详细估算模板和候选验证流程见 `references/operators/cv-fusion-tiling.md` 第 4-6 节。

## 8. 地址冲突避免原则

- **Ping/Pong 完全分离**：任何时刻 Cube 读的 buffer 与 Vector 写的 buffer 不能是同一个物理空间。
- **同 step 同 buffer**：C_i 的输出和后续 V_{i+1} 的输入必须绑定到同 step 的同一组 buffer。
- **最终写回顺序不能乱**：V_last 写 GM 的顺序必须和 step 顺序一致，不能为了重叠而破坏结果正确性。
- **free 信号必须配对**：每 set 一次 ready，必须对应一次 consumer 的 wait；每 set 一次 free，必须对应一次 producer 的 wait。

## 9. 编译器约束（平台实测）

以下约束在 Ascend950PR + triton 3.2.0 + CANN 9.0.0（bishengir）上实测，违反即编译崩溃（`ConvertLinalgRToBinary` / `hivm-graph-sync-solver` 阶段 SIGABRT：`dyn_cast on a non-existent value`）或编译失败：

1. **epilogue 必须留在 vector scope 内部**：归一化（`acc / l_i`）、LSE 计算、O 写回 GM 等收尾逻辑移到 scope 外时，在单迭代配置（`num_steps == 1`）下触发编译器崩溃。fallback 分支和 Batch 分支各自的 epilogue 都放在自己的 vector scope 内。
2. **helper `@triton.jit` 函数不能在运行时 `if i % 2` 分支内接收不同的 ping/pong buffer**：编译器无法稳定 lowering，QK/PV matmul（load + dot + fixpipe）必须在两个分支内分别内联。
3. **静态单迭代循环 + 死 pong 分支会崩溃**：见第 6 节，`num_steps` 为 constexpr 1 时必须用静态分支走 fallback，让 ping-pong 路径整体被剪枝。
4. **环境 API 名称**：本环境为 `bl.alloc` / `al.ascend_address_space.UB|L1` / `al.scope`（小写）；`bl.allocate_local_buffer` / `al.address_space` / `al.Scope` 不存在。

## 10. 验证要点

1. **精度**：重叠后计算逻辑、中间结果顺序、最终写回顺序必须和 baseline 一致。
2. **同步**：通过日志或工具确认每个 step 的读写没有冲突。
3. **性能**：用 `msprof op` 对比 Batch 流水线与基线，确认 latency 下降。
4. **内存**：确认 doubling 后的 On-Chip 占用没有超过硬件上限。
5. **Batch 流水线额外验证**：
   - 确认 `alpha` 等 loop-carried 状态在 Process batch 与 Accum batch 之间正确传递；
   - 确认单步 fallback 与多步 batch 路径精度一致；
   - 确认奇数 `num_steps`（batch 尾批 `batch_size=1`）精度与偶数步一致；
   - 确认运行时 `num_steps == 1`（如 causal 首块）直接走 Batch 路径不死锁、精度正确；
   - 使用 `msprof op` 对比 2-scope 基线与 Batch 流水线，确认 Task Duration 下降。
6. **平台 caveat**：`N_CTX == BLOCK_N`（K/V 整体单块）这类退化配置在基线上即可能存在非确定性精度错误（既有问题，非优化引入）；评测时应单独标注，不计入优化回归。

## 11. Batch 流水线实验记录（SFA 前向）

基于 `PIPE_STAGES=2` 的 Batch 流水线，沿用 ping/pong buffer 与 ready/free 信号：

| shape (b,m,h,d,n,topk) | Batch Pipeline (us) | 2-scope 基线 (us) | 原始 GM-buffer (us) | speedup vs 基线 |
|------------------------|---------------------|-------------------|---------------------|-----------------|
| 1,4096,32,512,4096,128 | 1704.18             | 1965.58           | 3003.84             | **1.15x** |
| 1,4096,32,512,4128,160 | 2542.62             | 2823.07           | 3958.28             | **1.11x** |
| 1,4096,32,512,5120,640 | 7488.22             | 8801.54           | 14760.51            | **1.17x** |
| 1,4096,64,512,5120,640 | 14635.96            | 17181.01          | 28992.44            | **1.17x** |
| 1,1,16,128,256,640     | 33.67               | 41.87             | 60.88               | **1.24x** |

结论：Batch 流水线在 5 个典型 shape 上均稳定优于 2-scope 基线，精度不变，是当前 SFA 前向 CV 融合的推荐调度模式。

## 12. 推荐实践

1. **先收敛 2-scope 基线**：所有 CV 通道都走 On-Chip，scope 已合并，精度全过。
2. **判断 phase 结构再选调度模式**：
   - 4-phase `C→V→C→V`（如 FA forward）→ 用 **Batch 流水线**（§4）。
   - 5-phase `V→C→V→C→V`（如 SFA BWD）→ 用 **T0–T5 逐拍交错**（§4.5），不要套用 Batch 流水线。
3. **每次改动后重新跑精度 + msprof**：任何重叠调度都改变了数据流时序，必须全量验证。
4. **保留单 step fallback**：`num_steps == 1` 时仍走简单路径，避免无谓开销。
5. **基线已开 auto-multibuffer 时先做对照**：若编译器已经自动覆盖迭代间重叠，Batch 流水线可能只增加同步、不增加重叠窗口；此时 T0–T5 或保持基线更优。
6. **d 很大时准备回退**：当 C2 等长 phase 导致短边空转占比高（如 d=512），直接回退 2-scope 基线往往是稳妥选择。

---

*参考实现（Batch 流水线）：SFA 前向 Batch 流水线代码*
*参考实现（T0–T5 BWD）：SFA BWD T0–T5 逐拍交错代码*
*性能报告（Batch 流水线）：SFA 前向 Batch 流水线性能报告*
