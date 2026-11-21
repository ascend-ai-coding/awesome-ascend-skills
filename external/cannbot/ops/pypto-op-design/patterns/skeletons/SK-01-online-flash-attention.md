---
type: pattern/skeleton
title: Online Flash Attention
description: KV 分块循环中的在线 softmax 注意力计算骨架。
tags:
- attention
flow_pattern:
- C1
- V1
- C2
examples:
- FA MHA
- FA Score
- BSA
- PageAttn
- SparseAttn
---

## SK-01: Online Flash Attention

**适用场景**: 需要对 KV 序列分块迭代、使用 online softmax 累积的注意力计算。

**CV 排布**: C1(QK^T) → V1(Softmax) → C2(PV) **在 KV tile 循环内重复**，形成 CVC... 循环模式。

展开因子候选为 8、4、2、1，初始设计每次只选一个值，并验证不能整除时的处理；其余值分别作为调优候选。

### 骨架结构

```python
@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {-1: N},
        "vec_nbuffer_setting": {-1: M},
        "cube_nbuffer_setting": {-1: K},
    },
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
    },
)
def flash_attention_kernel(Q, K, V, output, accumulators, seq_lens, ...):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes(c1_global, k_global, c2_global)
    pypto.set_vec_tile_shapes(v1_global, v2_global)

    for b_idx in pypto.loop(batch_size):                    # Loop: Batch
        seq_q, seq_kv = dynamic_seq_lens(...)
        q_tiles = ceildiv(seq_q, Q_TILE)
        kv_tiles = ceildiv(seq_kv, KV_TILE)

        for h_idx in pypto.loop(num_heads):                 # Loop: Head
            for q_idx in pypto.loop(q_tiles):               # Loop: Q Tile
                oi = pypto.tensor(...)  # accumulators
                li = pypto.tensor(...)
                mi = pypto.tensor(...)

                q_tile = view_or_gather(Q, ...)

                for kv_idx in pypto.loop(kv_tiles):         # Loop: KV Tile (C1-V1-C2)
                    k_tile = view_or_gather(K, ...)
                    v_tile = view_or_gather(V, ...)

                    # C1: QK^T MatMul
                    pypto.set_cube_tile_shapes(c1_tiles)
                    scores = pypto.matmul(q_tile, k_tile, ...)

                    # V1: Online Softmax
                    pypto.set_vec_tile_shapes(v1_tiles)
                    ... online softmax computation ...

                    # V: Quant (optional)
                    p_quant = pypto.cast(p, ...)

                    # C2: PV MatMul
                    pypto.set_cube_tile_shapes(c2_tiles)
                    oij = pypto.matmul(p_quant, v_tile, ...)

                    # V: Online Accumulator Update (three-way branch)
                    ... online softmax accumulator update ...
```

### 关键编码特征

| 特征 | 规则 |
|------|------|
| **TileShape 切换** | C1/V1/C2 各阶段前必须 `set_cube/vec_tile_shapes` |
| **累积器位置** | 在 Q tile loop 内、KV tile loop 外分配 |
| **三路分支** | `is_loop_begin` + `is_loop_end` 组合判断 |
| **数据加载** | `view` + `valid_shape` 处理边界 |
| **数据存储** | `assemble` 到全局输出张量 |
| **FP32 累积** | O/L/M 全程 FP32，仅最终 cast 输出 dtype |
| **Paged KV** | 替换 K/V 加载为 AT-17 Block Table Gather |
| **FP8 量化** | 在 V1→C2 之间插入 AT-08 (P 量化)，C2 后插入 AT-06 (反量化) |

### 变体矩阵

| 变体 | Loop 4 策略 | Mask | KV 来源 | 量化 |
|------|-----------|------|---------|------|
| FA MHA | `pypto.loop` | 无/因果 | 连续 view | 无 |
| FA Score | `loop_unroll(unroll_list=[4])` | 掩码张量 | 连续 view | 无 |
| BSA | `pypto.loop` | 稀疏掩码 | 紧凑 gather | 无 |
| PageAttn FP8 | `loop_unroll(unroll_list=[8])` | 无 | AT-17 分页 gather | FP8 |
| Sparse Attn | `pypto.loop` | topk 索引 | index_select | 无 |

### 开箱性能优化提示

> 实证来源：`models/deepseek_v4/sparse_compress_flash_attention_impl.py:141-149`、`models/deepseek_v4/win_attention_impl.py:54-59`、`models/deepseek_v4/compress_flash_attention_impl.py:109-117`

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pass_options.cube_l1_reuse_setting` | 必配 | `{-1: 2~4}` 或 `{-1: 2, 0: 8}` 分阶段 | C1(QK^T) 与 C2(P@V) 的权重/激活 L1 复用，FA 的核心 cube 优化 |
| `pass_options.cube_nbuffer_setting` | 推荐 | `{-1: 2}` 起步，head_dim 大时升 `{1: 2}` | Cube 双缓冲，掩盖 K/V 加载延迟 |
| `pass_options.vec_nbuffer_setting` | 必配 | `{-1: 4}`，分页/稀疏可升 `{-1: 6}` | Online softmax 阶段向量算子多，nbuffer 提升吞吐 |
| `runtime_options.stitch_function_max_num` | 必配 | `128` | 子图拼接上限，FA 经验值 |
| `runtime_options.device_sched_mode` | 推荐 | `1` | 启用设备侧并行调度 |
| `runtime_options.ready_on_host_tensors` | 推荐 | `["block_table", "kv_act_seqs"]` 等控制流读值 tensor | 控制流 tensor host 提前发射，消除调度等待气泡；paged/IFA 控制流必配 |
| `pypto.experimental.set_operation_options(combine_axis=True)` | 必配 | 在 jit 函数体首行 | 尾轴 broadcast 内联 brcb（如 `[M,N]*[M,1]`、`[M,N]/[M,1]`），见 F-15 |
| KV tile 循环展开 | 按变体选择单值 | PageAttn FP8 候选：8、4、2、1；FA Score 候选：4、2、1 | 每次采用一个因子，核对实际 KV 长度和余数处理 |
| TileShape 分阶段切换 | 强制 | C1 / V1 / C2 各自 `set_*_tile_shapes` | 不切换会导致 18000 表达式上限突破 |
| Online softmax 累积器 dtype | 强制 | 全程 FP32 | mi/li/oi 必须 FP32，最终仅 cast 输出 |
| Paged KV 模式 | 配套 | AT-17 Block Gather + `view valid_shape` | 避免单 block gather kernel 调用 |
| 强制合图 | 按场景 | softmax vec 链 `sg_set_scope=2` / 状态更新 `=1` / Cube 前后 `=-1`（范本见 AT-21）| 避免Vector图过小，调度缝隙变得多 |

**该骨架特有的性能方向**：**C1-V1-C2 三阶段 nbuffer 解耦** + **online softmax 累积器零冗余 cast**。瓶颈通常出在 V1 的 amax/exp/sum 流水，优先提升 `vec_nbuffer_setting`；KV 是稀疏/分页时优先配 `gather_in_ub`。

### 结构性能特征（调度等待主导时的结构诊断）

调度等待占总耗时较大、任务数量较多时，检查每个任务处理的数据量及 root 数，评估增大 tile 或减少计算趟数。等待源于跨 loop 累积器串行依赖且 head/batch 存在无依赖轴时，按 AT-23 多链交错并行化。原因仍需结合性能报告确认。

**结构性能特征表**：

| 维度 | 估算方式 | 检查内容 |
|---|---|---|
| root 数 | `batch × heads × ceildiv(seq_len_q, Q_TILE)` | 结合设备并行度和调度耗时判断 |
| 任务粒度 | 单 root 计算量 = `Q_TILE × seq_len_k × head_dim` | 权衡任务开销、资源占用与并行度 |
| 常驻 UB 状态 | 显式 buffer：`Q_TILE × head_dim × 4 B`（oi） | ≤ UB 容量 |
| 调度等待 | 从性能报告读取实际耗时 | 不仅凭 root 数预测等待比例 |

**性能导向形态（适用条件不满足时回退安全基线形态）**：

适用条件（全部满足才可用，任一不满足回退上方安全基线形态）：
1. k 侧段长编译期可确定为统一常量（varlen k 非均匀不适用）。
2. head_dim ∈ {64, 128}。
3. root 数按上表 ∈ [16, 256]。
4. 性能目标与基线差距 > 5×。

结构不变量（违反任一，性能导向形态失效）：
1. **Q_TILE = seq_len_q / 2**：每 (batch, head) 至多 2 个 Q 分块；禁用固定小 tile 基线（128/320）直接套用。
2. **k 侧循环静态展开**：k_tile_count ≤ 8 时必须用 Python `range`；禁用 `is_loop_begin` / `is_loop_end` 谓词。
3. **状态 SSA 重绑定**：首分块 `oi = oij`；禁止显式 `pypto.full` 常驻 buffer + `[:]` 写回（否则 UB 溢出回退安全基线）。
4. **softmax 链合**：必须用 `sg_set_scope(6)` 包裹 `mul→amax→sub→exp→sum→cast`，链尾复位 `sg_set_scope(-1)`。
5. **l/m 直写**：`assemble` 列写 `[total_q, num_heads]`；禁止 head-major 行写 + 转置桥。

**UB 预算强制**：设 tile 前列出常驻 buffer 字节数 ≤ UB 容量；SSA 形态无额外常驻，显式 buffer 形态按 `Q_TILE × head_dim × dtype_size` 计入。

### 反向传播结构决策（FA Grad 家族：单 Pass vs 双 Pass）

适用场景：attention 反向传播算子（输出 dQ/dK/dV）。dQ 的最优循环方向为 q-outer/kv-inner，dK/dV 为 kv-outer/q-inner；该方向冲突决定结构选型，必须在骨架匹配阶段显式决策并将结论记录到 DESIGN.md。

| 结构 | 循环组织 | 梯度写回 | softmax 统计量 | QK^T 次数 |
|------|---------|---------|---------------|-----------|
| 单 Pass（默认推荐） | `batch → head → q_tile → kv_tile` 单重嵌套，三路梯度在同一块对内完成 | `pypto.atomic_add` 直写 GM 输出，host 侧 `torch.zeros` 预清零 | 输入含 l/m 时直接消费（零重算）；无则块内在线自算 | 每 (q, kv) 块对 1 次 |
| 双 Pass（安全回退） | Pass1 q-outer 计算 dQ，Pass2 kv-outer 计算 dK/dV | UB 累积器 + 三路分支 + `assemble` 尾块写回 | Pass1 自算后经 GM scratch 传递至 Pass2 | 每块对 2~3 次（统计量趟/累加趟/Pass2 各一次） |

决策规则：

1. **默认选单 Pass**：`atomic_add` 消解循环方向冲突，消除双 Pass 的 QK^T 重算与 GM scratch 往返；FP32 输出与 varlen 动态 offset 均支持。**单 Pass 的具体实现骨架见 SK-16（Single-Pass Attention Backward）**——全动态四级 `pypto.loop` + l/m 直用 + 三路 atomic_add + sg_set_scope 分段。
2. **回退双 Pass 须写明合法理由**：平台不支持 `atomic_add`、精度语义要求确定性归约顺序、或输出 buffer 不可预清零。
3. **统计量来源联动**：签名含 l/m 统计量输入时禁止 kernel 内重算；重算会为每个 q_tile 增加一趟完整 QK^T。
4. **任务数预估**：`任务数 ≈ batch × heads × ⌈sq/Q_TILE⌉ × ⌈skv/KV_TILE⌉ × Pass 趟数`；结合实际调度开销，评估增大 tile 或减少计算趟数。

---

---

## Performance handoff

> 安全基线（cube `[128,128]×3`、vec 各轴 [16,64]）见 `constraints/tiling.md`；
> 实测性能结论归 `pypto-op-perf-tune`，本段不持有实测定值。

### 性能导向 Tile 粒度选型（attention / online softmax 家族）

适用条件：算子语义 = QK^T → 在线 softmax → P@V（MHA/GQA/cross-attention/causal 变体），且性能目标与基线差距 > 5×。不满足时回退安全基线。

1. **Q_TILE 按 root 粒度反推**：目标 root 数 ∈ [16, 256]；每 (batch, head) 至多 2 个 Q 分块（`Q_TILE = seq_len_q / 2`）。禁用固定小 tile 基线（128/320）直接套用。
2. **kv 循环静态化判据**：k_tile_count 编译期可确定且 ≤ 8 时，k 侧循环必须用 Python `range` 静态展开；禁用 `is_loop_begin` / `is_loop_end` 谓词。
3. **状态 SSA 重绑定前提**：性能导向粒度必须配合 SSA 状态管理（首分块 `oi = oij`），禁止显式 `pypto.full` 常驻 buffer + `[:]` 写回；否则 UB 溢出，回退安全基线。
4. **k 侧段长约束**：适用域 = k 侧段长编译期可确定为统一常量；varlen k 非均匀（各 k 段不等）不适用，回退安全基线 + 动态 loop。
5. **UB 预算强制**：设 tile 前列出常驻 buffer 字节数 ≤ UB 容量；SSA 形态无额外常驻，显式 buffer 形态按 `Q_TILE × head_dim × dtype_size` 计入。
6. **任务数量**：使用上面的公式估算，结合调度耗时评估 tile 大小及计算趟数。反向计算采用单趟还是多趟，需同时满足依赖和精度要求。
