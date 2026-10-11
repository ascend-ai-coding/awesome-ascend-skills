# reduction 范式 binary-base DAG 分析和 Buffer 划分

> UB buffer 的划分方案，包括 DAG 图分析（包括VF融合）、存活节点分析、buffer 分类表。

依赖输入：[reduction-template-overview.md](../reduction-template-overview.md)、[common/sync-and-consistency.md](../../../common/sync-and-consistency.md)

## 1 DAG 分析和 Buffer 划分方案

reduction 范式的计算图按三段式划分：**preReducePhase**（reduce 前）→ **reducePhase**（reduce）→ **postReducePhase**（reduce 后）。两个桥接节点将三段联系起来：
- **preReduceResult**：preReducePhase 的输出节点，作为 reducePhase 的输入
- **reduceResult**：reducePhase 完成后的节点，作为 postReducePhase 的输入

三段各自独立分析：
- **独立的存活节点分析**：preReducePhase 和 postReducePhase 各自追踪存活节点峰值，两阶段独立分配 buffer，不复用
- **独立的 buffer 空间分配**：两阶段空间大小不同——preReducePhase 的 UB buffer 含 R 轴维度，postReducePhase 不含 R 轴（reduce 后 R 轴被归约）
- **独立的 VF 融合**：preReducePhase 和 postReducePhase 各自做 VF 融合，不跨 reducePhase 融合
- reducePhase 内部引入的节点和 buffer 详见 §1.1.2 和 §1.3。


⛔ Tuple Reduce 场景（N 次独立归约、`Process`/`ProcessGroup` 按 `processIdx` 分发）：存活节点分析**分阶段**（pre/reduce/post，Group 为 Phase 1/Phase 2）各自画**一条流**，禁止按 for 循环展开成多轮：
1. **公共步骤**（所有过程都执行）在流中只出现一次。
2. **过程独有步骤**写成 `if (processIdx == …) {…} else if …` 并列分支，分割粒度 = 归约过程（同一过程的多个分叉输出在该分支内合并 trace）；分支互斥——任一时刻只执行一个分支，各分支的 buffer 不可能同时存活。
3. **分支峰值取 max**：每个分支单独 trace 其执行时的峰值，一处 if/else 的峰值 = 各分支峰值的 max，禁止相加；互斥分支共用同一份物理 UB，不按分支单独记账。
4. **存活期照常判定**：分支的中间 buffer 在分支结束即释放，仅被后续公共步骤消费的产出继续存活；释放后的普通复用池 buffer 可被后续任意步骤复用，与它来自哪个分支无关。既有复用例外不变（§1.1 各阶段复用规则、§1.3 固定份数 buffer 照旧）。
5. 每个阶段一条流走完，阶段峰值 = 该阶段存活节点数；UB 份数仍按 §1.5 汇总表分阶段相加。

**注**：以上规则对 L、P 均适用；"共用物理 UB""记账""UB 份数"只作用于 P——L 无物理复用概念，仍按公平基线逐 op 计节点；"按代码结构"指代码结构镜像算子公式的公共/独有子图划分，不违背 L 的纯拓扑定义。


**从GM视角看，不考虑UB，三段式数据流概览**：

```
┌─ preReducePhase ──────────────────────────────────────┐
│  GM_x1 --> preNode --PreElewise--> preReduceResult
│                            |-- preReduceOut,可选 --> GM_y1
└───────────────────────────────────┬───────────────────┘
                                    ↓
┌─ reducePhase ─────────────────────────────────────────┐
│  preReduceResult --> reduce --> reduceResult
└───────────────────────────────────┬───────────────────┘
                                    ↓
┌─ postReducePhase ─────────────────────────────────────┐
│  GM_x2 (可选)--> postInBuf --|
│  reduceResult --------------|--PostElewise--> outNode --> GM_y2
└───────────────────────────────────────────────────────┘
```

> **preReduceOut**：preReducePhase 计算流中任意阶段的中间值做为输出，不经过 reducePhase。
>
> **postReducePhase 也可能有其他 GM 输入**：postReducePhase 除了接收 reduceResult 的结果外，还可能有自己的 CopyIn 节点从 GM 搬入额外的输入。

> **preReducePhase buffer / postReducePhase buffer 是分类名**，各自可包含多个物理 buffer。例如 preReducePhase buffer 包含 `preInBuf`（搬入buffer）、`preReduceResult`（fp32 中间结果）、`preReduceResultTail`（二分尾块）等；postReducePhase buffer 包含 `outBuf`（输出buffer）、`postInBuf`（post_reduce_input 搬入）等。具体包含哪些 buffer 取决于算子的计算图。

### 1.1 存活节点分析

分两层：

**逻辑存活节点 L**：纯拓扑分析，不考虑 VF、融合、寄存器。画出 preReducePhase / postReducePhase 各自的完整计算图，按最优调度逐步追踪存活 tensor 数，取峰值。L 对给定计算图是固定的，不随硬件/API/实现改变。跨阶段驻留的 tensor 归属其分配阶段，不计入本阶段（如 reduceResult 驻 cacheBuf，计入 reduce 阶段）。

**物理存活节点 P**：`P = physical_min(L, API能力, 算法策略)`。P 的计数域为 UB buffer（寄存器不计入），buffer 计入其分配阶段的 P，跨阶段复用不占复用阶段的 P。P ≥ L 一般成立，VF 寄存器链融合可打破（中间值不落 UB，P < L 合法）。P 的具体值依赖 VF 融合策略，在 §1.2 中分析得出。L 是 P 的下界和收敛目标——P == L 时确认已到最优，P > L 时知道还有优化空间。


#### 1.1.1 preReducePhase 存活节点分析

**preReducePhase buffer 复用规则**：
- **preReduceResult 独占**：完全独立分配，不与 preReducePhase 内任何节点复用（含写入前）。原因：Phase A 主尾配对时 preReduceResult 要保持不动等待 preReduceResultTail，提前写入会被后续操作覆写。Group Phase 2 可跨阶段复用此物理槽
- **其余节点**：生命周期不重叠时可复用同一物理 buffer。所有 UB buffer 统一按本算子参与计算的最大 dtype 分配容量，不同 dtype 的 tensor 容量一致，跨 dtype 复用不受尺寸约束；buffer 份数同样统一覆盖全部 dtype 实例（含 Cast 中转需求，无 Cast 的 dtype 实例相应份数闲置），不按 dtype 分裂

分析方法参考 [reduction-survival-node.md](../reduction-survival-node.md)

**产出**：
1. **计算图** — 从算子公式推导 S1..Sn，标注分叉点。格式：`S<N>: <操作> [<输入> → <输出>]`，含 CopyIn 节点 + 中间计算节点 + preReduceResult + preReduceOut 节点
2. **L 调度 trace** — 逐操作追踪存活 tensor 数，每步标注执行前/中/后: 持有=[...] + 消费者推理，峰值标 `← L_pre=N 峰值`。L trace 仅基于纯数学拓扑，不考虑 VF、融合、寄存器
3. **L 结论** — 表格：L_pre 值 / 峰值出现步骤 / 瓶颈 / 不可降原因
4. **P_pre_baseline** — 不考虑 VF 融合时的基线 P：每个操作的 src/dst 各占独立 UB buffer，逐操作 trace 取峰值

#### 1.1.2 reducePhase 存活节点分析

> reducePhase仅涉及reduce，不涉及计算图的分析，下面一一说明占用的节点分析

**原则**:
- **preReduceResult**： preReduceResult 作为 reducePhase的输入，也是 preReducePhase 的输出，是同一个buffer

(1) **从UB视角看，reducePhase 需要拆出一个临时节点tmpResult做为单次reduce结果，reduceResult做为全局reduce结果**

> 受 UB 大小限制，必然存在无法在单次 UB 装载内做完整个 reduce 的场景——reduce 分多轮 for 循环做，循环间有依赖


```pseudocode
reduceResult = 0
for r in rLoops:                          // R 轴外层循环
    // 1. preReducePhase（每次迭代）

    // 2. reducePhase（每次迭代，跨循环累积）
    tmpResult = reduce(preReduce) // 每次迭代reduce
    reduceResult = reduceResult + tmpResult  // 累积历史结果

// 3. postReducePhase（循环结束后）
```

(2) **从Reduce API分析**

Reduce 高阶 API 占用的 buffer ，不依赖 VF 融合（API 详见 [reduction-highlevel-api.md](../reduction-highlevel-api.md)）：

本范式统一使用带 sharedTmpBuffer 的重载，`preReduceResultTail` 兼作 sharedTmpBuffer。固定 **3 个 buffer**：
- `preReduceResult`（输入）
- `preReduceResultTail`（兼作 sharedTmpBuffer，Phase A 配对后空闲）
- `reduceResult`（输出，复用 cacheBuf）

isReuseSource 仍按算子特征判定（true：src 调用后失效；false：src 保持不变），但不影响 buffer 份数。

（3）**总结**

以上是单纯从原始 reduce 角度分析的节点数，但为了保证精度而使用二分缓存树导致的节点变化，见 §1.3 分析


#### 1.1.3 postReducePhase 存活节点分析

**postReducePhase buffer 复用规则**：
- **reduceResult**： reduceResult 是 reducePhase 的输出，是同一个buffer
- **其余节点**：生命周期不重叠时可复用同一物理 buffer。所有 UB buffer 统一按本算子参与计算的最大 dtype 分配容量，不同 dtype 的 tensor 容量一致，跨 dtype 复用不受尺寸约束；buffer 份数同样统一覆盖全部 dtype 实例（含 Cast 中转需求，无 Cast 的 dtype 实例相应份数闲置），不按 dtype 分裂
- **preReducePhase 与 postReducePhase 独立分配**：两阶段 buffer 不复用

分析方法参考 [reduction-survival-node.md](../reduction-survival-node.md)

**产出**：
1. **计算图** — 从算子公式推导 S1..Sn，格式：`S<N>: <操作> [<输入> → <输出>]`，含 CopyIn 节点（可选）+ PostElewise + CopyOut 节点
2. **L 调度 trace** — 逐操作追踪存活 tensor 数，每步标注执行前/中/后: 持有=[...] + 消费者推理，峰值标 `← L_post=N 峰值`。L trace 仅基于纯数学拓扑
3. **L 结论** — 表格：L_post 值 / 峰值出现步骤 / 瓶颈 / 不可降原因
4. **P_post_baseline** — 不考虑 VF 融合时的基线 P


### 1.2 DAG VF融合分析

基于 §1.1 产出的计算图和 L 值，对 preReducePhase / postReducePhase 各自做 VF 融合分析，推导具体 P 值。

**VF 融合分析方法**：参考 reduction 范式的 VF 融合方法论（详见 [reduction-vf-fusion-rules.md](../reduction-vf-fusion-rules.md)）。穷举 preReducePhase / postReducePhase 计算图中所有连续 Vector 操作对，逐对判断是否可融。

跨阶段约束：preReducePhase 与 postReducePhase 不跨 reducePhase 融合 —— reducePhase 是天然断点，preReducePhase 的 PreElewise 与 postReducePhase 的 PostElewise 必须分属独立 VF 链。

**VF 融合对 buffer 的影响**：融合后中间值在寄存器内不落 UB，减少存活节点数。不融的相邻操作各占一个独立 buffer。如果融合后存活节点峰值过高，需要主动拆分 VF 降压。

#### 1.2.1 preReducePhase 产出

1. **可用 API 能力** — 表格：API | 能力 | 对物理节点的意义 | 来源。必须检查 RegBase 寄存器指令集（打开 [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) 逐条核对存在性）、Cast 可参与 VF
2. **VF 融合穷举分析** — 列出 preReducePhase 计算图中所有相邻 Vector 操作对，逐对判据，标明哪些可融、哪些断开
3. **VF 融合方案** — 基于判据结果，确定哪些操作融成一条 VF 链，哪些断开成多条
4. **{dtype} 物理 trace** — 基于 fusion 方案，每种 dtype 独立 trace，从 CopyIn S1 到 preReduceResult 每步展开，禁止写总结。每步标注执行前/中/后: 持有=[B0,B1,...] + 消费者推理，峰值标 `← P_pre=N 峰值`。Buffer 用 B0/B1/... 命名。各 dtype 峰值取最大者为统一 P_pre，不产生 per-dtype 份数
5. **结论** — 表格：P_pre 值（**包含 preReduceResult**）/ 峰值出现步骤 / 与 L_pre 的差距

#### 1.2.2 reducePhase 不需要做VF融合分析

#### 1.2.3 postReducePhase 产出

1. **可用 API 能力** — 表格：API | 能力 | 对物理节点的意义 | 来源
2. **VF 融合穷举分析** — 列出 postReducePhase 计算图中所有相邻 Vector 操作对，逐对判据
3. **VF 融合方案** — 基于判据结果，确定哪些操作融成一条 VF 链，哪些断开成多条
4. **{dtype} 物理 trace** — 基于 fusion 方案，每种 dtype 独立 trace，从 reduceResult/CopyIn 到 CopyOut 每步展开，禁止写总结。每步标注执行前/中/后: 持有=[B0,B1,...] + 消费者推理，峰值标 `← P_post=N 峰值`。Buffer 用 B0/B1/... 命名。各 dtype 峰值取最大者为统一 P_post，不产生 per-dtype 份数
5. **结论** — 表格：P_post 值 / 峰值出现步骤 / 与 L_post 的差距

### 1.3 额外buffer要求

所有 reduce 算子统一使用二分缓存树，算法详见 [reduction-binary-sum.md](../reduction-binary-sum.md)。

二分缓存树**改变**了 preReducePhase 和 reducePhase 原有的计算节点:
1. **preReduceResultTail（额外 +1）**：Phase A 需要主块和尾块配对，preReduceResult 和 preReduceResultTail 同时存活。preReduceResult 已计入 P_pre，preReduceResultTail 是额外引入的（记为 P_pre_ext，归属于 `preReducePhase` ），未计入 P_pre，单独分配。

2. **cacheBuf 兼作 tmpResult 和 reduceResult**：Reduce 直接将结果写到 cacheBuf 对应层位置（cacheBuf[levelOff]），吸收低层级时从低层位置读取、与当前层结果合并、覆写当前层。二分树的树根（reduceResult）就在 cacheBuf 的最高层，postReducePhase 读取的就是这个树根。因此 **tmpResult/reduceResult 不额外分配 buffer**，仅占用 cacheBuf 的部分。

3. **preReduceResultTail 兼作 sharedTmpBuffer**：Phase A preReduceResult 和 preReduceResultTail 合并到 preReduceResult 后，preReduceResultTail 临时空闲，兼作 Reduce API 的 sharedTmpBuffer

### 1.4 Double Buffer 决策

当前不开启double buffer.

### 1.5 强制产出清单

> agent 完成分析后，逐项核对，缺一项不算完成。

**preReducePhase**（§1.1.1 + §1.2.1）：
- [ ] 计算图（S1..Sn，标注 CopyIn/中间计算/preReduceResult/preReduceOut，标注分叉点）
- [ ] L 调度 trace（执行前/中/后: 持有=[...] + 消费者推理，标 L_pre 峰值）
- [ ] L 结论表（L_pre 值 / 峰值步骤 / 瓶颈 / 不可降原因）
- [ ] P_pre_baseline（不考虑 VF 融合的基线）
- [ ] 可用 API 能力表（API | 能力 | 对物理节点的意义 | 来源）
- [ ] VF 融合穷举分析（所有相邻操作对判据结果）
- [ ] VF 融合方案（哪些融成一条链，哪些断开）
- [ ] {dtype} 物理 trace（执行前/中/后: 持有=[B0,B1,...] + 消费者推理，标 P_pre 峰值）
- [ ] P 结论表（P_pre 值 / 峰值步骤 / 与 L_pre 的差距）
- [ ] 校验：preReduceResult 完全独立分配，不与 preReducePhase 内任何节点复用
- [ ] preReduceResultTail（P_pre_ext = 1）

**reducePhase**（§1.1.2 + §1.3）：
- [ ] isReuseSource 取值及理由
- [ ] sharedTmpBuffer = preReduceResultTail
- [ ] cacheBuf（1 份，§1.3）

**postReducePhase**（§1.1.3 + §1.2.3）：
- [ ] 计算图（S1..Sn，标注 CopyIn（可选）/PostElewise/CopyOut）
- [ ] L 调度 trace（执行前/中/后: 持有=[...] + 消费者推理，标 L_post 峰值）
- [ ] L 结论表（L_post 值 / 峰值步骤 / 瓶颈 / 不可降原因）
- [ ] P_post_baseline
- [ ] 可用 API 能力表
- [ ] VF 融合穷举分析
- [ ] VF 融合方案
- [ ] {dtype} 物理 trace（执行前/中/后: 持有=[B0,B1,...] + 消费者推理，标 P_post 峰值）
- [ ] P 结论表（P_post 值 / 峰值步骤 / 与 L_post 的差距）

**Double Buffer 决策**（§1.4）：
- [ ] 是否开启 double buffer 的结论及理由

**Buffer 产出汇总**：
- [ ] 汇总表：

| 阶段 | buffer | 份数 | 来源 |
|------|--------|------|------|
| preReducePhase | preReducePhase buffer（含 preReduceResult） | `P_pre` | §1.1.1 + §1.2.1，P_pre 包含 preReduceResult |
| preReducePhase | preReduceResultTail | P_pre_ext（固定 1） | §1.3，不计入 P_pre |
| reducePhase | cacheBuf | 1（固定） | §1.3，兼作 Reduce 输出落点和 reduceResult |
| postReducePhase | postReducePhase buffer | `P_post` | §1.1.3 + §1.2.3 |

- [ ] 总计 = `P_pre` + 1 + 1 + `P_post`


## 2 Buffer 划分产出校验

> 为了校验 agent 是否真正理解了 ub buffer 划分逻辑，给出一个复杂输入的样例。

> 严格按照 `1.5 强制产出清单` 执行。

以 `SyncBatchNormGatherStatsWithCounts` 为例。

**算子公式与输入输出**：

- **功能**：分布式 SyncBatchNorm 统计量聚合——将多设备的局部均值/方差按数据量加权，合并为全局方差，同时用指数移动平均更新 running variance。
**输入输出**：6 输入（`mean_all` 各设备均值、`invert_std_all` 标准差倒数、`count_all` 设备数据量、`mean_broadcast` 全局均值、`count_sum` 全局数据总数、`running_var` 运行时方差）+ 2 输出（`invert_std` 全局标准差倒数、`running_var_update` 更新后的运行时方差）。

- **核心公式**：

```
reduce 前（产生依赖树）：
  var_all_square_ε = (1/invert_std_all)² − ε
  mean_sub         = mean_all − mean_broadcast
  mean_var         = mean_sub²
  mean_var_sum     = var_all_square_ε + mean_var
  mean_var_count   = mean_var_sum × count_all       ← 喂给 ReduceSum axis=0

reduce 后（产生 postIn 路径）：
  var_sum            = ReduceSum(mean_var_count, axis=0)
  invert_std         = 1 / sqrt(var_sum/N + ε)      ← 输出 1
  unbiased_var       = var_sum / (N−1)
  running_var_update = (1−m)×running_var + m×unbiased_var  ← 输出 2
```

（1）preReducePhase

**计算图**（严格按公式 `var_all_square_ε = (1/invert_std_all)² − ε`，invert_std_all=1/σ，需先 Div(1,x) 还原 σ，再 Square，再减 ε）：

```
S1: CopyIn invert_std_all      [GM → invert_std_all]
S2: Cast                        [invert_std_all → invert_std_all_fp32]    // fp16→fp32
S3: Div(1, x)                  [invert_std_all_fp32 → std_all]           // 1/invert_std_all = σ
S4: Square                      [std_all → var_all_square_base]           // σ²
S5: Sub                         [var_all_square_base, ε → var_all_square_ε]  // σ² − ε（ε 标量，寄存器构造不占 UB）
S6: CopyIn mean_all             [GM → mean_all]
S7: CopyIn mean_broadcast       [GM → mean_broadcast]
S8: Sub                         [mean_all, mean_broadcast → mean_sub]     // μᵢ − μ_global
S9: Square                      [mean_sub → mean_var]                     // (μᵢ − μ_global)²
S10: Add                        [var_all_square_ε, mean_var → mean_var_sum]  // σ² + (μ−μ_global)²
S11: CopyIn count_all           [GM → count_all]
S12: Mul                        [mean_var_sum, count_all → mean_var_count]   // ← preReduceResult
```

分叉点：var_all_square_ε（S5 输出，S10 消费）、mean_var_sum（S10 输出，S12 消费）。

**L 调度 trace**（纯拓扑，不考虑 VF 融合；采用最优调度：先 mean 链（S6-S9）产出 mean_var，再 invert_std 链（S1-S5）产出 var_all_square_ε。理由：mean 链的 S8 Sub 是二元操作，执行瞬间占 mean_all+mean_broadcast+mean_sub 共 3 个；若先产 var_all_square_ε（须等 S10 才释放），此时会叠到 4。先做二元链、后做单输入链（每步只 +1），峰值压在 3）：

```pseudocode
// S6: CopyIn mean_all
// 执行前: 持有=[]
// 执行中: 持有=[mean_all]
// 执行后: 持有=[mean_all]               ← S8 会读，保留
// S7: CopyIn mean_broadcast
// 执行前: 持有=[mean_all]
// 执行中: 持有=[mean_all, mean_broadcast]
// 执行后: 持有=[mean_all, mean_broadcast]    ← S8 会读，保留
// S8: Sub → mean_sub
// 执行前: 持有=[mean_all, mean_broadcast]
// 执行中: 持有=[mean_all, mean_broadcast, mean_sub]
// 执行后: 持有=[mean_sub]               ← mean_all + mean_broadcast 无未来消费者，释放
// S9: Square → mean_var
// 执行前: 持有=[mean_sub]
// 执行中: 持有=[mean_sub, mean_var]
// 执行后: 持有=[mean_var]               ← mean_sub 无未来消费者，释放；mean_var S10 会读，保留
// S1: CopyIn invert_std_all
// 执行前: 持有=[mean_var]
// 执行中: 持有=[mean_var, invert_std_all]
// 执行后: 持有=[mean_var, invert_std_all]    ← S2 会读，保留
// S2: Cast → invert_std_all_fp32
// 执行前: 持有=[mean_var, invert_std_all]
// 执行中: 持有=[mean_var, invert_std_all, invert_std_all_fp32]
// 执行后: 持有=[mean_var, invert_std_all_fp32]    ← invert_std_all 无未来消费者，释放
// S3: Div(1,x) → std_all
// 执行前: 持有=[mean_var, invert_std_all_fp32]
// 执行中: 持有=[mean_var, invert_std_all_fp32, std_all]
// 执行后: 持有=[mean_var, std_all]          ← invert_std_all_fp32 无未来消费者，释放
// S4: Square → var_all_square_base
// 执行前: 持有=[mean_var, std_all]
// 执行中: 持有=[mean_var, std_all, var_all_square_base]
// 执行后: 持有=[mean_var, var_all_square_base]    ← std_all 无未来消费者，释放
// S5: Sub → var_all_square_ε
// 执行前: 持有=[mean_var, var_all_square_base]
// 执行中: 持有=[mean_var, var_all_square_base, var_all_square_ε]
// 执行后: 持有=[mean_var, var_all_square_ε]    ← var_all_square_base 无未来消费者，释放
// S10: Add → mean_var_sum
// 执行前: 持有=[mean_var, var_all_square_ε]
// 执行中: 持有=[mean_var, var_all_square_ε, mean_var_sum]    ← L_pre=3 峰值
// 执行后: 持有=[mean_var_sum]               ← mean_var + var_all_square_ε 无未来消费者，释放
// S11: CopyIn count_all
// 执行前: 持有=[mean_var_sum]
// 执行中: 持有=[mean_var_sum, count_all]
// 执行后: 持有=[mean_var_sum, count_all]    ← S12 会读，保留
// S12: Mul → mean_var_count (preReduceResult)
// 执行前: 持有=[mean_var_sum, count_all]
// 执行中: 持有=[mean_var_sum, count_all, mean_var_count]
// 执行后: 持有=[mean_var_count]            ← mean_var_sum + count_all 无未来消费者，释放
```

**L 结论**：

| L_pre | 峰值步骤 | 瓶颈 | 不可降原因 |
|-------|---------|------|-----------|
| 3 | S8 / S2-S5 / S10 / S12（代表性） | mean_all + mean_broadcast + 链上中间值 | 任一二元操作需 2 输入 + 1 产出 = 3（公平基线不假设 in-place） |

**P_pre_baseline**（不考虑 VF 融合）：= 4（复用池峰值 3 + preReduceResult 独立分配 1）。

**可用 API 能力**：

| API | 能力 | 对物理节点的意义 | 来源 |
|-----|------|----------------|------|
| Reg::Cast | b16↔fp32 寄存器转换 | Cast 可参与 VF，不占独立 buffer | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |
| Reg::Div | 寄存器内除法 | 可与 Cast/Square/Sub 串成 VF 链 | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |
| Reg::Mul | 寄存器内乘法（含 Square） | 可与其他 VF 操作串链 | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |
| Reg::Add | 寄存器内加法 | 可与其他 VF 操作串链 | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |
| Reg::Sub | 寄存器内减法 | 可与 Square 串成 VF 链 | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |

**VF 融合穷举分析**（按数据流相邻对）：

| 操作对 | 融后链长 ≤ 7？ | Cast 可融？ | RegBase 指令存在？ | 结论 |
|--------|----------|------------|------------------|------|
| Cast(S2) → Div(S3) | 2 ≤ 7 ✓ | ✓ | Reg::Cast ✓, Reg::Div ✓ | 可融 |
| Div(S3) → Square(S4) | 3 ≤ 7 ✓ | — | Reg::Div ✓, Reg::Mul(Square) ✓ | 可融 |
| Square(S4) → Sub(S5) | 4 ≤ 7 ✓ | — | Reg::Mul(Square) ✓, Reg::Sub ✓ | 可融 |
| Sub(S5) → Add(S10) | 5 ≤ 7 ✓ | — | Reg::Sub ✓, Reg::Add ✓ | 可融（S10 的另一输入来自链 2，方案层取 S10 独立） |
| Sub(S8) → Square(S9) | 2 ≤ 7 ✓ | — | Reg::Sub ✓, Reg::Mul(Square) ✓ | 可融 |
| Square(S9) → Add(S10) | 3 ≤ 7 ✓ | — | Reg::Mul(Square) ✓, Reg::Add ✓ | 可融 |
| Add(S10) → Mul(S12) | 2 ≤ 7 ✓ | — | Reg::Add ✓, Reg::Mul ✓ | 可融（S12 的第二输入 count_all 由 S11 CopyIn 提供，方案层取 S12 独立） |

**VF 融合方案**：
- VF 链 1：S2(Cast) + S3(Div) + S4(Square) + S5(Sub) → var_all_square_ε（链长 4 ≤ 7）
- VF 链 2：S8(Sub) + S9(Square) → mean_var（链长 2 ≤ 7）
- S10(Add)、S12(Mul) 独立

**{fp16} 物理 trace**（融合后，沿用最优调度：先 mean 链，再 invert_std 链。B0~B2 三块复用池覆盖 7 个逻辑 tensor，B3 独立承载 preReduceResult（共 4 块、8 个逻辑 tensor）。"执行中"取峰值瞬间=输入未释放+输出写入，输入仅在"执行后"其所有消费者完成时才移除，单步内不假设 dst 复用 src 的 in-place）：

```pseudocode
// S6: CopyIn mean_all
// 执行前: 持有=[]
// 执行中: 持有=[B0(mean_all)]
// 执行后: 持有=[B0(mean_all)]          ← VF 链2 会读，保留
// S7: CopyIn mean_broadcast
// 执行前: 持有=[B0(mean_all)]
// 执行中: 持有=[B0(mean_all), B1(mean_broadcast)]
// 执行后: 持有=[B0(mean_all), B1(mean_broadcast)]    ← VF 链2 会读，保留
// S8-S9: VF: Sub+Square → mean_var
// 执行前: 持有=[B0(mean_all), B1(mean_broadcast)]
// 执行中: 持有=[B0(mean_all), B1(mean_broadcast), B2(mean_var)]    ← 融掉 Sub 中间值；复用池峰值 3，P_pre = 3 + B3(preReduceResult) = 4
// 执行后: 持有=[B2(mean_var)]          ← B0+B1 所有消费者完成，释放；B2 S10 会读，保留
// S1: CopyIn invert_std_all
// 执行前: 持有=[B2(mean_var)]
// 执行中: 持有=[B2(mean_var), B0(invert_std_all)]    ← B0 复用
// 执行后: 持有=[B2(mean_var), B0(invert_std_all)]    ← VF 链1 会读，保留
// S2-S5: VF: Cast+Div+Square+Sub → var_all_square_ε
// 执行前: 持有=[B2(mean_var), B0(invert_std_all)]
// 执行中: 持有=[B2(mean_var), B0(invert_std_all), B1(var_all_square_ε)]    ← B0 仍存活（VF 输入），B1 复用承载输出，融掉 Cast/Div/Square 中间值；复用池峰值 3
// 执行后: 持有=[B2(mean_var), B1(var_all_square_ε)]    ← B0 所有消费者完成，释放；B1 S10 会读，保留
// S10: Add → mean_var_sum
// 执行前: 持有=[B2(mean_var), B1(var_all_square_ε)]
// 执行中: 持有=[B2(mean_var), B1(var_all_square_ε), B0(mean_var_sum)]    ← B2+B1 仍存活（Add 输入），B0 复用承载输出；复用池峰值 3
// 执行后: 持有=[B0(mean_var_sum)]    ← B2+B1 所有消费者完成，释放；B0 S12 会读，保留
// S11: CopyIn count_all
// 执行前: 持有=[B0(mean_var_sum)]
// 执行中: 持有=[B0(mean_var_sum), B1(count_all)]    ← B1 复用
// 执行后: 持有=[B0(mean_var_sum), B1(count_all)]    ← S12 会读，保留
// S12: Mul → mean_var_count (preReduceResult)
// 执行前: 持有=[B0(mean_var_sum), B1(count_all)]
// 执行中: 持有=[B0(mean_var_sum), B1(count_all), B3(preReduceResult)]    ← B0+B1 仍存活（Mul 输入），B3 独立分配承载输出；P_pre = 复用池峰值 3 + B3 = 4
// 执行后: 持有=[B3(preReduceResult)]    ← B0+B1 所有消费者完成，释放；B3 独立分配，不复用
```

物理 buffer 复用映射：

| 物理块 | 依次承载 | 复用合法性 |
|--------|---------|-----------|
| B0 | mean_all → invert_std_all → mean_var_sum | 三者生命周期不重叠 |
| B1 | mean_broadcast → var_all_square_ε → count_all | 三者生命周期不重叠 |
| B2 | mean_var | mean_var 在 S10 消费后释放 |
| B3 | preReduceResult | 独立分配，不与任何节点复用 |

**P 结论**：

| P_pre | 峰值步骤 | 与 L_pre 差距 |
|-------|---------|-------------|
| 4 | S8-S9 / S2-S5 / S10（复用池 3 + B3 全程驻留） | +1（P_pre = 复用池峰值 3 + 独占 B3；与 L_pre=3 的差距来自 preReduceResult 独占不复用） |

P_pre 包含 preReduceResult，不包含 preReduceResultTail（见 §1.3）。

（2）reducePhase

**isReuseSource**：true。理由：Reduce 后 preReduceResult（src）不再被后续操作读取。sharedTmpBuffer = preReduceResultTail。

**存活节点**：需要 3 个 buffer——preReduceResult（输入）+ preReduceResultTail（sharedTmpBuffer）+ reduceResult（输出）。reduceResult 复用 cacheBuf（见 §1.3），不额外分配。preReduceResult 已计入 P_pre，preReduceResultTail 已计入 P_pre_ext。

| 步骤 | 执行操作 | 存活 buffer | 存活数 |
|------|---------|------------|--------|
| 1 | Reduce → cacheBuf[levelOff]（sharedTmpBuffer=preReduceResultTail） | preReduceResult, preReduceResultTail, cacheBuf | 3 |
| 2 | 吸收低层，与当前层结果合并，覆写本层 | cacheBuf | 1 |

**额外 buffer**：preReduceResultTail（P_pre_ext = 1，见 §1.3）、cacheBuf（1 份，见 §1.3）。

（3）postReducePhase

**计算图**（严格按公式 `var_sum/N`、`var_sum/(N−1)`，用 Div 直接做 var_sum/N，避免方向反转错误）：

```
S1: CopyIn count_sum        [GM → count_sum]           // N，postIn 路 1
S2: CopyIn running_var      [GM → running_var]          // postIn 路 2
S3: Div                     [var_sum, count_sum → var_sum_n]        // var_sum/N（var_sum = reduceResult）
S4: Add                     [var_sum_n, ε → var_sum_n_eps]          // var_sum/N + ε
S5: Sqrt                    [var_sum_n_eps → sqrt_result]           // sqrt(var_sum/N + ε)
S6: Div(1, x)                  [sqrt_result → invert_std]              // 1/sqrt → 输出 1 (fp32)
S7: Sub                     [count_sum, 1 → count_sum_minus_1]      // N−1
S8: Div                     [var_sum, count_sum_minus_1 → unbiased_var]  // var_sum/(N−1)
S9: Mul                     [unbiased_var, m → scaled_var]          // m × unbiased_var
S10: Mul                    [running_var, (1−m) → scaled_running]    // (1−m) × running_var
S11: Add                    [scaled_running, scaled_var → running_var_update]  // 输出 2 (fp32)
S12: Cast                   [invert_std → invert_std_fp16]          // fp32→fp16
S13: Cast                   [running_var_update → running_var_update_fp16]  // fp32→fp16
S14: CopyOut invert_std     [invert_std_fp16 → GM]
S15: CopyOut running_var_update [running_var_update_fp16 → GM]
```

分叉点：var_sum（reduceResult，S3 和 S8 消费）、count_sum（S3 和 S7 消费）。

**L 调度 trace**（纯拓扑，不考虑 VF 融合；持有中的 var_sum(cacheBuf) 归属 reduce 阶段，不计入 L_post 计数。采用最优调度：推迟 S2 CopyIn running_var 至 invert_std 链完成并 CopyOut 之后——若提前搬入，running_var 与链上中间值叠加，峰值 3→4。"执行中"取峰值瞬间=输入未释放+输出写入，输入仅在"执行后"其所有消费者完成时才移除）：

```pseudocode
// S1: CopyIn count_sum
// 执行前: 持有=[var_sum(cacheBuf)]          ← var_sum(reduceResult) 驻 cacheBuf（归属 reduce 阶段，不计入 L_post）
// 执行中: 持有=[var_sum(cacheBuf), count_sum]
// 执行后: 持有=[var_sum(cacheBuf), count_sum]    ← count_sum S3/S7 会读，var_sum S3/S8 会读，均保留
// S3: Div → var_sum_n
// 执行前: 持有=[var_sum(cacheBuf), count_sum]
// 执行中: 持有=[var_sum(cacheBuf), count_sum, var_sum_n]
// 执行后: 持有=[var_sum(cacheBuf), count_sum, var_sum_n]    ← 三者均有未来消费者，保留
// S4: Add → var_sum_n_eps
// 执行前: 持有=[var_sum(cacheBuf), count_sum, var_sum_n]
// 执行中: 持有=[var_sum(cacheBuf), count_sum, var_sum_n, var_sum_n_eps]    ← L_post=3 峰值（var_sum 不计）
// 执行后: 持有=[var_sum(cacheBuf), count_sum, var_sum_n_eps]    ← var_sum_n 无未来消费者，释放
// S5: Sqrt → sqrt_result
// 执行前: 持有=[var_sum(cacheBuf), count_sum, var_sum_n_eps]
// 执行中: 持有=[var_sum(cacheBuf), count_sum, var_sum_n_eps, sqrt_result]    ← L_post=3 峰值（var_sum 不计）
// 执行后: 持有=[var_sum(cacheBuf), count_sum, sqrt_result]    ← var_sum_n_eps 无未来消费者，释放
// S6: Div(1,x) → invert_std
// 执行前: 持有=[var_sum(cacheBuf), count_sum, sqrt_result]
// 执行中: 持有=[var_sum(cacheBuf), count_sum, sqrt_result, invert_std]    ← L_post=3 峰值（var_sum 不计）
// 执行后: 持有=[var_sum(cacheBuf), count_sum, invert_std]    ← sqrt_result 无未来消费者，释放
// S12: Cast → invert_std_fp16
// 执行前: 持有=[var_sum(cacheBuf), count_sum, invert_std]
// 执行中: 持有=[var_sum(cacheBuf), count_sum, invert_std, invert_std_fp16]    ← L_post=3 峰值（var_sum 不计）
// 执行后: 持有=[var_sum(cacheBuf), count_sum, invert_std_fp16]    ← invert_std 无未来消费者，释放
// S14: CopyOut invert_std
// 执行前: 持有=[var_sum(cacheBuf), count_sum, invert_std_fp16]
// 执行中: 持有=[var_sum(cacheBuf), count_sum, invert_std_fp16]
// 执行后: 持有=[var_sum(cacheBuf), count_sum]    ← invert_std_fp16 无未来消费者，释放
// S7: Sub → count_sum_minus_1
// 执行前: 持有=[var_sum(cacheBuf), count_sum]
// 执行中: 持有=[var_sum(cacheBuf), count_sum, count_sum_minus_1]
// 执行后: 持有=[var_sum(cacheBuf), count_sum_minus_1]    ← count_sum 无未来消费者，释放
// S8: Div → unbiased_var
// 执行前: 持有=[var_sum(cacheBuf), count_sum_minus_1]
// 执行中: 持有=[var_sum(cacheBuf), count_sum_minus_1, unbiased_var]
// 执行后: 持有=[unbiased_var]    ← var_sum + count_sum_minus_1 无未来消费者，释放
// S2: CopyIn running_var
// 执行前: 持有=[unbiased_var]
// 执行中: 持有=[unbiased_var, running_var]
// 执行后: 持有=[unbiased_var, running_var]    ← running_var S10 会读，保留
// S9: Mul → scaled_var
// 执行前: 持有=[unbiased_var, running_var]
// 执行中: 持有=[unbiased_var, running_var, scaled_var]    ← unbiased_var 仍存活（正在被读）
// 执行后: 持有=[running_var, scaled_var]    ← unbiased_var 无未来消费者，释放；scaled_var S11 会读，保留
// S10: Mul → scaled_running
// 执行前: 持有=[running_var, scaled_var]
// 执行中: 持有=[running_var, scaled_var, scaled_running]
// 执行后: 持有=[scaled_var, scaled_running]    ← running_var 无未来消费者，释放
// S11: Add → running_var_update
// 执行前: 持有=[scaled_var, scaled_running]
// 执行中: 持有=[scaled_var, scaled_running, running_var_update]
// 执行后: 持有=[running_var_update]    ← scaled_var + scaled_running 无未来消费者，释放；S13 会读，保留
// S13: Cast → running_var_update_fp16
// 执行前: 持有=[running_var_update]
// 执行中: 持有=[running_var_update, running_var_update_fp16]
// 执行后: 持有=[running_var_update_fp16]    ← running_var_update 无未来消费者，释放
// S15: CopyOut running_var_update
// 执行前: 持有=[running_var_update_fp16]
// 执行中: 持有=[running_var_update_fp16]
// 执行后: 持有=[]    ← running_var_update_fp16 无未来消费者，释放
```

**L 结论**：

| L_post | 峰值步骤 | 瓶颈 | 不可降原因 |
|--------|---------|------|-----------|
| 3 | S4-S6 / S9-S12（代表性） | count_sum + 链上中间值 + 产出（var_sum 驻 cacheBuf 不计） | count_sum 分叉（S3, S7）与链上中间值同时存活，叠加产出，二元链不可降 |

**P_post_baseline**（不考虑 VF 融合）：= 3（按最优调度，各 op src/dst 各占独立 buffer，峰值在 S4 或 S12 = 3；var_sum 驻 cacheBuf 树根不计入）。

**可用 API 能力**：

| API | 能力 | 对物理节点的意义 | 来源 |
|-----|------|----------------|------|
| Reg::Div | 寄存器内除法 | 可与其他 VF 操作串链 | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |
| Reg::Mul | 寄存器内乘法 | 可与其他 VF 操作串链 | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |
| Reg::Add | 寄存器内加法 | 可与其他 VF 操作串链 | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |
| Reg::Sub | 寄存器内减法 | 可与其他 VF 操作串链 | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |
| Reg::Sqrt | 寄存器内开方 | 可与其他 VF 操作串链 | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |
| Reg::Cast | b16↔fp32 寄存器转换 | Cast 可参与 VF | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |

**VF 融合穷举分析**（按数据流相邻对；标量 ε/m/(1−m)/1 在寄存器构造，不占 UB，不影响链长判断）：

| 操作对 | 融后链长 ≤ 7？ | Cast 可融？ | RegBase 指令存在？ | 结论 |
|--------|----------|------------|------------------|------|
| Div(S3) → Add(S4) | 2 ≤ 7 ✓ | — | Reg::Div ✓, Reg::Add ✓ | 可融 |
| Add(S4) → Sqrt(S5) | 3 ≤ 7 ✓ | — | Reg::Add ✓, Reg::Sqrt ✓ | 可融 |
| Sqrt(S5) → Div(S6) | 4 ≤ 7 ✓ | — | Reg::Sqrt ✓, Reg::Div ✓ | 可融 |
| Div(S6) → Cast(S12) | 5 ≤ 7 ✓ | ✓ | Reg::Div ✓, Reg::Cast ✓ | 可融（数据流相邻，调度上 S6 后立即 S12，中间无 invert_std 消费者） |
| Sub(S7) → Div(S8) | 2 ≤ 7 ✓ | — | Reg::Sub ✓, Reg::Div ✓ | 可融 |
| Div(S8) → Mul(S9) | 3 ≤ 7 ✓ | — | Reg::Div ✓, Reg::Mul ✓ | 可融 |
| Mul(S9) → Mul(S10) | 不相邻（S9、S10 为并列支路，分别汇入 S11，非数据流直连） | — | — | 不融（并列支路） |
| Mul(S9) → Add(S11) | 4 ≤ 7 ✓ | — | Reg::Mul ✓, Reg::Add ✓ | 可融 |
| Mul(S10) → Add(S11) | 2 ≤ 7 ✓ | — | Reg::Mul ✓, Reg::Add ✓ | 可融 |
| Add(S11) → Cast(S13) | 5 ≤ 7 ✓ | ✓ | Reg::Add ✓, Reg::Cast ✓ | 可融 |

**VF 融合方案**：
- VF 链 1：S3(Div) + S4(Add) + S5(Sqrt) + S6(Div) + S12(Cast) → invert_std_fp16（链长 5 ≤ 7）
- VF 链 2：S7(Sub) + S8(Div) + S9(Mul) + S10(Mul) + S11(Add) + S13(Cast) → running_var_update_fp16（链长 6 ≤ 7，含并列支路在寄存器暂存后汇合）

**{fp16} 物理 trace**（融合后，沿用最优调度：先 invert_std 链输出并 CopyOut，再 CopyIn running_var，再做 running_var_update 链。"执行中"取峰值瞬间=输入未释放+输出写入，输入仅在"执行后"其所有消费者完成时才移除。持有中的 var_sum(cacheBuf) 驻 cacheBuf 树根，归属 reduce 阶段，不计入 P_post）：

```pseudocode
// S1: CopyIn count_sum
// 执行前: 持有=[var_sum(cacheBuf)]          ← 驻 cacheBuf 树根（归属 reduce 阶段，不计入 P_post）
// 执行中: 持有=[var_sum(cacheBuf), B0(count_sum)]
// 执行后: 持有=[var_sum(cacheBuf), B0(count_sum)]    ← 两者 VF 链1/链2 会读，保留
// S3-S6+S12: VF: Div+Add+Sqrt+Div+Cast → invert_std_fp16
// 执行前: 持有=[var_sum(cacheBuf), B0(count_sum)]
// 执行中: 持有=[var_sum(cacheBuf), B0(count_sum), B1(invert_std_fp16)]    ← 融掉 Div/Add/Sqrt/Div 中间值
// 执行后: 持有=[var_sum(cacheBuf), B0(count_sum), B1(invert_std_fp16)]    ← B0 VF 链2 会读，B1 待 CopyOut，均保留
// S14: CopyOut invert_std
// 执行前: 持有=[var_sum(cacheBuf), B0(count_sum), B1(invert_std_fp16)]
// 执行中: 持有=[var_sum(cacheBuf), B0(count_sum), B1(invert_std_fp16)]
// 执行后: 持有=[var_sum(cacheBuf), B0(count_sum)]    ← B1 无未来消费者，释放
// S2: CopyIn running_var
// 执行前: 持有=[var_sum(cacheBuf), B0(count_sum)]
// 执行中: 持有=[var_sum(cacheBuf), B0(count_sum), B1(running_var)]    ← B1 复用，承载 running_var
// 执行后: 持有=[var_sum(cacheBuf), B0(count_sum), B1(running_var)]    ← B1 VF 链2 会读，保留
// S7-S11+S13: VF: Sub+Div+Mul+Mul+Add+Cast → running_var_update_fp16
// 执行前: 持有=[var_sum(cacheBuf), B0(count_sum), B1(running_var)]    ← 三者均为 VF 链2 输入
// 执行中: 持有=[var_sum(cacheBuf), B0(count_sum), B1(running_var), B2(running_var_update_fp16)]    ← P_post=3 峰值（var_sum 不计）
// 执行后: 持有=[B2(running_var_update_fp16)]    ← B0+B1 所有消费者完成，释放
// S15: CopyOut running_var_update
// 执行前: 持有=[B2(running_var_update_fp16)]
// 执行中: 持有=[B2(running_var_update_fp16)]
// 执行后: 持有=[]                  ← B2 无未来消费者，释放
```

物理 buffer 复用映射：

| 物理块 | 依次承载 | 复用合法性 |
|--------|---------|-----------|
| B0 | count_sum | S1 CopyIn 至 VF 链2 消费完，单消费者链 |
| B1 | invert_std_fp16 → running_var | invert_std_fp16 在 S14 CopyOut 后释放，B1 转承载 S2 CopyIn 的 running_var |
| B2 | running_var_update_fp16 | VF 链2 产出，S15 CopyOut 后释放 |

var_sum(reduceResult)：驻 cacheBuf 树根（reduce 阶段已计入），postReducePhase 不分配，不计入 L_post/P_post。

**P 结论**：

| P_post | 峰值步骤 | 与 L_post 差距 |
|--------|---------|-------------|
| 3 | S7-S11+S13(VF) | 0（count_sum + running_var + 产出 = 3；var_sum 驻 cacheBuf 树根归属 reduce 阶段，不计入本阶段 L/P） |

（4）Buffer 划分产出

| 阶段 | buffer | 份数 | 来源 |
|------|--------|------|------|
| preReducePhase | preReducePhase buffer（含 preReduceResult） | 4（= P_pre） | §1.1.1 + §1.2.1，P_pre 包含 preReduceResult（独立分配） |
| preReducePhase | preReduceResultTail | P_pre_ext = 1 | §1.3，不计入 P_pre |
| reducePhase | cacheBuf | 1（固定） | §1.3，兼作 Reduce 输出落点和 reduceResult |
| postReducePhase | postReducePhase buffer | 3（= P_post） | §1.1.3 + §1.2.3 |

**总计 = P_pre(4) + P_pre_ext(1) + 1(cacheBuf) + P_post(3) = 9 份物理 buffer**。Reduce 输出落点和 reduceResult 都不额外分配，复用 cacheBuf；var_sum（reduceResult）驻 cacheBuf 树根，归属 reduce 阶段，不计入 P_post。
