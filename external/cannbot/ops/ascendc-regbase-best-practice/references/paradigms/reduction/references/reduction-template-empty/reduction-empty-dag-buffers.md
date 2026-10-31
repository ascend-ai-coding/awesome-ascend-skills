# reduction 范式 empty DAG 分析和 Buffer 划分

> UB buffer 的划分方案，包括 DAG 图分析（主要是VF融合）、存活节点分析、buffer 分类表。

依赖输入：[reduction-template-overview.md](../reduction-template-overview.md)

> Empty 模板与二分树无关（EMPTY_A 早退、EMPTY_R Duplicate 固化值）。本文档同时覆盖 EMPTY_A 和 EMPTY_R 两个子模板。

## 1 DAG 分析和 Buffer 划分方案

Empty 模板不做 reduce 计算。EMPTY_A 零计算零 IO；EMPTY_R 通过 Duplicate `empty_r_output_value` 替代 Reduce，然后做可选 PostElewise + CopyOut。

**从GM视角看，不考虑UB，数据流概览**：

```
EMPTY_A:
  无数据流（零计算、零 IO，所有核早退）

EMPTY_R:
  ┌─ 计算流程 ─────────────────────────────────────────────┐
  │  ── Duplicate(empty_r_output_value) → reduced_value
  │  GM_x2 (可选)--> postReducePhase buffer --|
  │  reduced_value --------------------------|--PostElewise--> outNode --> GM_y
  └────────────────────────────────────────────────────────┘
```

> **可能有其他 GM 输入**：除了 reduced_value 外，还可能有自己的 CopyIn 节点从 GM 搬入额外的输入。

### 1.1 存活节点分析

分两层：

**逻辑存活节点 L**：纯拓扑分析，不考虑 VF、融合、寄存器。画出计算流程的完整计算图，按最优调度逐步追踪存活 tensor 数，取峰值。L 对给定计算图是固定的，不随硬件/API/实现改变。

**物理存活节点 P**：`P = physical_min(L, API能力, 算法策略)`。P 的计数域为 UB buffer（寄存器不计入）。P ≥ L 一般成立，VF 寄存器链融合可打破（中间值不落 UB，P < L 合法）。P 的具体值依赖 VF 融合策略，在 §1.2 中分析得出。L 是 P 的下界和收敛目标——P == L 时确认已到最优，P > L 时知道还有优化空间。

⛔ Tuple Reduce 场景（N 次独立归约、`Process`/`ProcessGroup` 按 `processIdx` 分发）：存活节点分析**分阶段**（pre/reduce/post，Group 为 Phase 1/Phase 2）各自画**一条流**，禁止按 for 循环展开成多轮：
1. **公共步骤**（所有过程都执行）在流中只出现一次。
2. **过程独有步骤**写成 `if (processIdx == …) {…} else if …` 并列分支，分割粒度 = 归约过程（同一过程的多个分叉输出在该分支内合并 trace）；分支互斥——任一时刻只执行一个分支，各分支的 buffer 不可能同时存活。
3. **分支峰值取 max**：每个分支单独 trace 其执行时的峰值，一处 if/else 的峰值 = 各分支峰值的 max，禁止相加；互斥分支共用同一份物理 UB，不按分支单独记账。
4. **存活期照常判定**：分支的中间 buffer 在分支结束即释放，仅被后续公共步骤消费的产出继续存活；释放后的普通复用池 buffer 可被后续任意步骤复用，与它来自哪个分支无关。既有复用例外不变（§1.1 各阶段复用规则、§1.3 固定份数 buffer 照旧）。
5. 每个阶段一条流走完，阶段峰值 = 该阶段存活节点数；UB 份数仍按 §1.5 汇总表分阶段相加。

**注**：以上规则对 L、P 均适用；"共用物理 UB""记账""UB 份数"只作用于 P——L 无物理复用概念，仍按公平基线逐 op 计节点；"按代码结构"指代码结构镜像算子公式的公共/独有子图划分，不违背 L 的纯拓扑定义。

**buffer 复用规则**：
- **reduced_value**：Duplicate 写入后作为 PostElewise 的输入，PostElewise 消费后可释放
- **其余节点**：生命周期不重叠时可复用同一物理 buffer。所有 UB buffer 统一按本算子参与计算的最大 dtype 分配容量，不同 dtype 的 tensor 容量一致，跨 dtype 复用不受尺寸约束；buffer 份数同样统一覆盖全部 dtype 实例（含 Cast 中转需求，无 Cast 的 dtype 实例相应份数闲置），不按 dtype 分裂

分析方法参考 [reduction-survival-node.md](../reduction-survival-node.md)。

**EMPTY_A**：无 buffer 分配（零计算、零 IO）。

**EMPTY_R 产出**：
1. **计算图** — 从算子公式推导 S1..Sn，含 DuplicateEmptyROutputVf + 可选 CopyIn_post + PostElewise + CopyOut
2. **L 调度 trace** — 逐操作追踪存活 tensor 数，标 L_post 峰值
3. **L 结论** — L_post 值 / 峰值步骤 / 瓶颈 / 不可降原因
4. **P_post_baseline** — 不考虑 VF 融合时的基线

### 1.2 DAG VF融合分析

**EMPTY_A**：无 DAG（零计算、零 IO）。

**EMPTY_R**：R 轴有 0 → reduce 沿空集合 → 每个 output entry 取算子常量 `kEmptyROutputValue`。Kernel 直接 `Duplicate(kEmptyROutputValue)` 写目标 buf。

VF 融合分析方法参考 [reduction-vf-fusion-rules.md](../reduction-vf-fusion-rules.md)。

**VF 融合对 buffer 的影响**：融合后中间值在寄存器内不落 UB，减少存活节点数。不融的相邻操作各占一个独立 buffer。如果融合后存活节点峰值过高，需要主动拆分 VF 降压。

**产出**：
1. **可用 API 能力** — 表格：API | 能力 | 对物理节点的意义 | 来源
2. **VF 融合穷举分析** — 所有相邻操作对判据结果
3. **VF 融合方案** — 哪些融成一条链，哪些断开
4. **{dtype} 物理 trace** — 每步展开，标 P_post 峰值。各 dtype 峰值取最大者为统一 P_post，不产生 per-dtype 份数
5. **结论** — P_post 值 / 峰值步骤 / 与 L_post 的差距

### 1.3 额外buffer要求

Empty 模板只分配 postReducePhase buffer（`P_post` 份），不分配其他 buffer：
- 不搬入 reduce 输入（preReducePhase 无输入搬入）
- 无跨循环累加器
- reduce 输出由 `DuplicateEmptyROutputVf` 写算子常量 `kEmptyROutputValue`（取值见 empty kernel 模板 §5.1 表）

**单 buf 64KB 封顶**：`MAX_SINGLE_UB_BYTES = 65536`，每个 postReducePhase buffer 不超过 64KB。

### 1.4 Double Buffer 决策

**EMPTY_A**：无 buffer，无 DB。

**EMPTY_R**：不开 DB。

### 1.5 强制产出清单

> agent 完成分析后，逐项核对，缺一项不算完成。

**EMPTY_A**：
- [ ] 无 buffer 分配（零计算、零 IO）

**EMPTY_R**（§1.1 + §1.2）：
- [ ] 计算图（DuplicateEmptyROutputVf + 可选 CopyIn_post + PostElewise + CopyOut）
- [ ] L 调度 trace（执行前/中/后: 持有=[...] + 消费者推理，标 L_post 峰值）
- [ ] L 结论表（L_post 值 / 峰值步骤 / 瓶颈 / 不可降原因）
- [ ] P_post_baseline
- [ ] 可用 API 能力表
- [ ] VF 融合穷举分析
- [ ] VF 融合方案
- [ ] {dtype} 物理 trace（执行前/中/后: 持有=[B0,B1,...] + 消费者推理，标 P_post 峰值）
- [ ] P 结论表（P_post 值 / 峰值步骤 / 与 L_post 的差距）

**Double Buffer 决策**（§1.4）：
- [ ] 不开启

**Buffer 产出汇总**：
- [ ] 汇总表：

| buffer | 份数 | 来源 |
|--------|------|------|
| postReducePhase buffer | `P_post` | §1.1 + §1.2 |

- [ ] UB 总计 = `P_post`

## 2 Buffer 划分产出校验

> 为了校验 agent 是否真正理解了 ub buffer 划分逻辑，给出一个复杂输入的样例。

> 严格按照 `1.5 强制产出清单` 执行。

以 `SyncBatchNormGatherStatsWithCounts` 为例，EMPTY_R 场景。算子公式与输入输出同 [reduction-binary-base-dag-buffers.md](../reduction-template-binary-base/reduction-binary-base-dag-buffers.md) §2，此处不重复。

输入：fp16，6 输入 2 输出，R 轴含 0，触发 EMPTY_R。

（1）计算流程

**计算图**（cast 完全融 VF，两条 VF 链完成 PostElewise）：

```text
reduced_value (fp32) = Duplicate(empty_r_output_value)
count_sum   (fp16)   = CopyIn_post(postIn[1])
running_var (fp16)   = CopyIn_post(postIn[2])                 # 推迟至链1 CopyOut 之后（最优调度）
# 链1（长 5）：Load(reduced_value, count_sum) → Div+Add+Sqrt+Div+Cast → Store(invert_std)
# 链2（长 6）：Load(reduced_value, count_sum, running_var) → Sub+Div+Mul+Mul+Add+Cast → Store(running_var_update)
invert_std         (fp16) = PostElewise链1(...)
running_var_update (fp16) = PostElewise链2(...)
```

分叉点：reduced_value（S3 和 S8 消费）、count_sum（S3 和 S7 消费）。S 编号沿用 [reduction-binary-base-dag-buffers.md](../reduction-template-binary-base/reduction-binary-base-dag-buffers.md) §2(3) 计算图（Duplicate 无编号；CopyIn_post count_sum = S1、running_var = S2；链1 = S3-S6+S12，链2 = S7-S11+S13）。

**L 调度 trace**（纯拓扑，不考虑 VF 融合；采用最优调度：推迟 CopyIn_post running_var 至链1 完成并 CopyOut invert_std 之后——若提前搬入，running_var 与链1 中间值叠加，峰值 4→5。"执行中"取峰值瞬间=输入未释放+输出写入，输入仅在"执行后"其所有消费者完成时才移除）：

```pseudocode
// DuplicateEmptyROutputVf → reduced_value
// 执行前: 持有=[]
// 执行中: 持有=[reduced_value]
// 执行后: 持有=[reduced_value]               ← S3/S8 会读，保留
// CopyIn_post count_sum
// 执行前: 持有=[reduced_value]
// 执行中: 持有=[reduced_value, count_sum]
// 执行后: 持有=[reduced_value, count_sum]    ← S3/S7 会读，保留
// 链1: Div → var_sum_n
// 执行前: 持有=[reduced_value, count_sum]
// 执行中: 持有=[reduced_value, count_sum, var_sum_n]
// 执行后: 持有=[reduced_value, count_sum, var_sum_n]    ← 三者均有未来消费者，保留
// 链1: Add → var_sum_n_eps
// 执行前: 持有=[reduced_value, count_sum, var_sum_n]
// 执行中: 持有=[reduced_value, count_sum, var_sum_n, var_sum_n_eps]    ← L_post=4 峰值
// 执行后: 持有=[reduced_value, count_sum, var_sum_n_eps]    ← var_sum_n 无未来消费者，释放
// 链1: Sqrt → sqrt_result
// 执行前: 持有=[reduced_value, count_sum, var_sum_n_eps]
// 执行中: 持有=[reduced_value, count_sum, var_sum_n_eps, sqrt_result]    ← L_post=4 峰值
// 执行后: 持有=[reduced_value, count_sum, sqrt_result]    ← var_sum_n_eps 无未来消费者，释放
// 链1: Div(1,x) → invert_std
// 执行前: 持有=[reduced_value, count_sum, sqrt_result]
// 执行中: 持有=[reduced_value, count_sum, sqrt_result, invert_std]    ← L_post=4 峰值
// 执行后: 持有=[reduced_value, count_sum, invert_std]    ← sqrt_result 无未来消费者，释放
// 链1: Cast → invert_std_fp16
// 执行前: 持有=[reduced_value, count_sum, invert_std]
// 执行中: 持有=[reduced_value, count_sum, invert_std, invert_std_fp16]    ← L_post=4 峰值
// 执行后: 持有=[reduced_value, count_sum, invert_std_fp16]    ← invert_std 无未来消费者，释放
// CopyOut invert_std
// 执行后: 持有=[reduced_value, count_sum]    ← invert_std_fp16 无未来消费者，释放
// 链2: Sub → count_sum_minus_1
// 执行前: 持有=[reduced_value, count_sum]
// 执行中: 持有=[reduced_value, count_sum, count_sum_minus_1]
// 执行后: 持有=[reduced_value, count_sum_minus_1]    ← count_sum 无未来消费者，释放
// 链2: Div → unbiased_var
// 执行前: 持有=[reduced_value, count_sum_minus_1]
// 执行中: 持有=[reduced_value, count_sum_minus_1, unbiased_var]
// 执行后: 持有=[unbiased_var]    ← reduced_value + count_sum_minus_1 无未来消费者，释放
// CopyIn_post running_var
// 执行前: 持有=[unbiased_var]
// 执行中: 持有=[unbiased_var, running_var]
// 执行后: 持有=[unbiased_var, running_var]    ← running_var S10 会读，保留
// 链2: Mul → scaled_var
// 执行前: 持有=[unbiased_var, running_var]
// 执行中: 持有=[unbiased_var, running_var, scaled_var]    ← unbiased_var 仍存活（正在被读）
// 执行后: 持有=[running_var, scaled_var]    ← unbiased_var 无未来消费者，释放；scaled_var S11 会读，保留
// 链2: Mul → scaled_running
// 执行前: 持有=[running_var, scaled_var]
// 执行中: 持有=[running_var, scaled_var, scaled_running]
// 执行后: 持有=[scaled_var, scaled_running]    ← running_var 无未来消费者，释放
// 链2: Add → running_var_update
// 执行前: 持有=[scaled_var, scaled_running]
// 执行中: 持有=[scaled_var, scaled_running, running_var_update]
// 执行后: 持有=[running_var_update]    ← scaled_var + scaled_running 无未来消费者，释放；S13 会读，保留
// 链2: Cast → running_var_update_fp16
// 执行前: 持有=[running_var_update]
// 执行中: 持有=[running_var_update, running_var_update_fp16]
// 执行后: 持有=[running_var_update_fp16]    ← running_var_update 无未来消费者，释放
// CopyOut running_var_update
// 执行后: 持有=[]    ← running_var_update_fp16 无未来消费者，释放
```

**L 结论**：

| L_post | 峰值步骤 | 瓶颈 | 不可降原因 |
|--------|---------|------|-----------|
| 4 | 链1 中间步骤 | reduced_value + count_sum + 链上中间值 + 产出 | reduced_value 分叉（S3, S8）、count_sum 分叉（S3, S7）与链1 中间值同时存活，叠加产出，二元链不可降 |

**P_post_baseline**（不考虑 VF 融合）：= 4（按最优调度，各 op src/dst 各占独立 buffer，峰值在链1 中间步骤 = 4）。

**可用 API 能力**：

| API | 能力 | 对物理节点的意义 | 来源 |
|-----|------|----------------|------|
| Reg::Duplicate | 寄存器内标量广播 | 写 empty_r_output_value 不占独立 buffer | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |
| Reg::Cast | b16↔fp32 寄存器转换 | Cast 可参与 VF | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |
| Reg::Div / Reg::Sqrt / Reg::Mul / Reg::Add / Reg::Sub | 寄存器内算术 | 可与其他 VF 操作串链 | [regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md) |

**VF 融合方案**：PostElewise 拆两条 VF 链：链1 S3-S6+S12（长 5，cast 融链尾）→ invert_std_fp16；链2 S7-S11+S13（长 6，cast 融链尾）→ running_var_update_fp16。拆链原因：两路合计 11 条操作超过链长 ≤ 7 硬约束（[reduction-vf-fusion-rules.md](../reduction-vf-fusion-rules.md) §1.1），且两路在 S3/S7 处分叉、无数据流直连。两条链无跨 VF 中间值。

**{fp16} 物理 trace**（融合后，4 块物理 UB 覆盖 5 个逻辑 tensor：B0/B1 输入 + B2 输出兼复用 + B3 输出，单步内不假设 in-place）：

```pseudocode
// DuplicateEmptyROutputVf → reduced_value
// 执行前: 持有=[]
// 执行中: 持有=[B0(reduced_value)]
// 执行后: 持有=[B0(reduced_value)]          ← 链1/链2 会读，保留
// CopyIn_post count_sum
// 执行前: 持有=[B0(reduced_value)]
// 执行中: 持有=[B0(reduced_value), B1(count_sum)]
// 执行后: 持有=[B0(reduced_value), B1(count_sum)]    ← 链1/链2 会读，保留
// VF 链1: Div+Add+Sqrt+Div+Cast → invert_std_fp16
// 执行前: 持有=[B0(reduced_value), B1(count_sum)]
// 执行中: 持有=[B0(reduced_value), B1(count_sum), B2(invert_std_fp16)]    ← 融掉链内中间值
// 执行后: 持有=[B0(reduced_value), B1(count_sum), B2(invert_std_fp16)]    ← B0/B1 链2 会读，B2 待 CopyOut，均保留
// CopyOut invert_std
// 执行后: 持有=[B0(reduced_value), B1(count_sum)]    ← B2 无未来消费者，释放
// CopyIn_post running_var
// 执行前: 持有=[B0(reduced_value), B1(count_sum)]
// 执行中: 持有=[B0(reduced_value), B1(count_sum), B2(running_var)]    ← B2 复用，承载 running_var
// 执行后: 持有=[B0(reduced_value), B1(count_sum), B2(running_var)]    ← B2 链2 会读，保留
// VF 链2: Sub+Div+Mul+Mul+Add+Cast → running_var_update_fp16
// 执行前: 持有=[B0(reduced_value), B1(count_sum), B2(running_var)]    ← 三者均为链2 输入
// 执行中: 持有=[B0(reduced_value), B1(count_sum), B2(running_var), B3(running_var_update_fp16)]    ← P_post=4 峰值
// 执行后: 持有=[B3(running_var_update_fp16)]    ← B0+B1+B2 所有消费者完成，释放
// CopyOut running_var_update
// 执行后: 持有=[]                  ← B3 无未来消费者，释放
```

物理 buffer 映射：

| 物理块 | 承载 | 释放时机 |
|--------|------|---------|
| B0 | reduced_value | 链2 消费后释放 |
| B1 | count_sum | 链2 消费后释放 |
| B2 | invert_std_fp16 → running_var | invert_std_fp16 在 CopyOut 后释放，B2 转承载 running_var |
| B3 | running_var_update_fp16 | CopyOut 后释放 |

**P 结论**：

| P_post | 峰值步骤 | 与 L_post 差距 |
|--------|---------|-------------|
| 4 | 链2 | 0（reduced_value + count_sum + running_var + 产出 = 4；VF 融合消除链内中间值，B2 分时复用承载 invert_std/running_var） |

> **变体（cast 不融 VF）**：Cast 单独落 UB（fp32 结果先写 UB 再 Cast），P_post 会更高。算子作者按性能/UB 占用权衡。

（2）Buffer 划分产出

| buffer | 份数 | 来源 |
|--------|------|------|
| postReducePhase buffer | 4（= P_post） | (1) |

**UB 总计 = P_post(4) = 4 份物理 buffer**。

**按 P_post 平分 UB**：

```
maxBufSize = min(ubSize / P_post, 65536)        // 每个 buffer 不超过 64KB
maxDtypeSize = max(sizeof(D_T), sizeof(float))   // fp16: 4B, fp32: 4B
```

> `maxBufSize` 是每个 buffer 的上限。实际 `aUbFactor` 由 [reduction-empty-tiling.md](reduction-empty-tiling.md) §1.2 综合 4KB 下界/优先多核/上限/aTotal 兜底算出。
