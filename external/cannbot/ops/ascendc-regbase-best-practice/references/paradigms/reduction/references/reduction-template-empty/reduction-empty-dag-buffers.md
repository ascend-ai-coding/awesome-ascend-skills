# reduction 范式 empty DAG 分析和 Buffer 划分

> UB buffer 的划分方案，包括 DAG 图分析（段编排 + VF 融合）、物理存活节点 P 分析、buffer 分类表。

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

### 1.1 物理存活节点（P）分析

**物理存活节点 P**：`P = physical_min(计算图, API能力, 算法策略)`。P 的计数域为 UB buffer（寄存器不计入）；VF 寄存器链融合可消掉中间量（不落 UB）。P 的具体值依赖 VF 融合策略，在 §1.2 中分析得出。

**下界自检（必做）**：P < 输入 tensor 数 + 输出 tensor 数时，须解释成因（该 bound 假设 GM 输入输出同时驻 UB，**非硬性同时存活下界**）——合法成因含 ① VF 寄存器链消掉中间量 ② I/O 错峰调度（串行 CopyIn→计算→CopyOut 流水下输入输出非同时驻留）等；**无法归因到任一成因时才判 trace 漏记/异常**。P 的理论下界（原 L 值）**必须**由 [common/scripts/liveness.py](../../../common/scripts/liveness.py) 从计算图 JSON 算出——本节仅转述方法论，脚本调用时机、三步管线顺序（liveness.py → allocate.py → physical_p.py）与产物落盘以 [common/vf-split-fusion-rules.md](../../../common/vf-split-fusion-rules.md) §12 为准。

⛔ Tuple Reduce 场景（N 次独立归约、`Process`/`ProcessGroup` 按 `processIdx` 分发）：P 物理 trace **分阶段**（pre/reduce/post，Group 为 Phase 1/Phase 2）各自画**一条流**，禁止按 for 循环展开成多轮：
1. **公共步骤**（所有过程都执行）在流中只出现一次。
2. **过程独有步骤**写成 `if (processIdx == …) {…} else if …` 并列分支，分割粒度 = 归约过程（同一过程的多个分叉输出在该分支内合并 trace）；分支互斥——任一时刻只执行一个分支，各分支的 buffer 不可能同时存活。
3. **分支峰值取 max**：每个分支单独 trace 其执行时的峰值，一处 if/else 的峰值 = 各分支峰值的 max，禁止相加；互斥分支共用同一份物理 UB，不按分支单独记账。
4. **存活期照常判定**：分支的中间 buffer 在分支结束即释放，仅被后续公共步骤消费的产出继续存活；释放后的普通复用池 buffer 可被后续任意步骤复用，与它来自哪个分支无关。既有复用例外不变（§1.1 各阶段复用规则、§1.3 固定份数 buffer 照旧）。
5. 每个阶段一条流走完，阶段峰值 = 该阶段 P 值；UB 份数仍按 §1.5 汇总表分阶段相加。

**注**：以上规则作用于 P 的物理 trace；"按代码结构"指代码结构镜像算子公式的公共/独有子图划分。（逻辑存活节点 L 已从分析流程移除；P 的理论下界**必须**由 [common/scripts/liveness.py](../../../common/scripts/liveness.py) 从计算图 JSON 算出，脚本调用时机与三步管线以 [common/vf-split-fusion-rules.md](../../../common/vf-split-fusion-rules.md) §12 为准。）

**buffer 复用规则**：
- **reduced_value**：Duplicate 写入后作为 PostElewise 的输入，PostElewise 消费后可释放
- **其余节点**：生命周期不重叠时可复用同一物理 buffer。所有 UB buffer 统一按本算子参与计算的最大 dtype 分配容量，不同 dtype 的 tensor 容量一致，跨 dtype 复用不受尺寸约束；buffer 份数同样统一覆盖全部 dtype 实例（含 Cast 中转需求，无 Cast 的 dtype 实例相应份数闲置），不按 dtype 分裂

分析方法参考 [reduction-survival-node.md](../reduction-survival-node.md)。

**EMPTY_A**：无 buffer 分配（零计算、零 IO）。

**EMPTY_R 产出**：
1. **计算图** — 从算子公式推导 S1..Sn，含 DuplicateEmptyROutputVf + 可选 CopyIn_post + PostElewise + CopyOut
2. **P_post_baseline** — 不考虑 VF 融合时的基线（**必须用** [common/scripts/liveness.py](../../../common/scripts/liveness.py) 预计算并落盘，时机与三步管线以 [common/vf-split-fusion-rules.md](../../../common/vf-split-fusion-rules.md) §12 为准）

### 1.2 DAG VF融合分析

**EMPTY_A**：无 DAG（零计算、零 IO）。

**EMPTY_R**：R 轴有 0 → reduce 沿空集合 → 每个 output entry 取算子常量 `kEmptyROutputValue`。Kernel 直接 `Duplicate(kEmptyROutputValue)` 写目标 buf。

段边界先按 [common/vf-split-fusion-rules.md](../../../common/vf-split-fusion-rules.md)（规则 C1-C9）判定；段内融合方法参考 [reduction-vf-fusion-rules.md](../reduction-vf-fusion-rules.md)。

**VF 融合对 buffer 的影响**：融合后中间值在寄存器内不落 UB，减少存活节点数。不融的相邻操作各占一个独立 buffer。如果融合后存活节点峰值过高，需要主动拆分 VF 降压。

**产出**：
1. **可用 API 能力** — 表格：API | 能力 | 对物理节点的意义 | 来源
2. **VF 融合穷举分析** — 所有相邻操作对判据结果
3. **VF 融合方案** — 哪些融成一条链，哪些断开
4. **{dtype} 物理 trace** — 每步展开，标 P_post 峰值。各 dtype 峰值取最大者为统一 P_post，不产生 per-dtype 份数（B 名映射由 [common/scripts/allocate.py](../../../common/scripts/allocate.py) 生成，设计侧只声明调度顺序 / pinned / 互斥分支约束；P 峰值与 WAR / 下界自检交 physical_p.py 验证）
5. **结论** — P_post 值 / 峰值步骤 / 与 P_post_baseline 的差距 / 下界自检结论

### 1.3 额外buffer要求

Empty 模板只分配 postReducePhase buffer（**`S_post` 份 = allocate.py 实际槽数**，可 ≥ 变体峰值 P_post——分支/公共交错映射中槽数大于任一变体峰值，容量与份数以 S_post 为准），不分配其他 buffer：
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
- [ ] P_post_baseline（**必须用** common/scripts/liveness.py 预计算，脚本输入/输出落盘 `$WORK_DIR/design/liveness/`）
- [ ] 下界自检（P_post < 输入数+输出数 → 解释成因：VF 消中间量 / I/O 错峰等，无法归因才判漏记）
- [ ] 可用 API 能力表
- [ ] VF 融合穷举分析
- [ ] VF 融合方案
- [ ] {dtype} 物理 trace（执行前/中/后: 持有=[B0,B1,...] + 消费者推理，标 P_post 峰值）
- [ ] P 结论表（P_post 值 / 峰值步骤 / 与 P_post_baseline 的差距）

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

**调度决策**（供 P_baseline 与物理 trace 沿用）：推迟 CopyIn_post running_var 至链1 完成并 CopyOut invert_std 之后——若提前搬入，running_var 与链1 中间值叠加，峰值 4→5。"执行中"取峰值瞬间=输入未释放+输出写入，输入仅在"执行后"其所有消费者完成时才移除。

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

| P_post | 峰值步骤 | 与 P_post_baseline 差距 | 下界自检 |
|--------|---------|-------------|------|
| 4 | 链2 | 0（reduced_value + count_sum + running_var + 产出 = 4 = P_post_baseline；VF 融合消除链内中间值，B2 分时复用承载 invert_std/running_var） | 不触发（4 ≥ 输入 2 + 输出 2 = 4；reduced_value 为 Duplicate 常量，非 GM 输入） |

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
