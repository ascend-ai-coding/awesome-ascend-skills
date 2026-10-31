# reduction 范式 binary-group DAG 分析和 Buffer 划分

> UB buffer 的划分方案，包括 DAG 图分析（主要是VF融合）、存活节点分析、double buffer 决策, 需要产出 VF融合后的DAG 图和 buffer 分类表。

依赖输入：[reduction-template-overview.md](../reduction-template-overview.md)、[reduction-binary-base-dag-buffers.md](../reduction-template-binary-base/reduction-binary-base-dag-buffers.md)

> **本文档是差分文档**：preReducePhase / postReducePhase 的 VF 融合分析与 BinaryBase 完全一致，本文只描述 Phase 2 reducePhase 的差异和 workspace GM buffer 新增。

## 1 DAG 分析和 Buffer 划分方案

Group 模板将 Base 的三段式（preReducePhase → reducePhase → postReducePhase）拆成两阶段执行，通过 workspace GM buffer 传递中间结果。reducePhase 被拆成两段：Phase 1 做局部 R 段的 partial reduce（二分缓存树），Phase 2 做剩余 R 维的 final reduce（单 chunk，R 全载）。

- **Phase 1** = preReducePhase + 部分 reducePhase：完成 PreElewise，沿本核分配的局部 R 段做 partial reduce，结果写 workspace fp32。不做 PostElewise。
- **Phase 2** = 剩余 reducePhase + postReducePhase：从 workspace 读入 partial reduce 结果（fp32），做第二次 Reduce 消解 rGroupCnt 维，再做 PostElewise + CopyOut。

三个桥接节点将两阶段联系起来：
- **preReduceResult**：Phase 1 中 preReducePhase 的输出节点，作为 Phase 1 reducePhase 的输入
- **workspace**：Phase 1 reducePhase 的 partial reduce 结果（fp32 `[rGroupCnt, aTotal]`），作为 Phase 2 reducePhase 的输入
- **reduceResult**：Phase 2 reducePhase 完成后的节点（写入 cacheBuf[0]），作为 Phase 2 postReducePhase 的输入

preReducePhase 与 postReducePhase 各自独立分析：
- **独立的存活节点分析**：preReducePhase 和 postReducePhase 各自追踪存活节点峰值，两阶段独立分配 buffer，不复用
- **独立的 buffer 空间分配**：两阶段空间大小不同——preReducePhase 的 UB buffer 含 R 轴维度，postReducePhase 不含 R 轴（reduce 后 R 轴被归约）
- **独立的 VF 融合**：preReducePhase 和 postReducePhase 各自做 VF 融合，不跨 reducePhase 融合
- reducePhase 内部引入的节点和 buffer 详见 §1.1.2 和 §1.3。Phase 2 的 reducePhase 复用 Phase 1 的 preReduceResult 物理槽（workspace MTE2 搬入）和 cacheBuf（Reduce dst 写入 [0] 位置，R 全载无需二分树），详见 §1.1.2 和 §1.3。

⛔ Tuple Reduce 场景（N 次独立归约、`Process`/`ProcessGroup` 按 `processIdx` 分发）：存活节点分析**分阶段**（pre/reduce/post，Group 为 Phase 1/Phase 2）各自画**一条流**，禁止按 for 循环展开成多轮：
1. **公共步骤**（所有过程都执行）在流中只出现一次。
2. **过程独有步骤**写成 `if (processIdx == …) {…} else if …` 并列分支，分割粒度 = 归约过程（同一过程的多个分叉输出在该分支内合并 trace）；分支互斥——任一时刻只执行一个分支，各分支的 buffer 不可能同时存活。
3. **分支峰值取 max**：每个分支单独 trace 其执行时的峰值，一处 if/else 的峰值 = 各分支峰值的 max，禁止相加；互斥分支共用同一份物理 UB，不按分支单独记账。
4. **存活期照常判定**：分支的中间 buffer 在分支结束即释放，仅被后续公共步骤消费的产出继续存活；释放后的普通复用池 buffer 可被后续任意步骤复用，与它来自哪个分支无关。既有复用例外不变（§1.1 各阶段复用规则、§1.3 固定份数 buffer 照旧）。
5. 每个阶段一条流走完，阶段峰值 = 该阶段存活节点数；UB 份数仍按 §1.5 汇总表分阶段相加。

**注**：以上规则对 L、P 均适用；"共用物理 UB""记账""UB 份数"只作用于 P——L 无物理复用概念，仍按公平基线逐 op 计节点；"按代码结构"指代码结构镜像算子公式的公共/独有子图划分，不违背 L 的纯拓扑定义。

**从GM视角看，不考虑UB，两阶段数据流概览**：

```
┌─ Phase 1（preReducePhase + 部分 reducePhase）─────────────┐
│  GM_x1 --> preNode --PreElewise--> preReduceResult
│                            |-- preReduceOut,可选 --> GM_y1
│                                     ↓
│  preReduceResult --> Reduce(局部R段, 二分缓存树) --> cacheBuf 树根
│                                                       ↓ CopyOut(MTE3)
│                                            workspace (fp32)  ← partial reduce
└───────────────────────────────────┬───────────────────────┘
                                    ↓ SyncAll
┌─ Phase 2（剩余 reducePhase + postReducePhase）─────────────┐
│  workspace --> CopyIn(MTE2) --> preReduceResult(复用)
│                                  ↓
│  preReduceResult --> Reduce(rGroupCnt全载, 单chunk) --> cacheBuf[0]  ← reduceResult
│                                                              ↓
│  GM_x2 (可选)--> postInBuf --|
│  cacheBuf[0] ----------------|--PostElewise--> outNode --> GM_y2
└────────────────────────────────────────────────────────────┘
```

> **preReduceOut**：Phase 1 的 preReducePhase 计算流中任意阶段的中间值可作为输出，不经过 reducePhase。
>
> **postReducePhase 也可能有其他 GM 输入**：Phase 2 的 postReducePhase 除了接收 reduceResult 的结果外，还可能有自己的 CopyIn 节点从 GM 搬入额外的输入。

### 1.1 存活节点分析

**Phase 1**：同 [reduction-binary-base-dag-buffers.md](../reduction-template-binary-base/reduction-binary-base-dag-buffers.md) §1.1 的 preReducePhase + reducePhase 部分。`P_pre / P_pre_ext` 与 BinaryBase 一致，不涉及 postReducePhase（`P_post` 留给 Phase 2）。

**Phase 2**：Phase 2 复用 Phase 1 的 preReduceResult 和 cacheBuf 物理槽，postReducePhase buffer 是 Phase 2 专用（Init 时与 Phase 1 buffer 一起分配，Phase 1 不使用）：

| 类别 | 路数 | dtype | 大小 | 物理来源 |
|------|------|------|------|---------|
| preReduceResult | 1 | fp32 | `preBufSize`（Phase 1 已分配） | Phase 1 已分配，Phase 2 CopyIn 写入，作为 ReduceAPI 的 src |
| cacheBuf | 1 | fp32 | `a_len_ub × sizeof(fp32)`（写 [0] 位置） | 复用 Phase 1，ReduceAPI dst 写入 cacheBuf[0]（R 全载单 chunk，无需二分树层级），作为 reduceResult 供 postReducePhase 读取 |
| postReducePhase buffer | `P_post` | D_T | `postBufSize`（= CeilAlign(aUnit × maxDtypeSize, blockSize)，aUnit = aUbFactor × innerAProdAlign） | Phase 2 专用，Init 时分配，Phase 1 不使用 |

**为什么 Phase 2 复用 preReduceResult 槽**：同 fp32、同角色（Reduce src），Phase 1 SyncAll 后不再使用。Phase 2 优先全载 `rGroupCnt`（R 全载），剩余空间按 `a_len_ub` 切 A 轴。

**为什么 cacheBuf 写 [0] 位置**：R 全载（单 chunk），无需二分树层级，Reduce dst 直接写 cacheBuf[0] 作为 reduceResult。cacheBuf 16KB，`a_len_ub × sizeof(fp32)` 须 ≤ 16KB。

### 1.2 DAG VF融合分析

**Phase 1**：preReducePhase 的 VF 融合同 [reduction-binary-base-dag-buffers.md](../reduction-template-binary-base/reduction-binary-base-dag-buffers.md) §1.2.1，`P_pre` 与 BinaryBase 一致。

**Phase 2**：postReducePhase 的 VF 融合同 [reduction-binary-base-dag-buffers.md](../reduction-template-binary-base/reduction-binary-base-dag-buffers.md) §1.2.3，`P_post` 与 BinaryBase 一致。reduceResult 来源从 cacheBuf 树根改为 cacheBuf[0]，但不影响 postReducePhase 内部的 VF 融合分析。

跨阶段约束：preReducePhase 与 postReducePhase 不跨 reducePhase 融合。

### 1.3 额外buffer要求

UB buffer 部分同 [reduction-binary-base-dag-buffers.md](../reduction-template-binary-base/reduction-binary-base-dag-buffers.md) §1.3（preReduceResultTail、cacheBuf 兼作 reduceResult、preReduceResultTail 兼作 sharedTmpBuffer）。

**workspace GM buffer**（Group 模板新增）：
- 布局：`[rGroupCnt, aTotal]` fp32 dense，行优先（每行 aTotal 个 float，不是 a_len_ub）
- 大小：`rGroupCnt × aTotal × sizeof(fp32)`
- 用途：Phase 1 各核写 partial reduce 结果，Phase 2 各核读

### 1.4 Double Buffer 决策

同 [reduction-binary-base-dag-buffers.md](../reduction-template-binary-base/reduction-binary-base-dag-buffers.md) §1.4。

### 1.5 强制产出清单

> agent 完成分析后，逐项核对，缺一项不算完成。

**Phase 1 preReducePhase**（§1.1 + §1.2）：
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

**Phase 1 reducePhase**（§1.1 + §1.3）：
- [ ] isReuseSource 取值及理由
- [ ] sharedTmpBuffer = preReduceResultTail
- [ ] cacheBuf（1 份，§1.3）

**Phase 2 reducePhase**（§1.1 + §1.3）：
- [ ] isReuseSource 取值及理由（src 调用后失效，不后续读）
- [ ] sharedTmpBuffer = preReduceResultTail
- [ ] cacheBuf 写 [0] 作为 reduceResult（R 全载单 chunk，无需二分树层级）
- [ ] Phase 2 复用 Phase 1 的 preReduceResult 物理槽（workspace MTE2 搬入）

**Phase 2 postReducePhase**（§1.1 + §1.2）：
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
| Phase 1 preReducePhase | preReducePhase buffer（含 preReduceResult） | `P_pre` | §1.1 + §1.2，P_pre 包含 preReduceResult |
| Phase 1 preReducePhase | preReduceResultTail | P_pre_ext（固定 1） | §1.3，不计入 P_pre |
| Phase 1 reducePhase | cacheBuf | 1（固定） | §1.3，兼作 Reduce 输出落点和 reduceResult |
| Phase 2 reducePhase | preReduceResult（Phase 2 复用 Phase 1 槽） | 0（不新增） | §1.1 |
| Phase 2 reducePhase | cacheBuf（复用 Phase 1） | 0（不新增） | §1.1 |
| Phase 2 postReducePhase | postReducePhase buffer | `P_post` | §1.1 + §1.2 |
| GM | workspace | `rGroupCnt × aTotal × sizeof(fp32)` | §1.3 |

- [ ] UB 总计 = `P_pre` + 1 + 1 + `P_post`（Phase 2 复用，不额外增加 UB buffer）

## 2 Buffer 划分产出校验

> 为了校验 agent 是否真正理解了 ub buffer 划分逻辑，给出一个复杂输入的样例。

> 严格按照 `1.5 强制产出清单` 执行。

以 `SyncBatchNormGatherStatsWithCounts` 为例。算子公式与输入输出同 [reduction-binary-base-dag-buffers.md](../reduction-template-binary-base/reduction-binary-base-dag-buffers.md) §2，此处不重复。

输入：fp16，6 输入 2 输出，`aOuter=4, rOuter=8, coreNum=8`（A 用不满核，触发 Group）。

**Group 2D 分核**：
- totalOuter = 4 × 8 = 32
- perCoreNum = CeilDiv(32, 8) = 4
- numBlocks = CeilDiv(32, 4) = 8
- CeilAlign(8, 4) = 8 ≤ 8 → numBlocks = 8
- usedCoreNum = 8, rGroupCnt = 8 / 4 = 2

（1）Phase 1 preReducePhase + reducePhase

preReducePhase 的完整分析（计算图 / L trace / VF 融合 / 物理 trace）同 [reduction-binary-base-dag-buffers.md](../reduction-template-binary-base/reduction-binary-base-dag-buffers.md) §2 (1)(2)，`P_pre=4`、`P_pre_ext=1`。

（2）Phase 2 reducePhase

**isReuseSource**：true。理由：src（preReduceResult，workspace 搬入的 fp32）调用后失效，不后续读。sharedTmpBuffer = preReduceResultTail。

**存活节点**：

| 步骤 | 执行操作 | 存活 buffer | 存活数 |
|------|---------|------------|--------|
| 1 | CopyIn workspace → preReduceResult（复用 Phase 1 槽） | preReduceResult | 1 |
| 2 | Reduce → cacheBuf[0]（sharedTmpBuffer=preReduceResultTail） | preReduceResult, preReduceResultTail, cacheBuf | 3 |
| 3 | Reduce 完成，src 失效 | cacheBuf | 1 |

**cacheBuf 写 [0]**：Phase 2 的 R 维 = workspace 行数 `rGroupCnt`，一次全载单 chunk（无 R 切分 → 无二分树层级），Reduce dst 直接写 cacheBuf[0] 作为 reduceResult。

（3）Phase 2 postReducePhase

完整分析（计算图 / L trace / VF 融合 / 物理 trace）同 [reduction-binary-base-dag-buffers.md](../reduction-template-binary-base/reduction-binary-base-dag-buffers.md) §2 (3)，`P_post=3`。reduceResult 来源从 cacheBuf 树根改为 cacheBuf[0]，但不影响 postReducePhase 内部计算图和 VF 融合。

（4）Buffer 划分产出

| 阶段 | buffer | 份数 | 来源 |
|------|--------|------|------|
| Phase 1 preReducePhase | preReducePhase buffer（含 preReduceResult） | 4（= P_pre） | (1) |
| Phase 1 preReducePhase | preReduceResultTail | P_pre_ext = 1 | §1.3 |
| Phase 1 reducePhase | cacheBuf | 1（固定） | §1.3 |
| Phase 2 reducePhase | preReduceResult（Phase 2 复用 Phase 1 槽） | 0（不新增） | (2) |
| Phase 2 reducePhase | cacheBuf（复用 Phase 1） | 0（不新增） | (2) |
| Phase 2 postReducePhase | postReducePhase buffer | 3（= P_post） | (3) |
| GM | workspace | `rGroupCnt × aTotal × sizeof(fp32)` | §1.3 |

**UB 总计 = P_pre(4) + P_pre_ext(1) + 1(cacheBuf) + P_post(3) = 9 份物理 buffer**。Phase 2 复用 Phase 1 的 preReduceResult 和 cacheBuf，不额外增加 UB buffer。postReducePhase buffer 是 Phase 2 专用，Init 时分配。

**workspace GM**：`[rGroupCnt=2, aTotal]` fp32 dense，大小 = `2 × aTotal × 4` 字节。
