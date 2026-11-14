# A2/A3(`arch22`)FlashAttention 实现手册 — 范围与术语

> **适用范围**:Ascend 910B3(A2/A3 代际,`arch22` / `DAV_2201`):20 个 AI Core、每个 AI Core 配 2 个 AIV(Vector 子核);**每个 AIV 独立 UB 192 KiB**(两个 AIV 互不共享,实测 910B3 单卡 40 个 AIV 各占 190 KiB 同时常驻且互不覆盖)、L1 512 KiB、L0A 与 L0B 各 64 KiB、L0C 128 KiB;非 causal MHA;fp16 / bf16 输入输出。
> **不适用**:Ascend 950(A5,`arch35` / `DAV_3510`,UB 248 KiB)、其他 SKU 的核数与容量差异、causal / GQA / MLA / paged KV / varlen / 量化 / 多卡。跨代际、跨变体不继承本文任何取值。
> **性质**:单实现实证手册——只描述一个具体实现的做法、判据与已知失效,不构成跨实现约定,也不承担精度或性能的验收口径。
> **术语**:以 [`patterns.md`](../patterns.md) 与 [`fundamentals.md §0`](../foundation/fundamentals.md) 的抽象名为准;文中出现的符号名一律视为**角色名**,具体命名由实现自定。
> **最后核对**:2026-11-10;工具链 CANN 9.3.0 weekly / bisheng,编译目标 `dav-2201`。

---

## §1 覆盖范围

| 维度 | 覆盖 | 不覆盖(不要假设成立) |
|---|---|---|
| 硬件 | Ascend 910B3(A2/A3,`arch22`):20 个 AI Core、每核 2 个 AIV;UB 192 KiB **每 AIV 独立**(非两子核共享)、L1 512 KiB、L0A/L0B 64 KiB、L0C 128 KiB | Ascend 950(A5)、Kirin 系列、其他 SKU 的裁剪规格 |
| 形态 | 非 causal MHA(q/k/v 同 head 数),prefill 与 decode(Sq = 1) | causal mask、GQA/MQA、MLA、paged KV、varlen/TND、任意 mask、多卡与通算融合 |
| dtype | fp16、bf16(输入输出同型;中间段可降精度) | fp32 输入、int8 / fp8、混合 dtype |
| D | 大块几何约 64~256;小块几何约 64~608(受 UB 容量限制);要求 16 对齐 | 非 16 对齐;超过小块上限 |
| 软件 | CANN 9.3.0 weekly、bisheng、`ascendc.cmake`、Release 构建 | 其他 CANN 版本、增量构建 |

**这四类之外没有分支**:causal / GQA / paged KV / 量化 / 多卡在当前实现里都不存在,任何"应该也能用"的推断都只能当作假设。

---

## §2 与 Tiling 方法论的关系

本文是 `foundation/` + `design/` 的**落地补充**,不替代任何设计节点:

| 设计节点 | 本文对应内容 |
|---|---|
| N7 内存划分 | [`tiling_budget.md`](./tiling_budget.md):UB / L1 / L0 / workspace 的预算口径、回退判据 |
| N9 Host Tiling | [`tiling_budget.md`](./tiling_budget.md):决策顺序与判据形式 |
| N10 流水编排与跨核同步 | [`sync_pipeline.md`](./sync_pipeline.md):同步原语拆分判据、槽语义、节拍体结构 |
| N11 开发起点 | [`bringup_cube.md`](./bringup_cube.md) + [`bringup_crosscore.md`](./bringup_crosscore.md):从空目录到首个可运行算子 |
| N12 优化策略 | [`perf_levers.md`](./perf_levers.md):主杠杆与接入顺序;[`dead_ends.md`](./dead_ends.md):已关闭方向与翻案条件 |

其余文件:[`principles.md`](./principles.md)(成本模型与设计原理)、[`architecture.md`](./architecture.md)(核间分工与执行路径)、[`numerics.md`](./numerics.md)(数值链路与精度约束)。

---

## §3 术语(与 Tiling 方法论一致)

- **节拍**:处理一个 KV 分块的一次循环迭代;一拍内完成本轮的 C1 与领先若干拍的 C2,并配对 V1 / V2。
- **S / P / PV**:Q·Kᵀ 的结果 / 未归一化指数 / P·V 的结果。三段各占一条 GM 环形队列;最终输出写 `attentionOut`,累积量写 `O_acc`。
- **Bq / Bk / D**:Q 侧块行数 / KV 侧块行数 / head 维;Mmad 的有效 m 维为 `mEff = Bq × curG`,本形态 `curG = 1`。
- **预取深度**:为提前发射而多留的 L1 rotation 槽数;领先拍数决定重叠距离,槽数只决定容量。
- **低精度写回**:S 以半精度而非 fp32 写回,收益来自写回量与搬运量减半。
- **固定参考最大值**:用一个不随分块变化的上界常量替代在线 softmax 的 running max,使 P 与 PV 退化为纯累加。
- **L0C 跨分块累积**:PV 不逐分块撤出,而在累加区跨 KV 分块累加,一个任务只撤出一次。
- **执行路径**:由 host 判据选定、编译期分离、各自独立产物的实现分支;默认路径不启用任何性能开关。
- **三通道**:性能由算力、带宽、跨核片上吞吐三者中最紧的一个决定;主杠杆指该通道上的第一优先改动。

> 本文反复用到的四个**概念标签**——「S 半精度写回」「固定参考最大值」「L0C 跨分块累积」「扁平化节拍序列」——是便于叙述的名字,不是公开 API 名,也不是跨实现约定;实现侧的变量与宏命名自定。

---

## §4 与基线默认口径不同的三处(先看,避免误读)

1. **中间段落地位置**:本文的 AIC↔AIV 交换只经 GM workspace(上一代架构无跨核 UB 直通);基线 §3.5 的"默认优选片上 L0C→UB"适用于 950PR/950DT。两者不矛盾,是平台条件不同。
2. **中间段精度**:本文允许 S 与 PV 以半精度写回;基线要求中间矩阵保持 fp32、不使用随路 Cast。本文写法属**平台相关可选优化**,进入实现时必须连同回退路径一起规划。
3. **跨核 modeId**:本文只用 modeId 2(广播式,AIC 与所有 AIV);modeId 4(单 AIV 独立握手)仅 950PR 支持。
