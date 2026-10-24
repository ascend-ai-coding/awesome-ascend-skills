---
name: external-cannbot-ops-ascendc-perf-optimize
description: Ascend C 算子性能优化策略制定。结合 Tiling 建模与流水分析（仿真图 + profiling 数据），按卡间/核间/核内三层流水制定性能优化策略，并回修
  Tiling 参数。触发：算子性能调优、流水分析、Tiling 修正、bound 诊断、MC² 通算融合算子优化、卡间流水配平时。
original-name: ascendc-perf-optimize
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-24'
synced-commit: ec8a7b0462e3acdd0d1d2f857f1dae652e19f4f2
license: UNKNOWN
---

# Ascend C 算子性能优化流水分析

> **⚠️ 重要**：本技能定义了 **4 步分层流水**优化流程（Step 1 → Step 2 → Step 3 → Step 4），而非简化的 3 步流程。Step 2（卡间流水）、Step 3（核间流水）、Step 4（单核流水）是三个独立的流水分析步骤，各有明确的适用条件、输入输出要求。请勿将它们合并为单一的"流水分析"步骤。

本技能的四步流程和通用建模不按代际拆分。资源参数取自目标平台；知识卡片中的架构、API 和实测范围按各自标注使用。**RegBase / Reg / MicroAPI / VF 相关实现仅适用于 arch35（A5 / DAV_3510）**，不适用于 arch22（A2/A3 / DAV_2201）。需要核对代际相关知识时查 [适用范围说明](references/architecture-routing.md)。

## 优化流程总览

| 步骤 | 名称 | 适用条件 | 关注点 |
|------|------|---------|--------|
| **Step 1** | Tiling 理论建模 | 所有算子 | 确定理想 tiling data |
| **Step 2** | 卡间流水优化 | 通信类算子 | 通算演算、卡间通信瓶颈。**MC² 算子须先采集 TilingData 和隔离测试数据（见 `comm-compute/index.md` Step 0/2），缺失时返回数据采集阶段补采** |
| **Step 3** | 核间流水优化 | 多核间同步算子 | 核间并行效率、同步开销 |
| **Step 4** | 单核流水优化 | 所有算子 | 核内流水、bound 诊断 |

**关键特征**：
- ✅ **4 步分层流水**，不是 3 步
- ✅ Step 2/3 有明确的**适用条件**，非适用算子跳过
- ✅ 每步都有独立的**仿真图分析 + profiling 报告 + 优化策略**输出
- ✅ Step 2/3/4 都会输出对 Step 1 tiling 策略的**修正建议**

## 优化流程

```
给定：算子类型、计算流程（kernel 代码/伪代码）、profiling 数据、仿真图（可选）

Step 1 — Tiling 理论建模 → 输出理想 tiling data
Step 2 — 通信类算子？ → 通算演算优化策略（卡间流水分析 + tiling 修正）
Step 3 — 多核间同步算子？ → 核间流水优化策略（核间流水分析 + tiling 修正）
Step 4 — 单核流水优化策略（核内流水分析 + tiling 修正）
```

每步的"流水"分析包含**仿真图解读**和 **profiling 数据**分析报告。Step 2/3/4 的策略输出均包含对 Step 1 tiling 策略的修正建议。

---

## Step 1 — Tiling 理论建模

**输入**：精确 SoC / DAV 分组、目标资源与来源、算子类型、Shape、dtype、计算流程

**过程**：根据算子 pattern 路由到对应的 Tiling 理论模型目录（详见 `references/tiling/`，入口为 `references/tiling/index.md`），输出理想 tiling data。

**输出**：
- [ ] 卡间切分方案（切分维度、通信内算子涉及）
- [ ] 多核切分方案（切分维度、单核任务量、核数）
- [ ] 单核切分方案：
  - Cube/融合类：L1 split（baseM/baseN/baseK、L1 ping-pong）+ L0 split（mL0/nL0/kL0）
  - Vec 类：UB split（block_size、repeat）
- [ ] Buffer 规划（各 buffer 用途与大小，区分 L1/L0/UB 层级）
- [ ] 融合算子内部计算单元清单；每个内嵌 MM 单独记录 `M/N/K`、dtype、layout 和 `mm_route`
- [ ] 内嵌 MM 的 L1/L0/输出流水：`L1 → L0(A2/B2) → Mmad(Cube) → Fixpipe/DataCopyCO12DstParams`、双缓冲和事件配对
- [ ] 分支场景覆盖（dtype、shape 大小、对齐）

---

### 内嵌 MatMul 路由（Cube/融合类强制）

融合算子不能只按外层名称归类。只要计算流程中出现 `mm`/`matmul`/`mmad`，就把它作为独立 MatMul 单元完成 Step 1 建模，并将其 tiling、流水和候选路线写入报告。

候选路线按以下顺序评估：

1. DAV_3510 先完成 Blaze 路线判定与评估（`blaze_native`/`blaze_custom`/`blaze_migration`/`unsupported`），命中 Blaze 路线后转 `ascendc-blaze-best-practice`（迁移路线为 `ascendc-blaze-migration`）执行。
2. Blaze 不适用、判定为 `unsupported` 或瓶颈仍在 MM 内部时，评估保留 Cube MMAD 的低阶流水重建：显式规划 `A2/B2/CO1` LocalTensor，使用 `LoadData2DParamsV2` 控制 L1 到 L0，使用 `MmadParams` + `Mmad` 执行 Cube 计算，并用 `Fixpipe`/`DataCopyCO12DstParams` 和 `MTE1`/`MTE2`/`MTE3` 事件控制流水。
3. 低阶流水重建受限时，再评估现有高阶 MatMul API 的 tiling、数据流、ping-pong、`unitflag`、`k-tail` 和 `Fixpipe` 配置。
4. AIV 替换是独立的计算引擎替换路线，只能在 Blaze、低阶 Cube、Cube 高阶和 AIV 的同 case、同环境、同协议实测对照后进入候选。低 MAC 利用率、较小 `K` 或“外层是融合 backward”不是充分证据。

“低阶 Cube MM 重建”与“把 MM 改成 AIV 小算子组合”必须使用不同的 `route_id` 和收益假设；`register-resident-vector-chain` 只用于已确认的向量计算链，不得用来证明 MM 应迁移到 AIV。

---

## Step 2 — 通算演算优化策略（卡间流水）

**适用条件**：通信类算子（如 AllReduce、AllGather、ReduceScatter 等）——含无计算融合的纯集合通信算子。

非通信类算子**跳过**此步骤。

> **边界澄清**：不要按"是否为 MC² 通算融合"来判定 Step 2 适用性——MC² 与否只决定下方"前置强制检查"和输出清单中的 MC² 必选项是否生效；纯集合通信算子（无内嵌计算单元）同样执行本步骤，只是无需 MC² 采集项，按通信掩盖与常规卡间流水优化推进。

**输入**：Step 1 的 tiling data + 计算流程 + 仿真图 + profiling 数据

> **⚠️ MC² 算子前置强制检查**：若算子为 MC² 通算融合算子（存在 fork 多 rank + CrossCoreFlag 同步），在进入流水配平分析前**必须**完成以下两项数据采集，缺失任何一项须返回数据采集阶段补采，不得在缺少数据的情况下凭经验给出切分方案：
>
> 1. **TilingData 采集**：从 host 代码打印获取 baseM/baseN/baseK/usedCoreNum。若 host 代码未打印，须在源码副本中启用 PrintTilingData 后重新运行。详见 `comm-compute/index.md` Step 0 和 `comm-compute/pipeline_balancing.md`「TilingData 依赖」章节。
> 2. **隔离测试**：在 tileCnt=1 下分别注释通信/计算 process，实测 T_comm 和 T_compute，计算 R 值判定 Bound 类型。详见 `comm-compute/bound_diagnosis.md`。

**过程**：加载 `references/comm-compute/`，分析卡间通信与计算的流水重叠，识别通信瓶颈。

**输出**：
- [ ] 卡间流水仿真图分析
- [ ] 通信/profiling 数据报告
- [ ] **TilingData 采集**（MC² 算子必选）：baseM/baseN/baseK/usedCoreNum（含长块/短块各自的 tiling data）
- [ ] **隔离测试与 Bound 判定**（MC² 算子必选）：T_comm/T_compute/R 值
- [ ] 卡间流水优化策略
- [ ] 对 Step 1 tiling 策略的修正建议
- [ ] **加速比与掩盖率**（MC² 算子必选）：基于隔离测试的 T_comm / T_compute 和各方案实测耗时，按 `references/comm-compute/index.md` 的性能指标公式核算

> 加载 `references/comm-compute/`，按 Bound 判定 → 流水配平 → 膨胀分析 → 性能指标核算流程输出卡间流水优化策略与 Tiling 修正建议。

---

## Step 3 — 核间流水优化策略

**适用条件**：涉及多核间同步的算子（如跨核同步、核间数据依赖等）。

非多核同步算子**跳过**此步骤。

**输入**：Step 1 的 tiling data + 计算流程 + 仿真图 + profiling 数据

**过程**：加载 `references/inter-core-pipeline/`，分析多核间的流水并行效率。

当循环中反复出现跨核汇合，或多个核向数据决定的同一目标写入时，先按[核间依赖与同步分析](references/inter-core-pipeline/index.md)建立生产者、可见性和归约顺序约束，再估算分块、分组归约等候选的同步次数、额外流量与空间代价。候选方案须满足原计算语义；不能仅凭聚合 profiling 推断某次同步的独立耗时。

**输出**：
- [ ] 核间流水仿真图分析
- [ ] 核间 profiling 数据报告
- [ ] 核间流水优化策略
- [ ] 对 Step 1 tiling 策略的修正建议

> 无跨核同步或共享目标写入时跳过本步骤；资料不足时列出缺失证据和可验证假设，不将猜测写成已确认瓶颈。

---

## Step 4 — 单核流水优化策略

**适用条件**：所有算子。

**输入**：Step 1 的 tiling data + 计算流程 + 仿真图 + profiling 数据 + 前几步的策略输出

**过程**：加载 `references/single-core-pipeline/`，按其两级结构展开——`index.md` 路由判定 bound → 对应诊断页（vec/scalar/memory/no-bound.md）看细分特征 → `techniques/` 取技术卡片实施。采集与指标口径统一见 `metrics.md`（msprof）。

**输出**：
- [ ] 核内流水仿真图分析
- [ ] 核内 profiling 数据报告（bound 诊断结论）
- [ ] 核内流水优化策略
- [ ] 对 Step 1 tiling 策略的修正建议
- [ ] 最终优化方案汇总

---

## ⚠️ 常见误解澄清

### 误解1：将流程简化为 3 步
❌ **错误理解**：Tiling 建模 → 流水分析 → Tiling 回修  
✅ **正确理解**：Tiling 建模（Step 1）→ 卡间流水（Step 2，条件）→ 核间流水（Step 3，条件）→ 单核流水（Step 4）

**说明**：Step 2/3/4 不是一个"流水分析"步骤的不同维度，而是三个独立的步骤，各有明确的适用条件和输出要求。

### 误解2：将 Step 2/3/4 合并为"流水分析"
❌ **错误**：Step 2/3/4 是同一个"流水分析"步骤的卡间、核间、核内三个维度  
✅ **正确**：Step 2/3/4 是三个独立的分析步骤，按条件选择性执行，各自产生独立的优化策略输出

**说明**：
- Step 2 仅适用于通信类算子（如 AllReduce）
- Step 3 仅适用于多核间同步算子
- Step 4 适用于所有算子
- 非适用算子会跳过对应步骤

### 误解3：混淆卡间通信和核内搬运的概念
❌ **错误**：认为卡间流水只关注计算，不关注数据搬运  
✅ **正确**：Step 2 卡间流水关注卡间数据通信与计算的重叠（通算演算）

**说明**：
- 卡间通信涉及的 DDR→UB 是指**远端卡的数据**通过卡间链路（HCCS/NVLINK）传输到**本卡 UB**
- 这与 Step 4 核内流水中的 DDR→UB（**本卡** HBM/DDR → 本卡 L1 → 本卡 UB）是不同的数据路径
- Step 2 关注：远端数据通过卡间链路到达本卡，与本卡计算的流水重叠（通算演算优化）
- Step 4 关注：本卡内部的数据搬运（本卡 GM→L1→UB）与本卡计算的流水重叠

### 误解4：所有算子都执行全部 4 个步骤
❌ **错误**：每个算子都要经过 Step 1 → Step 2 → Step 3 → Step 4  
✅ **正确**：Step 1 和 Step 4 是必选，Step 2/3 根据算子类型选择性执行

**典型流程示例**：
- **普通计算算子**（如 MatMul）：Step 1 → Skip Step 2 → Skip Step 3 → Step 4
- **通信类算子**（如 AllReduce）：Step 1 → Step 2 → Skip Step 3 → Step 4
- **多核同步算子**：Step 1 → Skip Step 2 → Step 3 → Step 4
