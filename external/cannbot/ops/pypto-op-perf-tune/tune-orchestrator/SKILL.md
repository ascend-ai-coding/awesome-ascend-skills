---
name: external-cannbot-ops-pypto-op-perf-tune-tune-orchestrator
description: PyPTO 算子性能调优流程编排器。控制调优流程按固定顺序与完整迭代轮次执行、实时更新 Todo 清单。当需要编排 PyPTO 算子性能调优全流程（性能分析→调优→精度验证迭代）时使用此技能。
original-name: tune-orchestrator
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
license: UNKNOWN
---

# PyPTO 算子性能调优编排器

## 角色

你是性能调优流程控制器（orchestrator）。你的唯一职责是确保调优流程按照固定顺序严格执行、迭代轮次完整执行、Todo 清单实时更新。

你不负责具体的优化建议或代码修改，只负责：
1. 控制流程推进
2. 校验每一步的完成条件
3. 强制维护 Todo 清单
4. 阻止跳跃执行
5. 控制迭代轮次的完整执行

### ⛔ 禁止自主判断铁律

**agent 不得自主判断"该优化什么"或"是否可以停止了"，所有决策必须由以下三个依据驱动：**

| 决策 | 唯一依据 | 禁止的行为 |
|------|---------|-----------|
| 选择优化点 | **调优点清单**（Phase A/B 分析产出） | 禁止凭"感觉"或"经验"选择优化点 |
| 是否继续下一轮 | **退出条件表**（达标/连续无提升阈值/清单全尝试） | 禁止凭"觉得差不多了"提前终止 |
| 是否进入下一阶段 | **PHASE_SUMMARY 摘要完整性校验** | 禁止不做摘要直接推进 |
| 是否跳过某个优化点 | **调优点清单中的分析结论**（如"❌不适用"须标注原因） | 禁止跳过未尝试的优化点 |
| 是否进行第 2/3 轮外循环 | **外循环轮次计数器 n < 3** | 禁止跳过剩余轮次 |

**⛔ 所有流程推进必须基于状态机转移条件的硬性判断，禁止用"我认为"替代条件检查。**

---

## 三级状态机

调优流程是一个严格的三级状态机，只有一个方向，禁止回退或跳跃。

### 第一级：主流程状态机

```
INIT → S1_SETUP → S2_COLLECT → S3_ANALYZE → S4_TUNE → S5_REPORT → DONE
```

### 第二级：S4_TUNE 子阶段状态机

```
S4_TUNE = PHASE_FRONTEND → PHASE_SUMMARY_F
    → [PHASE_SWIMLANE_n → PHASE_SUMMARY_S_n → PHASE_INCORE_n → PHASE_SUMMARY_I_n] × ≤3
    → S5_REPORT

提前退出点：
  PHASE_SUMMARY_F:       达标 → S5_REPORT
  PHASE_SUMMARY_I_n (n=1,2): 达标 → S5_REPORT，未达标 → SWIMLANE_{n+1}
  PHASE_SUMMARY_I_3:     无条件 → S5_REPORT
```

每个 PHASE_SUMMARY 是独立状态，负责：1) 生成阶段交接摘要  2) 通过 Task 工具启动新 subagent 会话进行隔离

### 第三级：每个 PHASE 内部的迭代循环

```
每个 PHASE 内部反复执行以下循环，直到退出条件满足：

┌──────────────────────────────────────────────────────────┐
│  ITER_START（选择优化点）                                  │
│      │                                                   │
│      ▼                                                   │
│  ITER_MODIFY（修改代码，单参数）                            │
│      │                                                   │
│      ▼                                                   │
│  ITER_VERIFY（验证精度）                                   │
│      │                                                   │
│      ├─ 精度失败 → ITER_ROLLBACK（回退代码）               │
│      │                   │                                │
│      │                   ▼                                │
│      │               ITER_RECORD（记录失败）               │
│      │                   │                                │
│      │                   ▼                                │
│      │               ITER_JUDGE（判断退出条件）            │
│      │                   │                                │
│      ├───────────────────┘                                │
│      │                                                   │
│      ▼ （精度通过）                                        │
│  ITER_MEASURE（测试性能）                                  │
│      │                                                   │
│      ├─ 性能回退 → ITER_ROLLBACK → ITER_RECORD            │
│      │                   │                                │
│      │                   ▼                                │
│      │               ITER_JUDGE（判断退出条件）            │
│      │                   │                                │
│      ├───────────────────┘                                │
│      │                                                   │
│      ▼ （性能提升或持平）                                   │
│  ITER_RECORD（记录结果，更新Todo）                          │
│      │                                                   │
│      ▼                                                   │
│  ITER_JUDGE（判断退出条件）                                │
│      │                                                   │
│      ├─ 未达标且轮次未耗尽 → 回到 ITER_START               │
│      │                                                   │
│      └─ 达标或轮次耗尽 → 退出迭代循环                    │
│         （进入对应的 PHASE_SUMMARY 状态）                  │
└──────────────────────────────────────────────────────────┘

⛔ 所有 ROLLBACK 路径（精度失败/性能回退）统一经过：
  ITER_ROLLBACK → ITER_RECORD → ITER_JUDGE
  ITER_JUDGE 统一判定：失败是否超限、是否继续下一轮。
```

---

## 全状态定义与转移条件

### 第一级状态转移表

| 当前状态 | 完成条件 | 转移到 | 强制动作 | ⛔ state_transition 门控 |
|---------|---------|--------|---------|------------------------|
| INIT | 用户已提供目标（如"提升X倍"/"≤X us"）或用户确认 | S1_SETUP | 记录目标到 Todo，计算目标执行时间 | 完成时调用 `state_transition(opDir, "complete_stage", 0)` |
| S1_SETUP | 环境检查全部通过（见下方「环境检查清单」）+ 精度通过 | S2_COLLECT | ⛔ S1a 环境检查：加载 pypto-environment-setup 技能，逐项检查环境；⛔ S1b 精度校验：环境通过后运行精度校验；⛔ 全部通过后强制创建完整 Todo | S1a通过时调用 `state_transition(opDir, "start_stage", 1)`，全部通过后调用 `state_transition(opDir, "complete_stage", 1)` |
| S2_COLLECT | swimlane.json 存在 | S3_ANALYZE | 无 | 完成时调用 `state_transition(opDir, "start_stage", 2)` → 验证后调用 `complete_stage(2)` |
| S3_ANALYZE | 性能报告文件存在 | S4_TUNE | 记录基准性能 | 完成时调用 `state_transition(opDir, "start_stage", 3)` → 验证后调用 `complete_stage(3)` |
| S4_TUNE | ⛔ 提前达标（PHASE_SUMMARY_F / PHASE_SUMMARY_I_n 达标时直接→S5）或 外循环≤3轮全部完成（FRONTEND + [SWIMLANE→INCORE]×≤3 及其对应的 SUMMARY） | S5_REPORT | ⛔ 各PHASE间通过SUMMARY强制交接；⛔ 每轮外循环从SWIMLANE_n入口重新独立采集性能数据 | 每个 PHASE 完成时调用 `state_transition(opDir, "complete_stage", 4)` |
| S5_REPORT | 报告文件已保存 | DONE | ⛔ 还原 debug_options | 完成时调用 `state_transition(opDir, "complete_stage", 5)` |

### 第二级状态转移表（S4_TUNE 内部）

> **state_transition 调用说明**：表中所有 `state_transition` 调用均在**当前会话**中执行（即生成摘要的那个会话），在启动 Task subagent **之前**完成。新 subagent 通过摘要获取状态，不调用 state_transition。

| 当前 PHASE | 进入动作 | 退出条件 | 转移到 | ⛔ state_transition 门控 |
|-----------|---------|---------|--------|------------------------|
| PHASE_FRONTEND | 加载 tune-frontend 子技能 + ⛔ 执行阶段A(全局分析:Loop+常量+Reshape+TileShape) + 阶段B(局部分析:数据操作) + ⛔ 编排器核查分析制品 + 生成优化点清单 | ⛔ 见「退出条件详细定义」 | PHASE_SUMMARY_F | 退出时调用 `state_transition(opDir, "start_stage", 4)` |
| PHASE_SUMMARY_F | 生成 FRONTEND 阶段交接摘要 + ⛔ **检查是否已达标** + Task 启动新 subagent 进行隔离 | 摘要已生成 且 新 subagent 已启动 | 达标→S5_REPORT / 未达标→PHASE_SWIMLANE_1 | Task 启动前调用 `state_transition(opDir, "complete_stage", 4)` |
| PHASE_SWIMLANE_n (n=1,2,3) | ⛔ **外循环第{n}轮**：独立采集性能数据(S2型) → 独立分析泳道图(S3型) → 加载 tune-swimlane 子技能 → ⛔ **核使用率分析** → 生成调优点清单 | ⛔ 见「退出条件详细定义」 | PHASE_SUMMARY_S_n | 退出时调用 `state_transition(opDir, "complete_stage", 4)` |
| PHASE_SUMMARY_S_n | 生成 SWIMLANE 阶段交接摘要 + 检查达标状态 + 按序列推进（见「阶段交接流程」） | 摘要已生成 | 达标→S5_REPORT / 未达标→PHASE_INCORE_n | - |
| PHASE_INCORE_n (n=1,2,3) | ⛔ **外循环第{n}轮**：独立采集性能数据(S2型) → 独立分析性能数据(S3型) → 加载 tune-incore 子技能 → ⛔ 生成调优点清单 | ⛔ 见「退出条件详细定义」 | PHASE_SUMMARY_I_n | 退出时调用 `state_transition(opDir, "complete_stage", 4)` |
| PHASE_SUMMARY_I_n | 生成 INCORE 阶段交接摘要 + 检查达标状态 + 按序列推进（见「阶段交接流程」） | 摘要已生成 | 达标→S5_REPORT / 未达标→PHASE_SWIMLANE_{n+1}(若还有) / 队列空→S5_REPORT | - |

### 第三级状态转移表（迭代轮次内部）

⛔ 这是调优过程中最核心的控制，每一步都是独立操作，必须逐步执行。

| 迭代子状态 | 执行内容 | 完成标志 | 转移到 | 失败处理 |
|-----------|---------|---------|--------|---------|
| ITER_START | 根据子技能阶段A/B分析结论和 [shared/optimization_catalog.md](../shared/optimization_catalog.md) 选择一个优化点。⛔ 必须按调优点清单选择下一个未尝试的优化点，禁止凭直觉选题。⛔ FRONTEND阶段：优化点必须引用阶段A或B分析表格中的具体行号。⛔ 退出前提：只有调优点清单中所有项均已标记为 ✅已尝试 或 ❌已失败 或 ❌不适用(须注明原因) 时，才允许"无优化点可选→退出迭代循环"。**清单中存在未标记项时，禁止以"无优化点可选"为由退出，必须逐一尝试所有标记为 ⏳待尝试 的项。** | 确定了要改什么参数（含编号+分析依据） | ITER_MODIFY | 无优化点可选 **且 清单已全部标记** → 退出迭代循环（进入对应 PHASE_SUMMARY） |
| ITER_MODIFY | 修改代码，只改一个参数 | 代码已修改 | ITER_VERIFY | - |
| ITER_VERIFY | 运行测试用例验证精度 | 输出含 "passed" | ITER_MEASURE | → ITER_ROLLBACK |
| ITER_MEASURE | 运行用例采集性能数据 | 新的性能数据已获取 | ITER_RECORD | → ITER_ROLLBACK |
| ITER_RECORD | 更新 Todo 性能记录表 | Todo 已更新 | ITER_JUDGE | - |
| ITER_JUDGE | 检查退出条件 | 判定结果 | ITER_START 或 退出迭代循环（进入对应 PHASE_SUMMARY） | - |
| ITER_ROLLBACK | 回退代码修改 | 代码已恢复 | ITER_RECORD | - |

### S1_SETUP 结构

S1_SETUP 包含两个子阶段：
1. **S1a：环境检查** — 执行主技能步骤 1.0 的环境检查流程（含完整清单和修复步骤），⛔ 不可跳过
2. **S1b：精度校验** — 环境通过后执行主技能步骤 1.2-1.3 的精度校验流程

两个子阶段全部通过后，转移到 S2_COLLECT。

### ⛔ 通用原则：SWIMLANE/INCORE 进入时独立采集与分析

**⛔ ⛔ ⛔ 强制门控：PHASE_SUMMARY_F/S/I 之后的所有 PHASE（SWIMLANE_n、INCORE_n）进入时，必须重新采集性能数据并重新分析。禁止复用上一阶段或 S2_COLLECT 的数据！⛔ ⛔ ⛔**

**PHASE_FRONTEND 之后的 SWIMLANE 和 INCORE 每个外循环轮次进入时，都必须独立进行性能数据采集和泳道图分析。** PHASE_FRONTEND 直接复用 S2_COLLECT + S3_ANALYZE 的数据，无需重复采集。原因：

1. **性能特征逐轮变化**：SWIMLANE_n 基于 INCORE_{n-1} 修改后的代码运行，执行时间、核心利用率、气泡率等指标已显著改变，上一轮数据无法反映当前实现的实际表现
2. **决策依赖最新数据**：核使用率分析、瓶颈识别、合图策略等所有优化决策必须基于当前代码的泳道图数据
3. **NOT FULL 判定敏感**：TileShape 优化会改变任务数，进而改变核使用率。每次回到核使用率分析前都必须重新采集

**入口数据采集与分析的通用流程**（每个外循环轮次的 SWIMLANE_n 和 INCORE_n 入口执行）：

```
Step 0: 独立性能数据采集
  ├─ 确保 debug_options={"runtime_debug_mode": 1}
  ├─ 运行算子用例（python3 ... --run-mode npu）
  └─ 确认 swimlane.json 已基于当前代码生成

Step 0a: 独立性能数据分析
  ├─ 记录本轮入口基准：执行时间、核心利用率、气泡率等
  ├─ 识别当前瓶颈（最大耗时子图、最严重气泡等）
  └─ 将入口基准写入 Todo 作为本轮的 Baseline
```

> **与 S2_COLLECT / S3_ANALYZE 的关系**：S2+S3 提供全局初始基准（Initial Baseline），PHASE_FRONTEND 直接复用此基准。3 轮外循环中每轮的 SWIMLANE_n/INCORE_n 入口独立采集提供的是**当前轮次基准（Round_n Baseline）**，因每轮间代码已变化，不可复用历史数据。

### 各 PHASE 调优流程

#### PHASE_FRONTEND 开箱调优流程

```
┌──────────────────────────────────────────────────────────┐
│              开箱性能调优流程                              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ⛔ 阶段A: 全局分析（对应 F-1~F-10，必须先完成）            │
│     ├─ A1: 扫描所有 pypto.loop / Python for 调用          │
│     │      → 产出 Loop结构分析表                           │
│     ├─ A2: 扫描所有常量定义和 jit 参数                      │
│     │      → 产出 常量依赖关系图（对应F-11）                 │
│     ├─ A3: 扫描所有 pypto.reshape/squeeze/unsqueeze 调用    │
│     │      → 产出 Reshape全局分析表（对应F-4）               │
│     ├─ A4: 扫描所有 operation + TileShape                  │
│     │      → 产出 基本块(TileShape)审查表（对应F-9,F-10）   │
│     └─ A5: 汇总优化建议                                    │
│            → 产出 基于分析的优化建议清单                     │
│                                                          │
│     ⛔ 编排器核查：A1+A2+A3+A4+A5 制品齐全 → 进入阶段B      │
│                                                          │
│  ⛔ 阶段B: 局部分析（对应 F-11~F-15，F-11已在A2分析，必须先完成） │
│     ├─ B1: 数据操作分析(NZ/Transpose/冗余搬运)             │
│     │      → 产出 数据操作分析表（对应F-12~F-15）           │
│     └─ B2: 汇总最终优化点排序清单                           │
│            → 产出 最终优化点清单（引用catalog编号）          │
│                                                          │
│     ⛔ 编排器核查：B1+B2 制品齐全 → 进入阶段C               │
│                                                          │
│  阶段C: 逐项优化（基于A+B的分析结论执行）                  │
│     ├─ 按优化点排序清单逐项执行 ITER 循环                  │
│     │   ├─ P0 任务粒度: F-1~F-3                             │
│     │   ├─ P0 Reshape全局: F-4                              │
│     │   ├─ P1 Loop写法: F-5~F-8                             │
│     │   ├─ P2 TileShape: F-9, F-10                          │
│     │   ├─ P3 常量配置: F-11                                 │
│     │   └─ P3 其他: F-12~F-15                               │
│     ├─ 每项优化必须引用阶段A/B的分析结论                     │
│     └─ 每项优化后验证精度+性能                               │
│                                                          │
│  ⛔ 见「退出条件详细定义」→ 退出，进入 PHASE_SUMMARY_F        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**⛔ 阶段A/B 分析核查规则**：

编排器在阶段C开始前，必须确认以下制品已生成：

```
阶段A 核查清单：
□ A1: Loop结构分析表已填写（每个 pypto.loop/for 一行）
□ A2: 常量依赖关系图已填写（每个常量一行，含引用位置）
□ A3: Reshape全局分析表已填写（每个 reshape 一行，含冗余判定）
□ A4: 基本块(TileShape)审查表已填写（每个 operation 一行，含TileShape匹配检查）
□ A5: 基于分析的优化建议清单已列出

阶段B 核查清单：
□ B1: 数据操作分析表已填写（覆盖 F-12~F-15）
□ B2: 最终优化点排序清单已生成（引用 catalog 编号，按 P0→P3 排序）

⛔ 以上 7 项全部 ✅ 后，才允许进入阶段C 开始 ITER 循环。
⛔ 缺少任何一项 → 要求补充分析，禁止跳过进入优化。
```

#### PHASE_SWIMLANE 深度调优流程（外循环第 n 轮）

```
┌────────────────────────────────────────────────────┐
│          深度性能调优流程（外循环第n轮）              │
├────────────────────────────────────────────────────┤
│                                                    │
│  ⛔ 阶段入口: 按上方「通用原则」执行 Step 0+0a   │
│     SWIMLANE 特有: ⛔ 记录泳道图重采次数(Todo中)  │
│                      n=1时基于 FRONTEND 优化后代码  │
│                      n≥2时基于 INCORE_{n-1} 代码    │
│                                                    │
│  ⛔ 泳道图回流追踪：Step 1 和 Step 1a 中               │
│  回到 Step 0 时，Todo 中 SWIMLANE_n 累计泳道图重采次数│
│  自增。此回流不改变算子代码，只重新采集泳道图数据。    │
│  ⚠️ 此回流在当前会话内执行，不触发 Task subagent 隔离。│
│  只有 PHASE_SUMMARY 才触发 Task subagent 隔离。        │
│                                                    │
│  1. ⛔ 核使用率分析（合图前置条件）                   │
│     ├─ 运行 analyze_core_usage.py 统计核使用率        │
│     ├─ 核未满 → 调整 TileShape 用满核               │
│     │   → 回到 阶段入口(Step 0) 重新采集泳道图数据    │
│     │     ⛔ Todo: SWIMLANE_n 泳道图重采次数 +1       │
│     └─ 核已满/TileShape已充分调整 → 进入 Step 1a    │
│                                                    │
│  1a. ⛔ 负载均衡优化（核填充后强制执行）             │
│     ├─ 按 total(us) 排序识别瓶颈子图                │
│     ├─ 瓶颈差距 > 20% → 调整 TileShape             │
│     │   → 回到 阶段入口(Step 0) 重新采集泳道图数据   │
│     │     ⛔ Todo: SWIMLANE_n 泳道图重采次数 +1      │
│     └─ 瓶颈差距 < 20% 或 3轮无改善 → 进入 Step 2   │
│                                                    │
│  2. 合图调优                                       │
│     ├─ AIC 核满 → L1Reuse / CubeNBuffer            │
│     └─ AIV 核满 → VecNBuffer / sg_set_scope        │
│                                                    │
│  3. 调度策略调优                                   │
│     └─ device_sched_mode 调整                      │
│                                                    │
│  ⛔ 见「退出条件详细定义」→ 退出，进入 PHASE_SUMMARY_S_n  │
│                                                    │
└────────────────────────────────────────────────────┘
```

#### PHASE_INCORE 核内调优流程（外循环第 n 轮）

```
┌────────────────────────────────────────────┐
│        核内性能调优流程（外循环第n轮）       │
├────────────────────────────────────────────┤
│                                            │
│  ⛔ 阶段入口: 按上方「通用原则」执行 Step 0+0a    │
│     INCORE 特有: 基于 SWIMLANE_n 优化后的代码     │
│                   Step 0a 额外: 找耗时最长 task    │
│                                            │
│  1. 定位瓶颈 task                          │
│     └─ 通过泳道图找到耗时最长的 task       │
│                                            │
│  2. 分析 task 特征                         │
│     ├─ 输入输出 Shape                      │
│     ├─ Operation 类型                      │
│     └─ 依赖关系                            │
│                                            │
│  3. 选择优化策略                           │
│     ├─ 特殊 Shape → Vector 预处理          │
│     ├─ 尾轴过小 → concat/transpose/reshape │
│     ├─ 冗余依赖 → 增加冗余计算             │
│     ├─ L2 Cache → Cache 策略优化           │
│     └─ Operation → 检查实现/替代方案       │
│                                            │
│  ⛔ 见「退出条件详细定义」→ 退出，进入 PHASE_SUMMARY_I_n  │
│                                            │
└────────────────────────────────────────────┘
```

---

## ⛔ 每次回复前的强制自检

在生成任何回复之前，你必须完成以下自检。如果任何一项未通过，必须先处理该项，禁止继续执行其他操作。

```
□ Q1: 当前处于哪个状态？
      第一级: [INIT / S1 / S2 / S3 / S4 / S5 / DONE]
      第二级: [FRONTEND / SUMMARY_F / SWIMLANE_n(n=1,2,3) / SUMMARY_S_n / INCORE_n(n=1,2,3) / SUMMARY_I_{1,2} / SUMMARY_I_3]（仅 S4 时）
      第三级: [ANALYZE_A / ANALYZE_B / SWIMLANE_STEP0 / SWIMLANE_CORE_USAGE / SWIMLANE_LOAD_BALANCE / START / MODIFY / VERIFY / MEASURE / RECORD / JUDGE / ROLLBACK]
              （FRONTEND: ANALYZE_A/B; SWIMLANE: SWIMLANE_STEP0/SWIMLANE_CORE_USAGE/SWIMLANE_LOAD_BALANCE/ITER; INCORE: START/...）

□ Q1a: ⛔ 阶段交接检查（Task 返回后 / 启动新 Task 前必检）
        PHASE_QUEUE 的执行进度：已执行 [已完成的PHASE列表]，下一个是 [PHASE名]
        → Task 刚返回：摘要中达标状态 = ✅ → 跳出循环，进入 S5_REPORT
        → Task 刚返回：摘要中达标状态 = ❌ → 继续启动下一个 PHASE 的 Task
        → 队列已空（6个PHASE全部执行完）→ 进入 S5_REPORT

□ Q1b: ⛔ Task 粒度检查（启动 Task subagent 前必检）
        本次 Task 只包含一个 PHASE 吗？
        → 一个 Task 包含 SWIMLANE + INCORE → ⛔ 禁止！必须拆分为两个独立 Task
        → 一个 Task 只包含一个 PHASE → ✅ 允许

□ Q2: 当前子状态的完成条件是否满足？
       → 不满足 → 继续执行当前子状态，禁止推进

□ Q2a: （仅 FRONTEND 阶段的 ITER_START 子状态时检查）
        FRONTEND 有独立的第三级状态：ANALYZE_A → ANALYZE_B → ITER 循环。
        → ANALYZE_A 未完成（阶段A 分析表未填写）→ 当前状态应为 ANALYZE_A，禁止进入 ITER
        → ANALYZE_B 未完成（阶段B 分析表未填写）→ 当前状态应为 ANALYZE_B，禁止进入 ITER
        → 两者均已完成 → 允许进入 ITER_START

□ Q2b: （仅 SWIMLANE 阶段）当前处于哪个子阶段？
       → SWIMLANE_STEP0（泳道图采集/回流重采）: debug_options 已启用？泳道图重采次数已记录？
       → SWIMLANE_CORE_USAGE（核使用率分析）: analyze_core_usage.py 已运行？
       → SWIMLANE_LOAD_BALANCE（负载均衡分析）: 瓶颈子图已识别？
       → ITER（优化迭代）: 优化点清单已生成？

□ Q3: 本轮回复是否只推进了一个子状态？
       → ITER_START → ITER_MODIFY  ✅（一个）
       → ITER_START → ITER_MODIFY → ITER_VERIFY  ❌（三个，太多了）

□ Q4: 强制动作是否已完成？
        → S1→S2: Todo 已创建？state_transition 已调用？
        → FRONTEND 阶段C 前置: 阶段A+B 分析制品已核查？
        → PHASE→SUMMARY: 阶段交接摘要已生成？
        → SUMMARY_F/S→下一PHASE: Task 已启动新 subagent 完成隔离？
        → S5→DONE: debug_options 已还原？state_transition 已调用？
        → ⛔ 当前状态转移是否需要调用 `state_transition` 工具？
          查阅上方状态转移表的「state_transition 门控」列。需要则必须调用，禁止跳过。

□ Q5: Todo 是否最新？（⛔ 必须按下方规则逐项验证，仅问"是否已更新"不够）
       → 规则1: 刚完成 ITER_MEASURE/ROLLBACK → 迭代轮次记录表中必须有对应轮次的新行
       → 规则2: 刚完成 ITER_JUDGE → 当前迭代状态中的"连续无提升""失败累计"必须已更新
       → 规则3: 刚完成 PHASE_SUMMARY → 阶段状态（✅/⏸️）必须已更新
       → ⛔ 不确定时 → 通过 `bash` 执行 `todowrite` 重新读取 Todo 验证
         如果 `todowrite` 返回的 Todo 中缺少本轮记录 → **必须先补充，禁止推进**

□ Q6: （仅 ITER_START 时）本次选择的优化点是否来源于阶段A/B的分析结论？
        → 无分析结论支撑 → 禁止执行，回到分析阶段补充
        → 有分析结论支撑 → 确认引用了分析表格的具体行号

□ Q6a: （仅 ITER_START 判定"无优化点可选"时）调优点清单是否已全部标记？
        → 检查优化点清单中所有项的"状态"列：不存在 ⏳待尝试 或 空状态
        → 存在未标记项 → **禁止退出！必须按清单逐一尝试未标记项**
        → 所有项已标记（均为 ✅已尝试 / ❌已失败 / ❌不适用）→ 允许退出
```

---

## 迭代轮次执行规范

### ⛔ 单次回复执行边界

每次回复最多执行 **1-2 个相邻子状态**，具体规则：

```
允许的回复粒度（紧耦合的组合允许合并）：

✅ ITER_START + ITER_MODIFY
   → 选优化点 + 改代码（必须知道改什么才能改）

✅ ITER_VERIFY（单独执行，需要等待命令执行结果）

✅ ITER_MEASURE（单独执行，需要等待性能数据）

✅ ITER_RECORD + ITER_JUDGE（记录 + 判断，紧耦合）

✅ ITER_ROLLBACK + ITER_RECORD（回退 + 记录失败，紧耦合）

禁止的回复粒度：

❌ ITER_START → ITER_MODIFY → ITER_VERIFY → ITER_MEASURE
   （一次回复跨越4步，信息量过大无法校验）

❌ 跳过 ITER_VERIFY 直接 ITER_MEASURE
   （精度都没验就测性能）

❌ 跳过 ITER_RECORD 直接下一轮 ITER_START
   （没有记录就继续，Todo 会过时）
```

### ⛔ 单参数原则

```
ITER_MODIFY 阶段，每次只允许修改一个优化参数：

✅ 只改 BLOCK_SIZE: 128 → 64
✅ 只改 TileShape: (1,1,128) → (1,1,256)
✅ 只加一个 runtime_options 参数

❌ 同时改 BLOCK_SIZE + TileShape
❌ 同时改 unroll_list + stitch_max_num
❌ 同时加两个新参数
```

### 禁止跨阶段加载子技能

```
当前 PHASE 只能加载对应的子技能，禁止提前加载后续阶段的子技能：

PHASE_FRONTEND   → 只能加载 tune-frontend
PHASE_SWIMLANE_n → 只能加载 tune-swimlane
PHASE_INCORE_n   → 只能加载 tune-incore
```

### ⛔ 禁止跨 PHASE 打包 Task subagent

```
每个 Task subagent 只能执行一个 PHASE（SWIMLANE 或 INCORE），禁止将多个 PHASE 打包进同一个 Task。

✅ 正确：Task 执行 SWIMLANE_1 → 返回 PHASE_SUMMARY_S_1 → 主 agent 判定路由 → 启动新 Task 执行 INCORE_1
✅ 正确：Task 执行 INCORE_1 → 返回 PHASE_SUMMARY_I_1 → 主 agent 判定路由(未达标,n<3) → 启动新 Task 执行 SWIMLANE_2
❌ 禁止：Task 同时执行 SWIMLANE_1 + INCORE_1 → 绕过了 PHASE_SUMMARY_S_1 检查点
❌ 禁止：Task 同时执行 SWIMLANE_1 + INCORE_1 + SWIMLANE_2 + INCORE_2

原因：
1. 每个 PHASE_SUMMARY 是独立的路由决策点，主 agent 必须在检查点判定：达标→S5 / 未达标→下一PHASE
2. 跨 PHASE 打包导致主 agent 失去路由控制权，外循环计数器无法正确递增
3. 上下文膨胀：多个 PHASE 的调试日志、失败记录会使 subagent 上下文溢出
```

---

## ⛔ 退出条件详细定义

### ⛔ 核心原则：禁止询问用户"是否继续"

**退出由本表自动判定，满足任一条件就退出，不满足就继续。禁止向用户询问"是否继续优化"。只有在用户主动说"停"时才触发"用户要求停止"条件。**

### 各 PHASE 的退出条件

**PHASE_FRONTEND 退出条件（满足任一）**：

| 条件 | 判定方法 | 动作 |
|------|---------|------|
| 性能达标 | 当前执行时间 ≤ 目标值 | → 退出迭代循环 → PHASE_SUMMARY_F |
| 调优点清单已全部尝试 且 连续5轮无提升 | 调优点清单中所有项已标记为 ✅已尝试 或 ❌已失败，且最近5轮无提升 | → 退出迭代循环 → PHASE_SUMMARY_F |
| 用户主动要求停止 | 用户明确说"停止" | → 退出迭代循环 → PHASE_SUMMARY_F |

**PHASE_SWIMLANE 退出条件（满足任一）**：

| 条件 | 判定方法 | 动作 |
|------|---------|------|
| 性能达标 | 当前执行时间 ≤ 目标值 | → 退出迭代循环 → PHASE_SUMMARY_S |
| 调优点清单已全部尝试 且 连续8轮无提升 | 调优点清单中所有项已标记，且最近8轮无提升 | → 退出迭代循环 → PHASE_SUMMARY_S |
| 用户主动要求停止 | 用户明确说"停止" | → 退出迭代循环 → PHASE_SUMMARY_S |

**PHASE_INCORE 退出条件（满足任一）**：

| 条件 | 判定方法 | 动作 |
|------|---------|------|
| 性能达标 | 当前执行时间 ≤ 目标值 | → 退出迭代循环 → PHASE_SUMMARY_I_n |
| 调优点清单已全部尝试 且 达到理论性能上限 | 清单全部尝试，核心利用率 > 80% 且 气泡率 < 10% | → 退出迭代循环 → PHASE_SUMMARY_I_n |
| 调优点清单已全部尝试 且 连续5轮无提升 | 清单全部尝试，最近5轮无提升 | → 退出迭代循环 → PHASE_SUMMARY_I_n |
| 用户要求停止 | 用户明确说停止 | → 退出迭代循环 → PHASE_SUMMARY_I_n |

### ITER_JUDGE 判定流程

**⛔ 此流程由条件自动判定。禁止在条件未触发时询问用户是否继续。满足继续条件就直接下一轮，满足退出条件就自动退出。**

```
ITER_JUDGE 执行时，按以下顺序判断：

0. ⛔ 本轮的终端状态是什么？
   → ITER_ROLLBACK 后进入（编译失败/运行错误/精度失败）: 计入"失败累计"，递增失败累计计数器。**失败不计入"连续无提升"计数器**，但会消耗优化点清单中的一项。
   → ITER_MEASURE 后进入（正常执行）: 继续下方 1-7 判断流程。

0.5. ⛔ TODO 完整性闸门（必须通过才能继续判定）
    刚完成 ITER_MEASURE（正常执行）时，检查迭代轮次记录表：
    → 检查: 最新一轮（轮次编号最大）的"状态"列是否为 ✅（性能提升）或 ❌回退（性能回退）
    → 通过条件: 最新轮次的状态列已填写 ✅ 或 ❌
    → ❌ 未通过（状态列为空或标记为 🔄 进行中）: 
       ⛔ 禁止继续判定！必须先执行 ITER_RECORD 更新 Todo，再回到 ITER_JUDGE
       ⛔ 命令: `todowrite` 补充本轮记录

    刚完成 ITER_ROLLBACK（失败分支）时，检查迭代轮次记录表：
    → 检查: 最新一轮的"结果类型"列是否已标记失败原因
    → ❌ 未通过: 禁止继续！先补充记录再判定

    刚完成 ITER_VERIFY 后直接进入的 ROLLBACK 路径，没有 ITER_MEASURE 数据：
    → 结果类型: 编译失败 / 运行错误 / 精度失败
    → 禁止状态列为空就直接退出

1. 性能是否达标？
    当前执行时间 ≤ 目标值
    → 是: 输出"达标"，退出迭代循环，进入对应 PHASE_SUMMARY
    → 否: 继续

2. 当前轮次是否有提升？
   当前执行时间 < 上一轮执行时间
   → 是: 重置连续无提升计数器为 0
   → 否: 连续无提升计数器 +1

3. 连续无提升次数是否达到阈值？
    FRONTEND: 5次, SWIMLANE: 8次, INCORE: 5次
    → 是: 进入步骤 3a 核验调优点清单
    → 否: 继续

3a. ⛔ 调优点清单完整性闸门（连续无提升达阈值时的强制核验）
    检查优化点清单状态列：**所有项已标记为 ✅已尝试 或 ❌已失败 或 ❌不适用**
    → 清单已全部标记: 输出"阶段内优化耗尽"，退出迭代循环，进入对应 PHASE_SUMMARY
    → **清单存在未标记项（⏳待尝试 或 空状态）: ⛔ 禁止退出！必须回到 ITER_START 继续尝试未标记的优化点**

3b. 优化点清单是否已全部尝试？（仅当 3 未触发时执行）
    检查优化点清单状态列：所有项已标记为 ✅已尝试 或 ❌已失败
    → 是 且 连续无提升已达阈值: 退出迭代循环
    → 是 但 连续无提升未达阈值: 继续下一轮（ITR_START 会因无优化点可选自动退出）
    → 否: 继续

3c. （仅 INCORE 阶段）是否达到理论性能上限？
    核心利用率 > 80% 且 气泡率 < 10%
    → 是: 输出"达到理论性能上限"，退出迭代循环，进入 PHASE_SUMMARY_I_n
    → 否: 继续

4. 是否达到状态看板输出时机？
    总轮次 % 5 == 0
    → 是: 输出状态看板

5. 是否超过调优时间限制？（默认 12 小时）
    → 是: 输出"调优超时"，退出迭代循环，进入对应 PHASE_SUMMARY
    → 否: 继续

6. 继续下一轮
    → 转移至 ITER_START
```

---

## 完成条件校验方法

每个状态的完成条件必须通过实际命令或文件检查验证，禁止凭记忆或假设判断。

### S1_SETUP

> S1_SETUP 包含两个子阶段：S1a 环境检查 + S1b 精度校验。环境检查通过后，继续精度校验。
> **⛔ 门控要求**：`complete_stage(1)` 必须在 S1a（环境检查全部通过）和 S1b（精度校验通过）**两个子阶段均完成后**才能调用。任一子阶段未通过，禁止调用 `complete_stage(1)`。
> 子阶段失败时使用 `fail_stage(1)` 标记，并在 Todo 中记录失败原因（区分环境问题或精度问题）。
> 精度校验流程详见主技能步骤 1.3。

### S2_COLLECT

**⚠️ output 目录位于执行算子命令时的工作目录下，非固定位置。** 需在正确的工作目录下执行验证命令。

```bash
# 在执行算子命令时的工作目录下验证
ls output/output_*/merged_swimlane.json   # 必须存在

# 如果不确定位置，从项目根目录搜索
find . -name "merged_swimlane.json" -type f
```

### S3_ANALYZE

**⚠️ 同 S2_COLLECT，需在正确的工作目录下验证。**

```bash
# 在执行算子命令时的工作目录下验证
ls output/output_*/performance_analysis_report.md   # 必须存在

# 如果不确定位置，从项目根目录搜索
find . -name "performance_analysis_report.md" -type f
```

### S4_TUNE 各 PHASE

**外循环控制**：SWIMLANE→INCORE 为一个外循环轮次，最多 3 轮。

**单 PHASE 退出条件**：⛔ 见「退出条件详细定义」

**提前终止条件**：任意 PHASE_SUMMARY（F/S_n/I_n）在生成摘要时检查全局性能目标，达标则直接路由到 S5_REPORT，不继续后续阶段。

### S5_REPORT

```
报告文件已保存到算子目录下
debug_options 已还原（runtime_debug_mode 移除或置 0）
```

**S5_REPORT TODO 管理**：
1. 更新 Todo 状态机进度：将所有状态（S1-S5）标记为 ✅
2. 输出最终状态看板（见主技能步骤 5.1 的「最终调优状态」模板）
3. Todo 中记录最终性能结果（基准→最终执行时间、累计提升百分比）和迭代统计（总轮次、成功率、最佳优化）
4. 确认 debug_options 已还原的记录写入 Todo

---

## Todo 管理规范

### ⛔ TODO 创建与执行铁律

**⛔ 铁律 1：TODO 必须一次性创建完整，包含所有 3 轮外循环的每个子阶段**

S1_SETUP 全部通过后，必须立即创建包含以下所有阶段的完整 TODO，不得遗漏任何子阶段：

```
S1 → S2 → S3 → S4 → S5
  S4 = FRONTEND → SWIMLANE_1 → INCORE_1 → SWIMLANE_2 → INCORE_2 → SWIMLANE_3 → INCORE_3
  ⚠️ 任意 PHASE_SUMMARY 达标可提前跳到 S5，不要求全部跑完
```

TODO 中必须明确列出：
- ✅ 全部 3 轮外循环的 SWIMLANE_n/INCORE_n 子阶段（创建时全量列出，达标时标记 ✅ 并跳过后续）
- ✅ 每个子阶段的轮次数、连续无提升阈值（FRONTEND:5, SWIMLANE:8, INCORE:5）
- ✅ 失败累计计数器（所有 PHASE 共用）
- ✅ 基准性能值和目标性能值

**⛔ 铁律 2：严格按 TODO 清单执行，每个阶段不可跳过**

- 每步操作前必须确认当前状态与 TODO 中的"当前阶段"一致
- 未在 TODO 中标记为 ✅ 的阶段，禁止进入
- 每个 PHASE 退出时必须更新阶段状态为 ✅
- PHASE 之间必须生成 PHASE_SUMMARY 交接摘要

**⛔ 铁律 3：每次 ITER_RECORD 后必须立即刷新 TODO**

ITER_RECORD 执行后，必须立即通过 `todowrite` 更新以下内容（缺一不可）：
- 迭代轮次记录表新增一行（含轮次编号、阶段、优化内容、执行时间、变化率、状态）
- 当前迭代状态中的"当前最佳性能"和"累计提升%"
- "连续无提升"计数器
- "失败累计"计数器
- "当前轮次"编号

**⛔ 铁律 4：禁止在 TODO 未更新的情况下推进任何步骤**

任何时候对 TODO 状态不确定 → 必须通过 `todowrite` 读取确认 → 确认无未闭合行 → 才能推进。

### 创建时机

| 时机 | 动作 |
|------|------|
| S1_SETUP 全部通过后（S1a 环境检查 + S1b 精度校验） | ⛔ 立即创建完整 Todo（含全部 3 轮外循环子阶段），不创建禁止进入 S2 |
| 每轮 ITER_RECORD 后 | ⛔ 必须立即刷新：新行 + 当前最佳 + 连续无提升 + 失败累计 |
| 每个 PHASE 完成后 | ⛔ 更新阶段状态为 ✅ |
| PHASE_SUMMARY 后 | ⛔ 附加交接摘要 + 更新状态机进度 |

### ⛔ TODO 完整性自检（所有阶段切换/退出前的强制门控）

以下任何情况发生时，**必须先执行 `todowrite` 工具读取当前 Todo 列表，逐项核验**，核验通过后才能推进：

| 时机 | 核验内容 | 未通过处理 |
|------|---------|-----------|
| ITER_MEASURE 完成 → 进入 ITER_RECORD 前 | 本轮执行时间是否已记录？ | 先记录再进 RECORD |
| ITER_ROLLBACK 完成 → 进入 ITER_RECORD 前 | 回退原因(类型:列)是否已注明（编译失败/精度失败/性能回退）？ | 先注明原因再记录 |
| ITER_JUDGE 判定退出 → 进入 PHASE_SUMMARY 前 | 迭代轮次记录表中最后一行状态列是否为 ✅ 或 ❌，无空状态行 | ⛔ 禁止退出，先补齐记录 |
| PHASE_SUMMARY 完成 → 进入 Task subagent 前 | 阶段状态 ✅/⏸️ 与实际进度一致？轮次记录表无未闭合行？ | ⛔ 禁止进入 Task，先修正 Todo |
| S5_REPORT 完成 → DONE | debug_options 已还原？所有状态标记为 ✅？ | ⛔ 禁止完成 |

**⛔ 核验命令**：

```bash
# 通过 todowrite 读取当前 Todo 列表，检查是否有未闭合的轮次记录
todowrite  # 读取当前 Todo 清单
```

- 检查返回的 Todo 列表中的 "迭代轮次记录" 部分
- 最后一行状态列如果为空 或 标记为 🔄 → **未闭合轮次，禁止推进**
- 必须通过 `todowrite` 补充完整后再继续

**⛔ 强制原则**：任何时候，如果对 Todo 状态不确定，必须通过 `todowrite` 工具读取确认，禁止凭记忆假设 Todo 是最新的。

### Todo 完整模板（S1_SETUP 全部通过后立即创建）

```markdown
## 📊 [算子名称] 性能调优进度

### 目标
- 算子: [名称]
- 基准性能: [值] us
- 目标性能: [值] us (提升X倍)
- 当前最佳性能: [值] us
- 当前设备: NPU 卡 X

### 状态机进度
- ✅ S1: 环境检查 + 精度校验
- 🔄 S2: 性能数据采集
- ⏸️ S3: 性能数据分析
- ⏸️ S4: 外循环调优 (当前第 [n]/3 轮)
  - ⏸️ FRONTEND (轮次: 0, 连续无提升: 0/5, 失败累计: 0)
  - ⏸️ SWIMLANE_n (轮次: 0, 连续无提升: 0/8, 失败累计: 0)
  - ⏸️ INCORE_n (轮次: 0, 连续无提升: 0/5, 失败累计: 0)
- ⏸️ S5: 生成调优报告

### 当前迭代状态
- 外循环轮次: [n] / 3
- 当前阶段: [FRONTEND / SWIMLANE_n / INCORE_n]
- 当前第三级状态: [ANALYZE_A / ANALYZE_B / SWIMLANE_STEP0 / SWIMLANE_CORE_USAGE / SWIMLANE_LOAD_BALANCE / ITER_START / MODIFY / VERIFY / MEASURE / RECORD / JUDGE / ROLLBACK]
- 当前轮次: #N（全局递增 #1,#2,...，编号来自 catalog）
- 泳道图重采次数（SWIMLANE_n 内）: [m] 次
- 当前最佳性能: [XX] us（较基准提升 [XX]%）
- 优化点清单剩余: [X] 项（共 [Y] 项）
- 连续无提升: Z / [阈值]
- 失败累计: W 轮

### 迭代轮次记录
| 轮次 | 外循环 | 阶段 | 编号 | 优化内容 | 精度 | 执行时间(us) | 变化 | 结果类型 | 状态 |
|------|--------|------|------|---------|------|-------------|------|---------|------|
| 基准 | 0 | - | - | - | ✅ | XX | - | - | ✅ |
```

### 迭代轮次记录更新示例

```markdown
| 轮次 | 阶段 | 编号 | 优化内容 | 精度 | 执行时间(us) | 变化 | 结果类型 | 状态 |
|------|------|------|---------|------|-------------|------|---------|------|
| 基准 | 0 | - | - | - | ✅ | 79.34 | - | - | ✅ |
| #1 | 1 | FRONTEND | F-1 | BLOCK_SIZE 128→64 | ✅ | 68.54 | -13.6% | 性能提升 | ✅ |
| #2 | 1 | FRONTEND | F-8 | unroll_list [8,4,2,1] | ✅ | 74.44 | +8.6% | 性能回退 | ❌回退 |
| #3 | 1 | FRONTEND | F-10 | TileShape 2560 | ❌ | - | - | 编译失败 | ❌回退 |

### 当前迭代状态
- 外循环轮次: 1 / 3
- 当前阶段: FRONTEND
- 当前第三级状态: ITER_START
- 当前轮次: #3
- 当前最佳性能: 68.54 us（较基准 79.34 us 提升 13.6%）
- 连续无提升: 0 / 5
- 失败累计: 1 轮（#3 编译失败）
```

### 状态看板（每5轮输出）

```markdown
## 📊 调优状态看板

外循环: [n] / 3 | 当前阶段: [阶段名] | 轮次: [N]
当前最佳: [值] us | 基准: [值] us | 累计提升: [%]% | 目标: [值] us
进度: ████████░░░░░░░░ [%]%

### 本阶段统计
- 连续无提升: Z / [阈值]
- 最佳优化: [优化项] (+[%]%)
- 当前瓶颈: [描述]
```

---

## 阶段交接摘要模板（PHASE_SUMMARY_F / PHASE_SUMMARY_S / PHASE_SUMMARY_I）

以下是 PHASE_SUMMARY 状态的执行规范。每个 PHASE 退出时进入对应的 PHASE_SUMMARY，⛔ 强制按以下顺序执行，禁止跳过任何一步。每个阶段的迭代上下文通过 Task subagent 隔离为结构化交接文档，确保后续阶段以干净上下文继续。

### ⛔ 阶段交接流程（强制）

PHASE 执行顺序完全确定，主 agent 只需按序列依次启动 Task subagent，每次检查达标即可：

```
固定序列：
FRONTEND → SWIMLANE_1 → INCORE_1 → SWIMLANE_2 → INCORE_2 → SWIMLANE_3 → INCORE_3
                                                                         ↓
                                                                    全部完成 → S5_REPORT
任意阶段达标 → 提前退出 → S5_REPORT
```

主 agent 的执行逻辑：

```
PHASE_QUEUE = [SWIMLANE_1, INCORE_1, SWIMLANE_2, INCORE_2, SWIMLANE_3, INCORE_3]

FRONTEND 在主会话中执行（复用 S2+S3 的性能数据）。
FRONTEND 完成后：

for phase in PHASE_QUEUE:
    1. 生成阶段交接摘要
    2. 校验摘要完整性
    3. 启动 Task(phase)，subagent_type="pypto-op-perf-tuner"
       ⛔ 每个 Task 只执行一个 PHASE
    4. Task 返回后，检查摘要中「达标状态」：
       ✅已达标 → break（跳出循环，进入 S5_REPORT）
       ❌未达标 → continue（启动下一个 PHASE 的 Task）

循环结束（达标提前退出 或 队列耗尽）→ S5_REPORT
```

**步骤细化**：

```
每个 PHASE 的交接：
    ↓
步骤1: 生成阶段交接摘要（按下方模板）
    ↓
步骤2: 校验摘要完整性（按下方校验清单逐项确认）
    ↓
步骤3: 启动 Task subagent
    - 使用 Task 工具，subagent_type 选择 "pypto-op-perf-tuner"
    - ⛔ 每个 Task 只执行一个 PHASE，禁止跨 PHASE 打包
    - 将阶段交接摘要作为 prompt 的核心上下文传入
    ↓
步骤4: Task 返回后检查达标状态
    - 摘要中达标状态 = ✅已达标 → 结束循环，进入 S5_REPORT
    - 摘要中达标状态 = ❌未达标 → 继续下一个 PHASE（回到步骤1）
```

禁止事项：
❌ 只生成摘要不启动 Task subagent
❌ 跳过摘要直接启动 Task
❌ 不在 Task prompt 中传入完整摘要就继续
❌ 在当前会话中直接执行 SWIMLANE/INCORE（必须通过 Task subagent 隔离）
❌ 将多个 PHASE 打包进同一个 Task
❌ Task 返回后不检查达标状态就继续或跳到 S5

Task prompt 模板（两种变体）：

**变体 A — 继续下一阶段（PHASE_SUMMARY_F → SWIMLANE / PHASE_SUMMARY_S → INCORE / PHASE_SUMMARY_I_{1,2}→SWIMLANE_{n+1}）**：
"""
你是 PyPTO 算子性能调优执行器。请执行以下任务：

1. 加载下一阶段子技能：`[子技能路径]`
2. 读取当前算子代码：`[算子文件路径]`
3. 根据下方阶段交接摘要中的迭代统计，重建 Todo（状态机进度、当前迭代状态、迭代轮次记录）
4. 根据下方阶段交接摘要，从 [下一阶段名] 开始继续调优

## ⛔ 外循环状态（必须遵守）

- 当前外循环轮次: [n] / 3
- 当前执行阶段: [SWIMLANE_n / INCORE_n]
- ⛔ 本 Task 只负责当前这一个 PHASE 的执行和退出
- ⛔ 退出当前 PHASE 后，生成 PHASE_SUMMARY 摘要返回给主 agent，由主 agent 决定路由（继续下一 PHASE / 达标→S5 / 外循环耗尽→S5）
- ⛔ 禁止在 Task 内部自行决定进入下一 PHASE 或进入 S5_REPORT

## 调优执行规则（⛔ 必须遵守）

- ITER 循环：ITER_START(选优化点) → ITER_MODIFY(改1个参数) → ITER_VERIFY(验精度) → ITER_MEASURE(测性能) → ITER_RECORD(更新Todo) → ITER_JUDGE(判退出)
- 精度失败/性能回退 → ITER_ROLLBACK → ITER_RECORD → ITER_JUDGE
- 单参数原则：每次只改一个参数
- 每次回复最多推进 1-2 个子状态
- 退出条件：性能达标 / 连续无提升达阈值(FRONTEND:5,SWIMLANE:8,INCORE:5) + 清单全标记 / 用户说停
- 优化点必须来自调优点清单，禁止凭直觉选题
- 每次 ITER_RECORD 后必须立即更新 Todo
- SWIMLANE/INCORE 入口必须先独立采集性能数据

## 阶段交接摘要
[粘贴完整摘要]

## 关键文件
- 算子代码: [路径]
- 测试命令: [命令]
- 环境变量: [变量列表]

请从 [下一阶段名] 的入口开始执行（SWIMLANE/INCORE 入口需先独立采集性能数据）。
执行完成后，返回 PHASE_SUMMARY 摘要（含性能数据、已采纳/失败优化、迭代记录、达标状态），由主 agent 进行路由决策。
"""

**变体 B — 结束调优（PHASE_SUMMARY_F / PHASE_SUMMARY_S_n / PHASE_SUMMARY_I_n 达标→S5_REPORT）**：
"""
你是 PyPTO 性能调优报告生成器。请执行以下任务：

1. 根据下方阶段交接摘要，生成最终调优报告

## 报告生成步骤

1. 更新 Todo：将所有状态（S1-S5）标记为 ✅
2. 输出最终状态看板（目标达成、优化总结、性能趋势）
3. 填充最终调优报告模板（性能对比、已采纳/已失败优化、最佳配置、调优记录）
4. 保存报告到算子目录下：`{op_name}_tuning_report.md`
5. 还原 debug_options：将 `@pypto.frontend.jit` 中的 `debug_options` 移除

## 阶段交接摘要

[粘贴完整摘要]

## 关键文件

- 算子代码: [路径]
- 测试命令: [命令]
- 环境变量: [变量列表]

### 隔离规则（Task subagent 信息传递）

Task subagent 启动时，遵循以下信息传递规则：

| 信息类别 | 保留策略 | 说明 |
|---------|---------|------|
| 已采纳优化 | ✅ 完整保留 | 包含具体修改和代码位置 |
| 已失败优化 | ✅ 完整保留 | 避免重复尝试 |
| 约束与发现 | ✅ 完整保留 | 后续阶段必须遵守 |
| 当前代码配置 | ✅ 完整保留 | 需基于最新代码继续修改 |
| 性能趋势 | ✅ 浓缩保留 | 只保留趋势摘要，不保留每轮详细日志 |
| 中间调试日志 | ❌ 丢弃 | 如精度失败时的完整报错堆栈 |
| 失败代码完整内容 | ❌ 丢弃 | 已回退的代码不再需要 |
| 冗余性能对比细节 | ❌ 丢弃 | 只保留最终结果 |
| 子技能完整内容 | ❌ 不重新加载 | 下一阶段加载新的子技能 |

### 摘要校验清单

**⛔ 生成摘要后必须逐项确认，确认后立即启动 Task subagent：**

- [ ] 性能数据是否与实际一致？（执行时间、利用率、气泡率）
- [ ] 已采纳优化是否遗漏？（核对 Todo List 中的 ✅ 项）
- [ ] 已失败优化是否遗漏？（核对 Todo List 中的 ❌ 项）
- [ ] 约束与发现是否遗漏？（检查调优过程中的关键发现）
- [ ] 代码配置是否是最新的？（包含所有已采纳的修改）
- [ ] ⛔ Todo 完整性检查：
  - Todo 迭代轮次记录表中，所有已结束的轮次状态列是否均已填写 ✅ 或 ❌？
  - 阶段状态（S1-S5）是否与当前实际进度一致？
  - 当前迭代状态中的"当前轮次""连续无提升""失败累计"是否与轮次记录表一致？
  - ❌ 任何一项不一致 → **禁止进入 Task subagent！必须先修正 Todo**
- [ ] ⛔ 校验通过后，是否已启动 Task subagent？（未启动禁止进入下一阶段）
- [ ] ⛔ 达标状态校验（Task 返回后检查）：
  - 摘要中"达标状态"为 ✅已达标 → 进入 S5_REPORT
  - 摘要中"达标状态"为 ❌未达标 → 继续启动下一个 PHASE 的 Task

### 摘要模板

```markdown
## 📋 阶段交接摘要：[阶段名] → [下一阶段名/完成]

### 0. 执行进度
- 已完成阶段: [FRONTEND, SWIMLANE_1, ...]
- 当前 PHASE: [刚完成的 PHASE 名]
- 队列剩余: [INCORE_1, SWIMLANE_2, ...]（按序列）

### 1. 性能状态
- 基准: [值] us → 当前: [值] us (累计提升 [%]%)
- 本阶段提升: [%]%
- 核心利用率: [%]%
- 气泡率: [%]%
- 调优目标: [目标] us（差距: [%]%）
- 达标状态: ✅已达标 / ❌未达标

### 2. 已采纳优化
| # | 轮次 | 优化项 | 修改内容 | 收益 | 代码位置 |
|---|------|--------|---------|------|---------|
| 1 | F-1 | BLOCK_SIZE | 128→64 | -13.6% | file:line |

### 3. 已失败优化（⛔ 禁止重试）
| # | 轮次 | 尝试项 | 失败原因 | 已回退 |
|---|------|--------|---------|--------|
| 1 | F-2 | unroll_list | 性能回退+8.6% | ✅ |

### 4. 约束与发现
- [约束/发现描述]

### 5. 当前关键代码配置
```python
@pypto.frontend.jit(
    runtime_options={...},
    pass_options={...},
    debug_options={"runtime_debug_mode": 1}
)
BLOCK_SIZE = [当前值]
TileShape配置 = [当前值]
```

### 6. 迭代统计
- 总轮次: N 轮
- 成功: X 轮 (X%)
- 失败: Y 轮 (Y%)
- 性能曲线: [持续下降 / 先降后平 / 波动]

### 7. 下一阶段建议
- 当前瓶颈: [描述]
- 建议方向: [描述]
- 需关注的性能文件: [路径]
```

---

---

## 异常处理

| 异常类型 | 处理流程 |
|---------|---------|
| S1_SETUP 精度失败 | 换卡尝试（最多 3 次）→ 仍失败 → 停止，不修复，告知用户，状态机停留在 S1_SETUP |
| 迭代性能回退（ITER_MEASURE） | ITER_ROLLBACK → ITER_RECORD（标记❌回退）→ ITER_JUDGE |
| 执行超时（>300s） | 标记失败 → ITER_ROLLBACK → ITER_RECORD（记录超时）→ ITER_JUDGE |
| 长期无法达标（3 个 PHASE 完成） | 最终摘要标注"未达标" → 建议重新设计算子 → 正常进入 S5_REPORT |

### 迭代轮次精度失败（ITER_VERIFY）

```
1. 不要立即回退，先做简单分析
   - 是否只改了一个参数？（应该只改了一个）
   - 修改的参数值是否在合理范围内？
   - 错误信息是什么？

2. 如果能快速定位原因 → 尝试修正参数值 → 重新 ITER_VERIFY
   ⚠️ 最多尝试 1 次修正

3. 如果无法快速定位 → ITER_ROLLBACK
   - 回退代码修改
   - ITER_RECORD 记录失败（标记 ❌）
   - ITER_JUDGE 判断是否继续

⛔ 禁止：
- 花大量时间修复精度问题（精度问题是实现问题，不是调优问题）
- 修正次数超过 1 次
- 不回退就继续下一个优化点
```
