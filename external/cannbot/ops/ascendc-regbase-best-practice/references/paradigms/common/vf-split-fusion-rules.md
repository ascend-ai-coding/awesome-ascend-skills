# VF 段编排规则（C1-C9）

> regbase VF 的**段级编排**判据：决定"一个算子的计算流划分成几个段、段边界在哪里"。
> 每段在物理层落地为**一条或多条 VF 链**（`__simd_vf__` 函数、`asc_vf_call` 调用）；段内相邻 Vector 算子能否串成一条寄存器链由段内融合规则判定（见 §10 互引），两层判据不同、不混用。
> 规则 C1-C9 溯源自 AscendC-DevKit VF 编排决策层，已按本仓语境适配（VF 编排单元 → 段；golden 行号锚点 → 计算图 S 编号锚点；见各规则条目）。

## 1 定位：两层分工与优先级

VF 设计分两层，判据不同、顺序固定：

| 层 | 决策对象 | 判据 | 文档 |
|---|---|---|---|
| **段级编排（本文档）** | 计算流划几个段、段边界在哪 | 语义/结构：统计轴、遍历轮次、物化点、可内联性、跨核 | 本文档（C1-C9） |
| **段内融合** | 段内相邻算子对能否串成寄存器链 | 物理能力：白名单、链长 ≤7、UB 驻留峰值（P+3） | [reduction-vf-fusion-rules.md](../reduction/references/reduction-vf-fusion-rules.md) |

**两层优先级（冲突时按此裁决）**：

1. **段内物理硬规则是硬门**：段级融合结论（C1 等）若在段内超物理限（链长 >7 / 白名单外 / 超 P+3），在段内拆为多条 VF 链，**段边界不变**——一段落地为多条 VF 链是合法形态（如 PostElewise 段 11 条操作拆 2 链）。链长口径与段内融合层一致：Vector 计算指令条数，**含 Cast**。
2. **C6（物化点）先于段内"多消费者不打断链"规则**裁决段边界（见 §6）；C6 豁免保持融合后，物化点作为 P slot 记账点，多消费者规则此时适用。
3. 段级规则内部按序执行，**后判规则的拆分决定覆盖先判规则的融合结论**（如 C1 融合成立的链仍可被 C6 在物化点拆分）。

**默认哲学按层区分，不可混用**：

- 段级：两方案访存流量相等（性能平局）时**模块化优先拆**——单段寄存器压力可控、独立变化轴各自演化、段级可独立单测复用。
- 段内：先推满最长链（链长 ≤7），超 P+3 预算才在分叉输出最多处主动拆。

**决策记录要求**：每条切分/合并决策须引用规则编号 + 计算图节点锚点（如 `S3`、`S10`），裸写"融合更优"或"评估不出差异就拆"均不合格。

## 2 固定前提（RegBase 数据流模型）

- **段内中间量默认驻留寄存器、不占 UB**；两类合法 UB 触达：① **多消费者中途物化**——中间结果被段外引用时 `StoreAlign` 落 UB、链不因此断（见段内融合层 §1.1）；② **跨段/跨迭代的结构性驻留**——树形归约部分和（cacheBuf 类）、跨段交接量、Welford 跨迭代状态等，模式为 `StoreAlign` → `LocalMemBar<VEC_STORE, VEC_LOAD>` → `LoadAlign`。
- 因此 membase 的"**聚合边界 = 切分线**"在 regbase 下**不成立**：段内聚合结果可与紧随消费共存于同一段，切分只由 C1-C9 决定。
- **C1 适用域**：仅当聚合以 VF-safe 行内归约（`Reg::Reduce` / `ReduceDataBlock`，白名单 Reg reduction 族）实现时，聚合点才可能与紧随消费同段；主归约走高阶 API（LocalTensor 版 `ReduceXxx`、`adv_api/` norm 族，白名单 `VF Function? = No`）的场景——如 reduction 范式的主归约（结果写 cacheBuf）——聚合边界由 C8 接管，C1 不适用。
- 防退化：membase 平台中间结果须落 UB，聚合边界天然是切分线。本文档的融合结论（C1 等）**不可外推到 membase**；也不要把"见聚合就切"的 membase 习惯带回 regbase（混淆恢复见 [regbase_vs_membase_confusions.md](../../pitfalls/regbase_vs_membase_confusions.md)）。

## 3 规则总表

按序执行；后判规则的拆分决定覆盖先判规则的融合结论（优先级总纲见 §1）。

| 规则 | 判据 | 结论 |
|---|---|---|
| **C1（同轴消费融合）** | 聚合点与其紧随的广播消费：**统计轴相同、同一遍历轮次内**，且聚合以 VF-safe 行内归约实现（适用域见 §2） | 合入同一段（统计量驻留寄存器）；主归约走高阶 API 时本条不适用，边界由 C8 接管 |
| **C2（跨轴必切）** | 两个计算段的**统计轴不同**（如 row dim=-1 vs col dim=-2） | 必须分属不同段 |
| **C3（跨遍历不同语义必切）** | 两段操作序列不同且分属**不同遍历轮次**（如首轮 softmax 段 vs 后续归一段）。限定：轮次间须有数据依赖；无依赖的循环展开（Tuple Reduce 类互斥过程）不触发本条，骨架复用合法性由 C4 裁决 | 必须分属不同段 |
| **C4（控制流等价才可融合）** | 相同循环次数/嵌套、无（或**相同**）条件分支？ | 不等价 → 禁止合并。段内运行时 `if/else` 仍被 VF 编程硬约束禁止（用 `if constexpr`，见 [vf-programming-rules.md](vf-programming-rules.md) §9） |
| **C5（融合收益归零 → 模块化拆分，参考性判据）** | C1 融合对需**二次遍历**，且中间量跨遍驻留需求 ≥ 寄存器容量粗估口径（见 §4）或融合链活跃 RegTensor 粗估超上限 | 性能平局 → 建议在聚合边界拆分。**非强制**：容量口径为未验证经验值（§4.1），无法坐实时不得据此拆分 |
| **C6（独立输出物化点，弱默认拆分）** | 聚合派生统计量/中间量**本身是算子输出**（写入输出 tensor，如 rstd/scale/mean 输出给下游消费） | **默认在物化点拆分**；可显式豁免（条件见 §6）；无论拆合均须记录物化点 |
| **C7（多遍历归约链按统计阶段分段）**（归约类范式，见 §7） | 同时满足：①C6 判据成立；②串行依赖的多级归约（第二级归约吃第一级结果，如 `Var = Σ(x−Mean)²/r`），或归约段与应用段需**重复遍历 x** | **默认按统计阶段在物化点分段**（见 §7）；单遍归约豁免 |
| **C8（实现约束段边界）** | 计算链中必须调用白名单 `VF Function? = No` 的高级 API（LocalTensor 版 `ReduceXxx`、`adv_api/` norm/pad/transpose/quant 族等），或归约采用折叠/二分实现（VL 对齐约束使其无法与前后逐元素段共链） | 高级 API 调用点 / 折叠对齐点构成**天然段边界**，前后自定义 VF 链各自成段 |
| **C9（跨核归约终段独立成段）**（归约类范式） | 核间切分轴与统计轴相同（单核只产部分统计量），须 GM workspace + SyncAll 后做跨核最终归约 | 跨核 finalize 段计为独立段（串行依赖链的一环）；核内局部归约段与跨核 finalize 段分属两段，对应 Group 模板 Phase 1/Phase 2 |

## 4 C5：融合收益归零 → 模块化拆分（参考性判据）

**平局论证（C5/C6/C7 共用）**：C1 融合的收益 = 省掉中间量的 UB 往返。当中间量反正要落 UB 时（C5：二次遍历且驻留超限放不下寄存器；C6：本身是输出须物化；C7：多级归约须重复遍历 x），**且该物化值此后不再被后继段从 UB 重新读取**时，融合与拆分的访存流量相等（否则拆分侧多一次 Load，平局不成立）；同 V 引擎按序执行 + 统计量依赖屏障使两者均无流水重叠差异 → **性能平局 → 模块化优先拆**。流量平局成立；跨段 UB 交接需 `LocalMemBar<VEC_STORE, VEC_LOAD>` 类同步（同 V 单元内部，代价近似可忽略）——总代价近似平局。

**定位（降级声明）**：C5 为**参考性判据，非强制拆分依据**——其触发证据依赖寄存器容量口径，而该口径为开发经验值（验证状态见 §4.1）。无可靠容量数据坐实时，不得以 C5 为由拆分；可坐实时建议拆分，并在决策记录中附容量判定依据。

约束：

1. 拆分决策引用 C5 + 计算图节点锚点，并记录容量判定依据（中间量驻留字节数 / 活跃 RegTensor 粗估 vs 容量口径）。
2. C5 只在聚合边界拆分，不引入 C2-C4 之外的新边界。
3. 无二次遍历、或中间量驻留与活跃 RegTensor 均在容量口径以内时，C1 融合收益 > 0，**不触发 C5**（保持融合，防止退回"见聚合就切"的旧行为）。

### 4.1 寄存器容量粗估口径（⚠ 开发经验值，附验证状态）

> 样本来源：外部 ops-nn 库 arch35 实现（instance_norm / rms_norm_quant_v2 / bn_training_update_v3），**本仓不可达**，验证状态以本表标注为准。

| 口径 | 数值 | 验证状态 |
|---|---|---|
| 单 RegTensor 宽度 | 256B（fp32 = 64 lane） | ✅ 库上源码一致（`GetVRegSize()/sizeof(float)` = 64） |
| 寄存器文件容量粗估 | 32 RegTensor × 256B = 8KB（含等号） | ⚠ 未直接验证——库上样本单 VF 声明数从未逼近 32，该上限仅作为"够用"粗估与样本相容 |
| 活跃 RegTensor 分级 | <20 安全 / 20-30 关注 / >30 高风险 | ✅ "<20 安全"有支持（库上样本 VF 声明数 2~19，无寄存器压力绕行痕迹）；⚠ "20-30 / >30"两档无样本数据，待验证 |

计数口径注意：

- 用**同时活跃**数而非声明数——声明数会双向失真（末尾才用的寄存器拉高估；helper 内联瞬时占用拉低估）。
- 库上样本中，中间量落 UB 中转的写法确实存在且高度模式化（`StoreAlign` → `LocalMemBar<VEC_STORE, VEC_LOAD>` → `LoadAlign`），但驱动因素全部是**跨迭代/跨阶段的结构性驻留**，无一处是"单链寄存器计数超限被迫 spill"。

## 5 寄存器压力不是拆段的理由

即使单段很长，只要 C1-C9 未触发切分就不拆（C6 触发即默认拆，不再属于"未触发"情形）——额外 UB load/store 的代价通常大于节省的寄存器。寄存器压力是功能跑通后 profiling 阶段的优化维度，不属于段编排步骤。超阈时在决策记录中标注风险点，交由下游定量判断，不据此重决策。

（例外：C7 由**结构**判据触发——多级归约链 + 输出物化，非寄存器压力。）

## 6 C6：独立输出物化点（弱默认拆分）

动因是**输出契约**：统计量/中间量本身是算子输出（下游必须拿到它）→ 该值反正必须中途物化（经 UB 写出 GM）→ **物化点处拆/融代价近似相等**（前提见 §4 平局论证：融合侧 Store 物化值后可保留寄存器值继续计算，拆分侧若须从 UB 重新 Load 则多一次读——平局不普遍成立，最终以实测为准）→ **默认在物化点拆分**，另获精度派生段独立演进、多 dtype 解耦的工程收益。

- **与段内"多消费者"规则的优先级**：统计量既是算子输出、又被链内继续消费时（如 rstd 输出 + 链内算 y），**先按 C6 裁决段边界**（默认拆）；豁免保持融合后，物化点作为 P slot 记账点纳入段内物理 trace，段内"多消费者不打断链"规则此时适用（`StoreAlign` 落 UB、链继续、占 1 个 P slot，见 [reduction-vf-fusion-rules.md](../reduction/references/reduction-vf-fusion-rules.md) §1.1）。非输出型分叉不触发 C6，直接按段内多消费者规则处理。
- **豁免（保持融合）须同时满足**：① 融合后单段的 Vector 指令链长 ≤7（与段内融合硬规则一致，链长定义见 [reduction-vf-fusion-rules.md](../reduction/references/reduction-vf-fusion-rules.md)）；② 活跃 RegTensor 粗估 <20（口径见 §4.1）。豁免须在决策记录中写明理由 + C6 + 节点锚点，裸"融合更优"不合格。容量门槛仅用于豁免判定；C6 的拆分动因是物化平局而非寄存器压力（与 §5 不冲突）。
- **边界重合**：切分型规则（C2-C5、C7、C8）已在物化点切分时无需重复拆，引用编号取实际触发者。
- **记录义务**：凡触发 C6，设计文档必须记录物化点清单（输出名、锚点、是否与切分线重合、宽度），裸遗漏 = 自检不合格。

## 7 C7：多遍历归约链按统计阶段分段（归约类范式）

**适用范式**：Reduction / NumericalStable / ReductionComposite / FusedComposite（多级归约 + 统计量输出的 norm 家族是主场景）。

- **触发两条件（须同时满足，缺一不触发）**：① C6 判据成立（统计量/中间量是算子输出，无论 C6 最终拆分还是豁免）；② **串行依赖的多级归约**——第二级归约的输入含第一级结果（`Var = Σ(x−Mean)²/r`），或归约段与应用段需重复遍历 x。
- **单遍归约豁免**：同一遍历内同算 Σx 与 Σx²（矩法 `Var=E[x²]−E[x]²` / Welford）→ 归约级数折叠，条件②不成立，退回 C6 两段拆；判定时须在计算图中指出双累加器同遍历的节点锚点。单次归约且 x 驻留不重读（rms_norm 形态）不触发，防止退回"见聚合就切"。
- **默认分段**：每级归约（Mean 段 / Var 段）、应用归一化（Y 段）各自成段；统计量 finalize 段（如 Rstd 标量派生）默认独立成段。
- **finalize 并入豁免（三条件须同时成立，缺一保持独立）**：① finalize 为单遍标量一步（单条算术指令可完成，如 `s·avg`/`s+ε` 类——**含超越函数或迭代求精的不满足**）；② x 驻留不重读；③ 归约可内联（C8 证据支成立）。豁免并入时记录三条件核验 + 规则编号 + 节点锚点。
- **与 C5/C6 的关系**：C6 弱默认拆 → C7 细分到统计阶段；C5 看容量（参考性）、C7 看结构，判据独立、先 C5 后 C7，边界重合时引用实际触发者。

## 8 C8：实现约束段边界

- **默认假设**：当公式/参考实现展开需要 reduce/max/sum/softmax 类归约时，**先假设归约不可内联**进自定义 VF 函数体（`__simd_vf__`）；范式主归约默认走高阶 API（LocalTensor 版 `ReduceXxx`）+ cacheBuf。仅当逐案验证 VF-safe 行内归约（`Reg::Reduce` / `ReduceDataBlock`）能力后，才可走 C1 行内融合路径。
- **判定依据**：API 能否进入 VF 函数体，唯一入口是 API 白名单的 `VF Function?` 列——`No` 的族（LocalTensor 版 `ReduceXxx`（Reduction 族）、`adv_api/` norm/pad/transpose/quant 等高阶助手）即构成段边界。见 [regbase_api_whitelist.md](../../api/regbase_api_whitelist.md)。**同名辨析**：`Reg::Gather`（reg_compute load/store 族）是 VF-safe、可融；`DataCopyGather`（data movement 族）是 kernel-side、构成段边界。
- 决策记录须注明具体约束：哪个 API / 哪种对齐 / 为何不能证明可内联。

**与 reduction 范式 Phase 结构的关系**：reduction 范式既有的 PreElewise → Reduce → PostElewise 三段结构（reducePhase 是天然断点）正是 C8 边界的实例——主归约高阶 API 调用点将前后逐元素链切为两段。

## 9 C9：跨核归约终段独立成段（归约类范式）

核间切分轴与统计轴相同时（单核只产部分统计量，如 foreach 类多核归约），须 GM workspace + `SyncAll` 后做跨核最终归约：

- 跨核 finalize 段计为独立段（串行依赖链的一环）；核内局部归约段与跨核 finalize 段分属两段。
- 在本仓中对应 reduction 范式 **Group 模板的 Phase 1 / Phase 2 两阶段结构**（借 R 开多核 + 二次归约）；跨核实现的同步契约（`SetScheduleMode(1)` + `SyncAll`）见 [sync-and-consistency.md](sync-and-consistency.md)。
- 多核并行本身仍是一切编排的基底能力，C9 只针对"核间切分轴与统计轴相同"的两阶段归约形态。

## 10 与其他文档的关系

| 主题 | 文档 |
|---|---|
| **段内融合**（三硬规则、链长 ≤7、P+3 拆链、多消费者） | [reduction-vf-fusion-rules.md](../reduction/references/reduction-vf-fusion-rules.md) |
| **VF 编程硬约束**（循环范式、`__simd_vf__` 约束） | [vf-programming-rules.md](vf-programming-rules.md) |
| **VF 可用 API 边界**（C8 判定入口） | [regbase_api_whitelist.md](../../api/regbase_api_whitelist.md) |
| **同步与跨核契约**（C9 的 SyncAll/SetScheduleMode） | [sync-and-consistency.md](sync-and-consistency.md) |
| **Cast 参与 VF 融合的指令与写法** | [cast-rules.md](cast-rules.md) §4 |
| **membase 混淆恢复**（固定前提的反面教材） | [regbase_vs_membase_confusions.md](../../pitfalls/regbase_vs_membase_confusions.md) |

## 11 附：特殊值语义警示（Max/Min）

段内融合判据"RegBase 指令存在"遇 `Reg::Max` / `Reg::Min` 时需加**语义门禁**：需与参考实现逐位对齐的 maximum/minimum（及 foreach 变体），SIMD Max/Min 的 NaN 传播语义未经验证文档化、±0 tie（`-0/+0 → +0`）可能与参考实现的取 self 语义冲突——默认改走 `Compare(NE) + Compare(GT) + Select` 复合原语（仍可融为一条 VF 链，链长按实际指令数计）。

## 12 脚本调用时机与产物落盘

**调用时机（按设计流程顺序）**：

1. **P_baseline 预计算**（段编排方案定型后、buffer 分配定稿前）：用 `scripts/liveness.py` 对计算图 JSON 做 P_baseline 预计算（`--pinned` 传独占分配 tensor），得到不考虑 VF 融合的基线 P。
2. **槽位分配**（融合方案定稿后）：设计侧声明三项语义约束（调度顺序 = steps 顺序、pinned 独占集合、互斥分支结构），写 tensor 级物理流 JSON，用 `scripts/allocate.py` 生成 B0..Bn 复用映射与 B 名物理流（`--out`）。分配只执行不决策；选槽即 WAR 检查，正确性由构造保证。
3. **P 峰值校验**（槽位分配完成后）：用 `scripts/physical_p.py` 对 allocate.py 产出的 B 名物理流 JSON 做 P 峰值 + WAR 违规 + 下界自检。WAR 有违规或下界自检未通过（P < 输入数+输出数且未解释）时，**不得进入 tiling**；此时回退调整语义约束（改调度、拆 VF 链降压）后重跑 2→3。

**跨阶段驻留量的输入契约（三脚本统一口径）**：
1. 设计 trace 可保留跨阶段量（如驻 cacheBuf 的归约结果 var_sum/reduceResult）作为"阶段外上下文"，但必须标注**不参与计数、不进入脚本输入**——否则 physical_p.py 会把它当普通 buffer 计入 P（如 post 阶段 P_post=4 而非文档的 3）。
2. liveness 的 dag（inputs/outputs）、allocate 的 tensor 流（reads/writes）、physical_p 的 B 流对跨阶段量采用同一口径：**不进本阶段输入**（归属其分配阶段，见各范式 dag-buffers「跨阶段驻留归属分配阶段」条款）；如需在计算图中语义性表达读取，用 liveness.py 的 `excluded` 字段（可读但不计持有）。
3. cacheBuf / preReduceResultTail 等固定 extras 由范式文档 §1.5 Buffer 汇总表负责（单独预算外扣除），**不进入脚本输入**。

**产物落盘**：三脚本各自的输入 JSON（计算图 / tensor 级物理流 / B 名物理流）与输出结论（槽位映射、P 峰值、WAR 检查、下界自检结果）随交付件落盘，供后续评审复算（设计文档评审 / docs-consistency-check / code-review 等环节可对照脚本输出复核 P 值与 trace 一致性）。落盘目录的默认约定：`$WORK_DIR/design/liveness/`（各范式 patterns.md 的「脚本管线产物」产出项即按此目录约定；consume 方若有不同约定以其为准并在产出中注明）；**禁止以顺延声明推迟落盘——目录受限时在当前任务内落最近可用位置并在设计文档登记绝对路径**。
