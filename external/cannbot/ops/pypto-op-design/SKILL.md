---
name: external-cannbot-ops-pypto-op-design
description: 根据算子规格和 golden 完成 DESIGN.md 与模块接口；按需检索 API、约束和计算模式，形成可实现、可验证的设计。触发：根据算子规格与
  golden 进行算子设计、生成 DESIGN.md 与模块接口时使用。
original-name: pypto-op-design
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-24'
synced-commit: ec8a7b0462e3acdd0d1d2f857f1dae652e19f4f2
license: UNKNOWN
---

# PyPTO 算子设计

## 输入与输出

输入为 SPEC.md、golden 和用户约束；已有 API_REPORT.md 时一并读取，确认其结论适用于目标设备和版本。

输出为 `custom/<op>/DESIGN.md` 和 `custom/<op>/eval/module_interfaces.yaml`。设计文档解释计算方案，接口文件记录各模块的输入来源和输出。代码实现与设备验证交给后续流程。

本工作流要求至少一个实际动态轴，`dynamic_axes` 为非空列表；需求只给出固定尺寸时，先确认变化范围，不虚构动态轴。相关输入输出标注动态维度，用框架循环遍历动态轴；初始设计及实现中的 `unroll_list` 只选择一个展开因子。

## 获取资料

已知文件路径时直接读取；需要查 API、配置参数或相近实现时，使用 [pypto-docs-search](../pypto-docs-search/SKILL.md)。

- 按准确的 API 或参数名查文档，核对签名、设备支持、默认值和使用限制。
- 按算子名或计算结构找参考实现，对比 shape、dtype、布局及循环依赖；不把不同条件下的配置直接套用。
- 保留支持当前选择的文件链接和必要位置；文档与代码不一致时，注明版本并确认实际支持情况。

从 [DESIGN 模板](templates/DESIGN.md.tmpl)开始填写。以下方法对应模板各章，结果直接写入 DESIGN.md；不适用的可选内容注明原因或省略，不创建中间报告。

## 分析计算与模块

**算子契约**：对照 SPEC 和 golden 确认公式、输入输出、动态范围、特殊输入行为和精度要求；冲突先澄清，不修改参考结果来迁就实现。

**计算图与 API 映射**：沿 golden 的数据依赖逐步分析，每个关键中间张量推导 shape 和 dtype，再匹配 PyPTO API。对照 [API 约束](constraints/api.md)与目标版本文档检查类型转换、广播和归约；记录转换位置及原因，覆盖全部输出。

**分解与模块边界**：运行[复杂度估算器](scripts/estimate_decomposition.py)，以建议模块数为起点，按[模块分解方法](references/decomposition.md)选择边界：

```bash
python <skill目录>/scripts/estimate_decomposition.py <golden路径>
```

多个入口时用 `--function` 指定函数。工具只统计源码结构，不执行 golden；辅助函数、别名和循环状态需结合源码判断。记录 `Decision: module_count = N`、模块职责及输出，调整建议值时说明具体原因。

## 形成设计选择

**先读 [SK 索引](patterns/skeletons/index.md)和 [AT 索引](patterns/atoms/index.md)，再读取候选卡片**，按计算依赖和适用条件选用。SK 描述整体结构，AT 描述局部计算；无合适模式时直接设计。⛔ 「简单」不等于「无合适模式」——先查再选才是正确流程。

按下表完成“范式与设计决策”，只记录本算子实际采用的选择。

**结构选择原则：代码结构追求性能最优，配置项求稳。** 命中高置信度生产参考实现时，结构（数据布局、算子选型、循环组织、模块边界）默认对齐参考写法；偏离参考结构必须在设计中写明理由（约束不兼容 / 精度差异 / 平台限制），不得以“更稳妥”为由选择已知的低性能结构。配置项（tile 数值、pass/runtime options）则取参考值或保守默认值，性能收益留待调优阶段实测。

**⛔ AT 模式与 API 功能重叠时必须机制对比择优。** 同一计算段既可由 AT 模式实现、也可由单一 API 实现时（如 AT-17 view 拼装 vs `gather_in_l1` 分页装载），必须对比两者机制（搬运开销 / 指令开销 / 硬件适配）后择优选用，对比结论与理由写入 DESIGN.md。API 名称与功能描述匹配不等于机制等价；参考实现已使用其中一种时，默认对齐参考实现的选择。

| 内容 | 分析方法 |
|---|---|
| Tiling | 根据输入范围选择切分轴，区分读取的数据块与计算 TileShape；按 [Tiling 约束](constraints/tiling.md)核对秩、对齐和容量，将驻留张量及缓冲副本计入资源估算，确定当前值、硬性条件和可比较的候选。 |
| 实现配置 | 从选中模式和相近实现提取相关参数，按参数名查目标版本文档，确认含义及默认值；仅在当前计算确有需要时显式设置，记录选择原因，未经测量的性能收益标为待验证。 |
| 布局与量化 | 逐 tensor 采用 SPEC §5 声明的布局（ND / NZ），默认 ND；采用 NZ 时必须显式声明并注明依据。NZ tensor：JIT 签名声明 `format=pypto.TileOpFormat.TILEOP_NZ`，核对内轴 32B、外轴 16 元素对齐及 cube tile 兼容；不满足对齐时回退 ND 并写明原因。沿张量的读取、转换、计算和写回检查布局及索引；按 [数据流约束](constraints/dataflow.md)与相关 AT 分析分页、共享存储和尾块；量化时核对 scale 的轴、shape/dtype、乘除约定、舍入和饱和行为。 |
| 循环状态 | 根据动态轴和数据依赖确定循环顺序，找出下一轮需要的状态；按 [循环](constraints/loop.md)与 [符号值约束](constraints/symbolic.md)选择表达方式，明确状态的作用域、类型、初始值、更新和最终写回。 |
| 数值稳定性 | 沿计算检查相近数相减、累加误差、exp/log 范围及量化特殊值；结合参考计算选择累加类型或数学改写，并确认目标 API 支持，保持 SPEC 的语义和容差。 |

候选参数不代表已经验证可用；改变参数后需重新检查受影响的形状、资源和精度。需要了解约束字段时，参阅[约束格式](constraints/README.md)。

**⛔ golden 派生的 tile/分块参数要区分数据布局与执行切块。** 在 Tiling / 参数表中：
- **数据布局参数**（`block_size` 页大小、量化 scale 分组大小、packed_dim 布局）是数据格式/算法结构的一部分，改动会改变数据含义，可标 `constant`；
- **执行切块参数**（如 online softmax 的 `S2_TILE`、Flash Attention 的 `Br/Bc`、q/k_tile）是 golden 的一种等价实现写法——数学上与其余取值等价，属可调或结构性 tile，**一律标 `tunable`**，不得标 `constant`；若改它会切换结构（如 多块 running-max ⇄ 单块 full-softmax），在「可调整范围或候选」里注明联动项——尤其当把切块放大到 softmax 不再是 flash/online（单块覆盖全序列、bn=1）时，必须**同步删除**对应的 flash 操作（running-max 累加器 oi/li/mi、`is_loop_begin` 分支、跨块 rescale），改为普通 full softmax。
- `constant` 另加硬件/架构硬约束（32B/对齐、寄存器位宽、L0/L1/UB 尺寸上限）。

**⛔ kernel 需用、golden 可不用的输入，kernel 直接使用，禁止因 golden 重算而改为 kernel 重算。** 如 softmax `m`/`l`：kernel 直接读取（SK-16），golden 可重算；测试输入须提供真实值（见 `pypto-golden-generate` 的 reference-normalization.md）。

**伪代码**：将已确定的计算图、配置、循环和状态组合成完整计算过程，包含必要的预处理、数据访问、类型转换、后处理及全部输出。关键张量注明 shape/dtype，区分常量与符号值。发现必须新增的设计选择时，先明确选择，再同步伪代码；不要求每行附加编号或来源。

## 检查设计与接口

按[模块接口说明](references/module-interfaces.md)生成接口草稿，再按设计核对输入来源、输出 shape/dtype 及最终返回结果。

**设计检查结果**：对照 SPEC、golden 和已查资料复核计算路径、API 限制、状态依赖、资源估算及接口一致性，记录具体结论与依据；未检查的明确标记，不能预填“通过”。布局：逐 tensor 核对「布局继承」表与 SPEC §5 一致；NZ tensor 的 JIT 签名与 host 转换点已标注，对齐核对有结论。

**实现后的验证方案**：从规格中的输入范围选择典型值和端点，从切分方案选择整块及非整块输入，从数值风险选择特殊值；写出输入配置和比较标准，只覆盖已定义的语义。

**风险与开放问题**：汇总上述分析中尚未确认的 API 支持、资源假设、数值问题及需求冲突，说明影响和下一步；关键问题阻塞时返回具体原因，不用默认值掩盖。

按[输出文件格式](references/artifacts.yaml)运行[结构检查](scripts/validate_artifacts.py)：

```bash
python <skill目录>/scripts/validate_artifacts.py --op-dir <算子目录>
```

结构检查只确认文件格式和引用关系，不能证明算法正确或设备精度通过。

## 交接

返回两份文件路径、模块数和未解决问题，由独立 verifier 检查设计，再交给 develop 实现；实现完成后另行运行测试。引用路径相对于实际 DESIGN.md 解析。
