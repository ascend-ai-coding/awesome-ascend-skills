# 最佳实践的代际与验证范围

## 经验归类

| 经验族 | 名称与本库标识 | 使用边界 |
|---|---|---|
| arch22（A2/A3） | DAV_2201 / NPU 220x；910b、910_93 系列同族管理 | A2 实测不自动覆盖 A3 或所有 910b 型号 |
| arch35（A5） | DAV_3510 / dav-3510；A5、Ascend950 | 本库部分 `dav310/` 是历史目录名，实际范围看其使用指南 |

按用户指定的精确 SoC、工程分派与实际编译目标确认映射；不从模板目录名或本机设备反推用户目标。历史宏（如 `DAV_5102`）不新增别名映射，先查目标工程/SDK 定义。未知目标保留通用分析并上报缺口，多目标分别选路。

## 目录组织

- 多架构族的实现按 `references/<family>/arch22/` 和 `arch35/` 隔离，源码、模板和配套说明随实现存放；族级入口只负责选择目标目录。Sort 与 common 已分别收录两代内容。
- MatMul、FlashAttention 已分别收录 `arch22/` 与 `arch35/` 实现。`arch22/` 是 A2/A3 共用的 DAV_2201/220x 经验目录；样例/源码支持范围包含 A2/A3，不等于 A3 性能收益已验证。common 的 910B 搬运与 220x bank 经验归属 `arch22/`。
- 纯通用知识（如 Scalar 编码原则）保留共享目录；单代际族按分类表与文档的架构标注读取，扩展另一代实现时再拆分。不可把代际专用代码放进通用目录。
- RegBase / Reg / MicroAPI / VF 及 `regbase/` 范例库为 **arch35 独占**，不建立 arch22 副本。泛化的向量化思想不等于支持 Reg VF。

## 实现知识与证据范围

| 入口 | 已收录经验范围 | 跨代使用方式 |
|---|---|---|
| MatMul、FlashAttention | 已分别收录 `arch22/` 与 `arch35/` 指南；arch22 证据主要来自明确列出 A2/A3 的 `dav-2201` 样例和 FIA `arch22` 源码 | A2/A3 可引用实现/API 路由证据；现成收益数据不能自动覆盖 A3，A3 必须单独完成目标型号构建、精度、timeline 和逐 case 性能验证 |
| MC²、Reduction、Scan、Gather、Elementwise、Broadcast、Scatter、Conversion、Convolution、SIMT | 分类表中的实现指南/模板以 DAV_3510 为目标 | A2/A3 不自动进入该实现决策树；通用数学/数据流思路可参考，目标实现另取证 |
| [RadixSort](../sort/arch22/radix_sort.md) | 大规模排序经验在 A2 验证 | A3 需复验；A5 TopK 进入独立 [使用指南](../sort/arch35/templates/usage_guide.md)，不继承 A2 限制 |
| [Scalar](scalar/guide.md) | 通用编码原则；具体硬件解释/子族案例各有范围 | 先验证目标编译器与 profiling；不把“通常 spill”或某代 cache 参数当作所有 ScalarBound 根因 |
| common | [arch22 搬运](../common/arch22/datacopy_optimization_design.md)、[arch22 Bank 冲突](../common/arch22/ub_bank_conflict.md)、[arch22 检视适用性矩阵](../common/arch22/perf_review_applicability.md)、[arch35 UB 复用](../common/arch35/ub_resident_design.md) | 只取目标目录；不互换 bank/group 冲突规则或实测对齐阈值 |
| [Reg VF catalog](arch35/regbase/README.md) | DAV_3510 范例，另有逐项 readiness / 构建验证范围 | 先匹配目标代际，再匹配模式；compile_verified 不提供跨代授权 |

表中的“DAV_3510”是对应行的实现经验范围，不表示其他架构不具备该算子或优化思想。MatMul/FA 的 arch22 知识已收录，不再以“待收录”作为路由结论；若具体变体、dtype 或 SoC 仍无证据，记录 `architecture_reference_gap`，沿用目标原有实现并基于目标 SDK/源码补充方案，禁止用 A5 模板强行补位，也不因此终止整个优化任务。

## 读取与集成

1. 精确目标 → 匹配代际 → 算子族/计算模式 → 只读目标适用指南及引用模板。一个族含混合代际时按子文档的范围选择，不全目录预读。
2. 区分适用架构、实测设备和验证状态。A2 案例对 A3 只提供假设；A5 编译成功也不表示所有 A5 型号、布局或 dtype 已验证。
3. UB 总量、可用预算、预留空间分别取证；bank/cache/指令限制与性能阈值不得跨代复制。模板中的硬编码资源值须替换为目标查询结果并重新建模。
4. 新实现隔离于目标分支；target guard 只隔离编译/运行路径，不能让 A5 Reg VF、NDDMA 或 SIMT 模板变成 A2/A3 实现。保护原 legacy 和非目标依赖，真实构建/精度/性能验收绑定最终 SoC 与代码身份。
5. 扩充条目时标明适用代际、来源、实测 SoC、SDK/构建与验证状态。`evals` 平台标签只筛选评测，不证明任何模板已上板验证。
