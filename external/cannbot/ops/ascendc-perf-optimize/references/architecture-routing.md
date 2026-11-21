# 代际相关知识的适用范围

四步流程、算法分析和参数化容量公式共用，不以芯片代际划分流程或整套卡片。仅在涉及具体资源、API、模板、指标或实测结论时核对下述范围。

## 目标与知识范围

从用户指定的精确 SoC、工程 SocVersion 分派、实际编译目标和目标 SDK 取证。A2/A3 在本库归 DAV_2201 / 220x，A5 / Ascend950 归 DAV_3510；这些是经验分组，不是所有同族型号已验证的证明。`dav310` 在本库部分模板中是 A5 历史目录名，不按字面猜架构；`DAV_5102` 等历史宏以源工程定义为准。

目标未明时只做与架构无关的计算量、数据依赖和容量公式分析，返回缺失信息；不从运行 Agent 的机器、默认脚本值或原源码含 VF 推定目标。多个目标分别建模、选路和登记证据。

## 本库的读取边界

| 当前内容 | A2/A3 | A5 |
|---|---|---|
| 四步流程、依赖分析、参数化容量公式 | 可用；参数来自目标平台 | 可用；参数来自目标平台 |
| [Broadcast](tiling/broadcast/fallback/tiling-flow.md) | OneDim 或静态 UB Broadcast；核对 rank/对齐/目标 SDK | 按 DAV_3510 分支选择动态 UB / NDDMA 等 |
| Reduction / Scan / Gather / Scatter / Conversion 的现有模板决策树 | 先取通用 shape/容量分析；现有 A5 模板和阈值不能自动兜底 | 通过目标能力和模板前置条件后使用 |
| [MatMul](../../ascendc-performance-best-practices/references/matmul/arch22/guide.md) | 已收录 arch22 流水、预取、切分、驻留与 L2 复用；A2 样例证据不自动证明 A3 收益 | 使用 arch35 SWAT / FullLoad / StreamK 前仍需核对实际资源与变体，不照抄示例值 |
| [FlashAttention](../../ascendc-performance-best-practices/references/flash_attention/arch22/guide.md) | 已收录 FIA arch22 的 tiling/workspace、Split-KV、buffer/event 与 Scalar 启动知识；A3 需复验 | 使用 arch35 FA 子类和 Scalar/ScalarLDST 路由；不得把其阈值反投影到 arch22 |
| Convolution / MC² 的公式与参数 | 通用公式可参考；指令/dtype、常量和工程 ABI 须有目标实现证据 | 仍需核对实际目标资源与变体，不照抄示例值 |
| [Sort](tiling/sort/index.md) | 不能选择 A5 TopK；通用排序算法需目标 API/同步/临时空间证据 | TopK 专用路由按其条件；不继承 A2 实测限制 |
| [单核诊断](single-core-pipeline/index.md) | 使用目标实际 profiling 字段；通用融合/复用/减搬运思路重新取证 | 可继续读取本库 A5 技术卡片 |
| [技术卡片](single-core-pipeline/techniques/INDEX.md) | 可使用标为通用的算法/流水方法；API 与阈值按目标核验 | 通用方法及标为 arch35 的实现均按卡片条件选择 |
| RegBase / Reg / MicroAPI / VF | 不适用 | arch35 独占；仍须满足目标 SDK、布局与构建契约 |

“fallback”只表示已满足架构与能力条件后的算法兜底，不表示跨代兼容。目标无已收录实现时记录 `architecture_reference_gap`，保留现有目标实现，继续通用建模并向目标 SDK/工程取证；不把知识缺口解释为硬件不支持或终止整个优化任务。算子名相同、数学公式相同、加了 target guard 均不构成跨代实现证据。

## 参数、指标与输出

- UB/L1/L0、核数、向量宽度、bank/cache、同步与指令支持来自目标 SDK/工程查询或可信文档。区分硬件总容量、运行时可用容量和算法预留；例如资料中的 256 KB 与脚本的 248 KB 不能互换，192 KB/64 核等也不是同族默认配置。
- 参考脚本的样例常量不是平台探测。调用前显式填写目标资源，未提供时不上推理得到的具体 tiling 数值。Broadcast 脚本必须传 `is_dav3510`、`ub_size`、`core_num`，其输出保留代际标记。
- A5 RVEC/Reg、SIMT、NDDMA 路线和阈值只在匹配的目标下启用；A2/A3 缺少 A5 profiling 字段不等于利用率为零。Simulator 支持范围取对应 Skill，不把 950 仿真作为 A2/A3 验证。
- 每项策略附精确 SoC / DAV 分组、参数来源、引用适用范围、已验证范围和未决项。跨代任务分开产出 tiling 与收益结论，共享算法思路不共享未验证参数或 kernel 实现。
