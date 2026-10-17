# Step 4: Implementation

> **定位**：仅在完整四步流程中执行本算子的 DESIGN/PLAN：按已定稿的合同实现，持续更新 PLAN 并追加执行记录；不改设计、不换路线、不改接口/ABI、不扩大支持域。
>
> 父级流程定义以 `plugins-official/ops-direct-invoke/AGENTS.md` 为准。**开发/审查/修复/验收/汇报各阶段的 apace 场景要点与门禁，以 [`../workflow_integration.md`](../workflow_integration.md) Step 3-7 为唯一事实源**（双写已消除，防止漂移）；实现层问题按 [`../troubleshooting/failure-navigation.md`](../troubleshooting/failure-navigation.md) 定位修复。

## 阶段映射（模型参考）

| 本步骤子阶段 | 执行主体 | 事实源（workflow_integration） |
|:---|:---|:---|
| 开发（工程验收 + 场景增量项 + 开发期红线） | Developer | Step 3 |
| 审查（全局红线 + 场景约束逐项） | Reviewer | Step 4（清单：[`../review-checklist.md`](../review-checklist.md)） |
| 修复循环（≤3 轮，修不动表） | Developer ↔ Reviewer | Step 5 |
| 精度验收 + 性能采集 | Reviewer / Developer | Step 6 |
| 完成汇报 | CANNBot 主控 | Step 7 |

## 后续阅读

| 文档 | 何时读 |
|:---|:---|
| [`architecture.md`](../fundamentals/architecture.md) | 第一次了解 apace 三层架构 |
| [`development-guide.md`](../operator-design/development-guide.md) | 工程搭建时，定位改造验收标准 |
| [`communication.md`](../fundamentals/communication.md) | 通信接口与机制 |
| [`compute.md`](../fundamentals/compute.md) | 计算接口与 kernel 模式 |
| [`operator-anatomy.md`](../operator-design/operator-anatomy.md) | 算子骨架（tiling/Impl/入口规则） |
| [`fusion.md`](../fundamentals/fusion.md) | 通算融合组合模式 |
| [`host-and-testing.md`](../operator-design/host-and-testing.md) | host 序列与 ST 工程 |
| [`pipeline_tuning.md`](../../../shared/pipeline_tuning.md) | 通算并行调优 |
| [`profiling_mc2.md`](../../../shared/profiling_mc2.md) | 性能采集详细流程 |
