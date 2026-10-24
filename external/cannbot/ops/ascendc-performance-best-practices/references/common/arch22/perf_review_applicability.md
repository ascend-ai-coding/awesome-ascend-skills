# arch22（A2/A3）性能检视检查点适用性

本文是现有 A5/通用检查卡片到 arch22 的**架构投影**。通用卡片保持原文件为单一真源，不复制正文，避免同一 ID 在两处漂移；分析 arch22 目标时先读本矩阵，再读取 [ascendc-performance-review](skill:ascendc-performance-review) 的 `references/perf-review-checklist.md` 中的通用卡片和本库 MatMul/FA 专项检查点。

## 结论口径

| 结论 | 含义 |
|---|---|
| 机制可复用 | 检查方法可用于 arch22；其中的 A5 性能数字、核数、容量和阈值不迁移 |
| arch22 专项 | 卡片已包含 220x 规则或需要按 arch22 单独解释 |
| A5-only | API/编程模型在本库明确为 arch35 独占，arch22 不加载 |

## 现有卡片矩阵

| 结论 | 卡片 |
|---|---|
| 机制可复用 | `PRC-ARCH-001..002`；`PRC-MEM-001..007`、`010..013`、`015`；`PRC-VEC-001..003`；`PRC-SCA-001..009` |
| arch22 专项或需按 arch22 重判 | `PRC-MEM-008`（A2 收益证据，A3 未验证，保持 draft）；`PRC-MEM-009`（220x bank 规则与 A5 结论分开读取）；`PRC-MEM-014`（机制通用但性能未量化，保持 draft） |
| A5-only，不复制 | `PRC-VF-001..011`；`PRC-SIMT-001..006` |

合计：现有 46 条全部完成分类，其中 29 条可进入 arch22 检视（26 条通用 + 3 条专项/待验证），17 条明确排除。

## arch22 复用边界

1. “ready”表示检查结构和证据要求可用，不表示卡片中的 A5 加速比在 A2/A3 成立。
2. A2 与 A3 同属 `dav-2201` 编译架构，但 A2 的实测收益不能写成 A3 已验证。没有 A3 结果时报告 `A3_not_run`。
3. 核数、UB/L1/L2 容量、事件 ID、DataCopy/LoadData 能力和 dtype 支持由目标 SDK/平台接口取证。
4. `PRC-SCA-001` 在 FIA arch22 源码存在 `ICachePreLoad`，可进入检查；预取长度仍需目标 case A/B。
5. `PRC-SCA-002/004` 在 arch22 FA 中有静态 LocalTensor 和 GM TilingData 指针读法证据；不得全仓机械替换 `InitBuffer` 或 `GET_TILING_DATA_WITH_STRUCT`。
6. arch22 MatMul/FA 的族级检查点分别见 [MatMul](../../matmul/arch22/guide.md) 和 [FlashAttention](../../flash_attention/arch22/guide.md)；这些检查点补充 Cube/CV 检视，不改变通用 MEM/SCA 卡片的单一真源。

## 证据位置

通用卡片的状态和检查范围以 [ascendc-performance-review](skill:ascendc-performance-review) 的 `references/perf-review-checklist.md` 为准。arch22 专项的 book-910 条目和实现来源分别记录在 [MatMul 流水](../../matmul/arch22/pipeline_design.md)、[MatMul 复用与切分](../../matmul/arch22/reuse_and_split_design.md)、[FA Tiling 与流水](../../flash_attention/arch22/tiling_pipeline_design.md)及 [FA Scalar 与启动开销](../../flash_attention/arch22/scalar_overhead_design.md)分册中。
