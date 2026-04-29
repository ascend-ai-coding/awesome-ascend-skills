---
name: external-gitcode-ascend-triton-operator-design
description: 生成适用于 Ascend NPU 的 Triton 算子需求文档。当用户需要设计新的 Triton 算子、编写算子需求文档、进行算子性能优化设计时使用。核心产出：功能定义、API
  接口、Tiling 策略、Kernel 实现方案。
original-name: triton-operator-design
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-29'
synced-commit: 8faee0275e457955c8f50989aef8972c0838db31
license: UNKNOWN
---

# Triton 算子需求文档生成

## 工作流

1. **需求分析** → 产出：功能定义、数学公式、竞品对比
2. **原型设计** → 产出：API 接口定义
3. **规格约束** → 产出：输入输出约束、硬件限制
4. **特性实现** → 产出：Tiling 策略、Kernel 实现方案

## 参考资源加载

| 阶段 | 必须加载 | 不要加载 |
|------|----------|----------|
| 需求分析 | [`ascend-terminology.md`](references/ascend-terminology.md) | template, tiling-strategies |
| 特性实现 | [`tiling-strategies.md`](references/tiling-strategies.md), [`triton-operator-template.md`](references/triton-operator-template.md) | ascend-terminology |

**MANDATORY**：阶段 4 前，完整阅读 `tiling-strategies.md` 和 `triton-operator-template.md`，不设行数限制。按模板格式输出文档。

## 关键术语

- **GM**：全局内存（DDR），**UB**：Vector Core 高速缓存（192KB），**L1**：Cube Core 缓存（~1MB）
- **AI Core**：A2/A3 有 24 个，含 1 Cube + 2 Vector
- 归约操作必须升精度到 FP32

## 绝对不要做的事

- ❌ 使用模糊术语（"适当切分"、"合理分配"）— 必须给出具体计算方法
- ❌ 忽略 UB 大小（192KB）和对齐要求（32B）
- ❌ 不区分 Vector Core（向量计算）和 Cube Core（矩阵计算）
- ❌ 不标注数据流图中的数据类型和 GM↔UB 传输
- ❌ 归约操作不说明升精度策略

## 常见陷阱

| 陷阱 | 症状 | 解决 |
|------|------|------|
| UB 超限 | 方案不可实现 | 计算缓冲区总大小 < 192KB |
| 内存未对齐 | 硬件报错 | UB 缓冲区 32B 对齐，单值缓冲区分配 32B |
| 精度损失 | FP16 结果不准 | 归约前升 FP32，完成后降精度 |
| Tiling 不合理 | 性能差/大 shape 不支持 | 按维度切分，避免跨 Core 数据依赖 |
