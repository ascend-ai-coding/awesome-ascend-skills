# Triton-Ascend 文档知识库

本知识库为 Triton Agent 提供 Triton-Ascend 的详细技术文档，用于辅助人类编写和优化 Triton 算子。

## 知识库定位

- **目标用户**：Triton Agent（AI 辅助编程工具）
- **内容范围**：triton-ascend 特有知识（不重复 `docs_ascendnpu_ir/` 中已有的 HIVM/HFusion 等方言细节）
- **文档语言**：中文为主，技术术语保留英文原文
- **组织原则**：问题导向、代码驱动、高度结构化

## 快速导航

### 按使用场景

| 场景 | 推荐文档 |
|------|----------|
| 首次使用 Triton-Ascend | [环境搭建](00-Quick-Start/01-environment-setup.md) → [第一个 Kernel](00-Quick-Start/02-first-kernel.md) |
| 从 GPU 迁移算子 | [架构差异](06-Migration-from-GPU/01-architecture-differences.md) → [代码迁移模式](06-Migration-from-GPU/02-code-migration-patterns.md) → [常见问题](06-Migration-from-GPU/03-common-issues.md) |
| 查询 API 用法 | [内存操作](02-Core-API/01-memory-ops.md) / [数学运算](02-Core-API/02-math-ops.md) / [归约操作](02-Core-API/03-reduction-ops.md) / [矩阵乘法](02-Core-API/04-linear-algebra-ops.md) |
| 使用 Ascend 扩展 | [扩展总览](03-Ascend-Extensions/01-extension-overview.md) → [PIPE/CORE](03-Ascend-Extensions/02-pipe-and-core.md) → [fixpipe](03-Ascend-Extensions/03-fixpipe.md) → [同步操作](03-Ascend-Extensions/04-sync-operations.md) |
| 使用 libdevice 数学函数 | [libdevice 数学函数库](03-Ascend-Extensions/11-libdevice.md) |
| 优化 kernel 性能 | [优化总览](05-Performance-Optimization/01-optimization-overview.md) → [分块策略](05-Performance-Optimization/02-tiling-strategy.md) → [Autotune](05-Performance-Optimization/03-autotune-guide.md) |
| 调试编译/运行问题 | [调试总览](07-Debugging/01-debug-overview.md) → [编译错误](07-Debugging/03-compile-errors.md) / [运行时错误](07-Debugging/04-runtime-errors.md) |
| 理解编译流程 | [编译流程全景](00-Quick-Start/03-compilation-flow.md) → [编译流水线详解](04-Compilation-Pipeline/01-pipeline-overview.md) |
| 查找参考信息 | [API 支持矩阵](09-Reference/01-api-support-matrix.md) / [数据类型矩阵](09-Reference/02-data-type-matrix.md) / [环境变量](09-Reference/04-env-variables.md) |
| 学习算子模式 | [向量加法](08-Examples-Patterns/01-vector-add.md) / [矩阵乘法](08-Examples-Patterns/03-matmul.md) / [Flash Attention](08-Examples-Patterns/05-flash-attention.md) |

### 按目录

| 目录 | 主题 | 文件数 | 核心内容 |
|------|------|--------|----------|
| [00-Quick-Start](00-Quick-Start/) | 快速入门 | 3 | 环境搭建、第一个 Kernel、编译流程全景 |
| [01-Programming-Model](01-Programming-Model/) | 编程模型 | 4 | SPMD 映射、Grid/Program ID、内存模型、数据类型 |
| [02-Core-API](02-Core-API/) | 核心 API | 8 | 内存/数学/归约/线性代数/原子/形状/扫描排序/比较逻辑 |
| [03-Ascend-Extensions](03-Ascend-Extensions/) | Ascend 扩展 | 11 | PIPE/CORE/fixpipe/sync/sub_vec/custom_op/buffer/aux/vec/mem/libdevice |
| [04-Compilation-Pipeline](04-Compilation-Pipeline/) | 编译流水线 | 7 | 流水线总览、TTIR生成/优化、Ascend Passes、Linalg转换、编译选项 |
| [05-Performance-Optimization](05-Performance-Optimization/) | 性能优化 | 7 | 优化总览、分块、Autotune、care_padding、CV融合、数据搬运、Profiling |
| [06-Migration-from-GPU](06-Migration-from-GPU/) | GPU 迁移 | 4 | 架构差异、代码迁移模式、常见问题、Block Pointer |
| [07-Debugging](07-Debugging/) | 调试 | 5 | 调试总览、解释器、编译错误、运行时错误、环境变量 |
| [08-Examples-Patterns](08-Examples-Patterns/) | 示例模式 | 7 | 向量加法/Softmax/MatMul/LayerNorm/FlashAttention/归约/自定义算子 |
| [09-Reference](09-Reference/) | 参考 | 5 | API矩阵、数据类型矩阵、错误码、环境变量、FAQ |

## 文档模板

每个文档遵循统一结构，便于 Agent 预测和检索：

```
# [主题标题]
## 概述        — 1-2段概述，含关键词便于语义检索
## 关键概念    — 结构化表格呈现核心概念
## API 参考    — 函数签名、参数、返回值、约束
## 代码示例    — 至少2个可运行示例（基础+进阶）
## NPU 适配要点 — 与GPU差异、限制、注意事项
## 常见问题    — Q&A格式，Agent可直接匹配用户问题
## 相关文档    — 交叉引用链接
```

## 关键源码索引

| 知识领域 | 关键源码路径 |
|----------|-------------|
| 参考文档 | `docs_ascendnpu_ir/01-HIVM-Dialect/`, `docs_ascendnpu_ir/06-Compilation-Pipeline/` |

## 交叉引用

本知识库与 `docs_ascendnpu_ir/` 目录中的 HIVM/HFusion 等方言文档互补，不重复其内容。当需要深入了解底层 IR 方言时，请参考：

- [HIVM 方言文档](../docs_ascendnpu_ir/01-HIVM-Dialect/) — 华为中间虚拟机方言
- [HACC 方言文档](../docs_ascendnpu_ir/02-HACC-Dialect/) — 异构计算调用方言
- [HFusion 方言文档](../docs_ascendnpu_ir/03-HFusion-Dialect/) — 算子融合中间表示
- [编译流水线文档](../docs_ascendnpu_ir/06-Compilation-Pipeline/) — 完整编译流程
- [内存管理文档](../docs_ascendnpu_ir/07-Memory-Management/) — 内存空间与规划
- [硬件架构概览](../docs_ascendnpu_ir/00-Architecture/) — Ascend NPU 硬件架构
