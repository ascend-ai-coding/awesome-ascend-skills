---
name: external-cannbot-ops-ascendc-tiling-design
description: Ascend C 算子 Tiling 设计指南。提供算子分类体系和 Tiling 核心要素（多核切分、UB切分、Buffer规划、分支覆盖）的详细设计方法。触发：算子设计阶段、设计
  Tiling 策略（多核切分/UB切分）、规划 Buffer 分配、查阅某类算子的 Tiling 方法论时。
original-name: ascendc-tiling-design
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# Ascend C 算子 Tiling 设计指南

## 算子分类体系

| 类别 | 特征 | 典型算子 | 设计指南 |
|------|------|---------|---------|
| **Reduction 归约类** | 沿轴归约（含索引跟踪变体） | ReduceSum, Softmax, LayerNorm, ArgMax | ✅ [场景路由](references/reduction/patterns.md)（⚠️ 必须先读） / [算法](references/reduction/algorithms.md) |
| **Elementwise 逐元素类** | 输入输出Shape相同，逐元素独立计算 | Sin, Cos, Abs, Exp | ✅ [场景路由](references/elewise/patterns.md)（⚠️ 必须先读） |
| Broadcast 广播类 | 输入Shape不同，需广播对齐 | Add, Mul, Sub | ✅ [场景路由](references/broadcast/patterns.md)（⚠️ 必须先读） |
| Conversion 数据转换类 | 改变布局/形状，合并/拆分张量 | Transpose, Concat, Split | ⚠️ [场景路由](references/conversion/patterns.md)（当前仅 Transpose 部分支持） |
| Random 随机类 | 生成随机数，需种子管理 | RandomUniform, Dropout | 📋 规划中 |
| MatMul 矩阵乘类 | 矩阵乘法，高计算密度，用Cube单元 | MatMul, BatchMatMul | ✅ [场景路由](references/matmul/patterns.md)（mxfp8 + eltwise 融合；其它 matmul 形态规划中） |
| Convolution 卷积类 | 空间卷积，滑动窗口计算 | Conv2D, DepthwiseConv | 📋 规划中 |
| NN 神经网络类 | 神经网络专用，多种操作组合 | FlashAttention, GroupNorm | 📋 规划中 |

## 通用设计要素（所有类别必须）

以下设计要素是所有算子类别都必须考虑和完成的：

### 1. 多核切分策略

**核心问题**：任务如何分配给多个 AI Core？

**设计要点**：
- 负载均衡：每个核处理的任务量尽量相等
- 数据局部性：相邻数据尽量分配给同一核
- 粒度适中：tile 不能太小（调度开销大），不能太大（并行度低）

**输出**：
- [ ] 总任务切分方式（按哪个维度切）
- [ ] 每个 AI Core 处理的任务量
- [ ] 使用的 AI Core 数量

### 2. UB 切分策略

**核心问题**：单次能处理多少数据？

**设计要点**：
- UB 容量限制（DAV_2201: 192KB, DAV_3510: 248KB）
- 单次处理数据量
- 是否需要分 chunk 处理

**输出**：
- [ ] 单次处理的数据量
- [ ] 是否需要分 chunk
- [ ] chunk 大小计算公式

### 3. Buffer 规划

**核心问题**：需要哪些 buffer？各多大？

**设计要点**：
- 输入 buffer（inQueue）
- 输出 buffer（outQueue）
- 中间计算 buffer（tmpBuf, workBuf 等）
- Double Buffer 优化

**输出**：
- [ ] Buffer 列表及用途
- [ ] 各 Buffer 大小计算公式
- [ ] 总 UB 使用量

### 4. 分支场景覆盖

**核心问题**：需要处理哪些不同场景？

**常见分支维度**：
- 数据类型：FP32 / FP16 / BF16 / INT8
- Shape 大小：大 shape / 小 shape
- 数据对齐：32字节对齐 / 非对齐
- 边界情况：最小值 / 最大值 / 特殊值

**输出**：
- [ ] 分支决策条件
- [ ] 各分支的处理策略
- [ ] 边界测试用例
