---
name: external-cannbot-plugins-community-ops-perf-evolution-skills-evolution-strategies
description: AscendC kernel optimization strategy library with 61+ strategies across
  D/P/A/R/X series, supporting tiered retrieval, compatibility checking, and autonomous
  strategy discovery. 当需要为 AscendC kernel 选择优化策略、按瓶颈/算子族筛选策略、检查策略兼容性或登记进化发现的新策略时使用。
original-name: evolution-strategies
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# Evolution Strategies

AscendC 内核优化策略库，包含 61+ 策略，覆盖数据类型 (D)、性能 (P)、精度 (A)、A5 架构 (R)、进化发现 (X) 五大系列。

## 文件布局

```
references/
  strategy_index.md           # 主索引（可追加 X 系列条目）
  strategy_compatibility.md   # 互斥表、依赖关系、推荐组合
  strategy_index_a5.md        # A5 (351x/Regbase) R 系列扩展策略
  README.md                   # 策略使用指南
  IDEAS_SUMMARY.md            # 高层策略总览

  strategies/                 # [只读] 精选策略库
    perf_01_double_buffer.md ... perf_52_l2_cache_hint.md   # P 系列
    dtype_01_mixed_precision.md ... dtype_05_bf16_special_handling.md  # D 系列
    acc_01_fp32_intermediate.md ... acc_08_quant_precision.md  # A 系列
    a5_R1_vf_fusion.md ... a5_R8_mutex_sync.md  # R 系列 (A5)

  discovered/                 # [可写] 进化运行时发现的策略
    disc_X1.md ...            # X 系列，由 agent 动态写入
```

## 策略分层检索协议

### L0 (Universal) — 所有算子适用

18 个基础策略：D1-D5, P1-P13, A1-A8

**必读**: 读取 `references/strategy_index.md`，根据算子类型从 L0 策略中选择基础组合。

### L1 (Advanced) — 按算子类别选择

35+ 个进阶策略：P14-P98

根据算子类别（CV fusion, attention, matmul, quantization, reduction, sort等）选择适用的 L1 策略。参考 `strategy_index.md` 中的 "When to Apply" 列。

特别关注标量计算向量化系列 (P94-P98)：对任何新生成的 kernel 优先检查是否含有标量 for 循环，这是单次优化中收益最大的模式。

### L2 (Discovered) — 进化发现策略

X 系列（disc_X1.md, disc_X2.md, ...），由进化过程动态发现并注册。

## 策略兼容性检查

读取 `references/strategy_compatibility.md` 检查：

1. **互斥关系**: 部分策略不能同时使用（如 P1 vs P22/P19/P20）
2. **依赖关系**: 部分策略需要前置策略（如 P1 requires P5+P7）
3. **UB 预算**: Light (<50%) / Medium (50-75%) / Heavy (>75%)
4. **推荐组合**: 7 个经过验证的黄金组合

## 策略发现协议（写入 discovered/）

当 open_exploration 节点取得显著提升（speedup > best_score_before * 1.10）时：

1. 读取 `references/strategy_index.md` 检查新发现的优化方法是否已有覆盖
2. 如果是全新策略：
   - `Glob references/discovered/disc_X*.md` 获取当前最大编号
   - `Write references/discovered/disc_X{n+1}.md` 记录新策略
   - `Edit references/strategy_index.md` 追加新行到"探索发现策略"分类

> **可变性说明**: `strategies/` 目录为只读精选库（P/D/A/R 系列）。`discovered/` 目录为运行时可写（X 系列）。`strategy_index.md` 仅允许追加 X 系列条目。

## A5 架构扩展

当目标芯片为 Ascend 950 (351x/Regbase) 时，额外读取 `references/strategy_index_a5.md`：

- R1: VF fusion（合并多个 VF 函数，减少 Load/Store）
- R2: Register reuse（减少 RegTensor 溢出）
- R3: Dual-issue（指令调度优化）
- R4: Hardware loop（硬件循环构造优化）
- R5: Unaligned memory optimization
- R6: Low-latency tree reduction
- R7: SIMD/SIMT hybrid programming
- R8: Mutex synchronization
