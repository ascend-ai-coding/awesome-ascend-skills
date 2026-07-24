---
id: P86
bottlenecks: [mte2_stall, no_overlap, undersize_transfer]
op_families: [matmul]
complexity: L1
conflicts_with: []
synergizes_with: [P92, P93]
has_preconditions: true
has_playbook: true
---

# P86: Matmul Internal Data-Path Efficiency Tuning (Matmul 高阶 API 内部数据通路效率调优)

## 核心思想
该策略统一覆盖四类同源优化：选择更合适的调度模板（MDL/NBuffer33/Preload）、放大 base block 提升计算访存比、使能 UnitFlag 细粒度 MMAD-FIXPIPE 同步，以及开启 K 轴错峰访问缓解多核 GM 冲突。四者都作用于 Matmul 库内部的 GM→L1→L0→Cube 执行路径，目标是一致的：减少 MTE2 次数、缩短搬运空泡、提升 Cube 与搬运/搬出的重叠效率。它不包含手动 Mmad 流水、不包含跨核共享 L1 数据，也不改变分核拓扑。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：默认 Norm 模板
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM> matmulObj;
```

// === 改造后（专家模式）===
```cpp
// MDL：一次搬多个基本块
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulMdl;

// NBuffer33：基于 MDL 的 3x3 错开搬运
matmul_tiling::MatmulConfigParams matmulConfigParams(
    1, false, matmul_tiling::ScheduleType::N_BUFFER_33, ...);
cubeTiling.SetMatmulConfigParams(matmulConfigParams);

// Preload：在 MTE2 间隙预取下一轮
static constexpr MatmulConfig MM_CFG = GetMDLConfig(false, false, preloadMode);
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> matmulPreload;
```

## 关键修改点

1. 预期收益: 根据 bottleneck 选择最优调度模板，减少 MTE2 次数、缩短间隙并提升搬运与计算重叠度。

## 常见陷阱

⚠️ 不同模板、base block 和开关之间存在前置条件与互斥关系，错误组合可能无收益甚至退化
⚠️ 大 block、MDL、大包搬运或 preload 往往依赖更大的 L1/L0 预算，小 shape 容易被头开销抵消
⚠️ UnitFlag、Kdim reorder 等配置受模板能力约束，并非所有 Matmul 场景都支持

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|SyncAll\|PipeBarrier\|ExecuteTask\|PRELOAD\|tileSize" op_kernel/*.cpp op_host/*_tiling.cpp
```
