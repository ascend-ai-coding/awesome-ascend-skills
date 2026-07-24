---
id: P71
bottlenecks: [l2_cache_thrash, mte2_stall]
op_families: [matmul]
complexity: L1
conflicts_with: []
synergizes_with: [P72]
has_preconditions: true
has_playbook: true
---

# P71: Matmul IBShare L1 共享矩阵 (Matmul IBShare L1 Sharing)

## 核心思想
MIX 场景下多个 AIV 核使用相同的 A 或 B 矩阵数据时，默认每个 AIV 都独立从 GM 搬运到 L1。使能 IBShare 功能后，第一个 AIV 搬入的矩阵数据缓存在 L1 Buffer 上供其他 AIV 复用，避免重复 MTE2 搬运，减少搬运开销。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// 设置 B 矩阵的 IBSHARE 参数为 true
using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND,
    BType, false, LayoutMode::NONE, true>;  // 最后一个参数 true 使能 IBShare

AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_IBSHARE_NORM> matmulObj;
```

## 关键修改点

1. 预期收益: MTE2 搬运次数减半（AIV1 从 12 次降至 6 次），aic_mte2_time 从 5.56us 降至 4.71us（提升 15.46%）

## 常见陷阱

⚠️ 仅适用于 MIX 场景
⚠️ A 和 B 同时使能 IBShare 时只支持 IterateAll 输出到 GM
⚠️ 共享矩阵必须在 L1 上全载

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
