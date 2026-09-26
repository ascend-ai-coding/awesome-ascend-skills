---
id: P17
bottlenecks: [tiling_imbalance]
op_families: [cv_fusion, flash_attention, matmul]
complexity: L1
conflicts_with: []
synergizes_with: [P14]
has_preconditions: true
has_playbook: true
---

# P17: CV Parallel Block Sizing (CV并行基本块大小设计)

## 核心思想
在 Flash Attention 等需要 Cube 和 Vector 并行执行的融合算子中，合理设计 Cube 基本块（mBaseSize）和 Vector 基本块（s2BaseSize）的大小，使 Cube 的矩阵乘法耗时与 Vector 的 Softmax/Update 耗时尽可能匹配，从而最大化 CV 流水掩盖效果，减少流水线气泡。核心思路是：mBaseSize 控制 M 方向（Query 序列）每次处理的 token 数，s2BaseSize 控制 S2 方向（KV 序列）每次处理的 token 数，两者的乘积决定中间 workspace 大小，两者的比例决定 Cube 与 Vector 的工作量平衡。

## 代码骨架

// === 改造前（基线）===
```cpp
// 固定基本块大小, 不考虑 CV 平衡
constexpr uint32_t mBaseSize = 256;
constexpr uint32_t s2BaseSize = 512;
// 问题: Decode 场景 s2 很长时, Cube 工作量远大于 Vector
// 导致 Vector 空等, CV 流水掩盖差
```

// === 改造后（专家模式）===
```cpp
// Host 侧 Tiling: 根据 sInnerSizeAlign 反向推导 mBaseSize
void FiaTilingNonQuant::CalcMBaseSize()
{
    if (fiaInfo_->inputLayout == TilingKeyLayout::TND ||
        fiaInfo_->inputLayout == TilingKeyLayout::NTD) {
        mBaseSize_ = M_BASE_SIZE_512;  // TND/NTD 默认最大
    } else {
        if (fiaInfo_->s1Size <= S1_SIZE_16) {
            // Decode 场景: s1 很小, 根据 sInnerSizeAlign 分档
            if (sInnerSizeAlign_ <= 512)       mBaseSize_ = 512;
            else if (sInnerSizeAlign_ <= 1024) mBaseSize_ = 256;
            else if (sInnerSizeAlign_ <= 2048) mBaseSize_ = 128;
            else if (sInnerSizeAlign_ <= 4096) mBaseSize_ = 64;
            else                               mBaseSize_ = 32;
            // 关键: mBaseSize × sInnerSizeAlign ≈ 常数 (workspace 上限)
        } else {
            mBaseSize_ = M_BASE_SIZE_512;  // Prefill: M 方向给足
        }
    }
}

// s2BaseSize 根据 GQA group size 分档
uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
uint32_t idx = std::min(fiaInfo_->gSize / 5U, 2U);
sInnerSize_ = sInnerSize[idx];
// gSize=1~4 → 8192, gSize=5~9 → 4096, gSize≥10 → 2048
```

## 关键修改点

1. 预期收益: 通过 mBaseSize 与 s2BaseSize 的反比关系，在不同 shape 下自动平衡 Cube 和 Vector 的工作量，workspace 大小保...

## 常见陷阱

⚠️ mBaseSize 过大：Vector 侧 Softmax 工作量增大，Cube 等待 Vector 完成，流水气泡增加
⚠️ s2BaseSize 过大：Cube 侧 MatMul 工作量增大，Vector 等待 Cube 完成；同时 workspace 增大
⚠️ 基本块过小：每次迭代的启动开销占比增大，跨核同步频率增高，整体吞吐下降

## 代码搜索关键词

```bash
grep -n "SyncAll\|PipeBarrier\|ExecuteTask\|PRELOAD\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM" op_kernel/*.cpp op_host/*_tiling.cpp
```
