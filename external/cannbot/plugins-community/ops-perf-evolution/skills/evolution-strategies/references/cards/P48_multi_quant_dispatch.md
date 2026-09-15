---
id: P48
bottlenecks: [compute_bound, scalar_compute]
op_families: [matmul, moe, quantization]
complexity: L1
conflicts_with: []
synergizes_with: [P49, P70]
has_preconditions: true
has_playbook: true
---

# P48: 多量化模式编译期分发 (Multi-Quantization Mode Dispatch)

## 核心思想
通过 TilingKey + 编译期宏分发支持 A8W8/A4W4/A16W8/A8W4 MSD 等多种量化路径。编译期分支消除运行时开销，每种量化模式生成独立的内核模板。

## 代码骨架

// === 改造前（基线）===
```cpp
// 硬编码 int8×int8→int32 单一路径
// 手动 Cast(int32→float32) + Mul(scale) + Muls(per-token scale)
```

// === 改造后（专家模式）===
```cpp
#if defined(GMM_A8W8)
    // int8 × int8 → int32, 支持 O8/O16/O32 输出
    using aT = int8_t; using bT = int8_t; using cT = int32_t;
#elif defined(GMM_A4W4)
    // int4 × int4 直接计算
    using aT = int4b_t; using bT = int4b_t; using cT = int32_t;
#elif defined(GMM_A16W8)
    // Antiquant: int8 权重反量化到 fp16 后与 fp16 输入相乘
    using aT = half; using bT = int8_t; using cT = float;
#endif

// TilingKey 运行时选择
if (TILING_KEY_IS(2)) { /* A8W8 path */ }
if (TILING_KEY_IS(3)) { /* A4W4 path */ }
if (TILING_KEY_IS(4)) { /* A16W8 path */ }
```

## 关键修改点

1. 预期收益: 编译期分发零运行时开销，A4W4 吞吐 2x

## 常见陷阱

⚠️ 编译期分支增加二进制体积（每种模式一个内核）
⚠️ TilingKey 组合数可能爆炸（20+ 种），需要裁剪
⚠️ MSD 分解需要额外的中间 buffer

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM" op_kernel/*.cpp op_host/*_tiling.cpp
```
