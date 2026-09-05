---
id: R1
bottlenecks: []
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: false
has_playbook: false
deprecated: true
deprecated_reason: "Uses fictional APIs (__simd_vf__, RegTensor) not present in AscendC. Overlaps with P-series. Zero references in operator lookup tables."
---

# R1: VF 函数融合

## 核心思想
将多个独立的 `__simd_vf__` 函数合并为一个，减少中间结果的 Load/Store 操作，最大化寄存器复用。

## 代码骨架

// === 改造前（基线）===
```cpp
// 两个独立 VF: Add 和 Mul
__simd_vf__ inline void AddVF(__ubuf__ float* dst, __ubuf__ float* s0, __ubuf__ float* s1,
    uint32_t count, uint32_t oneRepSize, uint16_t repTimes) {
    RegTensor<float> r0, r1, d;
    MaskReg mask;
    for (uint16_t i = 0; i < repTimes; ++i) {
        mask = UpdateMask<float>(count);
        LoadAlign(r0, s0 + i * oneRepSize);
        LoadAlign(r1, s1 + i * oneRepSize);
        Add(d, r0, r1, mask);
        StoreAlign(dst + i * oneRepSize, d, mask);  // Store 到 UB
    }
}

__simd_vf__ inline void MulVF(__ubuf__ float* dst, __ubuf__ float* s0, __ubuf__ float* s1,
// ... (truncated)
```

// === 改造后（专家模式）===
```cpp
__simd_vf__ inline void FusedAddMulVF(__ubuf__ float* dst,
    __ubuf__ float* s0, __ubuf__ float* s1, __ubuf__ float* s2,
    uint32_t count, uint32_t oneRepSize, uint16_t repTimes) {
    RegTensor<float> r0, r1, r2, tmp, d;
    MaskReg mask;
    for (uint16_t i = 0; i < repTimes; ++i) {
        mask = UpdateMask<float>(count);
        LoadAlign(r0, s0 + i * oneRepSize);
        LoadAlign(r1, s1 + i * oneRepSize);
        LoadAlign(r2, s2 + i * oneRepSize);
        Add(tmp, r0, r1, mask);       // tmp 留在寄存器
        Mul(d, tmp, r2, mask);         // 直接使用 tmp
        StoreAlign(dst + i * oneRepSize, d, mask);  // 只 Store 最终结果
    }
}
// RegTensor: 5 (r0, r1, r2, tmp, d)
// 减少: 1x Store + 1x Load per iteration
```

## 关键修改点

1. 预期收益: **: 减少 2 次内存访问/iteration
**Trade-off**: RegTensor 从 3 增加到 5

## 常见陷阱

⚠️ 收益**: 减少内存访问 2-5x（取决于融合 VF 数量）
⚠️ 风险**: RegTensor 数量增加，可能超过 32 限制
⚠️ 复杂度**: 需确保控制流等价，循环结构一致

## 代码搜索关键词

```bash
grep -n "FusedAddMulVF" op_kernel/*.cpp op_host/*_tiling.cpp
```
