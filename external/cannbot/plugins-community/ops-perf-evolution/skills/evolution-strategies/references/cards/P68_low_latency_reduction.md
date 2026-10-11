---
id: P68
bottlenecks: [compute_bound]
op_families: [attention, normalization]
complexity: L1
conflicts_with: [P84]
synergizes_with: [P67, P69]
has_preconditions: true
has_playbook: true
---

# P68: 低延迟归约指令组合 (Low-Latency Reduction Instruction Combination)

## 核心思想
归约操作中，WholeReduceSum 等归约指令延迟约为 Add 指令的 2-5 倍。通过二分累加（Add 指令逐步折半）将数据缩减到 256B 以内，再用 WholeReduceSum 一次完成最终归约，性能优于纯 WholeReduceSum 方案。进一步地，BlockReduceSum + WholeReduceSum 组合在特定 shape 下优于纯 WholeReduceSum（8.44us vs 13us，循环 100 次）。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// 二分累加方案：shape(bsLength, hLength) → shape(bsLength, 1)
SetMaskCount();
for (uint32_t i = 0; i < bsLength; i++) {
    LocalTensor<float> srcTmp = src[i * hLength];
    LocalTensor<float> dstTmp = dst[i * hLength];
    uint32_t totalNum = hLength / 16 * 16;
    uint32_t remaining = hLength - totalNum;
    while (totalNum > ONE_REPEAT_FLOAT_SIZE) {
        uint32_t halfNum = DivCeil(totalNum, 16) * DEFAULT_REP_STRIDE;
        SetVectorMask<uint8_t, MaskMode::COUNTER>(0, totalNum - halfNum);
        Add<float, false>(dstTmp, srcTmp, srcTmp[halfNum], MASK_PLACEHOLDER, 1, binaryParams);
        totalNum = halfNum;
        srcTmp = dstTmp;
    }
    // 最终 WholeReduceSum
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, totalNum);
    WholeReduceSum<float, false>(dstTmp, srcTmp, MASK_PLACEHOLDER, 1, ...);
}
ResetMask(); SetMaskNorm();
```

## 关键修改点

1. 预期收益: shape=30000 float 场景下 172 cycle vs WholeReduceSum 的 242 cycle（提升 29%）

## 常见陷阱

⚠️ 二分累加方案代码复杂度较高
⚠️ 小数据量或特殊 shape 下需具体分析，不一定有收益
⚠️ 需配合 Counter 模式（SetMaskCount）使用效果更佳

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue" op_kernel/*.cpp op_host/*_tiling.cpp
```
