---
id: P32
bottlenecks: [mte3_stall, undersize_transfer]
op_families: [index_scatter]
complexity: L1
conflicts_with: []
synergizes_with: [P33]
has_preconditions: true
has_playbook: true
---

# P32: Scatter Block Update 的 stride 寻址

## 核心思想
支持输入张量 dim-0 非连续（通过 inputStride0_ 和 inputStride1_），实现 input[indices[k,0], indices[k,1], :] = update[k, :] 的散射写入。

## 代码骨架

// === 改造后（专家模式）===
```cpp
for (int64_t i = 0; i < loadCount; i++) {
    IndexT idx0 = indLocal.GetValue(i * INDICES_LAST_DIM);
    IndexT idx1 = indLocal.GetValue(i * INDICES_LAST_DIM + 1);
    int64_t gmOffset = static_cast<int64_t>(idx0) * inputStride0_
                     + static_cast<int64_t>(idx1) * inputStride1_;
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = updateDimSize_ * sizeof(T);
    DataCopyPad(inputGm_[gmOffset], updLocal[i * updateRowElements_], copyParams);
}
```

## 关键修改点

1. 预期收益: 支持非连续张量的散射更新，无需预先 reshape

## 常见陷阱

⚠️ 逐行写出每次只搬 1 行 MTE3 效率低
⚠️ indices 需要先搬入 UB 解析，增加额外的 MTE2 搬运和标量计算开销

## 代码搜索关键词

```bash
grep -n "DataCopy\|CopyIn\|CopyOut\|Fixpipe" op_kernel/*.cpp op_host/*_tiling.cpp
```
