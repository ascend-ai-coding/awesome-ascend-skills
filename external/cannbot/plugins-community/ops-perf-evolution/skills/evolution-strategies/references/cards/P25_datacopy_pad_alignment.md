---
id: P25
bottlenecks: [undersize_transfer]
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P25: DataCopyPad 带 Padding 的搬运

## 核心思想
当源数据列数不对齐 32B 时，使用 DataCopyPad + DataCopyPadExtParams 在搬运过程中自动补零对齐。

## 代码骨架

// === 改造后（专家模式）===
```cpp
uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
dataCopyParams.blockLen = info.s2dealNum;
dataCopyParams.srcStride = info.attenMaskStride - info.s2dealNum;
dataCopyParams.dstStride = 0;
DataCopyPadExtParams<bool> padParams{true, 0,
    static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 0};
DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
```

## 关键修改点

1. 预期收益: 一次搬运完成对齐，后续 Vector 计算无需额外处理非对齐尾部

## 常见陷阱

⚠️ padding 参数为 uint8_t 最大补零 255 字节；blockLen 单位为字节需注意转换
⚠️ 补零后的数据参与后续计算，需确保补零不影响算法正确性（如 ReduceSum 场景补零安全，但 ReduceMax 可能受影响）

## 代码搜索关键词

```bash
grep -n "DataCopy\|CopyIn\|CopyOut\|Fixpipe" op_kernel/*.cpp op_host/*_tiling.cpp
```
