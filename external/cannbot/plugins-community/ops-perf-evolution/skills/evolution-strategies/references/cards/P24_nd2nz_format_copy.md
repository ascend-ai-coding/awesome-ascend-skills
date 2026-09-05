---
id: P24
bottlenecks: [no_overlap]
op_families: [matmul]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P24: ND↔NZ 格式转换搬运

## 核心思想
在 GM→L1 搬运阶段同时完成 ND→NZ 格式转换，避免额外 TransData 指令。对齐到 16 元素（fp16）或 32 元素（fp8）的分形边界。反向地，写出时可通过 ScatterUpdateNZ 逐 token 完成 ND→NZ 格式转换，避免额外 TransData 算子（见 Variant B）。

## 代码骨架

// === 改造后（专家模式）===
```cpp
template<typename INPUT_T>
__aicore__ inline void CopyToL1Nd2Nz(const LocalTensor<INPUT_T> &l1Tensor,
    const GlobalTensor<INPUT_T> &gmTensor,
    uint32_t nValue, uint32_t dValue, uint32_t srcDValue) {
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.nValue = nValue;
    gm2L1Nd2NzParams.dValue = dValue;
    gm2L1Nd2NzParams.srcDValue = srcDValue;
    gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4;
    gm2L1Nd2NzParams.dstNzNStride = 1;
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}
```

## 关键修改点

1. 预期收益: 省去独立 TransData 指令，减少一次 L1 读写往返

## 常见陷阱

⚠️ 对齐 padding 浪费部分 L1 空间；不同数据类型对齐基数不同需条件编译
⚠️ 仅支持 GM→L1 方向的格式转换，L1→L0 仍需标准 LoadData 路径
⚠️ 写出方向逐 token 循环 MTE3 利用率低，需要 SToMTE3Sync 引入流水气泡

## 代码搜索关键词

```bash
grep -n "SyncAll\|SetFlag\|WaitFlag\|PipeBarrier\|GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
