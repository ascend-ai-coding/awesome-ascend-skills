---
id: P42
bottlenecks: [ub_memory_pressure]
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: [P8]
has_preconditions: false
has_playbook: false
deprecated: true
deprecated_reason: "Code skeleton is identical copy-paste of P85. P85 has more general and accurate description."
---

# P42: workspace buffer 多区域分时复用

## 核心思想
一块大的 TBuf<VECCALC> 被划分为多个逻辑区域，供 RmsNorm 和 RoPE 交替使用。RmsNorm 使用 zone0/1/2，RoPE 复用 zone1/2 做 cos/sin 和中间变量。

## 代码骨架

// === 改造后（专家模式）===
```cpp
int64_t xLocalFp32Offset = 0;                  // zone0: RmsNorm fp32
int64_t xSquareLocalOffset = rows * headSize;   // zone1: RmsNorm square / RoPE cos
int64_t xSumLocalOffset = rows * headSize * 2;  // zone2: RmsNorm sum / RoPE sin
LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
```

## 关键修改点

1. 预期收益: 一块 buffer 服务多个计算阶段，最大化 UB 利用率

## 常见陷阱

⚠️ 各阶段不能同时执行，zone 复用要求严格的阶段串行顺序
⚠️ wsBuffer 大小 = zone 数 × rows × headSize × sizeof(float)，headSize 较大时占用显著

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue" op_kernel/*.cpp op_host/*_tiling.cpp
```
