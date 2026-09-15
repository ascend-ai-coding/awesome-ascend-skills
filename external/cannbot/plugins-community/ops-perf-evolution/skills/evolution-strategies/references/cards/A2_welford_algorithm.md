---
id: A2
bottlenecks: []
op_families: [normalization]
complexity: L0
conflicts_with: []
synergizes_with: [A1, A6]
has_preconditions: true
has_playbook: true
---

# A2: Welford Numerically Stable Mean/Var (Welford数值稳定算法)

## 核心思想
在训练模式下，BatchNorm需要计算当前batch的均值和方差用于前向传播，同时更新running_mean和running_var。专家实现正确处理了样本方差（除以N-1）与总体方差（除以N）的区别：1）Batch统计量：使用batchVarScale系数将总体方差转换为样本方差（无偏估计），当patternR0 * patternR1 > 1时，batchVarScale = N / (N-1)；2）Running统计量：running_mean和running_var使用指数移动平均（EMA）更新，不涉及N-1校正；3）特殊值处理：当patternR0 * patternR1 == 1时（单样本场景），避免除以0，直接设置batchVarScale = 1.0。

## 代码骨架

// === 改造后（专家模式）===
```cpp
if (amsgrad_) {
    PipeBarrier<PIPE_V>();
    Max(dataOutLocal[maxGradOutOffset_], dataLocal[maxGradNormOffset_], 
        dataOutLocal[expAvgSqOffset_], dataCount);
    PipeBarrier<PIPE_V>();
    Sqrt(dataLocal[varOffset_], dataOutLocal[maxGradOutOffset_], dataCount);
    PipeBarrier<PIPE_V>();
    Muls(dataLocal[expAvgOffset_], dataLocal[varOffset_], biasCorrection2Sqrt_, dataCount);
} else {
    Sqrt(dataLocal[varOffset_], dataOutLocal[expAvgSqOffset_], dataCount);
    // ...
}
```

## 关键修改点

1. 用细粒度同步替代粗粒度 SyncAll
2. 预期收益: 解决Adam发散问题；提升训练稳定性；支持Adam/AMSGrad切换

## 常见陷阱

⚠️ 额外的Max操作；需要维护max_grad_norm状态
⚠️ 需要额外的乘法操作；需要特殊处理单样本场景
⚠️ 算法复杂度增加，需要维护额外的累加器状态

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM" op_kernel/*.cpp op_host/*_tiling.cpp
```
