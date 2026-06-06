# 案例：核内异步同步缺失

## 现象

- 同一输入多次运行结果不同（不稳定）
- 有时通过，有时失败，错误位置不固定
- block_dim=1 依然不稳定

## 误差分析输出

```
运行 1: MaxAbsErr=0.000000  ✅
运行 2: MaxAbsErr=3.45e+01  ❌  错误元素随机分布
运行 3: MaxAbsErr=1.23e-02  ❌  不同位置出错
```

## 根因

CopyIn 的 EnQue 和 Compute 的 DeQue 之间缺少正确的队列同步，Compute 读取到了 DMA 尚未搬运完成的 UB 脏数据。

```cpp
// ❌ 错误：绕过了队列机制直接访问
__aicore__ inline void Process() {
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);   // 可能 DMA 还没完成就开始读
        CopyOut(i);
    }
}
```

## 验证与修复

Phase 3 实验 B：将 Process 中每步之间加 `PipeBarrier<PIPE_ALL>()` → 稳定通过。

确认是同步问题后，检查队列使用：确保 CopyIn 中 `EnQue` 与 Compute 中 `DeQue` 正确配对，TQue 的 PIPE 类型正确（VECIN/VECOUT）。

## 定位关键

1. Phase 1：多次运行结果不同 → 高度怀疑同步
2. Phase 3 实验 B：PIPE_ALL 后稳定 → 确认核内同步
3. 逐步恢复细粒度同步，定位缺失的 EnQue/DeQue 配对
