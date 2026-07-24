---
id: P1
bottlenecks: [bus_contention, mte2_stall, mte3_stall, no_overlap, partial_overlap]
op_families: [datapath]
complexity: L0
conflicts_with: [P19, P20, P21]
synergizes_with: [P5, P8]
has_preconditions: true
has_playbook: true
---

# P1: 双缓冲机制 (Double Buffering)

## 核心思想
用两个 buffer 交替工作：一个用于计算时，另一个用于 DMA 搬运。隐藏数据传输延迟。

## 代码骨架
```cpp
// === 改造前（单缓冲，串行）===
for (int i = 0; i < N; i++) {
    CopyIn(i);      // DMA 搬运
    Compute(i);     // 计算
    CopyOut(i);     // DMA 搬出
}

// === 改造后（双缓冲，流水线）===
// Prologue: 预取第一块
CopyIn(0);
// Steady state: 计算当前块 + 预取下一块 并行
for (int i = 0; i < N - 1; i++) {
    CopyIn(i + 1);   // 预取下一块到空闲 buffer
    Compute(i);      // 计算当前块
    CopyOut(i);      // 搬出当前块
}
// Epilogue: 处理最后一块
Compute(N - 1);
CopyOut(N - 1);
```

## 关键修改点（必须全部完成）
1. **改 BUFFER_NUM**: 1 → 2（TQue/InitBuffer 的 buffer 数量）
2. **重构循环**: 增加 Prologue（预取第一块）和 Epilogue（处理最后一块）
3. **Tiling 联动**: tileSize 减半（双缓冲使 UB 用量翻倍，需保持总量不变）
4. **同步检查**: 确保 CopyIn/Compute/CopyOut 之间没有不必要的全局 SyncAll()

## 常见陷阱
- ❌ **只改 BUFFER_NUM 不改循环** → 无性能提升，buffer 只是空转
- ❌ **不改 tileSize** → UB 溢出，编译失败或运行时越界
- ❌ **Prologue 和 Epilogue 遗漏** → 第一块/最后一块数据错误

## 代码搜索关键词
用 Grep 定位当前代码中的对应结构：
```bash
grep -n "BUFFER_NUM\|bufferNum\|DB_BUFFER\|DOUBLE_BUFFER" *.cpp *.h
grep -n "InitBuffer.*TQue" *.cpp
grep -n "for.*loop\|CopyIn\|CopyOut\|Process()" *.cpp
```
