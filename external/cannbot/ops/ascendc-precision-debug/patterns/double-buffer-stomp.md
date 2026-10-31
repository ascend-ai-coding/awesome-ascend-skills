# double buffer 数据踩踏

## 形态指纹

- 小 shape 通过，**大 shape（需要多轮 tile 循环）才错**。
- 结果偶发不一致，或超差点分布在某些 tile 边界。
- 常伴随 pipe 同步问题，但根因在 buffer 生命周期而非单个 flag。

## 根因机理

开 double buffer 后，两块 buffer 乒乓使用：第 i 轮在算 buffer A 时，第 i+1 轮的搬运正在写 buffer B。若循环结构或同步设计有误（如复用深度不够、flag 配对错位），会发生"还在读就被写"的踩踏，部分 tile 数据被提前覆盖。

## 验证方法

1. 关闭 double buffer（改为单 buffer）重跑：通过 → 确认方向。
2. 出错 tile 序号是否呈规律分布（如每隔一个 tile 错）→ 乒乓错位特征。
3. 减小 tile、增大循环轮数，错误率上升 → 进一步佐证。

## 修复 pattern

```cpp
// 反例：两轮共用一套 eventId，flag 错位
for (i...) { CopyIn(i); Compute(i); CopyOut(i); }  // 无独立同步

// 正例：每个 buffer 独立的同步事件，或直接使用 TQue 的
// EnQue/DeQue 机制（队列深度 2 即 double buffer，框架管同步）
TQue<QuePosition::VECIN, 2> inQue;   // depth=2
// 循环中严格 Alloc/EnQue → DeQue/Compute/Free 配对
```

原则：优先用 TQue 机制而不是手写 TBuf + flag 实现 double buffer。

## 反例（像但不是）

- 单 buffer 也错 → 不是踩踏，查逻辑或尾块。
- 错误与 tile 边界无关、完全随机 → 更像 uninitialized-memory。
