# 双缓冲四区轮转协议

本文描述 dispatch/combine 多轮次执行时避免缓冲区踩踏的四区双缓冲方案。

## 问题背景

- dispatch 内部有全卡同步点，各 rank dispatch 进度趋于一致
- combine 无全卡同步，快卡可能在慢卡 dispatch1 结束前就进入 dispatch2
- 若不隔离，dispatch2 可能覆盖 dispatch1 仍在使用的缓冲区，导致数据竞争和死锁

## 四区结构

Win 区划分为四个独立存储块：

```
┌─────────────┬─────────────┬─────────────┬─────────────┬──────────┬──────────┐
│ dispatch-0  │ combine-0   │ dispatch-1  │ combine-1   │dispatch  │combine   │
│ (数据+状态) │ (数据+状态) │ (数据+状态) │ (数据+状态) │bufChosen │bufChosen │
└─────────────┴─────────────┴─────────────┴─────────────┴──────────┴──────────┘
```

- dispatch 和 combine 各自维护独立的 `bufferChosen` 标志位（0 或 1）
- 标志位存储在 Win 区状态区末尾，两张卡通过 MTE 可见

## 缓冲区选择规则

### Dispatch

```
使用区域：bufferChosen == 0 → dispatch-0 区
           bufferChosen == 1 → dispatch-1 区
完成后：   dispatch_bufferChosen = dispatch_bufferChosen ^ 1   // 翻转
```

### Combine

```
使用区域：bufferChosen == 0 → combine-0 区
           bufferChosen == 1 → combine-1 区
完成后：   combine_bufferChosen = combine_bufferChosen ^ 1    // 翻转
```

## 安全性保证

| 保证 | 机制 |
| --- | --- |
| 同一轮次内所有 rank 使用相同 dispatch 缓冲区 | dispatch 内部全卡同步 |
| dispatch 与 combine 不互相干扰 | 各自独立缓冲区，独立标志位 |
| 第 N+1 轮不覆盖第 N 轮 | 双缓冲交替；dispatch 全卡同步保证 N+2 轮时 N 轮已全部完成 |

**复用安全规则**：dispatch 第 N 轮用缓冲区 X，第 N+2 轮可安全复用缓冲区 X（双缓冲周期为 2 轮）。

## 实现要点

- bufferChosen 标志位的读取和翻转需要通过 MTE 写 GM，不能只改本地 UB
- dispatch 翻转标志位的时机：所有核完成本轮等待 + 回搬后，由指定核执行翻转
- combine 翻转标志位的时机：所有核完成本轮聚合输出后，由指定核执行翻转
- 翻转前需要 `SyncAll<true>()` 确保所有核都完成了本轮消费

## 与单缓冲的区别

- 单缓冲实现可忽略本文，只需在 ClearLocalStatus 后确保下一轮再使用
- 双缓冲四区方案适用于多轮次流水场景，需在 Init 阶段额外分配一组缓冲区和两个标志位

## 下一跳

- 布局公式：`window-memory-layout.md`
- 分核公式：`multi-core-formulas.md`
- 同步原语选择：`../api-rules/sync-and-visibility.md`
