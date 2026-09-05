# 管道同步操作 — set_flag / wait_flag / pipe_barrier

> 关键词：Pipe Sync, Event ID, set_flag, wait_flag, pipe_barrier, Pipeline Synchronization

## 概述

管道同步操作是 HIVM 中最细粒度的同步机制，用于同一 Block 内不同 Pipeline 之间的协调。NPU 的多 Pipeline 架构中，MTE1/MTE2/MTE3/M/V 等 Pipeline 可以并行执行，但存在数据依赖时需要通过 Event Flag 机制保证执行顺序。

三个核心操作：
- `hir.set_flag`：在 set_pipe 上设置 Event Flag，通知 wait_pipe 数据已就绪
- `hir.wait_flag`：在 wait_pipe 上等待 Event Flag，阻塞直到数据就绪
- `hir.pipe_barrier`：Pipeline 屏障，等待该 Pipe 上所有先前操作完成

## IR 操作定义

### SetFlagOp

从 [HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L34-L49) 提取：

```
def SetFlagOp : HIVM_SynchronizationOp<"set_flag"> {
  let arguments = (ins HIVM_PipeAttr:$set_pipe,
                       HIVM_PipeAttr:$wait_pipe,
                       OptionalAttr<HIVM_EventAttr>:$static_event_id,
                       Optional<I64>:$dynamic_event_id);
  let assemblyFormat = [{
    `[`
    $set_pipe
    `,` $wait_pipe
    `,` custom<EventID>($static_event_id, $dynamic_event_id)
    `]` attr-dict
  }];
}
```

### WaitFlagOp

从 [HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L51-L66) 提取：

```
def WaitFlagOp : HIVM_SynchronizationOp<"wait_flag"> {
  let arguments = (ins HIVM_PipeAttr:$set_pipe,
                       HIVM_PipeAttr:$wait_pipe,
                       OptionalAttr<HIVM_EventAttr>:$static_event_id,
                       Optional<I64>:$dynamic_event_id);
  let assemblyFormat = [{
    `[`
    $set_pipe
    `,` $wait_pipe
    `,` custom<EventID>($static_event_id, $dynamic_event_id)
    `]` attr-dict
  }];
}
```

### PipeBarrierOp

从 [HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L68-L72) 提取：

```
def PipeBarrierOp : HIVM_SynchronizationOp<"pipe_barrier"> {
  let arguments = (ins HIVM_PipeAttr:$pipe);
  let assemblyFormat = "`[` $pipe `]` attr-dict";
}
```

## 参数说明

### SetFlagOp / WaitFlagOp 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$set_pipe` | HIVM_PipeAttr | 是 | 发送信号的 Pipeline |
| `$wait_pipe` | HIVM_PipeAttr | 是 | 接收信号的 Pipeline |
| `$static_event_id` | HIVM_EventAttr | 否 | 静态 Event ID（编译时确定） |
| `$dynamic_event_id` | I64 | 否 | 动态 Event ID（运行时计算） |

### PipeBarrierOp 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$pipe` | HIVM_PipeAttr | 是 | 需要屏障的 Pipeline |

### Event ID 约束

- 每个 Pipe 有 8 个 Event ID（EVENT_ID0 ~ EVENT_ID7）
- `static_event_id` 和 `dynamic_event_id` 二选一，不能同时存在
- 同一对 set_flag/wait_flag 必须使用相同的 (set_pipe, wait_pipe, event_id) 三元组
- Event ID 在同一 (set_pipe, wait_pipe) 对中不能重复使用（除非前一对已完成 wait）

### 典型 Pipe 组合

| set_pipe | wait_pipe | 场景 |
|----------|-----------|------|
| PIPE_MTE1 | PIPE_M | L1 数据加载完成后通知 Cube 计算 |
| PIPE_MTE2 | PIPE_M | GM 数据加载完成后通知 Cube 计算 |
| PIPE_M | PIPE_MTE3 | Cube 计算完成后通知 GM 写回 |
| PIPE_M | PIPE_FIX | Cube 计算完成后通知 Fixpipe |
| PIPE_M | PIPE_V | Cube 计算完成后通知 Vector 后处理 |

## IR 示例

### 静态 Event ID

```mlir
hivm.hir.set_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
```

### 动态 Event ID

```mlir
%eventId = arith.constant 1 : i64
hivm.hir.set_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, %eventId]
hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, %eventId]
```

### Pipe Barrier

```mlir
hivm.hir.pipe_barrier [#hivm.pipe<PIPE_M>]
```

### 完整的 Load → Compute → Store 同步模式

```mlir
// 加载数据到 L1
hivm.hir.load ins(%src_gm : ...) outs(%dst_l1 : ...)
// 设置 flag 通知 M Pipe 数据已就绪
hivm.hir.set_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]

// 等待数据就绪后执行矩阵乘
hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
hivm.hir.mmadL1 ins(...) outs(...)

// 设置 flag 通知 MTE3 Pipe 计算结果已就绪
hivm.hir.set_flag [#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_MTE3>, #hivm.event<EVENT_ID1>]

// 等待计算完成后写回 GM
hivm.hir.wait_flag [#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_MTE3>, #hivm.event<EVENT_ID1>]
hivm.hir.store ins(...) outs(...)
```

## IR 层约束与验证

1. **Event ID 配对**：set_flag 和 wait_flag 必须使用相同的 (set_pipe, wait_pipe, event_id) 三元组。
2. **Event ID 唯一性**：同一 (set_pipe, wait_pipe) 对中，正在使用的 Event ID 不能重复。
3. **静态/动态互斥**：`static_event_id` 和 `dynamic_event_id` 不能同时存在。
4. **Pipe 合法性**：set_pipe 和 wait_pipe 必须是合法的 PIPE 枚举值。
5. **顺序约束**：set_flag 必须在对应的 wait_flag 之前执行（否则 wait_flag 会阻塞）。
6. **PipeBarrier**：pipe_barrier 会等待指定 Pipe 上所有先前操作完成，是比 set_flag/wait_flag 更重的同步操作。

## 常见问题

**Q: 什么时候用静态 Event ID，什么时候用动态 Event ID？**
A: 静态 Event ID 适用于编译时可以确定同步点的场景（如单次 load→compute→store）。动态 Event ID 适用于循环中需要复用 Event ID 的场景，运行时根据循环变量计算 Event ID。

**Q: Event ID 数量不够用怎么办？**
A: 每个 Pipe 只有 8 个 Event ID。在复杂场景中，InjectSync Pass 会自动管理 Event ID 的分配和复用。GraphSyncSolver Pass 可以更优化地使用 Event ID。

**Q: pipe_barrier 和 set_flag/wait_flag 的区别？**
A: pipe_barrier 是重量级同步，等待整个 Pipe 完成；set_flag/wait_flag 是轻量级同步，只同步特定的数据依赖。通常优先使用 set_flag/wait_flag。

## 相关文档

- 源码参考：[HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L34-L72)
- 测试用例：[sync-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/sync-ops.mlir)
- Event 枚举：[06-Attributes-Types/01-enumerations.md](../06-Attributes-Types/01-enumerations.md)
