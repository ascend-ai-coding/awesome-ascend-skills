# HIVM 同步体系总览

> 关键词：Synchronization, Pipe Sync, Block Sync, FFTS, UnitFlag, Event ID, InjectSync, GraphSyncSolver

## 概述

HIVM 同步体系是 AscendNPU-IR 中协调多 Pipeline、多 Core、多 Block 之间数据依赖的核心机制。由于 NPU 采用多 Pipeline 并行执行架构（MTE1/MTE2/MTE3/M/V 等），不同 Pipeline 上的操作之间存在数据依赖，需要通过同步原语来保证执行顺序的正确性。

HIVM 同步体系分为三个层次：

1. **管道同步（Pipe Synchronization）**：同一 Block 内不同 Pipeline 之间的同步，通过 Event ID 机制实现。这是最基础、最常用的同步层次。
2. **跨核同步（Block Synchronization）**：不同 Block 之间的同步，通过 FFTS（Fast Flag Transfer System）机制实现。用于多 Block 协作计算场景。
3. **UnitFlag 同步**：处理循环中"至少执行一次"依赖的特殊同步模式，通过 UnitFlag 条件控制。

## 同步操作层次

```
┌─────────────────────────────────────────────────┐
│              跨核同步 (Block Sync)                │
│  sync_block / sync_block_set / sync_block_wait   │
│  create_sync_block_lock / sync_block_lock/unlock │
├─────────────────────────────────────────────────┤
│              管道同步 (Pipe Sync)                 │
│  set_flag / wait_flag / pipe_barrier             │
├─────────────────────────────────────────────────┤
│              UnitFlag 同步                        │
│  unit_flag_cond / unit_flag_mode                 │
│  (嵌入在宏操作中)                                  │
└─────────────────────────────────────────────────┘
```

## 管道同步

管道同步是 HIVM 中最细粒度的同步机制，用于同一 Block 内不同 Pipeline 之间的协调。核心操作：

| 操作 | 助记符 | 说明 |
|------|--------|------|
| SetFlagOp | `hir.set_flag` | 在 set_pipe 上设置 Event Flag |
| WaitFlagOp | `hir.wait_flag` | 在 wait_pipe 上等待 Event Flag |
| PipeBarrierOp | `hir.pipe_barrier` | Pipeline 屏障，等待该 Pipe 所有操作完成 |

### Event ID 机制

每个 Pipe 有 8 个 Event ID（EVENT_ID0 ~ EVENT_ID7），用于区分不同的同步点。`set_flag` 和 `wait_flag` 通过 Event ID 配对使用：

- `set_flag[set_pipe, wait_pipe, event_id]`：在 set_pipe 上发送信号
- `wait_flag[set_pipe, wait_pipe, event_id]`：在 wait_pipe 上等待信号

Event ID 可以是静态的（编译时确定的 `#hivm.event<EVENT_IDx>`）或动态的（运行时计算的 i64 值）。

## 跨核同步

跨核同步用于不同 Block 之间的协调，基于 FFTS 机制。核心操作：

| 操作 | 助记符 | 说明 |
|------|--------|------|
| SyncBlockOp | `hir.sync_block` | 高层跨核同步，支持多种模式 |
| SyncBlockSetOp | `hir.sync_block_set` | 发送跨核同步信号 |
| SyncBlockWaitOp | `hir.sync_block_wait` | 等待跨核同步信号 |
| CreateSyncBlockLockOp | `hir.create_sync_block_lock` | 创建跨核锁 |
| SyncBlockLockOp | `hir.sync_block_lock` | 获取跨核锁 |
| SyncBlockUnlockOp | `hir.sync_block_unlock` | 释放跨核锁 |

### SyncBlockMode

| 模式 | 值 | 说明 |
|------|---|------|
| ALL_CUBE | 0 | 所有 Cube Core 同步 |
| ALL_VECTOR | 1 | 所有 Vector Core 同步 |
| ALL_SUB_VECTOR | 2 | 所有 Sub-Vector 同步 |
| BARRIER_CUBE | 3 | Cube-Cube 屏障同步 |
| BARRIER_VECTOR | 4 | Vector-Vector 屏障同步 |
| ALL | 5 | 所有 AIC/AIV 同步 |

### FFTS 机制

FFTS（Fast Flag Transfer System）是 AscendNPU 的硬件同步机制。每个 Block 有一个 FFTS 基地址（`ffts_base_addr`），FFTS 收集特定 flag_id 后将其设置回 Block 组中的 Block，实现同步。

在 Ascend910B 及以上平台上，`ffts_base_addr` 必须通过 `hir.set_ffts_base_addr` 设置。

## UnitFlag 同步

UnitFlag 是嵌入在宏操作（如 mmadL1）中的同步机制，用于处理循环中"至少执行一次"的依赖场景。详见 [03-unit-flag.md](03-unit-flag.md)。

## 自动同步注入

HIVM 编译器提供两个自动同步注入 Pass：

1. **InjectSync Pass**：基于启发式规则，在操作前后自动插入 `set_flag`/`wait_flag`。
2. **GraphSyncSolver Pass**：基于图的同步求解器，构建操作依赖图后求解最优同步方案。

详见 [04-sync-injection.md](04-sync-injection.md)。

## 操作列表

| 操作 | 助记符 | 详细文档 |
|------|--------|---------|
| SetFlagOp | `hir.set_flag` | [01-pipe-sync.md](01-pipe-sync.md) |
| WaitFlagOp | `hir.wait_flag` | [01-pipe-sync.md](01-pipe-sync.md) |
| PipeBarrierOp | `hir.pipe_barrier` | [01-pipe-sync.md](01-pipe-sync.md) |
| SyncBlockOp | `hir.sync_block` | [02-block-sync.md](02-block-sync.md) |
| SyncBlockSetOp | `hir.sync_block_set` | [02-block-sync.md](02-block-sync.md) |
| SyncBlockWaitOp | `hir.sync_block_wait` | [02-block-sync.md](02-block-sync.md) |
| CreateSyncBlockLockOp | `hir.create_sync_block_lock` | [02-block-sync.md](02-block-sync.md) |
| SyncBlockLockOp | `hir.sync_block_lock` | [02-block-sync.md](02-block-sync.md) |
| SyncBlockUnlockOp | `hir.sync_block_unlock` | [02-block-sync.md](02-block-sync.md) |

## 相关文档

- 源码参考：[HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td)
- 测试用例：[sync-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/sync-ops.mlir)
