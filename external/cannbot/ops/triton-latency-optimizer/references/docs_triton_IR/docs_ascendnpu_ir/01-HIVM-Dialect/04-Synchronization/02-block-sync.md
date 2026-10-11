# 跨核同步操作 — sync_block / sync_block_set / sync_block_wait

> 关键词：Block Sync, FFTS, SyncBlockMode, sync_block, sync_block_set, sync_block_wait, Lock

## 概述

跨核同步操作用于不同 Block 之间的协调，基于 FFTS（Fast Flag Transfer System）硬件机制实现。在多 Block 协作计算场景中（如大矩阵分块乘法、AllReduce 等），不同 Block 需要同步以避免数据竞争。

HIVM 提供两种跨核同步接口：
1. **高层接口**：`sync_block`，封装了 FFTS 的细节，通过 SyncBlockMode 指定同步模式
2. **低层接口**：`sync_block_set` / `sync_block_wait`，直接控制 FFTS 的 set/wait 操作

此外，还提供基于锁的同步原语：`create_sync_block_lock` / `sync_block_lock` / `sync_block_unlock`。

## IR 操作定义

### SyncBlockOp

从 [HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L78-L118) 提取：

```
def SyncBlockOp : HIVM_SynchronizationOp<"sync_block"> {
  let arguments = (ins HIVM_SyncBlockModeAttr:$sync_block_mode,
                       OptionalAttr<Builtin_IntegerAttr>:$flag_id,
                       Optional<I64>:$ffts_base_addr,
                       OptionalAttr<HIVM_PipeAttr>:$tcube_pipe,
                       OptionalAttr<HIVM_PipeAttr>:$tvector_pipe);
  let assemblyFormat = [{
    attr-dict `[` $sync_block_mode (`,` $flag_id^)?`]`
    (`ffts_base_addr` `=` $ffts_base_addr^)?
    (`tcube_pipe` `=` $tcube_pipe^)?
    (`tvector_pipe` `=` $tvector_pipe^)?
  }];
}
```

### SyncBlockSetOp

从 [HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L120-L148) 提取：

```
def SyncBlockSetOp : HIVM_SynchronizationOp<"sync_block_set", [AttrSizedOperandSegments]> {
  let arguments = (ins HIVM_TCoreTypeAttr:$tcore_type,
                       HIVM_PipeAttr:$tpipe,
                       HIVM_PipeAttr:$pipe,
                       OptionalAttr<Builtin_IntegerAttr>:$static_flag_id,
                       Optional<I64>:$dynamic_flag_id,
                       Optional<I64>:$ffts_base_addr,
                       DefaultValuedOptionalAttr<HIVM_SyncBlockInstrModeAttr,
                         "INTRA_BLOCK_SYNCHRONIZATION">:$tsync_instr_mode);
}
```

### SyncBlockWaitOp

从 [HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L150-L175) 提取：

```
def SyncBlockWaitOp : HIVM_SynchronizationOp<"sync_block_wait"> {
  let arguments = (ins HIVM_TCoreTypeAttr:$tcore_type,
                       HIVM_PipeAttr:$tpipe,
                       HIVM_PipeAttr:$pipe,
                       OptionalAttr<Builtin_IntegerAttr>:$static_flag_id,
                       Optional<I64>:$dynamic_flag_id,
                       DefaultValuedOptionalAttr<HIVM_SyncBlockInstrModeAttr,
                         "INTRA_BLOCK_SYNCHRONIZATION">:$tsync_instr_mode);
}
```

## 参数说明

### SyncBlockOp 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$sync_block_mode` | HIVM_SyncBlockModeAttr | 是 | 同步模式 |
| `$flag_id` | IntegerAttr | 否 | Flag ID（静态） |
| `$ffts_base_addr` | I64 | 否 | FFTS 基地址 |
| `$tcube_pipe` | HIVM_PipeAttr | 否 | Cube Core 等待的 Pipe |
| `$tvector_pipe` | HIVM_PipeAttr | 否 | Vector Core 等待的 Pipe |

### SyncBlockSetOp / SyncBlockWaitOp 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$tcore_type` | HIVM_TCoreTypeAttr | 是 | Core 类型（CUBE/VECTOR） |
| `$tpipe` | HIVM_PipeAttr | 是 | 目标 Core 的 Pipe |
| `$pipe` | HIVM_PipeAttr | 是 | 当前操作的 Pipe |
| `$static_flag_id` | IntegerAttr | 否 | 静态 Flag ID |
| `$dynamic_flag_id` | I64 | 否 | 动态 Flag ID |
| `$ffts_base_addr` | I64 | 否 | FFTS 基地址（仅 SetOp） |
| `$tsync_instr_mode` | HIVM_SyncBlockInstrModeAttr | 否 | 同步指令模式，默认 INTRA_BLOCK_SYNCHRONIZATION |

### SyncBlockMode 枚举

| 模式 | 值 | 说明 | 必需参数 |
|------|---|------|---------|
| ALL_CUBE | 0 | 所有 Cube Core 同步到同一点 | `tcube_pipe` |
| ALL_VECTOR | 1 | 所有 Vector Core 同步到同一点 | `tvector_pipe` |
| ALL_SUB_VECTOR | 2 | 所有 Sub-Vector 同步 | - |
| BARRIER_CUBE | 3 | Cube-Cube 屏障，lowering 为 barrier.pipe_all，仅复制到 AIC kernel | - |
| BARRIER_VECTOR | 4 | Vector-Vector 屏障，lowering 为 barrier.pipe_all，仅复制到 AIV kernel | - |
| ALL | 5 | 所有 AIC/AIV 同步到同一点 | `tvector_pipe` |

### SyncBlockInstrMode 枚举

| 模式 | 值 | 说明 |
|------|---|------|
| INTER_BLOCK_SYNCHRONIZATION | 0 | 跨 Block 同步 |
| INTER_SUBBLOCK_SYNCHRONIZATION | 1 | 跨 Sub-Block 同步 |
| INTRA_BLOCK_SYNCHRONIZATION | 2 | Block 内同步（默认） |

### 锁操作参数

| 操作 | 参数 | 说明 |
|------|------|------|
| create_sync_block_lock | `lockArg`: Optional\<AnyMemRef\> | 创建锁内存区域，返回 memref\<1xi64\> |
| sync_block_lock | `lock_var`: MemRefRankOf<[I64], [1]> | 获取锁，阻塞直到 lock_var == block_idx |
| sync_block_unlock | `lock_var`: MemRefRankOf<[I64], [1]> | 释放锁，lock_var 递增并释放 |

## IR 示例

### SyncBlockOp — ALL 模式

```mlir
%ffts_base_addr = arith.constant 0 : i64
hivm.hir.sync_block[#hivm.sync_block_mode<ALL>, 1 : i16]
          ffts_base_addr = %ffts_base_addr
          tcube_pipe=#hivm.pipe<PIPE_FIX>
          tvector_pipe=#hivm.pipe<PIPE_MTE3>
```

### SyncBlockOp — ALL_CUBE 模式

```mlir
hivm.hir.sync_block[#hivm.sync_block_mode<ALL_CUBE>, 1 : i16]
          ffts_base_addr = %ffts_base_addr
          tcube_pipe=#hivm.pipe<PIPE_FIX>
```

### SyncBlockSetOp — 静态 Flag ID

```mlir
%ffts_base_addr = arith.constant 0 : i64
hivm.hir.sync_block_set[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_FIX>, #hivm.pipe<PIPE_FIX>]
  flag = 1
  ffts_base_addr = %ffts_base_addr
  sync_instr_mode = #hivm.sync_block_instr_mode<INTER_BLOCK_SYNCHRONIZATION>
```

### SyncBlockSetOp — 动态 Flag ID

```mlir
%flag_id = arith.constant 0 : i64
hivm.hir.sync_block_set[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_FIX>, #hivm.pipe<PIPE_FIX>]
  flag = %flag_id
  ffts_base_addr = %ffts_base_addr
  sync_instr_mode = #hivm.sync_block_instr_mode<INTER_BLOCK_SYNCHRONIZATION>
```

### SyncBlockWaitOp

```mlir
hivm.hir.sync_block_wait[#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_V>] flag = 1
```

### 锁操作

```mlir
%lock = hivm.hir.create_sync_block_lock() : memref<1xi64>
hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
// ... 临界区代码 ...
hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
```

### 锁操作 — 从外部内存创建

```mlir
%lock = hivm.hir.create_sync_block_lock() from %arg : from memref<?xi8> to memref<1xi64>
```

## IR 层约束与验证

1. **FFTS 基地址**：在 Ascend910B 及以上平台，`ffts_base_addr` 必须设置。
2. **sync_block 限制**：只能在数据已搬运到 GM 后使用。
3. **Flag ID**：FFTS 收集特定 flag_id 后将其设置回 Block 组中的 Block，实现同步。
4. **tcube_pipe / tvector_pipe**：ALL_CUBE 模式需要 `tcube_pipe`，ALL_VECTOR/ALL 模式需要 `tvector_pipe`。
5. **锁操作**：`sync_block_lock` 阻塞直到 `lock_var == block_idx`，`sync_block_unlock` 递增并释放 `lock_var`。
6. **sync_block_set/wait 配对**：必须使用相同的 (tcore_type, tpipe, pipe, flag_id) 组合。

## 常见问题

**Q: sync_block 和 sync_block_set/wait 的区别？**
A: sync_block 是高层封装，自动处理 set/wait 配对；sync_block_set/wait 是低层接口，需要手动配对。通常推荐使用 sync_block。

**Q: 什么时候需要设置 ffts_base_addr？**
A: 在 Ascend910B 及以上平台，跨 Block 同步必须设置 FFTS 基地址。通常通过 `hir.set_ffts_base_addr` 在函数入口设置。

**Q: 锁操作的使用场景？**
A: 锁操作用于需要互斥访问共享资源的场景，如多个 Block 写入同一 GM 区域。`create_sync_block_lock` 分配锁内存，`sync_block_lock` 获取锁，`sync_block_unlock` 释放锁。

## 相关文档

- 源码参考：[HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L78-L244)
- 测试用例：[sync-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/sync-ops.mlir)
- SyncBlockMode 枚举：[06-Attributes-Types/01-enumerations.md](../06-Attributes-Types/01-enumerations.md)
