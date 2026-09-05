# 同步注入 Pass — InjectSync / GraphSyncSolver

> 关键词：InjectSync, GraphSyncSolver, SyncMode, Event ID Allocation, IR 变换

## 概述

HIVM 编译器提供自动同步注入机制，在用户编写的 IR 中自动插入 `set_flag`/`wait_flag` 等同步操作，保证多 Pipeline 执行的正确性。用户通常不需要手动编写同步操作，编译器会根据操作间的数据依赖自动注入。

HIVM 提供两种同步注入 Pass：

1. **InjectSync Pass**：基于启发式规则的同步注入，按操作顺序分析数据依赖并插入同步操作。
2. **GraphSyncSolver Pass**：基于图的同步求解器，构建操作依赖图后求解最优同步方案，通常能生成更优的同步代码。

## InjectSync Pass

### Pass 定义

从 [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L163-L183) 提取：

```
def InjectSync : Pass<"hivm-inject-sync", "func::FuncOp"> {
  let summary = "auto inject sync";
  let options = [
    Option<"syncMode", "sync-mode", "hivm::SyncMode",
           "hivm::SyncMode::NORMAL",
           "inject sync mode">,
    Option<"enableUnitFlag", "enable-unit-flag", "bool", "false",
           "Enable unit-flag modes for synchronization">,
    Option<"assumeAliveLoops", "assume-alive-loops", "bool", "false",
           "Assume that all loops will execute at least once.">
  ];
}
```

### Pass 选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sync-mode` | SyncMode | NORMAL | 同步模式：NORMAL（正常注入）或 BARRIERALL（注入 pipe_barrier，仅用于调试） |
| `enable-unit-flag` | bool | false | 启用 UnitFlag 模式，处理循环可能不执行的同步场景 |
| `assume-alive-loops` | bool | false | 假设所有循环至少执行一次，简化同步分析 |

### SyncMode 枚举

| 模式 | 说明 |
|------|------|
| NORMAL | 正常模式，根据数据依赖插入 set_flag/wait_flag |
| BARRIERALL | 调试模式，在每个操作前后插入 pipe_barrier |

### IR 变换效果

#### 变换前

```mlir
func.func @example(%src : memref<16x16xf16, #hivm.address_space<gm>>,
                   %dst_l1 : memref<16x16xf16>,
                   %dst_l0c : memref<16x16xf32>) {
  hivm.hir.load ins(%src : ...) outs(%dst_l1 : ...)
  hivm.hir.mmadL1 ins(%a, %b, %init, ...) outs(%dst_l0c : ...)
  hivm.hir.fixpipe ins(%dst_l0c : ...) outs(%res : ...)
  return
}
```

#### 变换后（NORMAL 模式）

```mlir
func.func @example(%src : memref<16x16xf16, #hivm.address_space<gm>>,
                   %dst_l1 : memref<16x16xf16>,
                   %dst_l0c : memref<16x16xf32>) {
  hivm.hir.load ins(%src : ...) outs(%dst_l1 : ...)
  hivm.hir.set_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
  hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
  hivm.hir.mmadL1 ins(%a, %b, %init, ...) outs(%dst_l0c : ...)
  hivm.hir.set_flag [#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_FIX>, #hivm.event<EVENT_ID1>]
  hivm.hir.wait_flag [#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_FIX>, #hivm.event<EVENT_ID1>]
  hivm.hir.fixpipe ins(%dst_l0c : ...) outs(%res : ...)
  return
}
```

#### 变换后（BARRIERALL 模式，调试用）

```mlir
func.func @example(...) {
  hivm.hir.load ins(%src : ...) outs(%dst_l1 : ...)
  hivm.hir.pipe_barrier [#hivm.pipe<PIPE_MTE2>]
  hivm.hir.pipe_barrier [#hivm.pipe<PIPE_M>]
  hivm.hir.mmadL1 ins(%a, %b, %init, ...) outs(%dst_l0c : ...)
  hivm.hir.pipe_barrier [#hivm.pipe<PIPE_M>]
  hivm.hir.pipe_barrier [#hivm.pipe<PIPE_FIX>]
  hivm.hir.fixpipe ins(%dst_l0c : ...) outs(%res : ...)
  return
}
```

## GraphSyncSolver Pass

### Pass 定义

从 [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L195-L213) 提取：

```
def GraphSyncSolver : Pass<"hivm-graph-sync-solver", "func::FuncOp"> {
  let summary = "graph sync solver";
  let options = [
    Option<"enableUnitFlag", "enable-unit-flag", "bool", "false",
           "Enable unit-flag modes for synchronization">,
    Option<"enableTesterMode", "enable-tester-mode", "bool", "false",
           "Enable sync-tester mode">,
    ListOption<"syncTesterOptions", "sync-tester-options", "int64_t",
               "Sync-tester options">
  ];
}
```

### Pass 选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable-unit-flag` | bool | false | 启用 UnitFlag 模式 |
| `enable-tester-mode` | bool | false | 启用同步测试模式 |
| `sync-tester-options` | list\<i64\> | [] | 同步测试选项（num_runs, init_seed, num_ops, num_ptrs, enable-multibuffer） |

### 工作原理

GraphSyncSolver 的工作流程：

1. **IR 转换**：将 MLIR IR 转换为内部同步图表示（SyncSolverIR）
2. **图构建**：构建操作依赖图，节点为操作，边为数据依赖
3. **求解**：在图上求解最优同步方案，包括 Event ID 分配和同步点选择
4. **代码生成**：将求解结果转换回 MLIR IR，插入 set_flag/wait_flag 操作

### 与 InjectSync 的对比

| 特性 | InjectSync | GraphSyncSolver |
|------|-----------|----------------|
| 算法 | 启发式规则 | 图求解 |
| Event ID 使用 | 可能较多 | 更优化 |
| 同步粒度 | 操作级 | 可优化到更细粒度 |
| 编译速度 | 较快 | 较慢 |
| 同步质量 | 基本正确 | 更优，减少不必要的同步 |
| 调试支持 | BARRIERALL 模式 | Tester 模式 |

## Pipeline 中的选择

从 [HIVMPipelines.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp#L81-L98) 可以看到，Pipeline 中的选择逻辑：

- 如果 `enableHIVMGraphSyncSolver=true` 且 `enableHIVMInjectBarrierAllSync=false`：使用 GraphSyncSolver
- 否则：使用 InjectSync

## InjectBlockSync Pass

除了 Pipe 内同步，还有跨 Block 同步注入 Pass：

```
def InjectBlockSync : Pass<"hivm-inject-block-sync", "func::FuncOp"> {
  let options = [
    Option<"blockAllSync", "block-all-sync", "bool", "false",
           "Enable inject all block sync">,
    Option<"assumeAliveLoops", "assume-alive-loops", "bool", "false",
           "Assume that all loops will execute at least once.">
  ];
}
```

## 常见问题

**Q: 应该使用 InjectSync 还是 GraphSyncSolver？**
A: 默认使用 InjectSync。如果需要更优的同步方案（减少同步开销），可以启用 GraphSyncSolver。GraphSyncSolver 编译时间较长，但生成的同步代码通常更高效。

**Q: assume-alive-loops 选项的作用？**
A: 当设置为 true 时，编译器假设所有循环至少执行一次，可以简化同步分析，避免插入不必要的 UnitFlag。但如果不满足假设，可能导致同步错误。

**Q: BARRIERALL 模式有什么用？**
A: BARRIERALL 是调试模式，在每个操作前后插入 pipe_barrier。虽然性能很差，但可以排除同步问题导致的错误，帮助定位问题。

## 相关文档

- 源码参考：[Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L163-L213)
- InjectSync 实现：[InjectSync.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/Transforms/InjectSync/InjectSync.cpp)
- GraphSyncSolver 实现：[GraphSyncSolver.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/Transforms/GraphSyncSolver/GraphSyncSolver.cpp)
- Pipeline 配置：[HIVMPipelines.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp)
