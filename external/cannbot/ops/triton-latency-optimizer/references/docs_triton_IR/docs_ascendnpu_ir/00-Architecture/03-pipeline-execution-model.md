# Pipeline 执行模型

> 关键词：PIPE, OpPipeTrait, MacroOpPipeTrait, SinglePipeOpTrait, set_flag, wait_flag, pipe_barrier, 同步, 流水线

## 概述

Ascend NPU 的 AI Core 内部采用多 Pipeline 并行执行架构。每个 Pipeline 对应一个硬件执行单元（如向量计算单元、矩阵计算单元、DMA 引擎等），不同 Pipeline 之间可以并行工作，但同一 Pipeline 内的操作是顺序执行的。

HIVM IR 通过 `PIPE` 枚举和一系列 TableGen Trait（`OpPipeTrait`、`MacroOpPipeTrait`、`SinglePipeOpTrait`）将每个操作绑定到特定的 Pipeline。编译器利用这些信息进行同步分析（`InjectSync` Pass），在需要时插入 `set_flag`/`wait_flag`/`pipe_barrier` 同步操作，确保数据依赖关系得到满足。

理解 Pipeline 执行模型对于编写正确的 HIVM IR 和调试同步问题至关重要。本文档从 [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td) 和 [HIVMTraits.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td) 中精确提取所有 Pipeline 定义和 Trait 机制。

## Pipe 枚举完整列表

源文件：[HIVMAttrs.td:203-236](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L203-L236)

| 枚举值 | C++ 符号 | 数值 | IR 标识符 | 硬件执行单元 | 说明 |
|--------|---------|------|-----------|-------------|------|
| PIPE_S | `PIPE::PIPE_S` | 0 | `PIPE_S` | Scalar 单元 | 标量流水线，执行标量计算和控制流 |
| PIPE_V | `PIPE::PIPE_V` | 1 | `PIPE_V` | Vector 单元 | 向量计算流水线，执行向量运算和 UB 内数据搬运 |
| PIPE_M | `PIPE::PIPE_M` | 2 | `PIPE_M` | Cube 单元 | 矩阵计算流水线，执行矩阵乘法 |
| PIPE_MTE1 | `PIPE::PIPE_MTE1` | 3 | `PIPE_MTE1` | MTE1 DMA 引擎 | L1 到 L0A/L0B/BT Buffer 的单向数据通路 |
| PIPE_MTE2 | `PIPE::PIPE_MTE2` | 4 | `PIPE_MTE2` | MTE2 DMA 引擎 | GM 与 L1/UB 之间的双向数据通路 |
| PIPE_MTE3 | `PIPE::PIPE_MTE3` | 5 | `PIPE_MTE3` | MTE3 DMA 引擎 | UB 到 GM 的单向数据通路 |
| PIPE_ALL | `PIPE::PIPE_ALL` | 6 | `PIPE_ALL` | 所有单元 | 所有流水线的统称，用于同步操作 |
| PIPE_MTE4 | `PIPE::PIPE_MTE4` | 7 | `PIPE_MTE4` | MTE4 DMA 引擎 | 额外的数据传输通路 |
| PIPE_MTE5 | `PIPE::PIPE_MTE5` | 8 | `PIPE_MTE5` | MTE5 DMA 引擎 | 额外的数据传输通路 |
| PIPE_V2 | `PIPE::PIPE_V2` | 9 | `PIPE_V2` | 第二 Vector 单元 | 第二向量流水线 |
| PIPE_FIX | `PIPE::PIPE_FIX` | 10 | `PIPE_FIX` | FixPipe 单元 | FixPipe 数据通路，L0C 到 GM/L1/UB |
| VIRTUAL_PIPE_MTE2_L1A | `PIPE::VIRTUAL_PIPE_MTE2_L1A` | 11 | `VIRTUAL_PIPE_MTE2_L1A` | 虚拟 Pipeline | 虚拟 MTE2 L1A 通路，用于编译器内部区分 |
| VIRTUAL_PIPE_MTE2_L1B | `PIPE::VIRTUAL_PIPE_MTE2_L1B` | 12 | `VIRTUAL_PIPE_MTE2_L1B` | 虚拟 Pipeline | 虚拟 MTE2 L1B 通路，用于编译器内部区分 |
| PIPE_NUM | `PIPE::PIPE_NUM` | 13 | `PIPE_NUM` | - | Pipeline 总数计数，非实际 Pipeline |
| PIPE_UNASSIGNED | `PIPE::PIPE_UNASSIGNED` | 99 | `PIPE_UNASSIGNED` | - | 未分配 Pipeline，用于无 Pipe 属性的操作 |

### Pipe 分类

**物理 Pipeline**（对应实际硬件执行单元）：PIPE_S, PIPE_V, PIPE_M, PIPE_MTE1, PIPE_MTE2, PIPE_MTE3, PIPE_MTE4, PIPE_MTE5, PIPE_V2, PIPE_FIX

**虚拟 Pipeline**（编译器内部使用，用于更细粒度的同步控制）：VIRTUAL_PIPE_MTE2_L1A, VIRTUAL_PIPE_MTE2_L1B

**特殊值**：PIPE_ALL（同步用）, PIPE_NUM（计数用）, PIPE_UNASSIGNED（未分配）

## 每个 Pipe 值与硬件执行单元的对应关系

| Pipe | 硬件执行单元 | 数据流 | 典型操作 |
|------|-------------|--------|---------|
| PIPE_S | Scalar | 标量计算 | 循环控制、条件判断 |
| PIPE_V | Vector (AIV) | UB -> UB | `hivm.vadd`, `hivm.vmul`, `hivm.vcast` 等 |
| PIPE_M | Cube (AIC) | L0A/L0B -> L0C | `hivm.mmadL1` 中的矩阵乘法部分 |
| PIPE_MTE1 | MTE1 DMA | L1 -> L0A/L0B/BT | `hivm.mmadL1` 中的数据加载部分 |
| PIPE_MTE2 | MTE2 DMA | GM <-> L1/UB | `hivm.load`, `hivm.nd2nz` |
| PIPE_MTE3 | MTE3 DMA | UB -> GM/L1 | `hivm.store`, `hivm.nz2nd` |
| PIPE_FIX | FixPipe | L0C -> GM/L1/UB | `hivm.fixpipe` |
| PIPE_V2 | 第二 Vector | UB -> UB | 第二向量单元操作 |
| PIPE_MTE4 | MTE4 DMA | 额外传输通路 | 预留 |
| PIPE_MTE5 | MTE5 DMA | 额外传输通路 | 预留 |

## IR 操作的 Pipe 属性映射表

### DMA 操作

源文件：[HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)

| 操作 | Pipe | Trait 声明 | 核心类型 |
|------|------|-----------|---------|
| `hivm.load` | PIPE_MTE2 | `OpPipeTrait<"PIPE::PIPE_MTE2">` | 可推断 |
| `hivm.store` | PIPE_MTE3 | `OpPipeTrait<"PIPE::PIPE_MTE3">` | 可推断 |
| `hivm.copy` | 动态推断 | `getPipe()` 方法 | 可推断 |
| `hivm.fixpipe` | PIPE_FIX | `OpPipeTrait<"PIPE::PIPE_FIX">` | CUBE |
| `hivm.nd2nz` | PIPE_MTE2 | `OpPipeTrait<"PIPE::PIPE_MTE2">` | CUBE |
| `hivm.nz2nd` | PIPE_MTE3 | `OpPipeTrait<"PIPE::PIPE_MTE3">` | CUBE |
| `hivm.l12ub` | PIPE_MTE1 | `OpPipeTrait<"PIPE::PIPE_MTE1">` | CUBE |

**`hivm.copy` 的 Pipe 推断逻辑**：

源文件：[HIVMDMAOps.cpp:616-622](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L616-L622)

| 源地址空间 | 目标地址空间 | 推断的 Pipe |
|-----------|-------------|------------|
| UB | UB | PIPE_V |
| L0C | GM | PIPE_FIX |
| GM | L1 | PIPE_MTE2 |
| UB | L1 | PIPE_MTE3 |

### 向量操作

源文件：[HIVMVectorOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td)

所有向量操作默认继承自 `HIVM_VectorOp`，其 Pipe 和核心类型通过基类 Trait 定义：

```tablegen
class HIVM_VectorOp<string mnemonic, list<Trait> traits = [],
  list<Trait> vecSpecialTraits=[OpPipeTrait<"PIPE::PIPE_V">, VectorCoreTypeTrait]> :
  HIVM_StructuredOp<mnemonic, !listconcat(!listconcat(
    [AlwaysSpeculatable, SinglePipeOpTrait
    ], vecSpecialTraits), traits)>
```

即所有向量操作默认 Pipe 为 `PIPE_V`，核心类型为 `VECTOR`。

| 操作类别 | 代表操作 | Pipe | 核心类型 |
|---------|---------|------|---------|
| 一元运算 | `vexp`, `vabs`, `vln`, `vrelu`, `vrsqrt`, `vsqrt`, `vtanh`, `vsin`, `vcos`, `verf`, `vrec`, `vnot`, `vcast` | PIPE_V | VECTOR |
| 二元运算 | `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vor`, `vand`, `vxor`, `vshl`, `vshr`, `vcmp`, `vpow`, `vmod`, `vmodui` | PIPE_V | VECTOR |
| 三元运算 | `vsel` | PIPE_V | VECTOR |
| 广播 | `vbrc` | PIPE_V | VECTOR |
| 规约 | `vreduce` | PIPE_V | VECTOR |
| 转置 | `vtranspose` | PIPE_V | VECTOR |
| 其他 | `varange`, `vinterleave`, `vdeinterleave`, `vflip`, `vmulextended`, `vpad`, `vconcat`, `vgather`, `vcumprod`, `vcumsum`, `vsort` | PIPE_V | VECTOR |

### 宏操作

源文件：[HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td)

宏操作使用 `MacroOpPipeTrait` 标注其输入和输出 Pipeline。

| 操作 | InPipe | OutPipe | Trait 声明 | 核心类型 |
|------|--------|---------|-----------|---------|
| `hivm.mmadL1` | PIPE_MTE1 | PIPE_M | `MacroOpPipeTrait<"PIPE::PIPE_MTE1, PIPE::PIPE_M">` | CUBE |
| `hivm.batchMmadL1` | PIPE_MTE1 | PIPE_M | `MacroOpPipeTrait<"PIPE::PIPE_MTE1, PIPE::PIPE_M">` | CUBE |
| `hivm.matmul` | PIPE_MTE2 | PIPE_MTE3 | `MacroOpPipeTrait<"PIPE::PIPE_MTE2, PIPE::PIPE_MTE3">` | 可推断 |
| `hivm.mix_matmul` | PIPE_MTE2 | PIPE_MTE3 | `MacroOpPipeTrait<"PIPE::PIPE_MTE2, PIPE::PIPE_MTE3">` | 可推断 |
| `hivm.mix_group_matmul` | PIPE_MTE2 | PIPE_MTE3 | `MacroOpPipeTrait<"PIPE::PIPE_MTE2, PIPE::PIPE_MTE3">` | 可推断 |

## OpPipeTrait / MacroOpPipeTrait / SinglePipeOpTrait 的 TableGen 定义

源文件：[HIVMTraits.td:157-173](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L157-L173)

### SinglePipeOpTrait

标记操作拥有单一 Pipeline 属性。

```tablegen
def SinglePipeOpTrait : NativeOpTrait<"SinglePipeOpTrait">;
```

### OpPipeTrait

参数化 Trait，将操作绑定到特定的单一 Pipeline。继承自 `SinglePipeOpTrait`。

```tablegen
class OpPipeTrait<string Pipe>
    : ParamNativeOpTrait<"OpPipeTrait", Pipe, [SinglePipeOpTrait]>;
```

使用示例：

```tablegen
OpPipeTrait<"PIPE::PIPE_MTE2">   // 操作属于 MTE2 Pipeline
OpPipeTrait<"PIPE::PIPE_FIX">    // 操作属于 FIX Pipeline
```

### MacroOpTrait

标记操作为宏操作（包含多个子操作，涉及多个 Pipeline）。

```tablegen
def MacroOpTrait : NativeOpTrait<"MacroOpTrait">;
```

### MacroOpPipeTrait

参数化 Trait，标注宏操作的输入 Pipeline 和输出 Pipeline。继承自 `MacroOpTrait`。

```tablegen
class MacroOpPipeTrait<string InOutPipes>
      : ParamNativeOpTrait<"MacroOpPipeTrait", InOutPipes,
                           [MacroOpTrait]>;
```

使用示例：

```tablegen
MacroOpPipeTrait<"PIPE::PIPE_MTE1, PIPE::PIPE_M">    // mmadL1: 输入=MTE1, 输出=M
MacroOpPipeTrait<"PIPE::PIPE_MTE2, PIPE::PIPE_MTE3"> // matmul: 输入=MTE2, 输出=MTE3
```

### OpPipeInterface

源文件：[OpPipeInterface.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Interfaces/OpPipeInterface.td)

`OpPipeInterface` 是统一的操作接口，提供以下方法：

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `isSinglePipeOp()` | `bool` | 判断是否为单 Pipe 操作 |
| `isMacroOp()` | `bool` | 判断是否为宏操作 |
| `getPipe()` | `mlir::hivm::PIPE` | 获取单 Pipe 操作的 Pipeline（非单 Pipe 返回 `PIPE_UNASSIGNED`） |
| `getInPipe()` | `mlir::hivm::PIPE` | 获取宏操作的输入 Pipeline（非宏操作返回 `PIPE_UNASSIGNED`） |
| `getOutPipe()` | `mlir::hivm::PIPE` | 获取宏操作的输出 Pipeline（非宏操作返回 `PIPE_UNASSIGNED`） |

## Pipeline 同步在 IR 中的表示

源文件：[HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td)

### set_flag

通知硬件某个 Pipeline 的操作已完成，设置事件标志。

```mlir
hivm.hir.set_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>, EVENT_ID0]
```

参数说明：
- `set_pipe`：发出通知的 Pipeline（数据生产方）
- `wait_pipe`：等待通知的 Pipeline（数据消费方）
- `static_event_id` / `dynamic_event_id`：事件 ID（0-7）

### wait_flag

等待某个 Pipeline 的事件标志，确保数据依赖满足。

```mlir
hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>, EVENT_ID0]
```

参数与 `set_flag` 一致。

### pipe_barrier

同一 Pipeline 内的屏障操作，确保该 Pipeline 之前的所有操作完成。

```mlir
hivm.hir.pipe_barrier [#hivm.pipe<PIPE_V>]
```

参数：
- `pipe`：需要执行屏障的 Pipeline

### 同步操作参数的 Pipe 含义

| 同步操作 | `set_pipe` 含义 | `wait_pipe` 含义 |
|---------|----------------|-----------------|
| `set_flag` | 完成数据生产的 Pipeline | 需要被通知的 Pipeline |
| `wait_flag` | 产生数据的 Pipeline | 等待数据的 Pipeline |

典型同步模式：

```
MTE2 加载数据 -> set_flag[MTE2, V, event0] -> V 使用数据前 -> wait_flag[MTE2, V, event0]
```

## 紧耦合缓冲区的 Pipeline 选择逻辑

紧耦合缓冲区是 950 架构特有的 CV 通信机制，其 Pipeline 选择取决于编译选项：

```
if (isAscend950(target)) {
    if (enableLayoutOptimization) {
        InsertCVDataMovement    // A5 新布局优化路径
    } else {
        InsertCVTightCoupledBuffer  // 传统紧耦合缓冲区路径
    }
} else {
    InsertLoadStoreForMixCV    // 非 950 设备的混合 CV 路径
}
```

在不同路径下，Cube 和 Vector 之间的数据传递使用不同的 Pipeline 组合：

| 路径 | 数据流 | 使用的 Pipeline |
|------|--------|----------------|
| InsertCVTightCoupledBuffer | L0C -> UB | PIPE_FIX (fixpipe) + PIPE_V (vector) |
| InsertCVDataMovement | L0C -> UB | PIPE_FIX (fixpipe) + PIPE_V (vector) |
| InsertLoadStoreForMixCV | L0C -> GM -> UB | PIPE_FIX (fixpipe) + PIPE_MTE2 (load) + PIPE_V (vector) |

## 执行顺序约束与同步要求

### Pipeline 间并行

不同 Pipeline 的操作可以并行执行。例如，MTE2 加载数据到 L1 的同时，V 可以在 UB 上执行向量计算。

### Pipeline 内顺序

同一 Pipeline 内的操作严格按程序顺序执行，无需额外同步。

### 跨 Pipeline 数据依赖

当数据生产者和消费者位于不同 Pipeline 时，必须插入同步操作：

| 场景 | 生产 Pipeline | 消费 Pipeline | 同步方式 |
|------|-------------|-------------|---------|
| GM 加载到 UB 后 V 计算 | PIPE_MTE2 | PIPE_V | set_flag/wait_flag |
| V 计算后写回 GM | PIPE_V | PIPE_MTE3 | set_flag/wait_flag |
| L1 加载到 L0A 后 M 计算 | PIPE_MTE1 | PIPE_M | set_flag/wait_flag |
| M 计算后 FIX 输出 | PIPE_M | PIPE_FIX | set_flag/wait_flag |
| FIX 输出到 UB 后 V 计算 | PIPE_FIX | PIPE_V | set_flag/wait_flag |

### Event ID 资源

源文件：[HIVMAttrs.td:479-498](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L479-L498)

硬件提供 8 个 Event ID（EVENT_ID0 ~ EVENT_ID7），用于区分不同的同步事件。编译器的 `SyncEventIdAllocation` Pass 负责分配和复用 Event ID。

### Unit Flag 模式

源文件：[HIVMAttrs.td:512-534](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L512-L534)

Unit Flag 是一种条件同步机制，用于循环中依赖首次迭代才能确定同步的场景。

| 模式 | C++ 符号 | 数值 | 说明 |
|------|---------|------|------|
| DISABLED | `UNIT_FLAG_DISABLED` | 0 | 禁用 Unit Flag |
| RESERVED | `UNIT_FLAG_RESERVED` | 1 | 保留 |
| ENABLED_WITHOUT_UPDATE | `ENABLED_WITHOUT_UPDATE` | 2 | 启用但不更新 |
| ENABLED_WITH_UPDATE | `ENABLED_WITH_UPDATE` | 3 | 启用并更新 |

### SyncBlock 模式

源文件：[HIVMAttrs.td:540-563](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L540-L563)

SyncBlock 用于不同 AI Core 之间的同步。

| 模式 | 说明 |
|------|------|
| ALL_CUBE | 所有 Cube 核心同步到同一点 |
| ALL_VECTOR | 所有 Vector 核心同步到同一点 |
| ALL_SUB_VECTOR | 所有子 Vector 核心同步到同一点 |
| BARRIER_CUBE | Cube-Cube 同步，降低为 barrier.pipe_all |
| BARRIER_VECTOR | Vector-Vector 同步，降低为 barrier.pipe_all |
| ALL | 所有 AIC/AIV 同步到同一点 |

## 常见问题

**Q: 为什么 `hivm.copy` 没有静态的 OpPipeTrait？**
A: `hivm.copy` 支持多种地址空间组合（UB->UB, UB->L1 等），其 Pipe 需要根据源和目标地址空间动态推断，因此通过 `getPipe()` 方法实现而非静态 Trait。

**Q: VIRTUAL_PIPE_MTE2_L1A 和 VIRTUAL_PIPE_MTE2_L1B 的用途是什么？**
A: 这两个虚拟 Pipeline 用于编译器内部更细粒度的同步控制。当 MTE2 同时向 L1 的不同区域（A 矩阵区域和 B 矩阵区域）搬运数据时，虚拟 Pipeline 允许编译器区分这两条数据流，实现更精确的同步。

**Q: 宏操作（如 mmadL1）的 InPipe 和 OutPipe 如何影响同步？**
A: 宏操作内部包含多个子操作（如 mmadL1 包含 MTE1 数据加载和 M 矩阵计算）。InPipe 表示宏操作的第一个子操作所属的 Pipeline，OutPipe 表示最后一个子操作所属的 Pipeline。同步分析使用这些信息确定宏操作与其他操作之间的依赖关系。

**Q: PIPE_MTE4 和 PIPE_MTE5 目前有操作使用吗？**
A: 从当前源码来看，PIPE_MTE4 和 PIPE_MTE5 已在枚举中定义，但尚未有 IR 操作显式使用。它们为未来硬件扩展预留。

## 相关文档

- 源码参考：[HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- 源码参考：[HIVMTraits.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td)
- 源码参考：[HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- 源码参考：[HIVMVectorOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td)
- 源码参考：[HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td)
- 源码参考：[HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td)
- 源码参考：[OpPipeInterface.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Interfaces/OpPipeInterface.td)
- 上一节：[02-memory-hierarchy.md](./02-memory-hierarchy.md) — 内存层次详解
- 下一节：[04-data-layout.md](./04-data-layout.md) — 数据布局详解
