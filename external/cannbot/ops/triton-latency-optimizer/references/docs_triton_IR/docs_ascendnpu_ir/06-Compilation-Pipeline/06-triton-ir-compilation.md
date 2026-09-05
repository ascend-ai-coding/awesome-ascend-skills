# Triton IR 编译路径

本文档详细描述 Triton IR 的编译路径：TTIR → TTGIR → LLVM IR，包括 Triton Passes、TritonGPU Passes 和 Conversion Passes。

源码参考：
- [Triton/Transforms/Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Triton/Transforms/Passes.td)
- [TritonGPU/Transforms/Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td)
- [TritonToTritonGPU/Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td)
- [TritonGPUToLLVM/Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Conversion/TritonGPUToLLVM/Passes.td)

## 1. 编译路径总览

```
┌────────────────────────────────────────────────────────────────────┐
│                      Triton Python Frontend                         │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Triton IR (TTIR)                                 │
│                    tt.load / tt.store / tt.dot / ...               │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │  Triton Transforms     │
                    │  (Combine, Reorder,    │
                    │   Rewrite, LICM, etc.) │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │  ConvertTritonTo       │
                    │  TritonGPU             │
                    └───────────┬───────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                   TritonGPU IR (TTGIR)                              │
│                   ttg.convert_layout / ttg.local_alloc / ...       │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │  TritonGPU Transforms  │
                    │  (Coalesce, Pipeline,  │
                    │   Accelerate, etc.)    │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │  TritonGPUToLLVM       │
                    │  Conversion             │
                    └───────────┬───────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                      LLVM IR                                       │
└────────────────────────────────────────────────────────────────────┘
```

## 2. Triton Transforms（TTIR 阶段）

### 2.1 TritonCombineOps

合并 Triton 操作，优化五种模式：

| 项目 | 内容 |
|------|------|
| CLI 参数 | `triton-combine` |
| 作用域 | `ModuleOp` |
| 依赖方言 | `arith::ArithDialect` |

优化模式：
1. `dot(a, b, 0) + c => dot(a, b, c)` — 将加法合并到 dot 的累加器
2. `addptr(addptr(ptr, idx0), idx1) => addptr(ptr, AddI(idx0, idx1))` — 合并连续指针偏移
3. `select(cond, load(ptrs, broadcast(cond), ???), other) => load(ptrs, broadcast(cond), other)` — 简化条件加载
4. `broadcast(constant) => reshaped_constant` — 常量广播简化
5. `sum(expand * expand) => dot` — 将扩展乘法求和转换为 dot

### 2.2 TritonReorderBroadcast

将 broadcast 和 splat 移到逐元素操作之后。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `triton-reorder-broadcast` |
| 作用域 | `ModuleOp` |
| 依赖方言 | `triton::TritonDialect` |

优化模式：
- `elementwise(broadcast(a)) => broadcast(elementwise(a))`
- `elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))`

### 2.3 TritonRewriteTensorPointer

将基于张量指针的 load/store 重写为传统 load/store。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `triton-rewrite-tensor-pointer` |
| 作用域 | `ModuleOp` |
| 依赖方言 | `triton::TritonDialect` |

重写后 `tt.make_tensor_ptr` 和 `tt.advance` 被消除，生成计算每个 load/store 的指针/掩码/other 的逻辑。

### 2.4 TritonRewriteTensorDescriptorToPointer

将基于张量描述符的 load/store 重写为指针 load/store。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `triton-rewrite-tensor-descriptor-to-pointer` |
| 作用域 | `ModuleOp` |
| 依赖方言 | `triton::TritonDialect` |

重写后 `tt.make_tensor_descriptor` 被消除。

### 2.5 TritonLoopUnroll

循环展开 Pass，根据 `tt.loop_unroll_factor` 属性展开循环。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `triton-loop-unroll` |
| 作用域 | `ModuleOp` |
| 依赖方言 | `triton::TritonDialect` |

### 2.6 TritonLoopInvariantCodeMotion

增强的 LICM Pass，在 MLIR 标准 LICM 基础上增加带掩码的 load 操作外提。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `triton-licm` |
| 作用域 | `ModuleOp` |
| 依赖方言 | `triton::TritonDialect` |

额外优化：
- 对于 `scf.for`：生成行程计数检查
- 对于 `scf.while`：从 before body 克隆条件

### 2.7 TritonLoopAwareCSE

循环体内的公共子表达式消除，可递归消除循环迭代参数和始终相同值的子计算。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `triton-loop-aware-cse` |
| 作用域 | `ModuleOp` |

## 3. ConvertTritonToTritonGPU

将 Triton IR 转换为 TritonGPU IR。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-triton-to-tritongpu` |
| 作用域 | `ModuleOp` |
| 依赖方言 | `arith::ArithDialect`, `math::MathDialect`, `scf::SCFDialect`, `triton::TritonDialect`, `triton::gpu::TritonGPUDialect` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target` | string | `""` | GPU 目标，如 `cuda:80`, `hip:gfx942` |
| `num-warps` | int32 | `4` | warp 数量 |
| `threads-per-warp` | int32 | `32` | 每 warp 线程数 |
| `num-ctas` | int32 | `1` | CGA 中 CTA 数量 |
| `enable-source-remat` | bool | `false` | 启用源重物化 |
| `shared-memory-size` | int32 | `122880` | SIMT 共享内存大小（昇腾条件编译） |

**转换语义**：
- 张量类型增强：`tensor<shape, type>` → `tensor<shape, type, #encoding>`
- 指针类型增强：`tt.ptr<tensor<...>>` → `tt.ptr<tensor<..., #encoding>>`
- 大部分 `tt.*` 操作直接映射，增加布局约束
- 新增 `ttg.convert_layout` 用于布局转换

### RelayoutTritonGPU

Warp 特化期间的重新布局 Pass。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `relayout-tritongpu` |
| 作用域 | `ModuleOp` |

## 4. TritonGPU Transforms（TTGIR 阶段）

### 4.1 布局优化 Pass

#### TritonGPUCoalesce

分析 load/store 操作，替换为合并访问布局（cache friendly）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-coalesce` |
| 作用域 | `ModuleOp` |
| 依赖方言 | `triton::gpu::TritonGPUDialect` |

#### TritonGPURemoveLayoutConversions

移除多余的布局转换，优先选择 `BlockedEncodingAttr`（利于合并访问）和 `NvidiaMmaEncodingAttr`（利于张量操作）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-remove-layout-conversions` |
| 作用域 | `ModuleOp` |

#### TritonGPUOptimizeDotOperands

重新排列矩阵乘法操作数的布局，促进硬件加速转置。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-optimize-dot-operands` |
| 作用域 | `ModuleOp` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hoist-layout-conversion` | bool | `true` | 是否将 convert_layout 提升到逐元素操作之前 |

#### TritonGPUReduceDataDuplication

减少寄存器中的数据重复，将 `convert[distributed -> dotOperand]` 分解为 `convert[distributed -> shared -> dotOperand]`。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-reduce-data-duplication` |
| 作用域 | `ModuleOp` |

#### TritonGPUCoalesceAsyncCopy

改善异步全局到本地拷贝的合并访问。当共享编码的 vec 小于 blocked 编码的 sizePerThread 时，裁剪 sizePerThread 值。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-coalesce-async-copy` |
| 作用域 | `ModuleOp` |

### 4.2 矩阵乘法优化 Pass

#### TritonGPUAccelerateMatmul

优化 dot 指令的输入/输出布局，使其兼容硬件加速器（如 NVIDIA Tensor Core）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-accelerate-matmul` |
| 作用域 | `ModuleOp` |

#### TritonGPUF32DotTC

3xTF32 技巧：将 FP32 dot 分解为 4 个逐元素操作和 3 个 FP16 dot，以使用 Tensor Core。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-F32DotTC` |
| 作用域 | `ModuleOp` |

#### TritonGPUOptimizeAccumulatorInit

将累加器零初始化替换为首次使用标志。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-optimize-accumulator-init` |
| 作用域 | `ModuleOp` |

### 4.3 流水线 Pass

#### TritonGPUPipeline

软件流水线化循环，将部分 load 转换为异步加载，多缓冲数据。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-pipeline` |
| 作用域 | `ModuleOp` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num-stages` | int32 | `3` | 流水线阶段数 |
| `dump-intermediate-steps` | bool | `false` | 转储中间步骤 |

#### TritonGPUAssignLatencies

为流水线化前的操作分配延迟。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-assign-latencies` |
| 作用域 | `ModuleOp` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num-stages` | int32 | `3` | 流水线阶段数 |

#### TritonGPUScheduleLoops

循环流水线调度。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-schedule-loops` |
| 作用域 | `ModuleOp` |

#### TritonGPUPrefetch

预取共享内存中 dot 操作数，在循环中将 dot 分解为更细粒度的操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-prefetch` |
| 作用域 | `ModuleOp` |

变换步骤：
1. 发出循环 prologue，预取第一次迭代的数据
2. 扩展循环参数，添加新的预取值
3. 更新 dot 参数
4. 添加下一次迭代的预取操作
5. 更新 yield，添加下一次迭代的预取值

### 4.4 Warp 特化 Pass

#### TritonGPUAutomaticWarpSpecialization

自动 warp 特化，分析循环并创建分区调度，将循环复制到 `ttg.warp_specialize` 分区区域。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-automatic-warp-specialization` |
| 作用域 | `ModuleOp` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num-stages` | int32 | `3` | 流水线阶段数 |

#### TritonGPURewritePartitionDependencies

重写分区间的 SSA 依赖，通过共享内存传递数据，根据分区阶段应用多缓冲。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-rewrite-partition-dependencies` |
| 作用域 | `ModuleOp` |

#### TritonGPUPartitionLoops

将已调度的循环分裂为 `ttg.warp_specialize` 分区区域。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-partition-loops` |
| 作用域 | `ModuleOp` |

#### TritonGPUOptimizePartitionWarps

优化分配给分区的 warp 数量，减少寄存器使用。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-optimize-partition-warps` |
| 作用域 | `ModuleOp` |

#### TritonGPUPartitionScheduling

分析循环中的 load/MMA/其他操作，确定每个操作的分区分配。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-partition-scheduling` |
| 作用域 | `ModuleOp` |

#### TritonGPULoadMMASpecialization

查找矩阵乘法循环，创建分区调度，将异步 load 和异步 MMA 分离到不同分区。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-load-mma-specialization` |
| 作用域 | `ModuleOp` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num-stages` | int32 | `3` | 流水线阶段数 |

#### TritonGPUHoistTMEMAlloc

将 TMEM 分配提升到循环外，尽可能保持 TMEM 中的值。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-hoist-tmem-alloc` |
| 作用域 | `ModuleOp` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hoist-out-of-if` | bool | `false` | 将 TMEM 分配提升出 if 语句 |

### 4.5 循环优化 Pass

#### TritonGPUFuseNestedLoops

融合嵌套循环以进行流水线化。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-fuse-nested-loops` |
| 作用域 | `ModuleOp` |

### 4.6 其他优化 Pass

#### TritonGPUOptimizeThreadLocality

减少 SM 内线程间的同步开销。优化归约和 gather 操作的布局。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-optimize-thread-locality` |
| 作用域 | `ModuleOp` |

#### TritonGPUReorderInstructions

重排指令以：(1) 降低寄存器压力 (2) 促进对 ptxas 友好的 LLVM 指令顺序。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-reorder-instructions` |
| 作用域 | `ModuleOp` |

昇腾条件编译选项：

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable-simt-reorder-instruction` | bool | `false` | 启用 SIMT 指令重排模式 |

#### TritonGPUCombineTensorSelectAndIf

合并 tensor select 和 if 操作，将 select 操作数合并到 then/else yield 中。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-combine-tensor-select-and-if` |
| 作用域 | `ModuleOp` |

## 5. TritonGPUToLLVM Conversion

### 5.1 AllocateSharedMemory

使用 ModuleAllocation 分析标注共享内存分配。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `allocate-shared-memory` |
| 作用域 | `ModuleOp` |

功能：
- 为模块添加属性，标注使用的共享/本地内存量
- 为操作添加偏移量属性，标注在总共享/本地内存中的偏移

### 5.2 TritonGPUGlobalScratchAllocationPass

分配全局 scratch 内存。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-global-scratch-memory-allocation` |
| 作用域 | `ModuleOp` |
| 依赖方言 | `triton::gpu::TritonGPUDialect` |

### 5.3 TritonGPUAllocateWarpGroups

为 GPU 程序分配 warp 组。分析 `ttg.warp_specialize` 操作，确定所需的总 warp 数，并将 warp ID 范围附加到每个 warpgroup 函数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `tritongpu-allocate-warp-groups` |
| 作用域 | `ModuleOp` |

## 6. 典型编译流程

### 6.1 标准 Triton Kernel 编译

```
1. TritonCombineOps
2. TritonReorderBroadcast
3. TritonRewriteTensorPointer
4. TritonLoopUnroll (可选)
5. TritonLoopInvariantCodeMotion
6. TritonLoopAwareCSE
7. ConvertTritonToTritonGPU
8. TritonGPUCoalesce
9. TritonGPURemoveLayoutConversions
10. TritonGPUAccelerateMatmul
11. TritonGPUOptimizeDotOperands
12. TritonGPUReduceDataDuplication
13. TritonGPUOptimizeThreadLocality
14. TritonGPUReorderInstructions
15. TritonGPUPipeline
16. AllocateSharedMemory
17. TritonGPUGlobalScratchAllocationPass
18. TritonGPUAllocateWarpGroups
19. ConvertTritonGPUToLLVM
```

### 6.2 昇腾适配编译

在昇腾适配场景下，编译路径可能通过 BishengIR：

```
1. TritonCombineOps
2. TritonReorderBroadcast
3. ConvertTritonToTritonGPU (shared-memory-size=122880)
4. TritonGPUCoalesce
5. TritonGPUAccelerateMatmul (使用 FractalSharedEncodingAttr)
6. TritonGPUReorderInstructions (enable-simt-reorder-instruction)
7. ConvertTritonAscendGPUToLLVM
```

或通过 HIVM 路径：

```
1. 前端 → HFusion → HIVM
2. ConvertHIVMToTritonGPU (SIMD 路径)
3. TritonGPU Passes
4. ConvertTritonAscendGPUToLLVM
```
