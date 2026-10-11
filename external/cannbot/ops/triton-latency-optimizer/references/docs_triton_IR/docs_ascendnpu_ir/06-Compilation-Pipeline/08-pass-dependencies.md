# Pass 依赖关系与执行顺序

本文档描述 AscendNPU-IR 编译流水线中各 Pass 的依赖关系和推荐执行顺序。

## 1. HFusion Pipeline 执行顺序

### 1.1 标准 HFusion Pipeline

`buildHFusionPipelines` 的典型执行顺序：

```
阶段 1：前端转换
├── ConvertLinalgToHFusion
├── ConvertArithToHFusion
├── ConvertMathToHFusion
├── ConvertTensorToHFusion
└── ConvertGPUToHFusion

阶段 2：规范化
├── HFusionNormalizeOps (enable-high-precision)
├── LegalizeBF16Pass
├── LegalizeFP8Pass
├── LegalizeBoolPass
├── LegalizeScalarPass
├── ConvertGenericToNamedOp
└── HFusionFoldUnitDims (enable-drop-unit-dims)

阶段 3：融合与调度
├── HFusionOpFusion (multi-kernel, enable-symbol-analysis)
├── ComposeMultiReduce
├── DecomposeMulti
├── HFusionInlineBrc
├── InferFuncFusionKind
└── AutoSchedule (block-dim, enable-deterministic-computing)

阶段 4：向量化
├── AutoVectorizeV2 (enable-auto-vectorize-v2)
│   或 AutoVectorize (disable-hfusion-vectorize)
├── VectorizeOps
├── PreVectorizationFusion (enable-triton-compile)
└── GenericUnroller

阶段 5：展平与优化
├── FlattenOps (enable-flatten)
├── HFusionDecompose
├── ReorderOpsByBFS (enable-ops-reorder)
└── SimplifyOps

阶段 6：函数管理
├── OutlineVectorFunction
├── OutlineSingleOp
├── PullSliceIntoVectorFunction
├── MergeVecScope (enable-vf-merge-level)
├── SimplifyVFArgs
├── EliminateDuplicateFuncs
└── WrapHostFunc (enable-manage-host-resources)

阶段 7：内存与形状
├── CacheIO / ReCacheIO
├── HoistTensorEmpty
├── TensorResToOutParams
├── PackTilingData
├── ConstantizeTilingData
├── InferOutShapesPass
├── DropSymbols / FoldSymbolicDim / UnfoldSymbolicDim
└── AddFFTSAddr

阶段 8：Triton 适配
└── AdaptTritonKernel (enable-triton-kernel-compile)
```

### 1.2 寄存器模式 HFusion Pipeline

`buildHFusionRegBasePipeline` 的执行顺序与标准 Pipeline 类似，但：
- 使用 `register-based` 模式的 FlattenOps
- 跳过 AutoSchedule 和 AutoVectorize
- 直接使用 ConvertHFusionToVector

## 2. HIVM Pipeline 执行顺序

### 2.1 ConvertToHIVM Pipeline

`buildConvertToHIVMPipeline` 的执行顺序：

```
阶段 1：前端到 HFusion（同上）

阶段 2：HFusion 到 HIVM
├── ConvertHFusionToHIVM (mm-map-mode)
└── ConvertToHIVMOp
```

### 2.2 HIVM Tensor 优化

`buildHIVMTensorOptimizations` 的执行顺序：

```
阶段 1：类型推断
├── InferFuncCoreType
├── InferHIVMMemScope
├── InferVFMode
└── InferHIVMDataLayout

阶段 2：规范化
├── NormalizeMatmul
├── Normalize
├── HIVMDecomposeOp
├── HIVMAggregatedDecomposeOp (decompose-phase)
├── HIVMRecognizeDeinterleaveOp
├── InlineOTFBroadcast
└── InlineFixpipe / InlineFixpipeV2

阶段 3：布局转换
├── InsertConvertLayout
├── PropagateConvertLayout (allow-agnostic-ops)
├── ConvertLayoutToTranspose
└── CombineOptimizedConvertLayout

阶段 4：Stride 对齐
├── MarkStrideAlign (enable-auto-storage-align)
├── EnableStrideAlign
├── LiftLowestStride
└── ReduceRankSubview

阶段 5：展平与分解
├── HIVMFlattenOps
├── HIVMAggregatedDecomposeOp (after-hivm-flatten-ops)
├── LiftZeroRank
└── ComposeCollapseExpand

阶段 6：缓冲与内存
├── CloneTensorEmpty
├── AllocExtraBuffer
├── AutoInferBufferSize
├── MarkMultiBuffer (enable-auto-multi-buffer)
├── EnableMultiBuffer
├── CloneSCFIfYieldOperand
├── NormalizeLoopIterator
└── PlanMemory (mem-plan-mode)
```

### 2.3 LowerHIVM Pipeline

`buildLowerHIVMPipelines` 的执行顺序（包含 ConvertToHIVM + TensorOptimizations + Lowering）：

```
阶段 1-6：同 ConvertToHIVM + TensorOptimizations

阶段 7：同步注入
├── InjectSync (sync-mode, enable-unit-flag)
├── InjectBlockSync (block-all-sync)
├── GraphSyncSolver (enable-hivm-graph-sync-solver)
└── CrossCoreGSS

阶段 8：Mix Kernel 处理
├── SplitMixKernel (enable-mixed-cv)
├── MarkRealCoreType
├── SplitMixedIfConditionals
├── InsertLoadStoreForMixCV
├── InsertCVTightCoupledBuffer
├── InsertWorkSpaceForMixCV
├── InsertCVDataMovement
└── CVPipelining (pipeline-depth)

阶段 9：后端转换
├── [SIMD 路径]
│   ├── ConvertHIVMToTritonGPU
│   ├── TritonGPU Passes
│   └── ConvertTritonAscendGPUToLLVM
├── [SIMT 路径]
│   ├── ArithToHIVMLLVMConversionPass
│   ├── GPUToDPX (use-dpx)
│   └── FuncToLLVM
└── [公共]
    ├── LowerMesh
    ├── LowerMemRefExt
    ├── StripMemRefAddressSpace
    └── InsertMemSemanticForSimtVF
```

## 3. Triton IR 编译路径执行顺序

### 3.1 标准 Triton 编译

```
阶段 1：Triton IR 优化
├── TritonCombineOps
├── TritonReorderBroadcast
├── TritonRewriteTensorPointer
├── TritonRewriteTensorDescriptorToPointer
├── TritonLoopUnroll (可选)
├── TritonLoopInvariantCodeMotion
└── TritonLoopAwareCSE

阶段 2：Triton → TritonGPU
└── ConvertTritonToTritonGPU (target, num-warps, threads-per-warp)

阶段 3：TritonGPU 优化
├── TritonGPUCoalesce
├── TritonGPURemoveLayoutConversions
├── TritonGPUAccelerateMatmul
├── TritonGPUOptimizeDotOperands
├── TritonGPUReduceDataDuplication
├── TritonGPUOptimizeThreadLocality
├── TritonGPUReorderInstructions
├── TritonGPUCoalesceAsyncCopy
├── TritonGPUCombineTensorSelectAndIf
├── TritonGPUOptimizeAccumulatorInit
└── TritonGPUF32DotTC (可选)

阶段 4：流水线化
├── TritonGPUAssignLatencies (num-stages)
├── TritonGPUScheduleLoops
├── TritonGPUPipeline (num-stages)
├── TritonGPUPrefetch (可选)
└── TritonGPUFuseNestedLoops (可选)

阶段 5：Warp 特化（可选）
├── TritonGPUPartitionScheduling
├── TritonGPULoadMMASpecialization
├── TritonGPUAutomaticWarpSpecialization
├── TritonGPURewritePartitionDependencies
├── TritonGPUPartitionLoops
├── TritonGPUOptimizePartitionWarps
├── TritonGPUHoistTMEMAlloc
└── RelayoutTritonGPU

阶段 6：内存分配
├── AllocateSharedMemory
├── TritonGPUGlobalScratchAllocationPass
└── TritonGPUAllocateWarpGroups

阶段 7：LLVM 转换
└── ConvertTritonGPUToLLVM / ConvertTritonAscendGPUToLLVM
```

## 4. Pass 依赖关系图

### 4.1 HFusion Pass 依赖

```
ConvertGenericToNamedOp → HFusionOpFusion → AutoSchedule → AutoVectorize
                                ↓
                        ComposeMultiReduce
                                ↓
                        DecomposeMulti
                                ↓
                        HFusionInlineBrc
                                ↓
                        FlattenOps → HFusionDecompose
                                ↓
                        OutlineVectorFunction → MergeVecScope
                                ↓
                        PackTilingData → ConstantizeTilingData
```

### 4.2 HIVM Pass 依赖

```
InferFuncCoreType → InferHIVMMemScope → InferVFMode → InferHIVMDataLayout
                                                              ↓
                    Normalize → HIVMDecomposeOp → HIVMFlattenOps
                                                              ↓
                    InsertConvertLayout → PropagateConvertLayout
                                                              ↓
                    MarkStrideAlign → EnableStrideAlign → LiftLowestStride
                                                              ↓
                    MarkMultiBuffer → EnableMultiBuffer → PlanMemory
                                                              ↓
                    InjectSync → InjectBlockSync → CrossCoreGSS
                                                              ↓
                    SplitMixKernel → CVPipelining
```

### 4.3 TritonGPU Pass 依赖

```
ConvertTritonToTritonGPU
        ↓
TritonGPUCoalesce → TritonGPURemoveLayoutConversions
        ↓
TritonGPUAccelerateMatmul → TritonGPUOptimizeDotOperands
        ↓
TritonGPUAssignLatencies → TritonGPUScheduleLoops → TritonGPUPipeline
        ↓
TritonGPUAutomaticWarpSpecialization → TritonGPUPartitionLoops
        ↓
AllocateSharedMemory → TritonGPUAllocateWarpGroups
        ↓
ConvertTritonGPUToLLVM
```

## 5. 关键约束

### 5.1 必须顺序执行的 Pass 对

| 前置 Pass | 后续 Pass | 原因 |
|-----------|-----------|------|
| `InferFuncCoreType` | `InferHIVMMemScope` | 需要知道 Core 类型才能推断内存作用域 |
| `InferHIVMDataLayout` | `InsertConvertLayout` | 需要数据布局信息才能插入布局转换 |
| `MarkStrideAlign` | `EnableStrideAlign` | 先标记后启用 |
| `MarkMultiBuffer` | `EnableMultiBuffer` | 先标记后启用 |
| `PlanMemory` | `InjectSync` | 需要内存规划完成才能注入同步 |
| `SplitMixKernel` | `CVPipelining` | 需要分裂后才能流水线化 |
| `ConvertTritonToTritonGPU` | `TritonGPUCoalesce` | 需要布局编码才能合并访问 |
| `TritonGPUAccelerateMatmul` | `TritonGPUPipeline` | 需要优化 dot 布局后才能流水线化 |
| `AllocateSharedMemory` | `ConvertTritonGPUToLLVM` | 需要 shared memory 偏移才能 LLVM 转换 |

### 5.2 可选 Pass 的条件启用

| Pass | 启用条件 | 对应选项 |
|------|----------|----------|
| `SplitMixKernel` | `enable-mixed-cv=true` | HIVMPipelineOptions |
| `CVPipelining` | `enable-mixed-cv=true` | HIVMPipelineOptions |
| `GraphSyncSolver` | `enable-hivm-graph-sync-solver=true` | HIVMPipelineOptions |
| `ConvertHIVMToTritonGPU` | `pure-simt=false` | HIVMPipelineOptions |
| `GPUToDPX` | `use-dpx=true` | HIVMPipelineOptions |
| `TritonGPUF32DotTC` | FP32 dot + Tensor Core | 自动检测 |
| `TritonGPUAutomaticWarpSpecialization` | Hopper+ 架构 | 自动检测 |
