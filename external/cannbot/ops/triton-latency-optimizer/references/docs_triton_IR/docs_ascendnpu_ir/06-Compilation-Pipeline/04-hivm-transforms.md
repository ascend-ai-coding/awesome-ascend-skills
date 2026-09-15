# HIVM 变换 Pass 详解

本文档列出 HIVM 方言的所有变换 Pass，包括名称、选项和功能简述。所有信息从 TableGen 源码精确提取。

源码参考：[HIVM/Transforms/Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td)

## 1. 类型推断与规范化 Pass

### 1.1 InferFuncCoreType

推断每个函数的 Core 类型（Cube/Vector/Mix）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-infer-func-core-type` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createInferFuncCoreTypePass()` |

### 1.2 ConvertToHIVMOp

将其他方言操作转换为 HIVM 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-to-hivm-op` |
| 构造函数 | `mlir::hivm::createConvertToHIVMOpPass()` |
| 依赖方言 | `arith::ArithDialect`, `hivm::HIVMDialect`, `tensor::TensorDialect` |

### 1.3 NormalizeMatmul

规范化 HIVM 矩阵乘法操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-normalize-matmul` |
| 构造函数 | `mlir::hivm::createNormalizeMatmulPass()` |
| 依赖方言 | `arith::ArithDialect`, `hivm::HIVMDialect`, `tensor::TensorDialect` |

### 1.4 Normalize

规范化 HIVM 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-normalize-ops` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createHIVMNormalizeOpsPass()` |
| 依赖方言 | `hivm::HIVMDialect`, `tensor::TensorDialect` |

### 1.5 TritonGlobalKernelArgsToHIVMOp

将 Triton 全局 kernel 参数转换为 HIVM 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `triton-global-kernel-args-to-hivm-op` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createTritonGlobalKernelArgsToHIVMOpPass()` |
| 依赖方言 | `hivm::HIVMDialect`, `hacc::HACCDialect`, `annotation::AnnotationDialect` |

## 2. 内存作用域与布局推断 Pass

### 2.1 InferHIVMMemScope

推断 HIVM 操作的内存作用域。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-infer-mem-scope` |
| 构造函数 | `mlir::hivm::createInferHIVMMemScopePass()` |
| 依赖方言 | `hivm::HIVMDialect`, `memref::MemRefDialect` |

### 2.2 InferHIVMDataLayout

推断 HIVM 操作的数据布局（ND/Fractal 等）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-infer-data-layout` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInferHIVMDataLayoutPass()` |
| 依赖方言 | `affine::AffineDialect`, `hivm::HIVMDialect` |

### 2.3 InferVFMode

推断操作的 VF 模式（SIMD/SIMT/MIX）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-infer-vf-mode` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInferVFModePass()` |
| 依赖方言 | `hivm::HIVMDialect` |

## 3. 多缓冲 Pass

### 3.1 MarkMultiBuffer

标记需要多缓冲的 HIVM 操作。L0C 作用域的缓冲不会被标记。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-mark-multi-buffer` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createMarkMultiBufferPass()` |
| 依赖方言 | `hivm::HIVMDialect`, `annotation::AnnotationDialect` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable-auto` | bool | `false` | 自动标记多缓冲 |
| `limit-auto-multi-buffer-only-for-local-buffer` | bool | `false` | 禁止 workspace 的多缓冲标记 |
| `limit-auto-multi-buffer-of-local-buffer` | MultiBufferStrategy | `no-l0c` | 限制 local buffer 自动多缓冲策略：`no-limit`/`no-l0c` |
| `limit-mix-auto-multi-buffer-buffer` | MultiBufferStrategy | `only-cube` | 限制 Mix 多缓冲策略：`only-cube`/`only-vector`/`no-limit` |
| `set-workspace-multibuffer` | unsigned | `2` | workspace 多缓冲数量覆盖 |

### 3.2 EnableMultiBuffer

启用标记为 `hivm.multi_buffer` 的操作的多缓冲。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-enable-multi-buffer` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createEnableMultiBufferPass()` |
| 依赖方言 | `affine::AffineDialect` |

## 4. 内存规划 Pass

### 4.1 PlanMemory

为 HIVM 操作规划内存分配。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-plan-memory` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createPlanMemoryPass()` |
| 依赖方言 | `hivm::HIVMDialect` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mem-plan-mode` | MemPlanMode | `local-mem-plan` | 内存规划模式：`local-mem-plan`（memref.alloc）/ `global-work-space-plan`（memref_ext.alloc_workspace） |
| `enable-global-workspace-reuse` | bool | `false` | 启用全局 workspace 重用 |
| `enable-print-memory-allocated-size` | bool | `false` | 打印内存分配大小 |
| `restrict-inplace-as-isa` | bool | `false` | 限制内存就地操作为 ISA |
| `simt-vf-dynamic-size` | int | `216` | SIMT VF 动态 UB 大小（KB） |
| `disable-tightly-coupled-buffer-reuse` | bool | `false` | 禁用紧耦合缓冲重用 |

### 4.2 CloneTensorEmpty

为不同 HIVM 操作输出克隆 tensor.empty。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-clone-tensor-empty` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createCloneTensorEmptyPass()` |

### 4.3 CloneSCFIfYieldOperand

克隆 scf.if.yield 操作数以避免 PlanMemory 就地操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-clone-scf-if-yield-operand` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createCloneSCFIfYieldOperandPass()` |

### 4.4 AllocExtraBuffer

为需要额外临时缓冲的操作分配缓冲。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-alloc-extra-buffer` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createAllocExtraBufferPass()` |

### 4.5 ConstantizeBufferSize

尝试将动态形状缓冲常量化，通过上界化原始形状。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-constantize-buffer-size` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createConstantizeBufferSizePass()` |

### 4.6 SetBufferSize

设置缓冲大小。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-set-buffer-size` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createSetBufferSizePass()` |

### 4.7 AutoInferBufferSize

通过插入 annotation.mark 自动推断缓冲大小。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-auto-infer-buffer-size` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createAutoInferBufferSizePass()` |

### 4.8 AlignAllocSize

自动对齐 memref.alloc 大小以适配硬件对齐要求。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-align-alloc-size` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createAlignAllocSizePass()` |

### 4.9 HIVMOptFuncOutput

优化缓冲化后的函数输出，移除不必要的地址返回。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-opt-func-output` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createHIVMOptFuncOutputPass()` |

## 5. 同步 Pass

### 5.1 InjectSync

自动注入同步操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-inject-sync` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInjectSyncPass()` |
| 依赖方言 | `affine::AffineDialect`, `hivm::HIVMDialect` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sync-mode` | SyncMode | `normal` | 同步模式：`normal`（正常）/ `barrier-all`（全屏障，仅调试） |
| `enable-unit-flag` | bool | `false` | 启用 unit-flag 同步模式 |
| `assume-alive-loops` | bool | `false` | 假设所有循环至少执行一次 |

### 5.2 InjectBlockSync

自动注入块级同步操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-inject-block-sync` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInjectBlockSyncPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `block-all-sync` | bool | `false` | 启用全块同步 |
| `assume-alive-loops` | bool | `false` | 假设所有循环至少执行一次 |

### 5.3 GraphSyncSolver

基于图的同步求解器。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-graph-sync-solver` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createGraphSyncSolverPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable-unit-flag` | bool | `false` | 启用 unit-flag 同步模式 |
| `enable-tester-mode` | bool | `false` | 启用同步测试模式 |
| `sync-tester-options` | int64 list | `[]` | 同步测试选项（num_runs, init_seed, num_ops, num_ptrs, enable-multibuffer） |

### 5.4 CrossCoreGSS

跨 Core 图同步求解器。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-cross-core-gss` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createCrossCoreGSSPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `block-all-sync` | bool | `false` | 启用全块同步 |
| `disable-auto-inject-block-sync` | bool | `false` | 禁用自动 set/wait 插入 |
| `always-use-pipe-s` | bool | `false` | 始终使用标量管道作为等待管道 |
| `use-different-multibuffer-flag-ids` | bool | `false` | 为多缓冲后向同步对使用不同 flag-id |
| `force-is-mem-based` | bool | `false` | 强制基于内存的 AI Core 架构模式 |
| `force-is-reg-based` | bool | `false` | 强制基于寄存器的 AI Core 架构模式 |

### 5.5 SyncBlockHoisting

将同步块锁/解锁操作提升到父区域（如果在 scf.for/while 中）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-sync-block-hoisting` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createSyncBlockHoistingPass()` |

### 5.6 LowerCreateSyncBlockLock

将 CreateSyncBlockLockOp 降低为 ViewOp。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-lower-create-sync-block-lock` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createSyncBlockLockLoweringPass()` |

### 5.7 InsertInferSyncBlockLockNumAndInitFunc

插入推断同步块锁数量和初始化的 host 回调函数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-insert-infer-sync-block-lock-num-and-init-func` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInsertInferSyncBlockLockNumAndInitFuncPass()` |

## 6. 算子分解与变换 Pass

### 6.1 HIVMDecomposeOp

根据硬件能力分解复合 HIVM 操作。例如 f32→i8 需分解为 f32→f16→i8。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-decompose-op` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createHIVMDecomposeOpPass()` |

### 6.2 HIVMAggregatedDecomposeOp

分解使用 AggregatedOpInterface 的 HIVM 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-aggregated-decompose-op` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createHIVMAggregatedDecomposeOpPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `decompose-phase` | DecomposePhase | `no-constraint` | 分解阶段：`no-constraint`/`before-hivm-align`/`after-hivm-recognize-deinterleave`/`after-hivm-recognize-broadcast`/`after-hivm-align`/`after-infer-hivm-data-layout`/`after-hivm-flatten-ops` |

### 6.3 HIVMLowerToLoops

将 HIVM 操作降低为循环。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-lower-to-loops` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createHIVMLowerToLoopsPass()` |

### 6.4 HIVMRecognizeDeinterleaveOp

优化非连续内存访问，使用 deinterleave。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-recognize-deinterleave-op` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createHIVMRecognizeDeinterleaveOpPass()` |

### 6.5 HIVMOptSinglePointOp

使用标量操作优化单点 HIVM 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-opt-single-point` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createHIVMOptSinglePointPass()` |

### 6.6 HIVMFlattenOps

展平 HIVM 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-flatten-ops` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createFlattenOpsPass()` |

### 6.7 ComposeCollapseExpand

组合 collapse 和 expand 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `compose-collapse-expand` |
| 构造函数 | `mlir::hivm::createComposeCollapseExpandPass()` |

## 7. Stride 对齐 Pass

### 7.1 MarkStrideAlign

为 HIVM 操作的 memref 操作数自动标注 stride_align 标记。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-mark-stride-align` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createMarkStrideAlignPass()` |

### 7.2 EnableStrideAlign

根据 stride_align 标记重新分配 memref。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-enable-stride-align` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createEnableStrideAlignPass()` |

### 7.3 LiftLowestStride

提升 HIVM 操作操作数的最低步幅。对于大多数结构化操作，如果最后维度不连续则提升。例外：MacroOp 和 VArangeOp。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-lift-lowest-stride` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createLiftLowestStridePass()` |

例如：`memref<16xf16, strided<[8]>>` → `memref<16x1xf32, strided<[8, 1]>>`

### 7.4 ReduceRankSubview

使用 subview 降低秩。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-reduce-rank-subview` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createReduceRankSubviewPass()` |

## 8. Mix Kernel Pass

### 8.1 SplitMixKernel

将 Mix 设备函数分裂为 AICube 和 AIVector 函数，并标记父模块为 Mix 模块。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-split-mix-kernel` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createSplitMixKernelPass()` |

### 8.2 MarkRealCoreType

用 core-type 属性标记标量操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-mark-real-core-type` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createMarkRealCoreTypePass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `remove-core-type-attrs` | bool | `false` | 移除所有 core type 属性（变为清理 Pass） |

### 8.3 SplitMixedIfConditionals

为 Mix CV 分裂 if 条件。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-split-mixed-if-conditionals` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createSplitMixedIfConditionalsPass()` |

### 8.4 InsertLoadStoreForMixCV

为 Mix CV 插入 load/store 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-insert-load-store-for-mix-cv` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInsertLoadStoreForMixCVPass()` |

### 8.5 InsertCVTightCoupledBuffer

为 Mix CV 插入紧耦合缓冲。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-insert-cv-tight-coupled-buffer` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInsertCVTightCoupledBufferPass()` |

### 8.6 InsertWorkSpaceForMixCV

为 Mix CV 插入 workspace。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `insert-workspace-for-mix-cv` |
| 构造函数 | `mlir::hivm::createInsertWorkSpaceForMixCVPass()` |

### 8.7 InsertCVDataMovement

Cube-Vector 数据搬运插入。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-insert-cv-data-movement` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInsertCVDataMovementPass()` |

### 8.8 CVPipelining

Cube 和 Vector 核心流水线化，用于多缓冲的 Mix CV 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `cv-pipelining` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createCVPipeliningPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable-auto-balance` | bool | `false` | 已弃用 |
| `pipeline-depth` | int | `-1` | 指定流水线深度，-1 为自动 |

## 9. 布局转换 Pass

### 9.1 InsertConvertLayout

为矩阵乘法操作插入布局转换。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-insert-convert-layout` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInsertConvertLayoutPass()` |

### 9.2 PropagateConvertLayout

传播矩阵乘法的布局转换。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-propagate-convert-layout` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createPropagateConvertLayoutPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `allow-agnostic-ops` | bool | `false` | 允许布局无关操作传播 reshape |

### 9.3 ConvertLayoutToTranspose

将布局转换分解为转置操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-convert-layout-to-transpose` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createConvertLayoutToTransposePass()` |

### 9.4 CombineOptimizedConvertLayout

折叠布局转换模式并优化。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-combine-optimized-convert-layout` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createCombineOptimizedConvertLayoutPass()` |

## 10. 外联与函数管理 Pass

### 10.1 OutlineAllocInVF

外联 VF 中静态形状的 memref.alloc。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-outline-alloc-in-VF` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createOutlineAllocInVFPass()` |

### 10.2 OutlineCopyInVF

当操作数为 VF 参数时，将 hivm.load 重写为 hivm.copy。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-outline-copy-in-VF` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createOutlineCopyInVFPass()` |

### 10.3 InsertInferWorkSpaceSizeFunc

插入推断 workspace 大小的 host 回调函数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-insert-infer-workspace-size-func` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInsertInferWorkSpaceSizeFuncPass()` |

### 10.4 InsertInferVFModeFunc

插入推断 VF 模式（SIMD/SIMT/MIX）的 host 回调函数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-insert-vf-mode-func` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInsertInferVFModeFuncPass()` |

### 10.5 BindWorkSpaceArg

将 hacc.workspace 函数参数绑定到 AllocWorkspaceOp。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-bind-workspace-arg` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createBindWorkSpaceArgPass()` |

### 10.6 BindSyncBlockLockArg

将 hacc.syncblocklock 函数参数绑定到 CreateSyncBlockLockOp。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-bind-sync-block-lock-arg` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createBindSyncBlockLockArgPass()` |

### 10.7 SplitSimtModule

为每个 SIMT VF 分裂 SIMT 模块。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `split-simt-module` |
| 构造函数 | `mlir::hivm::createSplitSimtModulePass()` |

## 11. 其他变换 Pass

### 11.1 InlineOTFBroadcast

内联 OTF 广播。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-inline-otf-broadcast` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInlineOTFBroadcastPass()` |

### 11.2 InitEntryKernel

在入口 kernel 开头插入 set_mask_norm()。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-init-entry-kernel` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInitEntryKernelPass()` |

### 11.3 InlineFixpipe / InlineFixpipeV2

将操作转换为 HIVM Fixpipe 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-inline-fixpipe` / `hivm-inline-fixpipe-v2` |
| 构造函数 | `mlir::hivm::createInlineFixpipePass()` / `mlir::hivm::createInlineFixpipeV2Pass()` |

### 11.4 TileBatchMMIntoLoop

将批量矩阵乘法 tiling 为循环，在批次维度上迭代。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-tile-batchmm-into-loop` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createTileBatchMMIntoLoopPass()` |

### 11.5 LiftZeroRank

提升零秩操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-lift-zero-rank` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createLiftZeroRankPass()` |

### 11.6 InsertLoadStoreForScalar

为标量操作插入 load/store。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-insert-load-store-for-scalar` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createInsertLoadStoreForScalarPass()` |

### 11.7 HIVMMapForallToBlocks

将 scf.forall 映射到 HIVM 块操作。一一映射，归纳变量重写为 hivm block idx。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-map-forall-to-blocks` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createHIVMMapForallToBlocksPass()` |

### 11.8 SinkOpToConsumerInLoop

将操作下沉到循环内的消费者。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-sink-op-to-consumer-in-loop` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createSinkOpToConsumerInLoopPass()` |

### 11.9 NormalizeLoopIterator

在 PlanMemory 前规范化循环迭代器的特殊状态。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-normalize-loop-iterator` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createNormalizeLoopIteratorPass()` |

### 11.10 HIVMInlineOTFLoadStore

即时内联 Load 和 Store 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-inline-otf-load-store` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createHIVMInlineOTFLoadStorePass()` |

### 11.11 AnnotateVFAlias

在 VF 内标注别名信息。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-annotate-vf-alias` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createAnnotateVFAliasPass()` |

### 11.12 RemoveCopyOps

移除冗余拷贝操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-remove-copy-ops` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createRemoveCopyOpsPass()` |

### 11.13 AnalyzeArithVectorMask

分析 arith/vector 掩码以消除 select 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `arith-vector-mask-analyze` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createArithVectorMaskAnalysisPass()` |

### 11.14 TileAndBindSubBlock

Tile 并绑定子块。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-bind-sub-block` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createTileAndBindSubBlockPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strict-mode` | bool | `true` | 严格模式运行 bubble up extract slice |
| `enable-tile` | bool | `true` | 启用 TileAndBindSubBlock pass |

### 11.15 HIVMBubbleUpExtractSlice

将 extract slice 冒泡提升。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-bubble-up-extract-slice` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createHIVMBubbleUpExtractSlicePass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strict-mode` | bool | `true` | 严格模式运行 |

### 11.16 HIVMVectorizeOps

向量化 HIVM 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-vectorize-ops` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createHIVMVectorizeOpsPass()` |

### 11.17 MarkDisableLoad

标记需要禁用 dcache 的 memref.load。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-mark-disable-load` |
| 构造函数 | `mlir::hivm::createMarkDisableLoadPass()` |

### 11.18 InsertNZ2NDForDebug

为调试插入 NZ→ND 转换。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-insert-nz2nd-for-debug` |
| 构造函数 | `mlir::hivm::createInsertNZ2NDForDebugPass()` |

### 11.19 InsertL12UBForDebug

为调试插入 L1→2UB 转换。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-insert-l12ub-for-debug` |
| 构造函数 | `mlir::hivm::createInsertL12UBForDebugPass()` |

### 11.20 AutoBlockifyParallelLoop

当逻辑块数大于物理块数时，自动在块上循环。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `auto-blockify-parallel-loop` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hivm::createAutoBlockifyParallelLoopPass()` |

### 11.21 InferSimtVFMemEffect

推断 SIMT VF 函数参数的内存效果。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `infer-simt-vf-memory-effect` |
| 构造函数 | `mlir::hivm::createInferSimtVFMemEffectPass()` |

### 11.22 StripMemRefAddressSpace

从当前模块的 memref 类型中移除内存空间。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-strip-memref-address-space` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createStripMemRefAddressSpacePass()` |

### 11.23 InsertMemSemanticForSimtVF

为 SIMT VF 插入内存语义。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `insert-memory-semantic-for-simtvf` |
| 构造函数 | `mlir::hivm::createInsertMemSemanticForSimtVFPass()` |

### 11.24 AutoScope

为 gather_load 和 scatter_store 创建 scope。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `auto-scope` |
| 构造函数 | `mlir::hivm::createAutoScopePass()` |

### 11.25 InsertAllocBasePlaceholder

插入具有虚拟大小的 memref.alloc 占位符。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `insert-alloc-base-placeholder` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createInsertAllocBasePlaceholderPass()` |

### 11.26 WriteBackShared

设置 SIMT VF 中共享内存 memref.alloc 的实际大小。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `write-back-shared` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createWriteBackSharedPass()` |
