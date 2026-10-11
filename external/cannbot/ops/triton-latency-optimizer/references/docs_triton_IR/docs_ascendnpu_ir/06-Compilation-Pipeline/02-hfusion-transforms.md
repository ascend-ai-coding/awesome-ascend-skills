# HFusion 变换 Pass 详解

本文档列出 HFusion 方言的所有变换 Pass，包括名称、选项和功能简述。所有信息从 TableGen 源码精确提取。

源码参考：[HFusion/Transforms/Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.td)

## 1. 融合与调度 Pass

### 1.1 HFusionOpFusion

算子融合 Pass，将多个算子融合为一个 kernel。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-fuse-ops` |
| 作用域 | `ModuleOp` |
| 构造函数 | `hfusion::createHFusionOpFusionPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output-mode` | `OutputMode` | `Multiple` | 外联函数输出模式：`multi`/`single`/`single-aggr` |
| `fusion-mode` | `FusionKind` | `Unknown` | 融合类型，由标签决定 |
| `always-inline` | bool | `false` | 启用外联函数始终内联 |
| `move-out-to-param` | bool | `true` | 是否将 tensor 输出移到参数 |
| `max-horizontal-fusion-size` | int | `-1` | 最大水平融合大小，-1 为无限制 |
| `multi-kernel` | bool | `false` | 启用多 kernel 融合 |
| `enable-symbol-analysis` | bool | `false` | 启用符号分析 |

### 1.2 AutoSchedule

自动调度融合后的 kernel。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-auto-schedule` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createHFusionAutoSchedulePass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `block-dim` | unsigned | `1` | 使用的块数 |
| `enable-auto-multi-buffer` | bool | `false` | 启用自动多缓冲 |
| `enable-deterministic-computing` | bool | `true` | 启用确定性计算 |
| `max-buffer-count-tuning` | int64 | `0` | 允许最大缓冲计数调优 |
| `enable-count-buffer-dma-opt` | bool | `false` | DMA 缓冲不被 Vector 重用 |
| `enable-manage-host-resources` | bool | `false` | 启用 Host 资源管理 |
| `cube-tiling-tuning` | int64 list | `[]` | Cube tiling 参数调优 |
| `external-tiling-func-path` | string | `"-"` | 外部 tiling 函数路径 |

## 2. 向量化 Pass

### 2.1 AutoVectorize

自动向量化器，将 linalg 命名算子转换为向量操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-auto-vectorize` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createHFusionAutoVectorizePass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vector-length` | unsigned | `256` | 向量长度（字节） |
| `peel-loops` | bool | `false` | 尝试剥离 tile 循环 |
| `max-vectorize-axes` | int64 | `-1` | 最大向量化轴数，-1 无限制 |
| `tree-reduce` | bool | `false` | 使用树形归约 |

### 2.2 AutoVectorizeV2

第二代自动向量化器，支持 tile、fuse 和 vectorize。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-auto-vectorize-v2` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createHFusionAutoVectorizeV2Pass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vector-length` | unsigned | `256` | 向量长度（字节） |
| `max-fused-ops` | unsigned | `15` | 单个融合节点最大算子数 |
| `enable-multiple-consumer-fusion` | bool | `false` | 启用多消费者融合 |

### 2.3 VectorizeOps

HFusion 算子向量化。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-vectorize-ops` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createHFusionVectorizeOpsPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `for-manual-scope` | bool | `false` | 仅对手动 scope 函数向量化 |

### 2.4 GenericUnroller

提供不可向量化 linalg 算子的展开（目前仅 linalg.reduce）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-generic-unroller` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createGenericUnrollerPass()` |

## 3. 函数外联与调度 Pass

### 3.1 OutlineVectorFunction

外联向量函数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `outline-vector-function` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createOutlineVectorFunctionPass()` |

### 3.2 OutlineSingleOp

将单个 linalg 算子外联为 kernel。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-outline-single-op` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createOutlineSingleOpPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `move-out-to-param` | bool | `true` | 是否将 tensor 输出移到参数 |

### 3.3 PullSliceIntoVectorFunction

将 tensor.extract_slice/insert_slice 拉入 VF 调用者以减少缓冲拷贝。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-pull-slice-into-vector-function` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createPullSliceIntoVectorFunctionPass()` |

## 4. 形状与类型规范化 Pass

### 4.1 FlattenOps

展平 linalg 和 hfusion 算子。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-flatten-ops` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createFlattenOpsPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `flatten-mode` | `FlattenMode` | `Greedy` | 展平模式：`greedy`/`tidy` |
| `skip-host` | bool | `false` | 是否跳过 host 函数 |
| `multi-dynamic-shape` | bool | `true` | 是否折叠多个动态形状 |
| `register-based` | bool | `false` | 是否使用寄存器模式展平 |
| `skip-scope` | bool | `true` | 存在 scope 时是否跳过 |

### 4.2 Normalize

规范化 HFusion 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-normalize-ops` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createHFusionNormalizeOpsPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable-high-precision` | bool | `true` | 启用 sin/cos 高精度计算 |

### 4.3 NormalizeSliceOps

规范化切片操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-normalize-slice-ops` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createHFusionNormalizeSliceOpsPass()` |

### 4.4 LegalizeScalarPass

将标量操作规范化为张量操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-legalize-scalar` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createLegalizeScalarPass()` |

### 4.5 LegalizeBF16Pass

将 BF16 规范化为 FP32。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-legalize-bf16` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createLegalizeBF16Pass()` |

### 4.6 LegalizeFP8Pass

将 FP8 规范化为 FP32。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-legalize-fp8` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createLegalizeFP8Pass()` |

### 4.7 LegalizeBoolPass

布尔类型规范化（int8 ↔ int1）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-legalize-bool` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createLegalizeBoolPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable-clamp` | bool | `false` | 启用 clamp 伪布尔算术模式 |

### 4.8 DowngradeFP64CstOpPass

将 FP64 常量降级为 FP32。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-downgrade-fp64` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createDowngradeFP64CstOpPass()` |

## 5. 算子变换 Pass

### 5.1 ConvertGenericToNamedOp

将 linalg generic 算子转换为命名算子。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-convert-generic-to-named` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createConvertGenericToNamedOpPass()` |

### 5.2 HFusionInlineBrc

内联广播类操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-inline-brc` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `hfusion::createHFusionInlineBrcPass()` |

### 5.3 SimplifyOps

简化操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-simplify-ops` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createSimplifyOpsPass()` |

### 5.4 ReorderOpsByBFS

按 BFS 顺序重排操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-reorder-ops` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createReorderOpsByBFS()` |

### 5.5 Decompose

分解实现了 AggregatedOpInterface 的算子。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-decompose` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createDecomposePass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hfusion-decompose-phase` | `DecomposePhase` | `NO_CONSTRAINT` | 分解阶段：`no-constraint`/`after-hfusion-flatten` |

### 5.6 HFusionFoldUnitDims

移除张量中的单位维度。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-fold-unit-dims` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `hfusion::createHFusionFoldUnitDimsPass()` |

### 5.7 HFusionGeneralizePass

将 hfusion 算子转换为 linalg.generic。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-generalize` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `::mlir::hfusion::createHFusionGeneralizePass()` |

### 5.8 RemoveMaskFromUnalignedReductionLoop

从非对齐归约循环中移除掩码。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `remove-mask-from-unaligned-reduction-loop` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `::mlir::hfusion::createRemoveMaskFromUnalignedReductionLoopPass()` |

## 6. 归约优化 Pass

### 6.1 ComposeMultiReduce

多归约组合优化。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-compose-multi-reduce` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createComposeMultiReduce()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max-compose` | int | `-1` | 最大组合归约数，-1 无限制 |
| `max-dist-diff` | int | `-1` | 距公共祖先最大距离差 |
| `aggressive` | bool | `false` | 激进模式：尝试 reshape 匹配 |

### 6.2 DecomposeMulti

将多输出算子分解为单输出算子。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-decompose-multi` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createDecomposeMulti()` |

## 7. 缓冲与内存管理 Pass

### 7.1 CacheIO

缓存输入输出参数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-cache-io` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `::mlir::hfusion::createCacheIO()` |

### 7.2 CacheIOForReturnArg

缓存直接返回的参数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-cache-io-for-return-arg` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `::mlir::hfusion::createCacheIOForReturnArg()` |

### 7.3 ReCacheIO

重新缓存 IO。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-recache-io` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `::mlir::hfusion::createReCacheIO()` |

### 7.4 RemoveCacheIO

移除缓存 IO。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-remove-cache-io` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `::mlir::hfusion::createRemoveCacheIO()` |

### 7.5 HoistTensorEmpty

将 tensor.empty 提升到函数参数并合并为一个参数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-hoist-tensor-empty` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createHoistTensorEmptyPass()` |

### 7.6 TensorResToOutParams

将张量结果移到函数输出参数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-tensor-results-to-out-params` |
| 作用域 | `ModuleOp` |
| 构造函数 | `hfusion::createTensorResToOutParamsPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `include-symbols` | string list | `[]` | 应用转换的符号列表，空则全部 |
| `enable-manage-host-resources` | bool | `false` | 启用 Host 资源管理 |

## 8. Tiling 与形状推断 Pass

### 8.1 PackTilingData

将动态 tiling 信息打包为结构体。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-pack-tiling-data` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createPackTilingDataPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `include-symbols` | string list | `[]` | 应用转换的符号列表 |
| `emit-get-tiling-struct-size-function` | bool | `false` | 发出返回 tiling 结构大小的函数 |
| `pack-tiling-key` | bool | `true` | 是否将 tiling key 打包到结构体 |

### 8.2 ConstantizeTilingData

在 tiling 函数和设备函数间传播常量。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-constantize-tiling-data` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createConstantizeTilingDataPass()` |

### 8.3 InferOutShapesPass

为 kernel 生成输出张量形状函数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-infer-out-shapes` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createInferOutShapesPass()` |

## 9. 符号与函数管理 Pass

### 9.1 FoldSymbolicDim

将 tensor.dim 源操作数替换为 hfusion::SymbolicDimOp。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-fold-symbolic-dim` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createFoldSymbolicDimPass()` |

### 9.2 UnfoldSymbolicDim

将 hfusion::SymbolicDimOp 替换为相同的符号参数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-unfold-symbolic-dim` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createUnfoldSymbolicDimPass()` |

### 9.3 DropSymbols

从操作中移除 ranked tensor 符号。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-drop-symbols` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createDropSymbolsPass()` |

### 9.4 EliminateDuplicateFuncs

消除融合后的重复函数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-eliminate-duplicate-funcs` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createEliminateDuplicateFuncsPass()` |

### 9.5 InferFuncFusionKind

推断函数的融合类型。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-infer-func-fusion-kind` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `mlir::hfusion::createInferFuncFusionKind()` |

### 9.6 SimplifyVFArgs

简化 VF 函数参数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-simplify-vf-arg` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createSimplifyVFArgsPass()` |

### 9.7 MergeVecScope

合并 VF 函数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-merge-vf` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createMergeVecScopePass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `merge-level` | int | `1` | 合并级别：0=不合并，1=无依赖合并，2=全部合并 |
| `merge-vf-num-limit` | int | `4` | 允许的最大合并 VF 数 |

### 9.8 WrapHostFunc

为 host 相关函数创建包装器。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-wrap-host-func` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createWrapHostFuncPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `remove-unused-arguments` | bool | `false` | 是否移除 host 包装器中未使用的参数 |

## 10. Triton 适配 Pass

### 10.1 AdaptTritonKernel

适配 Triton kernel。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `adapt-triton-kernel` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createAdaptTritonKernelPass()` |

### 10.2 PreVectorizationFusion

向量化前的逐元素融合和泛化。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-pre-vectorization-fusion` |
| 作用域 | `func::FuncOp` |
| 构造函数 | `::mlir::hfusion::createPreVectorizationFusionPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable-triton-compile` | bool | `false` | 启用 Triton 编译 |
| `max-fused-elementwise-ops` | int | `-1` | 最大融合逐元素算子数 |

### 10.3 AddFFTSAddr

添加 FFTS 基地址到函数参数和注解。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hfusion-add-ffts-addr` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hfusion::createAddFFTSAddrPass()` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `force-add-ffts-addr` | int | `-1` | 强制添加 FFTS 基地址的位置，-1 不插入 |
