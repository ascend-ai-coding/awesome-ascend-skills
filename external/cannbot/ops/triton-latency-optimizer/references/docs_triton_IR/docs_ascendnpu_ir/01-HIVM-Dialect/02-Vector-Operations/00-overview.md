# HIVM 向量操作总览

> 关键词：HIVM, Vector Operations, PIPE_V, Elementwise, TableGen, AscendNPU

## 概述

HIVM（Hybrid Intelligence Virtual Machine）方言的向量操作是 AscendNPU 向量计算单元（AIV）的核心 IR 抽象。所有向量操作在硬件层面映射到 Vector Pipe（PIPE_V），由向量计算核心执行。这些操作遵循 MLIR 的 DestinationStyleOpInterface，同时支持 tensor 和 memref 两种语义。

> Python API 对应：Triton Ascend 的 `tl` 原子操作最终通过编译流水线降级为 HIVM 向量操作。详见 docs_triton_ascend 相关文档。

## 操作继承层次

HIVM 向量操作采用 TableGen 多层继承定义，层次结构如下：

```
HIVM_Op                          -- 所有 HIVM 操作的基类（前缀 "hir."）
  └── HIVM_StructuredOp          -- 结构化操作基类（实现 HIVMStructuredOpInterface 等）
        └── HIVM_VectorOp        -- 向量操作基类（PIPE_V, VectorCoreTypeTrait）
              ├── HIVM_ElementwiseNaryOp   -- 逐元 N-ary 操作模板
              │     ├── HIVM_ElementwiseUnaryOp   -- 一元操作（N=1）
              │     ├── HIVM_ElementwiseBinaryOp  -- 二元操作（N=2）
              │     └── HIVM_ElementwiseTernaryOp -- 三元操作（N=3）
              └── [独立向量操作]           -- vbrc, vreduce, vtranspose, varange 等
```

### 基类定义

| 基类 | 关键特性 | 源码位置 |
|------|---------|---------|
| [HIVM_Op](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td#L52-L58) | 操作前缀 `hir.`，命名空间 `::mlir::hivm` | HIVMBase.td |
| [HIVM_StructuredOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td#L60-L68) | 实现 HIVMStructuredOpInterface, MemoryEffectsOpInterface, FlattenInterface, LibraryFunctionOpInterface | HIVMBase.td |
| [HIVM_VectorOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L24-L40) | PIPE_V, VectorCoreTypeTrait, AlwaysSpeculatable, SinglePipeOpTrait, DestinationStyleOpInterface | HIVMVectorOps.td |
| [HIVM_ElementwiseNaryOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L46-L95) | AttrSizedOperandSegments, HIVMOpSameOperandsAndResultRank, UniformReassociationFlattenTrait, CollapsibleConsecutiveTargetDimsTrait, TransposableOTF | HIVMVectorOps.td |

## Pipeline 归属

所有 HIVM 向量操作归属于 **PIPE_V**（Vector Pipe），对应硬件的向量计算单元。这在 `HIVM_VectorOp` 基类中通过 `OpPipeTrait<"PIPE::PIPE_V">` 静态指定。

```
PIPE 枚举值    含义              适用操作类别
──────────────────────────────────────────────
PIPE_S         Scalar Pipe       标量操作
PIPE_V         Vector Pipe       向量操作（本文档范围）
PIPE_M         Matrix Pipe       矩阵乘操作
PIPE_MTE1      数据搬入 L1       Load 类操作
PIPE_MTE2      数据搬入 L0A/L0B  Load 类操作
PIPE_MTE3      数据搬出          Store 类操作
```

源码参考：[HIVMAttrs.td - Pipe 枚举定义](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L203-L236)

## Traits 分类

HIVM 向量操作使用以下关键 Traits 进行约束：

### 核心 Traits

| Trait | 说明 | 依赖 | 源码位置 |
|-------|------|------|---------|
| ElementwiseNaryOpTrait\<N\> | 逐元 N-ary 操作约束：N 个输入，1 个输出，相同 rank | HIVMStructuredOpInterface, HIVMOpSameOperandsAndResultRank | [HIVMTraits.td#L53-L56](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L53-L56) |
| CommutativeOpTrait | 交换律：输入操作数可交换 | 无 | [HIVMTraits.td#L76](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L76) |
| OperElemTypeConstraints\<indices, types\> | 操作数元素类型约束 | 无 | [HIVMTraits.td#L90-L106](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L90-L106) |
| VectorOnlyTrait\<idx\> | 指定操作数只能为向量类型 | 无 | [HIVMTraits.td#L80-L81](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L80-L81) |
| ScalarOnlyTrait\<idx\> | 指定操作数只能为标量类型 | 无 | [HIVMTraits.td#L87](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L87) |
| StaticMaxRankTrait\<N\> | 静态已知最大 rank 限制 | 无 | [HIVMTraits.td#L64](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L64) |
| InferMaxRankTrait | 运行时推断最大 rank | 无 | [HIVMTraits.td#L67](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L67) |

### OTF（On-The-Fly）Traits

| Trait | 说明 | 源码位置 |
|-------|------|---------|
| BroadcastableOTF | 支持 OTF 广播：在计算时自动扩展指定维度 | [HIVMTraits.td#L130-L132](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L130-L132) |
| TransposableOTF | 支持 OTF 转置：在计算时自动重排维度 | [HIVMTraits.td#L153-L155](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L153-L155) |

### Flatten 相关 Traits

| Trait | 说明 | 源码位置 |
|-------|------|---------|
| UniformReassociationFlattenTrait | 所有操作数和结果可以使用相同的维度重关联进行展平 | [HIVMTraits.td#L183](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L183) |
| CollapsibleConsecutiveTargetDimsTrait | 标记操作在展平时必须保持目标维度的独立性和秩 | [HIVMTraits.td#L191-L192](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L191-L192) |

## 与 HFusion 逐元操作的对应关系

HIVM 向量操作在编译流水线中会降级到 HFusion 方言的逐元操作。主要对应关系：

| HIVM 操作 | HFusion 操作 | 说明 |
|-----------|-------------|------|
| hir.vrelu | hfusion.elemwise_unary {fun = relu} | ReLU 激活 |
| hir.vnot | hfusion.elemwise_unary {fun = vnot} | 按位取反 |
| hir.vcast | hfusion.cast {round_mode = ...} | 类型转换 |
| hir.vcmp | hfusion.compare {compare_fn = veq/vne/vlt/vle/vgt/vge} | 比较操作 |
| hir.vreduce | hfusion.reduce / hfusion.reduce_with_index | 归约操作 |
| hir.varange | hfusion.arange | 范围生成 |
| hir.vinterleave | hfusion.interleave | 交错合并 |
| hir.vdeinterleave | hfusion.deinterleave | 交错分离 |

对于标准的算术操作（vadd, vsub, vmul, vdiv, vabs, vexp, vln 等），HIVM 操作降级到 `linalg` 方言的对应操作（linalg.add, linalg.sub, linalg.abs, linalg.exp 等）。

## 所有向量操作总表

### 一元运算（Unary Ops）

| 操作 | 助记符 | 说明 | 支持元素类型 | 最大 Rank |
|------|--------|------|-------------|-----------|
| VExpOp | hir.vexp | 逐元指数运算 | F16, F32 | 3 |
| VAbsOp | hir.vabs | 逐元绝对值 | F16, F32, I8, I16, I32, I64 | 3 |
| VLnOp | hir.vln | 逐元自然对数 | F16, F32 | 3 |
| VReluOp | hir.vrelu | 逐元 ReLU | F16, F32, I32 | 3 |
| VRsqrtOp | hir.vrsqrt | 逐元倒数平方根 | F16, F32 | 3 |
| VSqrtOp | hir.vsqrt | 逐元平方根 | F16, F32 | 3 |
| VTanhOp | hir.vtanh | 逐元双曲正切 | 浮点类型 | - |
| VSinOp | hir.vsin | 逐元正弦 | 浮点类型 | - |
| VCosOp | hir.vcos | 逐元余弦 | 浮点类型 | - |
| VErfOp | hir.verf | 逐元误差函数 | 浮点类型 | - |
| VRecOp | hir.vrec | 逐元倒数 | F16, F32 | 3 |
| VNotOp | hir.vnot | 逐元按位取反 | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32 | 3 |
| VCastOp | hir.vcast | 逐元类型转换 | 多种（见 04-cast-ops.md） | 2 |

### 二元运算（Binary Ops）

| 操作 | 助记符 | 说明 | 支持元素类型 | 最大 Rank | 交换律 |
|------|--------|------|-------------|-----------|--------|
| VAddOp | hir.vadd | 逐元加法 | I8, I16, I32, F16, F32, I64 | 3 | 是 |
| VSubOp | hir.vsub | 逐元减法 | I8, I16, I32, F16, F32, I64 | 3 | 否 |
| VMulOp | hir.vmul | 逐元乘法 | I16, I32, F16, F32, I64 | 3 | 是 |
| VDivOp | hir.vdiv | 逐元除法 | F16, F32, I16, I32, I64 | 3 | 否 |
| VMaxOp | hir.vmax | 逐元最大值 | I16, I32, F16, F32, I64 | 3 | 是 |
| VMinOp | hir.vmin | 逐元最小值 | I16, I32, F16, F32, I64 | 3 | 是 |
| VOrOp | hir.vor | 逐元按位或 | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32 | 3 | 是 |
| VAndOp | hir.vand | 逐元按位与 | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32 | 3 | 是 |
| VXorOp | hir.vxor | 逐元按位异或 | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64 | 2 | 否 |
| VModOp | hir.vmod | 逐元取模（有符号） | I16, I32, I64 | 1 | 否 |
| VModUIOp | hir.vmodui | 逐元取模（无符号） | I16, I32, I64 | 1 | 否 |
| VPowOp | hir.vpow | 逐元幂运算 | I32 | 1 | 否 |
| VShLOp | hir.vshl | 逐元左移 | I16, I32, I64 | 3 | 否 |
| VShROp | hir.vshr | 逐元右移 | I16, I32, I64 | 3 | 否 |
| VCmpOp | hir.vcmp | 逐元比较 | F16, F32, I8, I16, I32, I64（输入）; I1, I8（输出） | 1 | 否 |
| VMulExtOp | hir.vmulext | 逐元乘法高32位 | I32 | 3 | 否 |

### 三元运算（Ternary Ops）

| 操作 | 助记符 | 说明 | 支持元素类型 | 最大 Rank |
|------|--------|------|-------------|-----------|
| VSelOp | hir.vsel | 逐元条件选择 | 条件: I1/I8; 数据: I1, AnyI8, AnyI16, F16, BF16, AnyI32, F32, I64 | 1 |

### 归约运算（Reduction Ops）

| 操作 | 助记符 | 说明 | 支持元素类型 |
|------|--------|------|-------------|
| VReduceOp | hir.vreduce | 向量归约 | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32 |

### 数据搬移（Data Movement Ops）

| 操作 | 助记符 | 说明 | 支持元素类型 |
|------|--------|------|-------------|
| VBrcOp | hir.vbrc | 向量广播 | I8, UI8, I16, F16, UI16, I32, F8E4M3FN, F8E5M2, F32, UI32, BF16, I64, UI64, I1 |
| VTransposeOp | hir.vtranspose | 维度转置 | AnyI8, AnyI16, AnyI32, F16, BF16, F32, I64, UI64, F8E4M3FN, F8E5M2 |
| VInterleaveOp | hir.vinterleave | 交错合并 | I16, F16, UI16, I32, F32, UI32, BF16, I64, UI64, F8E4M3FN, F8E5M2 |
| VDeinterleaveOp | hir.vdeinterleave | 交错分离 | I8, I16, F16, UI16, I32, F32, UI32, BF16, I64, UI64 |
| VFlipOp | hir.vflip | 维度翻转 | I8, UI8, I16, I32, UI16, UI32, I64, UI64, F16, F32, BF16 |
| VPadOp | hir.vpad | 填充 | 无类型约束（由 pad_value 决定） |
| VConcatOp | hir.vconcat | 拼接 | SameOperandsElementType |
| VGatherOp | hir.vgather | 按索引收集 | 数据: I1, I8, I16, UI16, I32, UI32, F16, BF16, F32, F8E4M3FN, F8E5M2; 索引: I32 |

### 累积与排序（Cumulative & Sort Ops）

| 操作 | 助记符 | 说明 | 支持元素类型 | 最大 Rank |
|------|--------|------|-------------|-----------|
| VCumsumOp | hir.vcumsum | 累积求和 | I1, I8, I16, I32, I64, F16, F32, BF16 | 2 |
| VCumprodOp | hir.vcumprod | 累积求积 | I1, I8, I16, I32, I64, F16, F32, BF16 | 1 |
| VSortOp | hir.vsort | 排序 | F16, F32, I32, I64 | 1 |

### 特殊操作（Special Ops）

| 操作 | 助记符 | 说明 | 支持元素类型 | 最大 Rank |
|------|--------|------|-------------|-----------|
| VArangeOp | hir.varange | 范围序列生成 | I16, I32, F16, F32, I64 | 3 |
| VMulextendedOp | hir.vmulextended | 扩展乘法（高低位） | I16 | 1 |
| VMulExtOp | hir.vmulext | 乘法高32位 | I32 | 3 |

## 通用操作数结构

### Elementwise N-ary 操作通用参数

所有继承自 `HIVM_ElementwiseNaryOp` 的操作共享以下参数结构：

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| $src | Variadic\<AnyShaped\> 或 Variadic\<AnyType\> | 是 | 输入操作数（支持向量-向量或向量-标量） |
| $dst | Variadic\<AnyShaped\> | 是 | 输出操作数（DestinationStyle） |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | 临时缓冲区（部分操作需要） |
| $transpose | DenseI64ArrayAttr (默认 {}) | 否 | OTF 转置维度排列 |
| $broadcast | DenseI64ArrayAttr (默认 {}) | 否 | OTF 广播维度索引 |
| $result | Variadic\<AnyRankedTensor\> | 是（tensor 语义） | 结果类型 |

### 通用 Assembly Format

```
hir.<op> attr-dict
  ins(<src> : <src_type>)
  outs(<dst> : <dst_type>)
  [temp_buffer(<temp_buffer> : <temp_buffer_type>)]
  [broadcast = <broadcast>]
  [transpose = <transpose>]
  [-> <result_type>]
```

## 标量降级

部分 HIVM 向量操作在特定条件下会被降级为标量循环执行，而非使用硬件向量指令。这通常发生在硬件向量计算单元不支持某些数据类型和操作组合时（如 i64 类型的算术运算、整数大小比较等）。标量降级会显著影响性能。

实现标量降级的操作均声明了 `ImplByScalarOpInterface`，通过 `shouldLowerToScalarLoops()` 方法判断是否需要降级。

**详细文档**：[11-scalar-lowering.md](../../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/11-scalar-lowering.md)

### 支持标量降级的操作一览

| 操作 | 助记符 | 标量降级主要触发条件 |
|------|--------|-------------------|
| VAddOp | hir.vadd | i64 或 SIMT VF |
| VSubOp | hir.vsub | i64 或 SIMT VF |
| VMulOp | hir.vmul | i64 或 SIMT VF |
| VMinOp | hir.vmin | i64 或 SIMT VF |
| VMaxOp | hir.vmax | i64 或 SIMT VF |
| VAbsOp | hir.vabs | i64 或 SIMT VF |
| VShLOp | hir.vshl | i64 或 SIMT VF |
| VShROp | hir.vshr | i64 或 SIMT VF |
| VInterleaveOp | hir.vinterleave | i64 或 SIMT VF |
| VDeinterleaveOp | hir.vdeinterleave | i64 或 SIMT VF |
| VCmpOp | hir.vcmp | 整数类型 且 (非 i32 或 非 EQ/NE) |
| VMulExtOp | hir.vmulext | i32 或 i64（即始终降级） |
| VCumsumOp | hir.vcumsum | i64 或 累积维度为最后维度 |
| VCumprodOp | hir.vcumprod | i64 或 累积维度为最后维度 |
| VReduceOp | hir.vreduce | 依架构和 reduceOp 类型 |

## 相关文档

- 标量降级详解：[11-scalar-lowering.md](../../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/11-scalar-lowering.md)
- Python API：docs_triton_ascend 中的原子操作文档
- 源码参考：
  - [HIVMVectorOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td) — 向量操作 TableGen 定义
  - [HIVMTraits.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td) — Traits 定义
  - [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td) — 枚举属性定义
  - [HIVMBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td) — 基类定义
  - [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir) — IR 测试示例
