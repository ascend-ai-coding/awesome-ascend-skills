# Triton 操作详解

本文档详细描述 Triton 方言（`tt`）中所有操作的定义、签名和语义。所有操作签名均从 TableGen 源码精确提取。

源码参考：[TritonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Triton/IR/TritonOps.td)

## 1. 类型转换操作

### 1.1 tt.int_to_ptr

将 int64 值转换为指针类型。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.int_to_ptr` |
| 输入 | `$src`: `TT_I64Like` (i64 或 i64 张量) |
| 输出 | `$result`: `TT_PtrLike` (指针或指针张量) |
| Traits | `Elementwise`, `SameOperandsAndResultShape`, `SameOperandsAndResultEncoding`, `Pure` |

```mlir
%ptr = tt.int_to_ptr %src : i64 -> !tt.ptr<f32>
```

### 1.2 tt.ptr_to_int

将指针值转换为 int64 类型。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.ptr_to_int` |
| 输入 | `$src`: `TT_PtrLike` |
| 输出 | `$result`: `TT_I64Like` |
| Traits | `Elementwise`, `SameOperandsAndResultShape`, `SameOperandsAndResultEncoding`, `Pure` |

```mlir
%int = tt.ptr_to_int %ptr : !tt.ptr<f32> -> i64
```

### 1.3 tt.bitcast

在相同位宽的类型之间进行位转换。`arith.bitcast` 不支持指针类型，因此 Triton 提供了此操作。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.bitcast` |
| 输入 | `$src`: `TT_Type` |
| 输出 | `$result`: `TT_Type` |
| Traits | `Elementwise`, `SameOperandsAndResultShape`, `SameOperandsAndResultEncoding`, `Pure` |
| 验证器 | `hasVerifier = 1` |

```mlir
%result = tt.bitcast %src : f32 -> i32
```

### 1.4 tt.fp_to_fp

浮点类型之间的转换，支持自定义类型（F8）和非默认舍入模式。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.fp_to_fp` |
| 输入 | `$src`: `TT_FloatLike` |
| 属性 | `$rounding`: `TT_RoundingModeAttr` (可选) |
| 输出 | `$result`: `TT_FloatLike` |
| Traits | `Elementwise`, `SameOperandsAndResultShape`, `SameOperandsAndResultEncoding`, `Pure` |
| 验证器 | `hasVerifier = 1` |
| 优化 | `hasFolder = 1` |

支持的转换：F8 ↔ FP16, BF16, FP32, FP64

```mlir
%result = tt.fp_to_fp %src, rounding = #tt<rounding rtne> : f16 -> f32
```

## 2. 算术运算

### 2.1 tt.clampf

浮点数限幅操作，将值限制在 [min, max] 范围内。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.clampf` |
| 输入 | `$x`: `TT_FloatLike`, `$min`: `TT_FloatLike`, `$max`: `TT_FloatLike` |
| 属性 | `$propagateNan`: `TT_PropagateNanAttr` |
| 输出 | `$result`: `TT_FloatLike` |
| Traits | `Elementwise`, `SameOperandsAndResultType`, `Pure` |

```mlir
%result = tt.clampf %x, %min, %max, propagateNan = #tt<propagate_nan all> : f32
```

### 2.2 tt.precise_sqrt

高精度浮点平方根。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.precise_sqrt` |
| 输入 | `$x`: `TT_FloatLike` |
| 输出 | `$result`: `TT_FloatLike` |
| Traits | `Elementwise`, `SameOperandsAndResultType`, `Pure` |

### 2.3 tt.precise_divf

高精度浮点除法。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.precise_divf` |
| 输入 | `$x`: `TT_FloatLike`, `$y`: `TT_FloatLike` |
| 输出 | `$result`: `TT_FloatLike` |
| Traits | `Elementwise`, `SameOperandsAndResultType`, `Pure` |

### 2.4 tt.mulhiui

计算两个无符号整数乘积的高 N 位。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.mulhiui` |
| 输入 | `$x`: `TT_IntLike`, `$y`: `TT_IntLike` |
| 输出 | `$result`: `TT_IntLike` |
| Traits | `Elementwise`, `SameOperandsAndResultType`, `Pure` |

## 3. 指针算术

### 3.1 tt.addptr

指针偏移计算，将指针按元素偏移量前进。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.addptr` |
| 输入 | `$ptr`: `TT_PtrLike`, `$offset`: `TT_IntLike` |
| 输出 | `$result`: `TT_PtrLike` (类型与 `$ptr` 相同) |
| Traits | `Pure`, `Elementwise`, `SameOperandsAndResultShape`, `SameOperandsAndResultEncoding` |
| 优化 | `hasFolder = 1` |

```mlir
%new_ptr = tt.addptr %ptr, %offset : !tt.ptr<f32>, i32
```

### 3.2 tt.advance

沿指定偏移推进张量指针。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.advance` |
| 输入 | `$ptr`: `TT_TensorPtr`, `$offsets`: `Variadic<I32>` |
| 输出 | `$result`: `TT_TensorPtr` (类型与 `$ptr` 相同) |
| Traits | `Pure` |
| 优化 | `hasFolder = 1` |

```mlir
%new_ptr = tt.advance %ptr, [%offset0, %offset1] : !tt.ptr<tensor<8x8xf16>>
```

## 4. 内存访问操作

### 4.1 tt.load

从指针张量或张量指针加载数据。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.load` |
| 输入 | `$ptr`: `TT_PtrLike \| TT_TensorPtr`, `$mask`: `TT_BoolLike` (可选), `$other`: `TT_Type` (可选) |
| 属性 | `$boundaryCheck`: `DenseI32ArrayAttr` (默认 `[]`), `$padding`: `TT_PaddingOptionAttr` (可选), `$cache`: `TT_CacheModifierAttr` (默认 `NONE`), `$evict`: `TT_EvictionPolicyAttr` (默认 `NORMAL`), `$isVolatile`: `BoolAttr` (默认 `false`) |
| 输出 | `$result`: `TT_Type` |
| Traits | `SameLoadStoreOperandsAndResultShape`, `SameLoadStoreOperandsAndResultEncoding`, `AttrSizedOperandSegments` |

```mlir
%result = tt.load %ptr, %mask, %other cacheModifier = #tt<cache ca> evictionPolicy = #tt<evict evict_normal> : !tt.ptr<tensor<128xf32>>
```

### 4.2 tt.store

通过指针张量或张量指针存储数据。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.store` |
| 输入 | `$ptr`: `TT_PtrLike \| TT_TensorPtr` (MemWrite), `$value`: `TT_Type`, `$mask`: `TT_BoolLike` (可选) |
| 属性 | `$boundaryCheck`: `DenseI32ArrayAttr` (默认 `[]`), `$cache`: `TT_CacheModifierAttr` (默认 `NONE`), `$evict`: `TT_EvictionPolicyAttr` (默认 `NORMAL`) |
| Traits | `SameLoadStoreOperandsShape`, `SameLoadStoreOperandsEncoding` |

```mlir
tt.store %ptr, %value, %mask cacheModifier = #tt<cache cg> : !tt.ptr<tensor<128xf32>>
```

## 5. 原子操作

### 5.1 tt.atomic_rmw

原子读-改-写操作。从 `$ptr` 加载数据，与 `$val` 执行 `$rmw_op`，存储结果到 `$ptr`，返回旧值。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.atomic_rmw` |
| 属性 | `$atomic_rmw_op`: `TT_AtomicRMWAttr` (AND/OR/XOR/ADD/FADD/MAX/MIN/UMAX/UMIN/XCHG) |
| 输入 | `$ptr`: `TT_PtrLike` (MemRead+MemWrite), `$val`: `TT_Type`, `$mask`: `TT_BoolLike` (可选) |
| 属性 | `$sem`: `TT_MemSemanticAttr`, `$scope`: `TT_MemSyncScopeAttr` |
| 输出 | `$result`: `TT_Type` |

```mlir
%old = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (tensor<128x!tt.ptr<f32>>, tensor<128xf32>) -> tensor<128xf32>
```

### 5.2 tt.atomic_cas

原子比较-交换操作。将 `$ptr` 处数据与 `$cmp` 比较，相等则存储 `$val`，否则保持不变，返回旧值。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.atomic_cas` |
| 输入 | `$ptr`: `TT_PtrLike` (MemRead+MemWrite), `$cmp`: `TT_Type`, `$val`: `TT_Type` |
| 属性 | `$sem`: `TT_MemSemanticAttr`, `$scope`: `TT_MemSyncScopeAttr` |
| 输出 | `$result`: `TT_Type` |

```mlir
%old = tt.atomic_cas acquire_release, gpu, %ptr, %cmp, %val : (tensor<128x!tt.ptr<i32>>, tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
```

## 6. 形状操作

### 6.1 tt.splat

将标量值广播为张量。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.splat` |
| 输入 | `$src`: `TT_Type` (标量) |
| 输出 | `$result`: `TT_Tensor` |
| Traits | `Pure`, `SameOperandsAndResultElementType`, `SameOperandsAndResultEncoding` |
| 优化 | `hasFolder = 1` |

### 6.2 tt.unsplat

将单元素张量转换为标量。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.unsplat` |
| 输入 | `$src`: `TT_Tensor` |
| 输出 | `$result`: `TT_Type` |
| Traits | `Pure` |
| 验证器 | `hasVerifier = 1` |

### 6.3 tt.expand_dims

在指定轴插入大小为 1 的维度。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.expand_dims` |
| 输入 | `$src`: `TT_Tensor`, `$axis`: `I32Attr` |
| 输出 | `$result`: `TT_Tensor` |
| Traits | `Pure`, `SameOperandsAndResultElementType` |
| 优化 | `hasCanonicalizeMethod = 1`, `hasFolder = 1` |

### 6.4 tt.reshape

将张量重新解释为不同形状。设置 `allow_reorder` 允许编译器改变元素顺序以生成更高效的代码。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.reshape` |
| 输入 | `$src`: `TT_Tensor` |
| 属性 | `$allow_reorder`: `UnitAttr`, `$efficient_layout`: `UnitAttr` |
| 输出 | `$result`: `TT_Tensor` |
| Traits | `Pure`, `SameOperandsAndResultElementType` |
| 验证器 | `hasVerifier = 1` |

```mlir
%result = tt.reshape %src allow_reorder efficient_layout : tensor<64xf32> -> tensor<8x8xf32>
```

### 6.5 tt.broadcast

将张量中大小为 1 的维度广播到新大小。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.broadcast` |
| 输入 | `$src`: `TT_Tensor` |
| 输出 | `$result`: `TT_Tensor` |
| Traits | `Pure`, `SameOperandsAndResultElementType`, `SameOperandsAndResultEncoding` |
| 验证器 | `hasVerifier = 1` |

### 6.6 tt.cat

沿最内层维度连接两个张量。不是 Pure 操作，因为可能重排元素。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.cat` |
| 输入 | `$lhs`: `TT_Tensor`, `$rhs`: `TT_Tensor` |
| 输出 | `$result`: `TT_Tensor` |
| Traits | `NoMemoryEffect`, `SameTypeOperands`, `SameOperandsAndResultElementType` |

### 6.7 tt.join

沿新的最内层维度合并两个张量。输入张量必须形状相同。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.join` |
| 输入 | `$lhs`: `TT_Tensor`, `$rhs`: `TT_Tensor` |
| 输出 | `$result`: `TT_Tensor` |
| Traits | `Pure`, `SameTypeOperands` |

例如：两个 `4x8xf32` 张量 → `4x8x2xf32` 张量。

### 6.8 tt.split

沿最内层维度拆分张量。输入最内层维度大小必须为 2。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.split` |
| 输入 | `$src`: `TT_Tensor` |
| 输出 | `$outLHS`: `TT_Tensor`, `$outRHS`: `TT_Tensor` |
| Traits | `Pure`, `InferTypeOpWithLayoutEquivalence` |

例如：`4x8x2xf32` → 两个 `4x8xf32` 张量。

### 6.9 tt.trans

重排张量的维度顺序。实现 `tl.trans()` 和 `tl.permute()` 的语义。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.trans` |
| 输入 | `$src`: `TT_Tensor` |
| 属性 | `$order`: `DenseI32ArrayAttr` |
| 输出 | `$result`: `TT_Tensor` |
| Traits | `Pure`, `TransposeOpInterface`, `InferTypeOpWithLayoutEquivalence`, `SameOperandsAndResultElementType` |

```mlir
%result = tt.trans %src {order = [2, 0, 1]} : tensor<1x2x4xf32> -> tensor<4x1x2xf32>
```

## 7. SPMD 操作

### 7.1 tt.get_program_id

获取当前程序在指定维度上的 ID。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.get_program_id` |
| 属性 | `$axis`: `TT_ProgramDim` (X/Y/Z) |
| 输出 | `$result`: `I32` |
| Traits | `Pure` |

```mlir
%pid = tt.get_program_id x : i32
```

### 7.2 tt.get_num_programs

获取指定维度上的程序总数。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.get_num_programs` |
| 属性 | `$axis`: `TT_ProgramDim` (X/Y/Z) |
| 输出 | `$result`: `I32` |
| Traits | `Pure` |

## 8. 矩阵乘法操作

### 8.1 tt.dot

矩阵乘法加累加：`d = matmul(a, b) + c`。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.dot` |
| 输入 | `$a`: `TT_FpIntTensor`, `$b`: `TT_FpIntTensor`, `$c`: `TT_FpIntTensor` |
| 属性 | `$inputPrecision`: `TT_InputPrecisionAttr` (默认 `IEEE`), `$maxNumImpreciseAcc`: `I32Attr` (默认 `0`) |
| 输出 | `$d`: `TT_FpIntTensor` (类型与 `$c` 相同) |
| Traits | `Pure`, `DotOpInterface` |

`inputPrecision` 选项（仅当输入为 f32 时有效）：
- `tf32`：使用 TF32 Tensor Core
- `tf32x3`：3xTF32 技巧
- `ieee`：不使用 Tensor Core，软件实现

```mlir
%d = tt.dot %a, %b, %c, inputPrecision = #tt<input_precision tf32> : tensor<16x32xf16> * tensor<32x64xf16> -> tensor<16x64xf32>
```

### 8.2 tt.dot_scaled

带缩放的矩阵乘法：`d = matmul(scale(a, a_scale), scale(b, b_scale)) + c`。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.dot_scaled` |
| 输入 | `$a`: `RankedTensorOf<[TT_Float, I8]>`, `$b`: `RankedTensorOf<[TT_Float, I8]>`, `$c`: `TT_FloatTensor` |
| 输入 | `$a_scale`: `Optional<RankedTensorOf<[TT_Float, I8]>>`, `$b_scale`: `Optional<RankedTensorOf<[TT_Float, I8]>>` |
| 属性 | `$a_elem_type`: `TT_ScaleDotElemTypeAttr`, `$b_elem_type`: `TT_ScaleDotElemTypeAttr`, `$fastMath`: `BoolAttr`, `$lhs_k_pack`: `BoolAttr` (默认 `true`), `$rhs_k_pack`: `BoolAttr` (默认 `true`) |
| 输出 | `$d`: `TT_FloatTensor` |
| Traits | `Pure`, `AttrSizedOperandSegments`, `DotOpInterface` |

```mlir
%d = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %c lhs = #tt<scale_dot_elem_type e4m3> rhs = #tt<scale_dot_elem_type e5m2> : tensor<16x32xi8>, tensor<1x32xf32> * tensor<32x64xi8>, tensor<1x64xf32> -> tensor<16x64xf32>
```

## 9. 归约与扫描操作

### 9.1 tt.reduce

使用通用组合算法进行归约。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.reduce` |
| 输入 | `$srcs`: `Variadic<TT_Tensor>` |
| 属性 | `$axis`: `I32Attr` |
| 输出 | `$result`: `Variadic<TT_Type>` |
| 区域 | `$combineOp`: `SizedRegion<1>` |
| Traits | `Pure`, `SameOperandsShape`, `SameOperandsEncoding`, `SingleBlock` |

```mlir
%result = tt.reduce %src {axis = 0 : i32}  {
^bb0(%arg0: f32, %arg1: f32):
  %0 = arith.addf %arg0, %arg1 : f32
  tt.reduce.return %0 : f32
} : tensor<128xf32> -> f32
```

### 9.2 tt.scan

使用通用组合算法进行关联前缀扫描。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.scan` |
| 输入 | `$srcs`: `Variadic<TT_Tensor>` |
| 属性 | `$axis`: `I32Attr`, `$reverse`: `BoolAttr` |
| 输出 | `$result`: `Variadic<TT_Tensor>` |
| 区域 | `$combineOp`: `SizedRegion<1>` |
| Traits | `Pure`, `SameOperandsAndResultEncoding`, `SameOperandsAndResultShape`, `SingleBlock` |

## 10. 映射操作

### 10.1 tt.map_elementwise

将标量子区域映射到张量的每个元素上。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.map_elementwise` |
| 输入 | `$srcs`: `Variadic<TT_Tensor>` |
| 属性 | `$pack`: `I32Attr` |
| 输出 | `$result`: `Variadic<TT_Tensor>` |
| 区域 | `$scalarOp`: `AnyRegion` |
| Traits | `SameOperandsAndResultEncoding`, `SameOperandsAndResultShape`, `RecursiveMemoryEffects` |

## 11. 范围与统计操作

### 11.1 tt.make_range

生成从 `$start` 到 `$end`（不含）的 1D int32 张量，步长为 1。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.make_range` |
| 属性 | `$start`: `I32Attr`, `$end`: `I32Attr` |
| 输出 | `$result`: `TT_IntTensor` |
| Traits | `Pure` |

```mlir
%range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
```

### 11.2 tt.histogram

计算输入张量的直方图。bin 数量等于输出张量维度，bin 宽度为 1，从 0 开始。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.histogram` |
| 输入 | `$src`: `TT_IntTensor`, `$mask`: `TT_BoolLike` (可选) |
| 输出 | `$result`: `TT_IntTensor` |
| Traits | `Pure` |

### 11.3 tt.gather

按索引沿指定轴从输入张量收集元素。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.gather` |
| 输入 | `$src`: `TT_Tensor`, `$indices`: `TT_IntTensor`, `$axis`: `I32Attr` |
| 属性 | `$efficient_layout`: `UnitAttr` |
| 输出 | `$result`: `TT_Tensor` |
| Traits | `Pure` |
| 验证器 | `hasVerifier = 1` |

```mlir
%result = tt.gather %src[%indices] {axis = 1 : i32, efficient_layout} : (tensor<4x8xf32>, tensor<4x4xi32>) -> tensor<4x4xf32>
```

## 12. 调试操作

### 12.1 tt.print

设备端打印操作，用于调试。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.print` |
| 属性 | `$prefix`: `StrAttr`, `$hex`: `BoolAttr`, `$isSigned`: `DenseI32ArrayAttr` |
| 输入 | `$args`: `Variadic<TT_Type>` |
| 内存效果 | `MemWrite<GlobalMemory>` |

```mlir
tt.print "value:" hex : %arg0 : tensor<128xf32>
```

### 12.2 tt.assert

设备端断言操作。条件为假时打印消息并中止程序。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.assert` |
| 输入 | `$condition`: `I1 \| I1Tensor` |
| 属性 | `$message`: `StrAttr` |
| 内存效果 | `MemWrite<GlobalMemory>` |

```mlir
tt.assert %cond, "assertion failed" : i1
```

## 13. 张量指针与描述符操作

### 13.1 tt.make_tensor_ptr

构造张量指针，包含父张量的元信息和块张量信息。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.make_tensor_ptr` |
| 输入 | `$base`: `TT_Ptr`, `$shape`: `Variadic<I64>`, `$strides`: `Variadic<I64>`, `$offsets`: `Variadic<I32>` |
| 属性 | `$order`: `DenseI32ArrayAttr` |
| 输出 | `$result`: `TT_TensorPtr` |
| Traits | `Pure`, `SameVariadicOperandSize` |

```mlir
%ptr = tt.make_tensor_ptr %base, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1] {order = [1, 0]} : !tt.ptr<tensor<8x8xf16>>
```

### 13.2 tt.make_tensor_descriptor

构造张量描述符，用于 TMA 操作。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.make_tensor_descriptor` |
| 输入 | `$base`: `TT_Ptr`, `$shape`: `Variadic<I32>`, `$strides`: `Variadic<I64>` |
| 属性 | `$padding`: `TT_PaddingOptionAttr` (默认 `PAD_ZERO`) |
| 输出 | `$result`: `TT_TensorDescType` |
| Traits | `Pure`, `SameVariadicOperandSize` |

```mlir
%desc = tt.make_tensor_descriptor %base, [%shape0, %shape1], [%stride0, %stride1] : !tt.ptr<f16>, !tt.tensordesc<tensor<8x8xf16>>
```

### 13.3 tt.descriptor_load

基于张量描述符加载数据（TMA Load）。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.descriptor_load` |
| 输入 | `$desc`: `TT_TensorDescType` (MemRead), `$indices`: `Variadic<I32>` |
| 属性 | `$cache`: `TT_CacheModifierAttr` (默认 `NONE`), `$evict`: `TT_EvictionPolicyAttr` (默认 `NORMAL`) |
| 输出 | `$result`: `TT_Tensor` |

### 13.4 tt.descriptor_store

基于张量描述符存储数据（TMA Store）。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.descriptor_store` |
| 输入 | `$desc`: `TT_TensorDescType` (MemRead+MemWrite), `$src`: `TT_Tensor`, `$indices`: `Variadic<I32>` |

### 13.5 tt.descriptor_reduce

基于描述符的归约存储操作。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.descriptor_reduce` |
| 属性 | `$kind`: `TT_DescriptorReduceKindAttr` (ADD/MIN/MAX/INC/DEC/AND/OR/XOR) |
| 输入 | `$desc`: `TT_TensorDescType`, `$src`: `TT_Tensor`, `$indices`: `Variadic<I32>` |

### 13.6 tt.descriptor_gather

基于描述符的多行收集操作。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.descriptor_gather` |
| 输入 | `$desc`: `TT_TensorDescType`, `$x_offsets`: `RankedTensorOf<[I32]>`, `$y_offset`: `I32` |
| 输出 | `$result`: `TT_Tensor` |

### 13.7 tt.descriptor_scatter

基于描述符的多行散射操作。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.descriptor_scatter` |
| 输入 | `$desc`: `TT_TensorDescType`, `$x_offsets`: `RankedTensorOf<[I32]>`, `$y_offset`: `I32`, `$src`: `TT_Tensor` |

## 14. 函数操作

### 14.1 tt.func

Triton 函数定义。与 MLIR `func.func` 类似，但使用 Triton 方言命名空间。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.func` |
| 属性 | `$sym_name`: `SymbolNameAttr`, `$function_type`: `TypeAttrOf<FunctionType>`, `$sym_visibility`: `StrAttr` (可选) |
| 区域 | `$body`: `AnyRegion` |
| Traits | `AffineScope`, `AutomaticAllocationScope`, `CallableOpInterface`, `FunctionOpInterface`, `IsolatedFromAbove`, `HasParent<"ModuleOp">` |

### 14.2 tt.call

函数调用操作。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.call` |
| 属性 | `$callee`: `FlatSymbolRefAttr` |
| 输入 | `$operands`: `Variadic<AnyType>` |
| 输出 | `Variadic<AnyType>` |
| Traits | `CallOpInterface`, `SymbolUserOpInterface` |

### 14.3 tt.return

函数返回操作。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.return` |
| 输入 | `$srcs`: `Variadic<AnyType>` |
| Traits | `Pure`, `HasParent<"FuncOp">`, `ReturnLike`, `Terminator` |

## 15. 外部调用操作

### 15.1 tt.extern_elementwise

调用外部函数。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.extern_elementwise` |
| 输入 | `$srcs`: `Variadic<TT_Type>` |
| 属性 | `$libname`: `StrAttr`, `$libpath`: `StrAttr`, `$symbol`: `StrAttr`, `$pure`: `BoolAttr` |
| 输出 | `$result`: `TT_Type` |
| Traits | `Elementwise`, `SameOperandsAndResultEncoding`, `ConditionallySpeculatable` |

### 15.2 tt.elementwise_inline_asm

内联汇编操作，对打包的元素组执行逐元素操作。

| 项目 | 内容 |
|------|------|
| 操作名 | `tt.elementwise_inline_asm` |
| 属性 | `$asm_string`: `StrAttr`, `$constraints`: `StrAttr`, `$pure`: `BoolAttr`, `$packed_element`: `I32Attr` |
| 输入 | `$args`: `Variadic<TT_Type>` |
| 输出 | `$result`: `Variadic<TT_Type>` |
| Traits | `Elementwise`, `SameOperandsAndResultEncoding` |
