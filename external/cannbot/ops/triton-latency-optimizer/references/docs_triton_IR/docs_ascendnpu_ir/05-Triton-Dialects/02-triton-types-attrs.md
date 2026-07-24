# Triton 类型与属性详解

本文档详细描述 Triton 方言中定义的所有类型和属性。所有定义均从 TableGen 源码精确提取。

源码参考：
- [TritonTypes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Triton/IR/TritonTypes.td)
- [TritonAttrDefs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)

## 1. 核心类型

### 1.1 PointerType (`!tt.ptr`)

Triton 指针类型，可指向标量或张量。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$pointeeType` | `Type` | 指向的目标类型 |
| `$addressSpace` | `int` | 地址空间编号 |

```mlir
!tt.ptr<f32>                    ; 指向 f32 的指针
!tt.ptr<tensor<8x8xf16>>       ; 指向张量的指针（张量指针）
!tt.ptr<f32, 1>                ; 指向地址空间 1 的指针
```

PointerType 使用自定义汇编格式，构建器可从上下文推断：

```cpp
PointerType::get(Type pointeeType, int addressSpace)
```

### 1.2 TensorDescType (`!tt.tensordesc`)

张量描述符类型，是 NVIDIA TMA 描述符的可移植抽象。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$blockType` | `RankedTensorType` | 描述符对应的块张量类型 |

```mlir
!tt.tensordesc<tensor<8x8xf16>>
```

构建器支持有符号/无符号整数语义：

```cpp
TensorDescType::get(RankedTensorType blockType, bool isSigned)
```

额外方法：
- `getSignlessBlockType()`：返回无符号版本的 blockType
- `getTensorShape()`：返回块张量的形状

## 2. 类型约束别名

Triton 方言通过 TableGen 定义了一系列类型约束别名，用于操作签名中的类型匹配：

### 2.1 浮点类型约束

| 约束名 | 定义 | 说明 |
|--------|------|------|
| `TT_Float` | `AnyTypeOf<[F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64]>` | 所有支持的浮点类型 |
| `TT_FloatTensor` | `RankedTensorOf<[TT_Float]>` | 浮点张量 |
| `TT_FloatLike` | `AnyTypeOf<[TT_Float, TT_FloatTensor]>` | 浮点标量或张量 |

支持的浮点格式：

| 类型 | 位宽 | 说明 |
|------|------|------|
| `F8E4M3FN` | 8 | E4M3 格式（标准 FP8） |
| `F8E4M3FNUZ` | 8 | E4M3 格式（无偏移） |
| `F8E5M2` | 8 | E5M2 格式（标准 FP8） |
| `F8E5M2FNUZ` | 8 | E5M2 格式（无偏移） |
| `F16` | 16 | 半精度浮点 |
| `BF16` | 16 | BFloat16 |
| `F32` | 32 | 单精度浮点 |
| `F64` | 64 | 双精度浮点 |

### 2.2 整数类型约束

| 约束名 | 定义 | 说明 |
|--------|------|------|
| `I4` | `I<4>` | 4 位整数 |
| `TT_Int` | `AnyTypeOf<[I1, I4, I8, I16, I32, I64]>` | 所有支持的整数类型 |
| `TT_IntTensor` | `RankedTensorOf<[TT_Int]>` | 整数张量 |
| `TT_IntLike` | `AnyTypeOf<[TT_Int, TT_IntTensor]>` | 整数标量或张量 |

### 2.3 布尔类型约束

| 约束名 | 定义 | 说明 |
|--------|------|------|
| `TT_BoolTensor` | `RankedTensorOf<[I1]>` | 布尔张量 |
| `TT_BoolLike` | `AnyTypeOf<[I1, TT_BoolTensor]>` | 布尔标量或张量 |

### 2.4 指针类型约束

| 约束名 | 定义 | 说明 |
|--------|------|------|
| `TT_Ptr` | `TT_PtrOf<[AnyType]>` | 标量指针 `ptr<>` |
| `TT_PtrTensor` | `RankedTensorOf<[TT_Ptr]>` | 指针张量 `tensor<ptr<>>` |
| `TT_PtrLike` | `AnyTypeOf<[TT_Ptr, TT_PtrTensor]>` | 指针或指针张量 |
| `TT_TensorPtr` | `TT_PtrOf<[TT_Tensor]>` | 张量指针 `ptr<tensor<>>` |

### 2.5 复合类型约束

| 约束名 | 定义 | 说明 |
|--------|------|------|
| `TT_FpIntTensor` | `RankedTensorOf<[TT_Float, TT_Int]>` | 浮点或整数张量 |
| `TT_Tensor` | `RankedTensorOf<[TT_Float, TT_Int, TT_Ptr]>` | 通用张量（含指针张量） |
| `TT_Type` | `AnyTypeOf<[TT_FloatLike, TT_IntLike, TT_PtrLike, TT_TensorPtr]>` | Triton 中任意类型 |
| `TT_I32Like` | `AnyTypeOf<[I32, I32Tensor]>` | i32 标量或张量 |
| `TT_I64Like` | `AnyTypeOf<[I64, I64Tensor]>` | i64 标量或张量 |

## 3. 枚举属性

### 3.1 CacheModifier

控制加载/存储操作的缓存行为。

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `NONE` | 1 | `none` | 无缓存修饰 |
| `CA` | 2 | `ca` | 缓存所有级别（Cache All） |
| `CG` | 3 | `cg` | 缓存全局级别（Cache Global） |
| `WB` | 4 | `wb` | 写回缓存（Write-Back） |
| `CS` | 5 | `cs` | 流式缓存（Cache Streaming） |
| `WT` | 6 | `wt` | 写穿透缓存（Write-Through） |
| `CV` | 7 | `cv` | 易失性缓存（Cache Volatile） |

C++ 命名空间：`::mlir::triton`

### 3.2 EvictionPolicy

控制缓存行的驱逐优先级。

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `NORMAL` | 1 | `evict_normal` | 正常驱逐 |
| `EVICT_FIRST` | 2 | `evict_first` | 优先驱逐 |
| `EVICT_LAST` | 3 | `evict_last` | 最后驱逐 |

### 3.3 PaddingOption

控制张量指针加载时越界元素的填充方式。

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `PAD_ZERO` | 1 | `zero` | 用零填充 |
| `PAD_NAN` | 2 | `nan` | 用 NaN 填充 |

### 3.4 RMWOp

原子读-改-写操作的类型。

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `AND` | 1 | `and` | 按位与 |
| `OR` | 2 | `or` | 按位或 |
| `XOR` | 3 | `xor` | 按位异或 |
| `ADD` | 4 | `add` | 整数加法 |
| `FADD` | 5 | `fadd` | 浮点加法 |
| `MAX` | 6 | `max` | 有符号最大值 |
| `MIN` | 7 | `min` | 有符号最小值 |
| `UMAX` | 8 | `umax` | 无符号最大值 |
| `UMIN` | 9 | `umin` | 无符号最小值 |
| `XCHG` | 10 | `exch` | 交换 |

### 3.5 DescriptorReduceKind

基于描述符的归约操作类型。

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `ADD` | 1 | `add` | 加法 |
| `MIN` | 2 | `min` | 最小值 |
| `MAX` | 3 | `max` | 最大值 |
| `INC` | 4 | `inc` | 递增 |
| `DEC` | 5 | `dec` | 递减 |
| `AND` | 6 | `and` | 按位与 |
| `OR` | 7 | `or` | 按位或 |
| `XOR` | 8 | `xor` | 按位异或 |

### 3.6 MemSemantic

原子操作的内存排序语义。

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `RELAXED` | 1 | `relaxed` | 宽松排序 |
| `ACQUIRE` | 2 | `acquire` | 获取语义 |
| `RELEASE` | 3 | `release` | 释放语义 |
| `ACQUIRE_RELEASE` | 4 | `acq_rel` | 获取-释放语义 |

### 3.7 MemSyncScope

原子操作的同步范围。

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `GPU` | 1 | `gpu` | GPU 级同步 |
| `CTA` | 2 | `cta` | CTA（线程块）级同步 |
| `SYSTEM` | 3 | `sys` | 系统级同步 |

### 3.8 ProgramIDDim

程序 ID 的维度标识。

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `X` | 0 | `x` | X 维度 |
| `Y` | 1 | `y` | Y 维度 |
| `Z` | 2 | `z` | Z 维度 |

### 3.9 RoundingMode

浮点转换的舍入模式。

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `RTZ` | 0 | `rtz` | 向零舍入（Round Toward Zero） |
| `RTNE` | 1 | `rtne` | 向最近偶数舍入（Round To Nearest Even） |

### 3.10 PropagateNan

NaN 传播策略，用于 `tt.clampf` 操作。

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `NONE` | 0 | `none` | 不传播 NaN |
| `ALL` | 0xFFFF | `all` | 传播所有 NaN |

### 3.11 InputPrecision

`tt.dot` 操作的输入精度控制，仅当输入为 f32 时有效。

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `TF32` | 0 | `tf32` | 使用 TF32 Tensor Core |
| `TF32x3` | 1 | `tf32x3` | 3xTF32 技巧实现高精度 |
| `IEEE` | 2 | `ieee` | IEEE 754 精确实现，不使用 Tensor Core |

### 3.12 ScaleDotElemType

`tt.dot_scaled` 操作的缩放元素类型。

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `E4M3` | 0 | `e4m3` | FP8 E4M3 格式 |
| `E5M2` | 1 | `e5m2` | FP8 E5M2 格式 |
| `E2M3` | 2 | `e2m3` | FP6 E2M3 格式 |
| `E3M2` | 3 | `e3m2` | FP6 E3M2 格式 |
| `E2M1` | 4 | `e2m1` | FP4 E2M1 格式 |
| `BF16` | 5 | `bf16` | BFloat16 |
| `FP16` | 6 | `fp16` | 半精度浮点 |

## 4. 属性在操作中的使用

### 4.1 CacheModifier 使用场景

| 操作 | 属性名 | 默认值 |
|------|--------|--------|
| `tt.load` | `$cache` | `NONE` |
| `tt.store` | `$cache` | `NONE` |
| `tt.descriptor_load` | `$cache` | `NONE` |
| `ttg.async_copy_global_to_local` | `$cache` | `NONE` |

### 4.2 EvictionPolicy 使用场景

| 操作 | 属性名 | 默认值 |
|------|--------|--------|
| `tt.load` | `$evict` | `NORMAL` |
| `tt.store` | `$evict` | `NORMAL` |
| `tt.descriptor_load` | `$evict` | `NORMAL` |
| `ttg.async_copy_global_to_local` | `$evict` | `NORMAL` |

### 4.3 PaddingOption 使用场景

| 操作 | 属性名 | 默认值 |
|------|--------|--------|
| `tt.load` | `$padding` | 可选（无默认） |
| `tt.make_tensor_descriptor` | `$padding` | `PAD_ZERO` |

### 4.4 MemSemantic / MemSyncScope 使用场景

| 操作 | 属性名 |
|------|--------|
| `tt.atomic_rmw` | `$sem`, `$scope` |
| `tt.atomic_cas` | `$sem`, `$scope` |

## 5. 类型系统设计要点

### 5.1 指针类型的双重角色

Triton 指针类型有两种使用模式：

1. **标量指针** (`!tt.ptr<elementType>`)：指向单个值或一维连续内存
2. **张量指针** (`!tt.ptr<tensor<shape, elementType>>`)：指向多维张量，附带形状和步长信息

`TT_PtrLike` 约束同时接受标量指针和指针张量，使得 `tt.load`/`tt.store` 可以处理两种内存访问模式。

### 5.2 TensorDescType 与 PointerType 的区别

| 特性 | `!tt.ptr<tensor<...>>` | `!tt.tensordesc<tensor<...>>` |
|------|------------------------|-------------------------------|
| 创建方式 | `tt.make_tensor_ptr` | `tt.make_tensor_descriptor` |
| 偏移推进 | `tt.advance` | 索引参数 |
| 加载方式 | `tt.load` | `tt.descriptor_load` |
| 存储方式 | `tt.store` | `tt.descriptor_store` |
| 硬件映射 | 通用 | NVIDIA TMA 硬件加速 |
| 边界检查 | `boundaryCheck` 属性 | 描述符内置 |

### 5.3 TT_Type 的覆盖范围

`TT_Type = AnyTypeOf<[TT_FloatLike, TT_IntLike, TT_PtrLike, TT_TensorPtr]>` 覆盖了 Triton IR 中几乎所有数据类型，但不包括：
- `TT_BoolLike` 中的 `I1` 标量（不在 `TT_IntLike` 中的 `I1`... 实际上 `TT_Int` 包含 `I1`，所以 `TT_IntLike` 包含 `I1`）
- `TT_TensorDescType`（描述符类型独立于 `TT_Type`）
- `TTG_AsyncToken`（TritonGPU 方言的异步令牌类型）
