# HIVM 类型系统约束

> 关键词：MemRef, Tensor, Vector, Element Type, AddressSpace, DataLayout, ShapedType

## 概述

HIVM 在标准 MLIR 类型系统（MemRef、Tensor、Vector）上增加了特定约束，以表达 NPU 硬件的存储层次、数据布局和元素类型限制。这些约束通过 AddressSpace 属性、DataLayout 属性和元素类型限制来体现。

## MemRef 在 HIVM 中的约束

### AddressSpace 标注

HIVM 中的 MemRef 通常需要通过 `#hivm.address_space<...>` 属性标注其所在的存储层次：

```mlir
memref<16x16xf16, #hivm.address_space<gm>>     // 全局内存
memref<16x16xf16, #hivm.address_space<cbuf>>   // L1 缓存
memref<16x16xf16, #hivm.address_space<ca>>     // L0A 缓存
memref<16x16xf16, #hivm.address_space<cb>>     // L0B 缓存
memref<16x16xf16, #hivm.address_space<cc>>     // L0C 缓存
memref<16x16xf16, #hivm.address_space<ub>>     // 统一缓冲区
```

### 存储层次与操作约束

| 存储层次 | AddressSpace | 可用操作 | 说明 |
|---------|-------------|---------|------|
| GM | `gm` | load, store, matmul, mix_matmul | 全局内存，所有 Block 共享 |
| L1 | `cbuf` | nd2nz, copy, mmadL1 输入 | CBuffer，Block 内共享 |
| L0A | `ca` | mmadL1 内部使用 | Cube A 矩阵缓冲区 |
| L0B | `cb` | mmadL1 内部使用 | Cube B 矩阵缓冲区 |
| L0C | `cc` | fixpipe 输入 | Cube C 矩阵/累加器 |
| UB | `ub` | vector ops, fixpipe 输出 | 统一缓冲区，Vector Core 使用 |

### Strided MemRef

HIVM 支持 strided MemRef，用于表达子视图：

```mlir
memref<256x128xf16, strided<[2048, 1], offset: 0>>
memref<256x128xf16, strided<[2048, 1]>, #hivm.address_space<ub>>
```

## Tensor 在 HIVM 中的约束

### Tensor 语义

HIVM 中的 Tensor 遵循 MLIR 标准 Tensor 语义，但增加了以下约束：

1. **DestinationStyleOpInterface**：HIVM 操作通常实现 DPS 接口，Tensor 通过 `outs` 操作数传入
2. **RankedTensor**：HIVM 操作要求使用 RankedTensor（有确定形状的 Tensor）
3. **Tensor/MemRef 双语义**：大多数 HIVM 操作同时支持 Tensor 和 MemRef 语义

### Tensor 与 MemRef 的选择

| 场景 | 推荐类型 | 原因 |
|------|---------|------|
| 函数间传递数据 | MemRef + AddressSpace | 明确存储层次 |
| 操作间传递中间结果 | Tensor | 更安全的类型系统 |
| 需要子视图 | MemRef + subview | Tensor 使用 extract_slice |
| Bufferization 后 | MemRef | Bufferization 将 Tensor 转换为 MemRef |

## Vector 在 HIVM 中的约束

HIVM 中的 Vector 类型主要用于 Vector Core 上的计算操作。约束包括：

1. **元素类型限制**：Vector 的元素类型必须符合硬件支持的类型
2. **形状约束**：Vector 的形状必须符合 Vector Core 的计算单元大小

## 元素类型约束

### 支持的元素类型

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| f16 | 半精度浮点 | 矩阵乘法、通用计算 |
| bf16 | BFloat16 | 矩阵乘法、推理 |
| f32 | 单精度浮点 | 累加器、通用计算 |
| f64 | 双精度浮点 | 有限支持 |
| i8 | 8-bit 整数 | 量化计算 |
| i16 | 16-bit 整数 | 通用计算 |
| i32 | 32-bit 整数 | 通用计算、索引 |
| i64 | 64-bit 整数 | 地址、索引 |
| i1 | 布尔 | 条件、mask |

### 矩阵乘法元素类型组合

| A 类型 | B 类型 | C 类型 | 说明 |
|--------|--------|--------|------|
| f16 | f16 | f32 | 标准 FP16 矩阵乘，累加为 FP32 |
| f16 | f16 | f16 | FP16 矩阵乘，累加为 FP16 |
| bf16 | bf16 | f32 | BF16 矩阵乘 |
| i8 | i8 | i32 | INT8 量化矩阵乘 |

### 类型转换约束

HIVM 提供以下类型转换操作：

- **bitcast**：位模式不变，重新解释类型（要求总位宽相同）
- **cast_signed / cast_unsigned**：有符号/无符号类型转换
- **FixpipePreQuantMode**：Fixpipe 中的在线类型转换（F322F16、S322I8 等）

## 类型约束 Trait

HIVM 定义了以下类型约束相关的 Trait：

| Trait | 说明 |
|-------|------|
| `HIVMOpSameOperandsAndResultRank` | 所有操作数和结果的 rank 相同（除临时缓冲区） |
| `ElementwiseNaryOpTrait<N>` | N 元素逐元素操作，rank 相同 |
| `StaticMaxRankTrait<MaxRank>` | 静态已知最大 rank |
| `NoMaxRankTrait` | 无 rank 限制 |
| `VectorOnlyTrait<idx>` | 指定操作数只支持 Vector 类型 |
| `ScalarOnlyTrait<idx>` | 指定操作数只支持标量类型 |
| `OperElemTypeConstraints<indices, types>` | 指定操作数的元素类型约束 |

## 常见问题

**Q: 什么时候需要标注 AddressSpace？**
A: 当 MemRef 需要明确存储层次时（如 load/store 操作的 GM 参数），必须标注 AddressSpace。编译器在 Bufferization 后会自动推断和添加 AddressSpace。

**Q: Tensor 和 MemRef 可以混用吗？**
A: 大多数 HIVM 操作同时支持 Tensor 和 MemRef 语义。但在同一操作中，输入和输出通常使用相同的语义（全 Tensor 或全 MemRef）。

**Q: 为什么 L0C 的 AddressSpace 是 `cc`？**
A: L0C 是 Cube Core 的累加器缓冲区，`cc` 代表 Cube C 矩阵。类似地，`ca` 是 Cube A，`cb` 是 Cube B。

## 相关文档

- 源码参考：[HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- 源码参考：[HIVMTraits.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td)
- 测试用例：[ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/ops.mlir)
