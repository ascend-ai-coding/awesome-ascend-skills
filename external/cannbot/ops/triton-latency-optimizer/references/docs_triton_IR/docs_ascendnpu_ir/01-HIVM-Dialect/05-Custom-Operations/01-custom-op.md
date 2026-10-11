# hir.custom — 自定义单 Pipe 操作

> 关键词：custom, CustomOp, Builtin, __builtin_gather_load, __builtin_index_select, SinglePipe

## 概述

`hir.custom` 是 HIVM 方言中的自定义单 Pipe 操作，允许用户在现有操作无法满足需求时编写自定义实现。该操作在单个 Pipeline 上执行，通过 `name` 属性标识操作类型。

Custom 操作适用于以下场景：
1. 现有操作无法实现所需功能
2. 现有操作可以实现但性能不佳
3. 需要私有操作

内置操作（name 以 `__builtin` 开头）由编译器自带实现，用户无需指定属性即可使用。

> Python API 对应：Triton 的自定义操作可以通过 `tl.extern` 等机制映射为 hir.custom。

## IR 操作定义

从 [HIVMOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L1069-L1103) 提取：

```
def CustomOp : HIVM_CustomOp<"custom", [SinglePipeOpTrait]> {
  let arguments = (ins StrAttr:$name,
                       Variadic<AnyType>:$inputs,
                       Variadic<AnyType>:$outputs);
  let results = (outs Variadic<AnyType>:$results);
}
```

基类 `HIVM_CustomOp` 定义（[HIVMOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L904-L967)）：

```
class HIVM_CustomOp<string mnemonic, list<Trait> traits = []>
    : HIVM_StructuredOp<mnemonic,
                        !listconcat([AttrSizedOperandSegments,
                                     MemoryEffects<[MemRead, MemWrite]>,
                                     HIVMInferCoreTypeInterface],
                                    traits)> {
  dag args = (ins StrAttr:$name, Variadic<AnyType>:$inputs,
      Variadic<AnyType>:$outputs);
  dag res = (outs Variadic<AnyType>:$results);
}
```

## 参数说明

### 输入操作数（ins）

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$name` | StrAttr | 是 | 操作名称，`__builtin_` 前缀为内置操作 |
| `$inputs` | Variadic\<AnyType\> | 是 | 输入参数（可变数量） |
| `$outputs` | Variadic\<AnyType\> | 是 | 输出/初始化参数（DestinationStyle） |

### 输出结果（outs）

| 参数 | 类型 | 说明 |
|------|------|------|
| `$results` | Variadic\<AnyType\> | 结果值 |

### 必需属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `hivm.tcore_type` | TCoreTypeAttr | 执行的 Core 类型 |
| `hivm.pipe` | PipeAttr | 执行的 Pipeline |
| `hivm.vf_mode` | VFModeAttr | Vector 运行模式（Cube Core 时忽略） |

### 可选属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `gm_addr_args_indices` | DenseI32ArrayAttr | 指示哪些参数包含 GM 纯地址 |

### 额外类方法

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getCoreType()` | `optional<TCoreType>` | 获取 Core 类型 |
| `setCoreType(TCoreType)` | void | 设置 Core 类型 |
| `getPipe()` | PIPE | 获取 Pipe |
| `setPipe(PIPE)` | void | 设置 Pipe |
| `getVFMode()` | `optional<VFMode>` | 获取 VF 模式 |
| `setVFMode(VFMode)` | void | 设置 VF 模式 |
| `getGMAddrArgsIndices()` | `optional<SmallVector<size_t>>` | 获取 GM 地址参数索引 |
| `setGMAddrArgsIndices(SmallVector<size_t>)` | void | 设置 GM 地址参数索引 |
| `isBuiltin()` | bool | 判断是否为内置操作 |
| `getDpsInitsMutable()` | MutableOperandRange | DestinationStyleOpInterface |

### 内置操作常量

```cpp
static constexpr StringLiteral kBuiltinGatherLoadName = "__builtin_gather_load";
static constexpr StringLiteral kBuiltinIndexSelectName = "__builtin_index_select";
```

## IR 示例

### 自定义操作

```mlir
%empty = tensor.empty() : tensor<3x3xf32>
%0 = hivm.hir.custom
     { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD> }
     "my_custom_op"
     ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
         : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
     outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
```

### 内置操作（无需指定属性）

```mlir
%empty = tensor.empty() : tensor<3x3xf32>
%0 = hivm.hir.custom
     "__builtin_gather_load"
     ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
         : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
     outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
```

### __builtin_index_select

```mlir
%0 = tensor.empty() : tensor<1x4x32xf32>
%1 = hivm.hir.custom
       {extra_attr = "srcStrideLength=3", hivm.vf_mode = #hivm.vf_mode<SIMT>}
       "__builtin_index_select"
       ins(%arg0, %arg1, %c0_i32, %c9000_i64, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c32_i32, %c4000_i32, %c32_i32
       : memref<?xf32>, tensor<16x400xi32>, i32, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32)
       outs(%0 : tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
```

## IR 层约束与验证

1. **SinglePipeOpTrait**：操作在单个 Pipeline 上执行。
2. **AttrSizedOperandSegments**：inputs 和 outputs 的数量由属性决定。
3. **DestinationStyleOpInterface**：outputs 作为 DPS init 操作数。
4. **Core Type**：必须通过 `hivm.tcore_type` 属性指定（内置操作可省略，编译器自动推断）。
5. **Pipe**：必须通过 `hivm.pipe` 属性指定（内置操作可省略）。
6. **VFMode**：必须通过 `hivm.vf_mode` 属性指定（内置操作可省略，Cube Core 时忽略）。
7. **Builtin 验证**：内置操作编译器会检查参数正确性并规范化属性。

## 常见问题

**Q: 自定义操作如何提供实现？**
A: 当前自定义操作的用户实现机制尚未完全开放（标记为 TODO）。内置操作由编译器自动链接到模板库。

**Q: __builtin_gather_load 和 hir.gather_load 的区别？**
A: `__builtin_gather_load` 是通过 custom 操作机制实现的 gather load，而 `hir.gather_load` 是独立的 HIVM 操作。两者功能类似，但 custom 版本更灵活，可以通过属性配置。

**Q: gm_addr_args_indices 的用途？**
A: 指示哪些输入参数包含 GM 的纯地址值。Lowering Pass 在处理这些参数时会保留地址值，而不是尝试从 memref 中提取数据。

## 相关文档

- 源码参考：[HIVMOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L1069-L1103)
- 测试用例：[custom-op.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/custom-op.mlir)
- CustomMacroOp：[02-custom-macro-op.md](02-custom-macro-op.md)
