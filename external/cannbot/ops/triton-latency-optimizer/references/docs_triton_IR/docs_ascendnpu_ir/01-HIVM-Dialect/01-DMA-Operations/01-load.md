# hir.load

> 关键词：Load、DMA、MTE2、GM 到 UB、Padding、Eviction Policy

## 概述

`hir.load` 是 HIVM 方言中的核心 DMA 操作，用于将数据从全局内存（GM）加载到本地缓冲区（目前仅支持加载到统一缓冲区 UB）。该操作映射到硬件的 MTE2 Pipeline，是大多数内核中最频繁执行的 DMA 操作。

`hir.load` 支持丰富的随路功能，包括 Padding（填充）、Eviction Policy（驱逐策略）和缓冲区初始化，使其能够处理边界不齐、数据对齐等常见场景。

> Python API 对应：tl.load -- 详见 [docs_triton_ascend/02-Core-API/01-memory-ops.md](../../../docs_triton_ascend/02-Core-API/01-memory-ops.md)

## IR 操作定义

### TableGen 定义

来源：[HIVMDMAOps.td:L53-L136](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L53-L136)

```tablegen
def LoadOp : HIVM_DmaOp<"load", [
  AttrSizedOperandSegments, StaticMaxRankTrait<3>,
  SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE2">,
  DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
  UniformReassociationFlattenTrait,
  DeclareOpInterfaceMethods<FlattenInterface, ["getLimitedAxes"]>,
  OperElemTypeConstraints<[0], [I8, UI8, I16, UI16, F16, BF16,
                                I32, UI32, F32, UI64, I64, F8E4M3FN, F8E5M2]>,
  DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
    ["decomposeOperation", "getDecomposePhase"]>,
  DeclareOpInterfaceMethods<HIVMStructuredOpInterface, ["getIndexingMaps"]>,
]> {
  let summary = "HIVM data load operation";
  let description = [{
    Loads the data from the global memory to the local buffer.
    Currently only support loading to the unified buffer.

    Examples:
    ```mlir
    hivm.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
    ```

    Constraints:
    - `src` and `dst` are expected to have the same element type.
    - If `pad_mode` is not set, `src` and `dst` shape should be the same.
    - Supports both left and right padding.
    - `pad_value` should have the same element type as `src` and `dst`.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       OptionalAttr<HIVM_PadModeAttr>:$pad_mode,
                       Optional<AnyType>:$pad_value,
                       Optional<Index>:$left_padding_num,
                       Optional<AnyType>:$right_padding_num,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$init_out_buffer,
                       Optional<AnyType>:$init_condition,
                       OptionalAttr<HIVM_EvictionPolicyAttr>:$eviction_policy
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`pad_mode` `=` $pad_mode^)?
    (`pad_value` `=` $pad_value^ `:` type($pad_value))?
    (`left_padding_num` `=` $left_padding_num^ `:` type($left_padding_num))?
    (`init_out_buffer` `=` $init_out_buffer^ )?
    (`right_padding_num` `=` $right_padding_num^ `:` type($right_padding_num))?
    (`init_condition` `=` $init_condition^ `:` type($init_condition))?
    (`eviction_policy` `=` $eviction_policy^)?
    (`->` type($result_tensor)^)?
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = DmaOpBaseDecl;
}
```

### MLIR 语法

```mlir
hivm.hir.load ins(%src : memref<MxNxf16, #hivm.address_space<gm>>)
               outs(%dst : memref<MxNxf16, #hivm.address_space<ub>>)

hivm.hir.load ins(%src : tensor<MxNxf16>)
               outs(%dst : tensor<MxNxf16>)
               pad_mode = #hivm.padmode<PadValue>
               pad_value = %val : f16
               -> tensor<MxNxf16>
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| `src` | TensorOrMemref | 是 | 源数据缓冲区 | 必须为 GM 地址空间（memref 语义下） |
| `dst` | TensorOrMemref | 是 | 目标数据缓冲区 | 必须为 UB 地址空间（memref 语义下） |
| `pad_mode` | HIVM_PadModeAttr | 否 | 填充模式 | PadNull / PadFirstElem / PadValue |
| `pad_value` | AnyType | 否 | 填充值 | 类型须与 src/dst 元素类型相同 |
| `left_padding_num` | Index | 否 | 左侧填充元素数 | 仅在 pad_mode 设置时有意义 |
| `right_padding_num` | AnyType | 否 | 右侧填充元素数 | 仅在 pad_mode 设置时有意义 |
| `init_out_buffer` | BoolAttr | 否 | 是否初始化输出缓冲区 | 默认 false |
| `init_condition` | AnyType | 否 | 初始化条件 | 用于条件化初始化 |
| `eviction_policy` | HIVM_EvictionPolicyAttr | 否 | 缓存驱逐策略 | EvictFirst / EvictLast |

### 结果说明

| 结果 | 类型 | 说明 |
|------|------|------|
| `result_tensor` | Optional<AnyRankedTensor> | Tensor 语义下的结果张量 |

### 属性说明

| 属性 | 类型 | 默认值 | 说明 | 可选值 |
|------|------|--------|------|--------|
| `pad_mode` | HIVM_PadModeAttr | 无 | 填充模式 | `PadNull`(0)、`PadFirstElem`(1)、`PadValue`(2) |
| `init_out_buffer` | BoolAttr | `false` | 是否在加载前初始化输出缓冲区 | `true` / `false` |
| `eviction_policy` | HIVM_EvictionPolicyAttr | 无 | 数据缓存驱逐策略 | `EvictFirst`(0)、`EvictLast`(1) |

### 数据类型约束

来源：`OperElemTypeConstraints<[0], [...]>`，约束操作数索引 0（src）的元素类型。

| 支持的元素类型 | 说明 |
|---------------|------|
| `i8`, `ui8` | 8 位整数 |
| `i16`, `ui16` | 16 位整数 |
| `f16` | 半精度浮点 |
| `bf16` | BFloat16 |
| `i32`, `ui32` | 32 位整数 |
| `f32` | 单精度浮点 |
| `ui64`, `i64` | 64 位整数 |
| `f8E4M3FN` | 8 位浮点 E4M3 格式 |
| `f8E5M2` | 8 位浮点 E5M2 格式 |

### PadMode 枚举

定义于 [HIVMAttrs.td:L330-L349](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L330-L349)

| 枚举值 | 数值 | IR 字面量 | 说明 |
|--------|------|----------|------|
| `PadNull` | 0 | `PadNull` | 不填充 |
| `PadFirstElem` | 1 | `PadFirstElem` | 使用第一个元素填充 |
| `PadValue` | 2 | `PadValue` | 使用指定值填充 |

### EvictionPolicy 枚举

定义于 [HIVMAttrs.td:L356-L372](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L356-L372)

| 枚举值 | 数值 | IR 字面量 | 说明 |
|--------|------|----------|------|
| `EvictFirst` | 0 | `EvictFirst` | 优先驱逐 |
| `EvictLast` | 1 | `EvictLast` | 最后驱逐 |

## IR 示例

### 基础加载

最简单的 GM 到 UB 数据加载，源和目标形状相同：

```mlir
func.func @hivm_memref_load_gm_to_ub() {
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
  return
}
```

### 带 Padding 的加载

当源数据形状小于目标缓冲区时，使用 Padding 填充多余位置：

```mlir
func.func @hivm_memref_copy_gm_to_ub_pad_value() {
  %val = arith.constant 10.0 : f16
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
                pad_mode = #hivm.padmode<PadValue>
                pad_value = %val : f16
  return
}
```

使用 PadFirstElem 模式（用第一个元素填充）：

```mlir
func.func @hivm_memref_copy_gm_to_ub_pad_first() {
  %src = memref.alloc() : memref<16x15xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<16x15xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
                pad_mode = #hivm.padmode<PadFirstElem>
  return
}
```

带左侧填充数和仅指定 pad_value（自动推断 PadValue 模式）：

```mlir
func.func @hivm_memref_copy_gm_to_ub_pad_value_only() {
  %val = arith.constant 10.0 : f16
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
                pad_value = %val : f16
  return
}
```

### 带 Eviction Policy 的加载

```mlir
func.func @hivm_memref_load_with_eviction() {
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
                eviction_policy = #hivm.evictionpolicy<EvictLast>
  return
}
```

### Tensor 语义

```mlir
func.func @hivm_tensor_load() -> tensor<16x16xf32> {
  %src = tensor.empty() : tensor<16x16xf32>
  %dst = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.load ins(%src : tensor<16x16xf32>)
                        outs(%dst : tensor<16x16xf32>)
                        -> tensor<16x16xf32>
  return %res : tensor<16x16xf32>
}
```

## IR 层约束与验证

来源：[HIVMDMAOps.cpp:L219-L272](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L219-L272)

`hir.load` 的验证器执行以下检查：

1. **元素类型一致性**：`src` 和 `dst` 的元素类型必须相同
2. **Rank 一致性**：`src` 和 `dst` 必须具有相同的 rank
3. **形状兼容性**：如果未设置 `pad_mode`，`src` 和 `dst` 的形状必须兼容（相同）
4. **PadValue 必要性**：如果 `pad_mode` 为 `PadValue`，则 `pad_value` 必须提供
5. **PadValue 类型一致性**：`pad_value` 的类型必须与 `dst` 的元素类型相同
6. **地址空间约束**（memref 语义下）：
   - `src` 必须为 GM 地址空间
   - `dst` 不能为 GM 地址空间
7. **Tensor 语义约束**：
   - `result_tensor` 的元素类型必须与 `dst` 相同
   - `result_tensor` 的 rank 必须与 `dst` 相同
   - 如果未设置 `pad_mode`，`result_tensor` 的形状必须与 `dst` 兼容

### 验证错误示例

```
error: element types of dst and src should be the same!
error: src and dst should have the same dimensions!
error: if pad_mode is not set, src and dst shape should be the same!
error: if padmode is PadValue, pad_value is required!
error: dtype of pad_value and element type of dst/src should be the same!
error: only support src == gm and dst != gm currently!
```

## 与其他 IR 操作的关系

### 从 Triton 到 HIVM

```
tt.load  -->  hivm.hir.load
                (带 padding 参数映射)
```

Triton 的 `tt.load` 中的 `padding_option` 和 `other` 参数映射到 HIVM 的 `pad_mode` 和 `pad_value`。

### 从 HFusion 到 HIVM

HFusion 方言中的加载操作在 lowering 到 HIVM 时，会根据目标地址空间和布局需求生成 `hir.load` 或 `hir.nd2nz`。

### 后续 Lowering

`hir.load` 最终被 lowering 为库函数调用，函数名格式为：

```
load_gm_to_ubuf_{rank}d_{datatype}
```

## 常见问题

### Q: 为什么 load 只支持 GM -> UB？

A: 这是当前硬件的限制。如果需要将数据加载到 L1，应使用 `hir.nd2nz`；如果需要在本地层级间搬运，应使用 `hir.copy`。

### Q: pad_mode 和 pad_value 的关系是什么？

A: `pad_mode` 决定填充策略。如果仅指定 `pad_value` 而不指定 `pad_mode`，编译器会自动推断为 `PadValue` 模式。如果 `pad_mode` 为 `PadFirstElem`，则不需要 `pad_value`。

### Q: init_out_buffer 的作用是什么？

A: 当设置为 `true` 时，会在加载数据之前先将目标缓冲区初始化为零。这在某些需要确保缓冲区干净的场景下有用，例如当加载的数据不能完全覆盖目标缓冲区时。

### Q: eviction_policy 如何选择？

A: `EvictFirst` 表示该数据在缓存中优先被驱逐，适用于只使用一次的数据；`EvictLast` 表示该数据尽量保留在缓存中，适用于会被反复访问的数据。

## 相关文档

- Python API：[docs_triton_ascend/02-Core-API/01-memory-ops.md](../../../docs_triton_ascend/02-Core-API/01-memory-ops.md)
- DMA 操作总览：[00-overview.md](00-overview.md)
- 随路功能详解：[10-padding-quantization.md](10-padding-quantization.md)
- 源码参考：
  - [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td) - TableGen 定义
  - [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp) - 验证逻辑实现
  - [dma-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/dma-ops.mlir) - IR 测试用例
