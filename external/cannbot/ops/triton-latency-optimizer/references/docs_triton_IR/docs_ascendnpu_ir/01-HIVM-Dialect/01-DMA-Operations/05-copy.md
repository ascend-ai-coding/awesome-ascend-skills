# hir.copy

> 关键词：Copy、DMA、本地搬运、UB 到 UB、Padding

## 概述

`hir.copy` 是 HIVM 方言中的 DMA 操作，用于在本地内存层级之间拷贝数据。与 `hir.load` 和 `hir.store` 不同，`hir.copy` 的 Pipeline 归属是动态的，取决于源和目标地址空间的组合。

当前支持的拷贝通路包括：
- UB 到 UB（Vector Pipeline）
- GM 到 L1（MTE2 Pipeline）
- UB 到 L1（MTE3 Pipeline，仅 Ascend950 系列）

`hir.copy` 支持随路 Padding 功能，可以在拷贝过程中填充目标缓冲区的多余位置。

> Python API 对应：无直接对应，由编译器在 bufferization 阶段自动生成

## IR 操作定义

### TableGen 定义

来源：[HIVMDMAOps.td:L192-L244](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L192-L244)

```tablegen
def CopyOp : HIVM_DmaOp<"copy", [
  SinglePipeOpTrait, StaticMaxRankTrait<3>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
  UniformReassociationFlattenTrait,
  DeclareOpInterfaceMethods<FlattenInterface, ["getLimitedAxes"]>,
  OperElemTypeConstraints<[0], [I1, I8, UI8, I16, UI16, F16, BF16,
                                I32, UI32, F32, UI64, I64, F8E4M3FN, F8E5M2]>,
  DeclareOpInterfaceMethods<HIVMStructuredOpInterface, ["getIndexingMaps"]>,
]> {
  let summary = "HIVM data copy operation";
  let description = [{
    Copy the data between local memory hierarchies.
    Currently support:
      - UB to UB
      - UB to L1 (for Ascend950 series)

    Examples:
    ```mlir
    hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
    ```

    Constraints:
    - `src` and `dst` are expected to have the same element type.
    - If `pad_mode` is not set, `src` and `dst` shape should be the same.
    - Only support left padding.
    - `pad_value` should have the same element type as `src` and `dst`.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       OptionalAttr<HIVM_PadModeAttr>:$pad_mode,
                       Optional<AnyType>:$pad_value
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let builders = [
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst)>
  ];
  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`pad_mode` `=` $pad_mode^)?
    (`pad_value` `=` $pad_value^ `:` type($pad_value))?
    (`->` type($result_tensor)^)?
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = DmaOpBaseDecl # [{
    // Declare functions necessary for SinglePipeOpTrait.
    PIPE getPipe();
  }];
}
```

### MLIR 语法

```mlir
hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>)
               outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)

hivm.hir.copy ins(%src : tensor<16x16xf32>)
               outs(%dst : tensor<16x16xf32>)
               -> tensor<16x16xf32>
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| `src` | TensorOrMemref | 是 | 源数据缓冲区 | 取决于通路 |
| `dst` | TensorOrMemref | 是 | 目标数据缓冲区 | 取决于通路 |
| `pad_mode` | HIVM_PadModeAttr | 否 | 填充模式 | PadNull / PadFirstElem / PadValue |
| `pad_value` | AnyType | 否 | 填充值 | 类型须与 src/dst 元素类型相同 |

### 结果说明

| 结果 | 类型 | 说明 |
|------|------|------|
| `result_tensor` | Optional<AnyRankedTensor> | Tensor 语义下的结果张量 |

### 属性说明

| 属性 | 类型 | 默认值 | 说明 | 可选值 |
|------|------|--------|------|--------|
| `pad_mode` | HIVM_PadModeAttr | 无 | 填充模式 | `PadNull`(0)、`PadFirstElem`(1)、`PadValue`(2) |

### 数据类型约束

来源：`OperElemTypeConstraints<[0], [...]>`

| 支持的元素类型 | 说明 |
|---------------|------|
| `i1` | 布尔类型 |
| `i8`, `ui8` | 8 位整数 |
| `i16`, `ui16` | 16 位整数 |
| `f16` | 半精度浮点 |
| `bf16` | BFloat16 |
| `i32`, `ui32` | 32 位整数 |
| `f32` | 单精度浮点 |
| `ui64`, `i64` | 64 位整数 |
| `f8E4M3FN` | 8 位浮点 E4M3 格式 |
| `f8E5M2` | 8 位浮点 E5M2 格式 |

### 动态 Pipeline 归属

来源：[HIVMDMAOps.cpp:L607-L632](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L607-L632)

| 源地址空间 | 目标地址空间 | Pipeline |
|-----------|-------------|----------|
| UB | UB | `PIPE_V` |
| L0C | GM | `PIPE_FIX` |
| GM | L1 | `PIPE_MTE2` |
| UB | L1 | `PIPE_MTE3` |

## IR 示例

### UB 到 UB 拷贝

```mlir
func.func @hivm_memref_copy_ub_to_ub() {
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
  return
}
```

### UB 到 L1 拷贝（Ascend950）

```mlir
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @hivm_memref_copy_ub_to_l1() {
    %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
    %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>)
                  outs(%dst : memref<16x16xf16, #hivm.address_space<cbuf>>)
    return
  }
}
```

### Tensor 语义拷贝

```mlir
func.func @hivm_tensor_copy() -> tensor<16x16xf32> {
  %src = tensor.empty() : tensor<16x16xf32>
  %dst = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.copy ins(%src : tensor<16x16xf32>)
                        outs(%dst : tensor<16x16xf32>)
                        -> tensor<16x16xf32>
  return %res : tensor<16x16xf32>
}
```

## IR 层约束与验证

来源：[HIVMDMAOps.cpp:L552-L605](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L552-L605)

1. **元素类型一致性**：`src` 和 `dst` 的元素类型必须相同
2. **Rank 一致性**：`src` 和 `dst` 必须具有相同的 rank
3. **形状兼容性**：如果未设置 `pad_mode`，`src` 和 `dst` 的形状必须兼容
4. **PadValue 必要性**：如果 `pad_mode` 为 `PadValue`，则 `pad_value` 必须提供
5. **PadValue 类型一致性**：`pad_value` 的类型必须与 `dst` 的元素类型相同
6. **地址空间约束**（memref 语义下）：
   - 仅支持以下通路：UB->UB, GM->L1, UB->L1（Ascend950）
7. **Padding 限制**：仅支持左侧 Padding

### 支持的地址空间组合

来源：[HIVMDMAOps.cpp:L440-L448](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L440-L448)

| 组合 | 支持情况 |
|------|---------|
| UB -> UB | 所有型号 |
| GM -> L1 | 所有型号 |
| UB -> L1 | 仅 Ascend950 |

## 与其他 IR 操作的关系

### 与 load/store 的区别

| 操作 | 数据通路 | Pipeline | Padding | 布局转换 |
|------|---------|----------|---------|---------|
| `hir.load` | GM -> UB | MTE2 | 支持（左+右） | 无 |
| `hir.store` | UB -> GM | MTE3 | 不支持 | 无 |
| `hir.copy` | 本地 <-> 本地 | 动态 | 仅左侧 | 无 |

### 后续 Lowering

`hir.copy` 最终被 lowering 为库函数调用，函数名格式为：

```
copy_{src_space}_to_{dst_space}_{rank}d_{datatype}
```

## 常见问题

### Q: copy 和 load 的 Padding 有什么区别？

A: `hir.copy` 仅支持左侧 Padding，而 `hir.load` 同时支持左侧和右侧 Padding。这是因为 copy 操作的硬件实现在 Padding 方面的限制。

### Q: 什么时候使用 copy 而不是 load/store？

A: 当数据搬运不涉及 GM 时（如 UB 到 UB），应使用 `hir.copy`。当涉及 GM 时，应使用 `hir.load`（GM 到本地）或 `hir.store`（本地到 GM）。

### Q: 为什么 UB 到 L1 的拷贝仅支持 Ascend950？

A: 这是硬件约束。Ascend950 系列的 Vector 核心可以直接写入 L1 缓存，而其他型号不支持此功能。

## 相关文档

- DMA 操作总览：[00-overview.md](00-overview.md)
- hir.load：[01-load.md](01-load.md)
- hir.store：[02-store.md](02-store.md)
- 随路功能详解：[10-padding-quantization.md](10-padding-quantization.md)
- 源码参考：
  - [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td) - TableGen 定义
  - [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp) - 实现代码
