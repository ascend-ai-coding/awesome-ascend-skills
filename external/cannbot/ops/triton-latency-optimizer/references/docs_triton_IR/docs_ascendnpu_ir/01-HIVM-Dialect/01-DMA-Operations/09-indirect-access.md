# hir.indirect_load / hir.indirect_store

> 关键词：Indirect、间接访问、SIMT、Vector Pipeline、Mask

## 概述

`hir.indirect_load` 和 `hir.indirect_store` 是 HIVM 方言中的间接内存访问操作，用于按偏移张量从全局内存（GM）中加载数据或向 GM 存储数据。这两个操作使用 SIMT（Single Instruction Multiple Thread）模板执行，归属于 Vector Pipeline（PIPE_V），支持 1D 到 5D 的数据访问。

与 `hir.gather_load`/`hir.scatter_store` 类似，`indirect_load`/`indirect_store` 也支持非连续的按索引内存访问，但它们使用 Destination Style 接口，具有更结构化的操作语义。

> Python API 对应：无直接对应，由编译器在 lowering 阶段自动生成

## hir.indirect_load

### 概述

间接内存加载操作。从源内存缓冲区中按偏移张量指定的位置收集元素到目标张量，支持掩码和回退值。

### TableGen 定义

来源：[HIVMOps.td:L524-L592](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L524-L592)

```tablegen
def IndirectLoadOp : HIVM_Op<"indirect_load",[
                            StaticMaxRankTrait<5>,
                            DestinationStyleOpInterface,
                            OpPipeInterface,
                            SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_V">,
                            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                            DeclareOpInterfaceMethods<LibraryFunctionOpInterface>,
                            AttrSizedOperandSegments
                            ]> {
  let summary = [{
    Performs indirect memory loading with masking and fallback values.
  }];

  let description = [{
    This operation takes a source memory buffer and a tensor of offsets,
    and produces an output tensor by gathering elements from the source
    at the specified offset positions. The operation supports masking
    and provides fallback values for masked-out positions.
    This operation supports 1D-5D.

    For each position in the output tensor:
      1D:
          dst[i] = mask[i] ? src[offsets[i]] : other[i]
      2D:
          dst[i][j] = mask[i][j] ? src[offsets[i][j]] : other[i][j]
      3D:
          dst[i][j][k] = mask[i][j][k] ? src[offsets[i][j][k]] : other[i][j][k]
      4D:
          dst[i][j][k][l] = mask[i][j][k][l] ? src[offsets[i][j][k][l]] : other[i][j][k][l]
      5D:
          dst[i][j][k][l][m] = mask[i][j][k][l][m] ? src[offsets[i][j][k][l][m]] : other[i][j][k][l][m]

    Where:
      - src: source memory buffer to load from
      - offsets: indices specifying positions in the source buffer
      - mask: boolean mask controlling which elements to load
      - other: fallback values used when mask is false
      - dst: destination tensor specifying output shape and type

    This operation is useful for sparse data access patterns and
    gather operations with conditional loading semantics.
  }];

  let arguments = (ins AnyMemRef:$src,
                       TensorOrMemref:$offsets,
                       TensorOrMemref:$dst,
                       Optional<TensorOrMemref>:$mask,
                       Optional<TensorOrMemref>:$other
  );
  let results = (outs Optional<TensorOrMemref>:$result);

  let assemblyFormat = [{
      `ins` `(` $src `:` type($src) `,` $offsets `:` type($offsets)
      (`,` $mask^ `:` type($mask))?
      (`,` $other^ `:` type($other))? `)`
      `outs` `(` $dst `:` type($dst) `)`
      attr-dict
      (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
      static StringRef getOpName() { return "indirect_load"; }
      ::mlir::MutableOperandRange getDpsInitsMutable() {
          return getDstMutable();
      }
  }];

  let hasVerifier = 1;
}
```

### MLIR 语法

```mlir
hivm.hir.indirect_load ins(%src : memref<?xf32>, %offsets : tensor<16xi32>)
                       outs(%dst : tensor<16xf32>)

hivm.hir.indirect_load ins(%src : memref<?xf32>, %offsets : tensor<16xi32>, %mask : tensor<16xi1>, %other : tensor<16xf32>)
                       outs(%dst : tensor<16xf32>) -> tensor<16xf32>
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| `src` | AnyMemRef | 是 | 源内存缓冲区 | GM 地址空间 |
| `offsets` | TensorOrMemref | 是 | 偏移索引张量 | 指定加载位置 |
| `dst` | TensorOrMemref | 是 | 目标张量 | 指定输出形状和类型 |
| `mask` | TensorOrMemref | 否 | 布尔掩码张量 | 控制哪些位置加载 |
| `other` | TensorOrMemref | 否 | 掩码位置的回退值 | 与 dst 元素类型相同 |

### 结果说明

| 结果 | 类型 | 说明 |
|------|------|------|
| `result` | Optional<TensorOrMemref> | Tensor 语义下的结果张量 |

### 语义说明

```
dst[i] = mask[i] ? src[offsets[i]] : other[i]
```

当 mask 为 true 时，从 src 的 offsets[i] 位置加载数据；当 mask 为 false 时，使用 other[i] 作为回退值。

## hir.indirect_store

### 概述

间接内存存储操作。将源张量中的元素按偏移张量指定的位置写入目标 GM 缓冲区，支持掩码。

### TableGen 定义

来源：[HIVMOps.td:L598-L660](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L598-L660)

```tablegen
def IndirectStoreOp : HIVM_Op<"indirect_store",[
                            StaticMaxRankTrait<5>,
                            DestinationStyleOpInterface,
                            OpPipeInterface,
                            SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_V">,
                            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                            DeclareOpInterfaceMethods<LibraryFunctionOpInterface>,
                            ]> {
  let summary = [{
    Performs indirect memory storing with SIMT template.
  }];

  let description = [{
    This operation takes a source tensor and a tensor of offsets from UB,
    and stores elements from the source into the destination GM buffer
    at the specified offset positions with SIMT template. The operation
    supports masking to conditionally control which elements are stored.
    This operation supports 1D-5D.

    For each position in the source tensor:
      1D:
          if (mask[i]) dst[offsets[i]] = src[i]
      2D:
          if (mask[i][j]) dst[offsets[i][j]] = src[i][j]
      3D:
          if (mask[i][j][k]) dst[offsets[i][j][k]] = src[i][j][k]
      4D:
          if (mask[i][j][k][l]) dst[offsets[i][j][k][l]] = src[i][j][k][l]
      5D:
          if (mask[i][j][k][l][m]) dst[offsets[i][j][k][l][m]] = src[i][j][k][l][m]

    Where:
      - src: source tensor containing values to store from UB
      - offsets: indices specifying positions in the destination buffer
      - dst: destination memory buffer to store into GM
      - mask: optional boolean mask controlling which elements to store

    When no mask is provided, all elements from the source tensor are stored
    to the corresponding offset positions in the destination buffer.
  }];

  let arguments = (ins AnyMemRef:$dst,
                       TensorOrMemref:$offsets,
                       TensorOrMemref:$src,
                       Optional<TensorOrMemref>:$mask
  );

  let assemblyFormat = [{
      `ins` `(` $src `:` type($src) `,` $offsets `:` type($offsets)
      (`,` $mask^ `:` type($mask))? `)`
      `outs` `(` $dst `:` type($dst) `)`
      attr-dict
  }];

  let extraClassDeclaration = [{
      static StringRef getOpName() { return "indirect_store"; }
      ::mlir::MutableOperandRange getDpsInitsMutable() {
          return getDstMutable();
      }
  }];

  let hasVerifier = 1;
}
```

### MLIR 语法

```mlir
hivm.hir.indirect_store ins(%src : tensor<16xf32>, %offsets : tensor<16xi32>)
                        outs(%dst : memref<?xf32>)

hivm.hir.indirect_store ins(%src : tensor<16xf32>, %offsets : tensor<16xi32>, %mask : tensor<16xi1>)
                        outs(%dst : memref<?xf32>)
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| `dst` | AnyMemRef | 是 | 目标内存缓冲区 | GM 地址空间 |
| `offsets` | TensorOrMemref | 是 | 偏移索引张量 | 指定存储位置 |
| `src` | TensorOrMemref | 是 | 源数据张量 | UB 中的数据 |
| `mask` | TensorOrMemref | 否 | 布尔掩码张量 | 控制哪些位置存储 |

### 语义说明

```
if (mask[i]) dst[offsets[i]] = src[i]
```

当 mask 为 true 时，将 src[i] 存储到 dst 的 offsets[i] 位置。当无 mask 时，所有元素都被存储。

## IR 示例

### indirect_load 基础用法

```mlir
func.func @test_indirect_load(%src: memref<?xf32>, %offsets: tensor<16xi32>) -> tensor<16xf32> {
  %dst = tensor.empty() : tensor<16xf32>
  %result = hivm.hir.indirect_load ins(%src : memref<?xf32>, %offsets : tensor<16xi32>)
                                    outs(%dst : tensor<16xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}
```

### indirect_load 带 Mask 和 Other

```mlir
func.func @test_indirect_load_masked(%src: memref<?xf32>, %offsets: tensor<16xi32>, %mask: tensor<16xi1>, %other: tensor<16xf32>) -> tensor<16xf32> {
  %dst = tensor.empty() : tensor<16xf32>
  %result = hivm.hir.indirect_load ins(%src : memref<?xf32>, %offsets : tensor<16xi32>, %mask : tensor<16xi1>, %other : tensor<16xf32>)
                                    outs(%dst : tensor<16xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}
```

### indirect_store 基础用法

```mlir
func.func @test_indirect_store(%dst: memref<?xf32>, %offsets: tensor<16xi32>, %src: tensor<16xf32>) {
  hivm.hir.indirect_store ins(%src : tensor<16xf32>, %offsets : tensor<16xi32>)
                          outs(%dst : memref<?xf32>)
  return
}
```

### indirect_store 带 Mask

```mlir
func.func @test_indirect_store_masked(%dst: memref<?xf32>, %offsets: tensor<16xi32>, %src: tensor<16xf32>, %mask: tensor<16xi1>) {
  hivm.hir.indirect_store ins(%src : tensor<16xf32>, %offsets : tensor<16xi32>, %mask : tensor<16xi1>)
                          outs(%dst : memref<?xf32>)
  return
}
```

## IR 层约束与验证

### indirect_load 约束

1. **Rank 限制**：最大 rank 为 5（`StaticMaxRankTrait<5>`）
2. **Pipeline**：固定为 Vector Pipeline（`OpPipeTrait<"PIPE::PIPE_V">`）
3. **Destination Style**：实现 `DestinationStyleOpInterface`
4. **AttrSizedOperandSegments**：可变长度操作数段的大小由属性控制
5. **元素类型一致性**：`src`、`dst`、`other` 的元素类型必须一致

### indirect_store 约束

1. **Rank 限制**：最大 rank 为 5（`StaticMaxRankTrait<5>`）
2. **Pipeline**：固定为 Vector Pipeline（`OpPipeTrait<"PIPE::PIPE_V">`）
3. **Destination Style**：实现 `DestinationStyleOpInterface`
4. **元素类型一致性**：`src` 和 `dst` 的元素类型必须一致

## 与其他 IR 操作的关系

### indirect_load/store vs gather_load/scatter_store

| 特性 | indirect_load/store | gather_load/scatter_store |
|------|-------------------|--------------------------|
| Pipeline | PIPE_V | 无固定 |
| 接口风格 | Destination Style | 非 Destination Style |
| Rank 支持 | 1D-5D | 无限制 |
| 库函数 | 有（LibraryFunctionOpInterface） | 无 |
| 缓存属性 | 不支持 | 支持 cache/evict |
| 边界检查 | 不支持 | 支持 boundaryCheck |

### 典型转换关系

```
hir.gather_load --> hir.indirect_load (分解后)
hir.gather_load --> hir.custom name="__builtin_gather_load" (内置 Custom Op)
```

## 常见问题

### Q: indirect_load 和 gather_load 应该使用哪个？

A: `indirect_load` 是更结构化的操作，属于 Vector Pipeline，有对应的库函数实现。`gather_load` 是更通用的操作，支持缓存属性和边界检查。在编译流程中，`gather_load` 可能被分解为 `indirect_load` 或 Custom Op。

### Q: indirect_load/store 为什么属于 Vector Pipeline？

A: 因为间接内存访问使用 SIMT（Single Instruction Multiple Thread）模板执行，每个线程独立处理一个索引位置，这是 Vector 核心的执行模式。

### Q: 5D 限制是否足够？

A: 对于大多数深度学习场景，5D 已经足够覆盖常见的张量维度（batch, seq_len, head, height, width）。如果需要更高维度的间接访问，可能需要先 reshape 数据。

## 相关文档

- DMA 操作总览：[00-overview.md](00-overview.md)
- Gather/Scatter：[08-gather-scatter.md](08-gather-scatter.md)
- 源码参考：
  - [HIVMOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td) - TableGen 定义
