# hir.nz2nd

> 关键词：NZ2ND、布局转换、MTE3、L1 到 GM、Fractal 到 ND

## 概述

`hir.nz2nd` 是 HIVM 方言中的 DMA 操作，用于将数据从 L1 缓存搬运到全局内存（GM），同时执行 NZ（NzFormat/Z-Fractals）到 ND（Normal Data）的布局转换。该操作映射到硬件的 MTE3 Pipeline。

`hir.nz2nd` 是 `hir.nd2nz` 的逆操作，通常用于将 Cube 计算产生的 NZ 格式中间结果从 L1 写回 GM，同时恢复为 ND 格式以便后续处理。

> Python API 对应：无直接对应，由编译器自动插入

## IR 操作定义

### TableGen 定义

来源：[HIVMDMAOps.td:L370-L386](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L370-L386)

```tablegen
def NZ2NDOp : HIVM_DmaOp<"nz2nd",
    [StaticMaxRankTrait<2>,
     SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE3">,
     HIVMCoreTypeInterface, CubeCoreTypeTrait
    ]> {
  let summary = "HIVM data copy operation from L1 to Global Memory with NZ2ND conversion";
  let description = [{ NZ2ND does data movement from L1 to OUT with NZ2ND conversion. }];
  let arguments = (ins TensorOrMemref:$src, TensorOrMemref:$dst);
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`->` type($result_tensor)^)?
  }];
  let extraClassDeclaration = DmaOpBaseDecl;
}
```

### MLIR 语法

```mlir
hivm.hir.nz2nd ins(%src : memref<MxNxf16, #hivm.address_space<cbuf>>)
                outs(%dst : memref<MxNxf16, #hivm.address_space<gm>>)
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| `src` | TensorOrMemref | 是 | 源数据缓冲区 | 必须为 L1 地址空间 |
| `dst` | TensorOrMemref | 是 | 目标数据缓冲区 | 必须为 GM 地址空间 |

### 结果说明

| 结果 | 类型 | 说明 |
|------|------|------|
| `result_tensor` | Optional<AnyRankedTensor> | Tensor 语义下的结果张量 |

### 属性说明

`hir.nz2nd` 没有额外属性，是最简洁的 DMA 操作之一。

### 数据类型约束

`hir.nz2nd` 继承自 `HIVM_DmaOp`，受 `HIVM_StructuredOp` 的通用类型约束。源和目标的元素类型必须相同。

### Rank 约束

`StaticMaxRankTrait<2>` 限制操作数的最大 rank 为 2。

## IR 示例

### 基础 NZ 到 ND 转换

```mlir
func.func @hivm_nz2nd_l1_to_gm() {
  %src = memref.alloc() : memref<256x128xf16, #hivm.address_space<cbuf>>
  %dst = memref.alloc() : memref<256x128xf16, #hivm.address_space<gm>>
  hivm.hir.nz2nd ins(%src : memref<256x128xf16, #hivm.address_space<cbuf>>)
                 outs(%dst : memref<256x128xf16, #hivm.address_space<gm>>)
  return
}
```

### Tensor 语义

```mlir
func.func @hivm_nz2nd_tensor() -> tensor<256x128xf16> {
  %src = tensor.empty() : tensor<256x128xf16>
  %dst = tensor.empty() : tensor<256x128xf16>
  %res = hivm.hir.nz2nd ins(%src : tensor<256x128xf16>)
                         outs(%dst : tensor<256x128xf16>)
                         -> tensor<256x128xf16>
  return %res : tensor<256x128xf16>
}
```

## IR 层约束与验证

1. **元素类型一致性**：`src` 和 `dst` 的元素类型必须相同
2. **地址空间约束**（memref 语义下）：
   - `src` 必须为 L1 地址空间
   - `dst` 必须为 GM 地址空间
3. **Rank 约束**：最大 rank 为 2
4. **Core Type**：属于 Cube 核心类型（`CubeCoreTypeTrait`）
5. **Pipeline**：固定为 MTE3（`OpPipeTrait<"PIPE::PIPE_MTE3">`）

### 库函数命名

`hir.nz2nd` 的库函数名格式为：

```
nz2nd_{src_rank}d_to_{dst_rank}d_{datatype}
```

来源：[HIVMDMAOps.cpp:L693-L714](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L693-L714)

## 与其他 IR 操作的关系

### 与 nd2nz 的关系

```
hir.nd2nz (GM -> L1, ND -> NZ)     hir.nz2nd (L1 -> GM, NZ -> ND)
       |                                    ^
       v                                    |
   [Cube 计算]                          [Cube 结果写出]
   hir.mmadL1                             |
       |                                    |
       v                                    |
   hir.fixpipe (L0C -> L1, NZ)  -----------+
```

### 与 fixpipe 的关系

`hir.nz2nd` 仅执行 L1 到 GM 的布局转换搬运。如果需要从 L0C 搬运数据并同时执行 NZ 到 ND 的布局转换，应使用 `hir.fixpipe` 并设置 `dma_mode = NZ2ND`。

### 典型使用场景

`hir.nz2nd` 通常用于以下场景：
1. 将 Cube 计算的中间结果从 L1 写回 GM
2. 在多轮矩阵乘法之间，将 NZ 格式的部分和写回 GM 暂存

## 常见问题

### Q: nz2nd 和 fixpipe 的 NZ2ND 模式有什么区别？

A: `hir.nz2nd` 是独立的 L1 -> GM 搬运操作，仅执行布局转换。`hir.fixpipe` 的 NZ2ND 模式从 L0C 搬运数据到 GM，同时支持随路量化和激活。如果数据源是 L0C，应优先使用 `hir.fixpipe`；如果数据源已经是 L1 中的 NZ 格式数据，则使用 `hir.nz2nd`。

### Q: 为什么 nz2nd 的最大 rank 限制为 2？

A: 这是由硬件约束决定的。NZ 到 ND 的布局转换在 2D 矩阵上最为常见，更高维度的数据通常需要先 reshape 为 2D 再进行转换。

## 相关文档

- DMA 操作总览：[00-overview.md](00-overview.md)
- hir.nd2nz：[03-nd2nz.md](03-nd2nz.md)
- hir.fixpipe：[06-fixpipe.md](06-fixpipe.md)
- 源码参考：
  - [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td) - TableGen 定义
  - [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp) - 实现代码
