# hir.nd2nz

> 关键词：ND2NZ、布局转换、MTE2、GM 到 L1、Fractal 布局

## 概述

`hir.nd2nz` 是 HIVM 方言中的 DMA 操作，用于将数据从全局内存（GM）搬运到 L1 缓存，同时执行 ND（Normal Data）到 NZ（NzFormat/Z-Fractals）的布局转换。该操作映射到硬件的 MTE2 Pipeline。

ND 到 NZ 的布局转换是 Ascend NPU 矩阵计算的关键前置步骤。Cube 核心的矩阵乘法操作（MMAD）要求输入数据以 NZ（Fractal）格式存储在 L1 缓存中，因此 `hir.nd2nz` 是从 GM 加载矩阵数据到 Cube 计算流水线的必经之路。

> Python API 对应：无直接对应，由编译器自动插入

## IR 操作定义

### TableGen 定义

来源：[HIVMDMAOps.td:L328-L368](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L328-L368)

```tablegen
def ND2NZOp : HIVM_DmaOp<"nd2nz", [
  AttrSizedOperandSegments, SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE2">,
  HIVMCoreTypeInterface, CubeCoreTypeTrait, NoMaxRankTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
  ["decomposeOperation", "getDecomposePhase"]>
]> {
  let summary = "HIVM data copy operation with on-the-fly ND to NZ layout transformation";
  let description = [{
    - `dst_continuous`: if present, signify that the source data is stored continuously
      in the destination buffer. This must be set in order for this op to be converted to
      library function call.
    Constraints:
    - if `init_out_buffer` is true, `pad_value` should have value.
  }];
  let arguments = (ins AnyShaped:$src,
                       AnyShaped:$dst,
                       OptionalAttr<UnitAttr>:$dst_continuous,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$init_out_buffer,
                       Optional<AnyType>:$pad_value,
                       Optional<AnyType>:$init_condition
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensor);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`init_out_buffer` `=` $init_out_buffer^ )?
    (`pad_value` `=` $pad_value^ `:` type($pad_value))?
    (`init_condition` `=` $init_condition^ `:` type($init_condition))?
    (`->` type($result_tensor)^)?
  }];
  let builders = [
    OpBuilder<(ins "TypeRange" : $res, "Value" : $src, "Value" : $dst,
                   "UnitAttr" : $dst_continuous)>,
    OpBuilder<(ins "TypeRange" : $res, "Value" : $src, "Value" : $dst,
                   "UnitAttr" : $dst_continuous, "bool" : $init_out_buffer,
                   "Value" : $pad_value)>,
  ];
  let extraClassDeclaration = DmaOpBaseDecl;
}
```

### MLIR 语法

```mlir
hivm.hir.nd2nz ins(%src : memref<MxNxf16>) outs(%dst : memref<MxNxf16>)

hivm.hir.nd2nz {dst_continuous}
               ins(%src : memref<MxNxf16>) outs(%dst : memref<MxNxf16>)

hivm.hir.nd2nz {dst_continuous}
               ins(%src : memref<?x?x?x?xf32, #hivm.address_space<gm>>)
               outs(%dst : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>)
               init_out_buffer = true pad_value = %cst : f32
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| `src` | AnyShaped | 是 | 源数据缓冲区 | 通常为 GM 地址空间 |
| `dst` | AnyShaped | 是 | 目标数据缓冲区 | 通常为 L1 地址空间 |
| `dst_continuous` | UnitAttr | 否 | 目标数据是否连续存储 | 设置后才能转换为库函数调用 |
| `init_out_buffer` | BoolAttr | 否 | 是否初始化输出缓冲区 | 默认 false；若为 true 则 pad_value 必须有值 |
| `pad_value` | AnyType | 否 | 填充值 | 类型须与 src/dst 元素类型相同 |
| `init_condition` | AnyType | 否 | 初始化条件 | 用于条件化初始化 |

### 结果说明

| 结果 | 类型 | 说明 |
|------|------|------|
| `result_tensor` | Variadic<AnyRankedTensor> | Tensor 语义下的结果张量 |

### 属性说明

| 属性 | 类型 | 默认值 | 说明 | 可选值 |
|------|------|--------|------|--------|
| `dst_continuous` | UnitAttr | 无 | 目标数据连续存储标志 | 存在即表示连续 |
| `init_out_buffer` | BoolAttr | `false` | 是否在搬运前初始化输出缓冲区 | `true` / `false` |

## IR 示例

### 基础 ND 到 NZ 转换

```mlir
func.func @test_nd2nz() {
  %gmA = memref.alloc() : memref<1024x2048xf16>
  %gmASubview = memref.subview %gmA[0, 0][256, 128][1, 1]
                       : memref<1024x2048xf16> to
                         memref<256x128xf16, strided<[2048, 1], offset: 0>>
  %l1A = memref.alloc() : memref<256x128xf16>
  hivm.hir.nd2nz ins(%gmASubview : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
                 outs(%l1A: memref<256x128xf16>)
  return
}
```

### 带连续存储标志

```mlir
hivm.hir.nd2nz {dst_continuous}
               ins(%gmASubview : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
               outs(%l1A: memref<256x128xf16>)
```

### 带缓冲区初始化

```mlir
func.func @test_nd2nz_tensor_init_out_buffer(%arg0: memref<?x?x?x?xf32, #hivm.address_space<gm>>) {
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous}
                 ins(%arg0 : memref<?x?x?x?xf32, #hivm.address_space<gm>>)
                 outs(%alloc : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>)
                 init_out_buffer = true pad_value = %cst : f32
  return
}
```

### Tensor 语义

```mlir
func.func @test_nd2nz_tensor() {
  %gmA = tensor.empty() : tensor<1024x2048xf16>
  %gmASubview = tensor.extract_slice %gmA[0, 0][256, 128][1, 1]
                       : tensor<1024x2048xf16> to tensor<256x128xf16>
  %l1A = tensor.empty() : tensor<256x128xf16>
  %ret = hivm.hir.nd2nz ins(%gmASubview : tensor<256x128xf16>)
                        outs(%l1A: tensor<256x128xf16>) -> tensor<256x128xf16>
  return
}
```

## IR 层约束与验证

1. **元素类型一致性**：`src` 和 `dst` 的元素类型必须相同
2. **init_out_buffer 与 pad_value**：如果 `init_out_buffer` 为 true，则 `pad_value` 必须提供
3. **Core Type**：该操作属于 Cube 核心类型（`CubeCoreTypeTrait`）
4. **Pipeline**：固定为 MTE2（`OpPipeTrait<"PIPE::PIPE_MTE2">`）
5. **无最大 Rank 限制**：`NoMaxRankTrait` 表示不限制操作数的最大 rank

### 库函数命名

`hir.nd2nz` 的库函数名会根据下游操作自动调整。如果下游是 `MmadL1Op` 且当前操作提供 per-channel bias，则函数名会添加 `_forbias` 后缀（[HIVMDMAOps.cpp:L660-L671](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L660-L671)）。

## 与其他 IR 操作的关系

### ND 与 NZ 布局

```
ND 布局 (Normal Data):          NZ 布局 (Fractal):
+---+---+---+---+               +===+===+
| 0 | 1 | 2 | 3 |               |0,1|2,3|
+---+---+---+---+               +===+===+
| 4 | 5 | 6 | 7 |     -->      |4,5|6,7|
+---+---+---+---+               +===+===+
| 8 | 9 |10 |11 |               |8,9|10,11|
+---+---+---+---+               +===+===+

行优先连续存储                    按 Fractal Block 重新排列
                                 适配 Cube 矩阵计算引擎
```

### 典型使用模式

```
hir.nd2nz (GM -> L1, ND -> NZ)
     |
     v
hir.mmadL1 (L1 -> L0A/L0B -> L0C, 矩阵乘加)
     |
     v
hir.fixpipe (L0C -> GM/UB, 可选 NZ -> ND)
```

## 常见问题

### Q: 什么时候需要使用 nd2nz 而不是 load？

A: 当数据需要被 Cube 核心用于矩阵乘法时，必须使用 `nd2nz` 将数据从 GM 搬运到 L1 并转换为 NZ 格式。如果数据只是被 Vector 核心使用，则应使用 `hir.load` 将数据搬运到 UB。

### Q: dst_continuous 标志的作用是什么？

A: `dst_continuous` 表示目标缓冲区中的数据是连续存储的。设置此标志后，操作才能被转换为优化的库函数调用。如果未设置，编译器可能需要生成更通用的搬运代码。

### Q: 为什么 nd2nz 属于 Cube 核心类型？

A: 因为 ND 到 NZ 的布局转换是 Cube 矩阵计算流水线的专用操作，转换后的 NZ 格式数据仅供 Cube 核心的 MMAD 操作使用。

## 相关文档

- DMA 操作总览：[00-overview.md](00-overview.md)
- hir.nz2nd：[04-nz2nd.md](04-nz2nd.md)
- hir.fixpipe：[06-fixpipe.md](06-fixpipe.md)
- 源码参考：
  - [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td) - TableGen 定义
  - [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp) - 实现代码
  - [dma-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/dma-ops.mlir) - 测试用例
