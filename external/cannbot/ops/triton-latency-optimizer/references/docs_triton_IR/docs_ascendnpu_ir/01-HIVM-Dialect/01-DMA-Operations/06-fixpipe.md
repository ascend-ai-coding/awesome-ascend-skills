# hir.fixpipe

> 关键词：Fixpipe、DMA、FIX Pipeline、L0C、量化、ReLU、双目标、NZ2ND

## 概述

`hir.fixpipe` 是 HIVM 方言中最复杂的 DMA 操作，负责将数据从 L0C（Cube 累加器缓存）搬运到其他内存层级（GM/UB/L1），同时支持丰富的随路功能：

- **布局转换**：NZ 到 ND、NZ 到 DN、NZ 到 NZ（无转换）
- **随路量化**：F32 到 F16、F32 到 I8、F32 到 BF16 等类型转换
- **随路激活**：ReLU、Leaky ReLU、P-ReLU
- **双目标输出**：将 L0C 数据拆分到两个 UB 缓冲区

`hir.fixpipe` 映射到硬件的 FIX Pipeline，属于 Cube 核心类型，是矩阵乘法计算结果写出的核心操作。

> Python API 对应：al.fixpipe -- 详见 [docs_triton_ascend/03-Ascend-Extensions/03-fixpipe.md](../../../docs_triton_ascend/03-Ascend-Extensions/03-fixpipe.md)

## IR 操作定义

### TableGen 定义

来源：[HIVMDMAOps.td:L246-L326](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L246-L326)

```tablegen
def FixpipeOp : HIVM_DmaOp<"fixpipe", [
  AttrSizedOperandSegments,
  SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_FIX">,
  HIVMCoreTypeInterface, CubeCoreTypeTrait, NoMaxRankTrait,
  HIVMUnitFlagEnabledInterface, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<HIVMStructuredOpInterface, ["getIndexingMaps"]>
]> {
  let summary =
      "HIVM data copy operation from L0C to other memory hierarchies.";
  let description = [{
    Fixpipe is pipeline that performing data movement from L0C to other memory hierarchies,
    with on-the-fly fixed function of pre-stage quantization,
    pre-stage ReLU, element-wise add, post-stage ReLU, post-stage quantization.
    Currently support:
      - L0C to OUT
      - L0C to L1
      - L0C to UB (for Ascend950 series)

    Additionally, Fixpipe is also capable of layout transform.

    ### Attributes

    #### 'dma_mode'
    HIVM data movement model from L0C to destination, There are three values: NZ2DN, NZ2ND, and NZ2NZ(normal).

    #### `dual_dst_mode`
    HIVM dual destination mode control. dual destination mode can be enabled only when nz2nd or normal data
    movement mode is enabled, and data movement is being performed from L0C to UB. Only supported on Ascend950 series.
  }];
  let arguments = (ins AnyShaped:$src,
                       AnyShaped:$dst,
                       Arg<Variadic<I1>, [{
                        An optional condition to enable unit-flag mode, 
                        useful if there is a dependency on a for loop to run at least once.
                       }]>:$unit_flag_cond,
                       DefaultValuedAttr<HIVM_FixpipeDMAModeAttr,
                         "FixpipeDMAMode::NZ2NZ">:$dma_mode,
                       DefaultValuedOptionalAttr<HIVM_FixpipeDualDstModeAttr,
                         "FixpipeDualDstMode::NO_DUAL">:$dual_dst_mode,
                       DefaultValuedOptionalAttr<HIVM_FixpipePreQuantModeAttr,
                         "FixpipePreQuantMode::NO_QUANT">:$pre_quant,
                       DefaultValuedOptionalAttr<HIVM_FixpipePreReluModeAttr,
                         "FixpipePreReluMode::NO_RELU">:$pre_relu,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$channel_split,
                       OptionalAttr<UnitFlagArrayAttr>:$unit_flag_mode,
                       Optional<AnyFloat>:$quant_scale
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`quant_scale` `=` $quant_scale^ `:` type($quant_scale) )?
    (`dual_dst_mode` `=` $dual_dst_mode^)?
    (`unit_flag_mode` `(` $unit_flag_mode^ `)` )?
    (`unit_flag_cond` `(` $unit_flag_cond^ `)` )?
    (`->` type($result_tensor)^)?
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = DmaOpBaseDecl # [{
    int getFixpipeState();
    int needFixpipePreFuse();
    bool hasStore();
  }];
}
```

### MLIR 语法

```mlir
hivm.hir.fixpipe ins(%src : memref<256x128xf16>)
                  outs(%dst : memref<256x128xf16>)

hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
                  ins(%src : memref<256x128xf16>)
                  outs(%dst : memref<256x128xf16>)

hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
                  ins(%src : tensor<256x128xf32>)
                  outs(%dst : tensor<256x128xf16>)
                  -> tensor<256x128xf16>
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| `src` | AnyShaped | 是 | 源数据缓冲区（L0C） | 通常为 L0C 地址空间 |
| `dst` | AnyShaped | 是 | 目标数据缓冲区 | GM/UB/L1 |
| `unit_flag_cond` | Variadic<I1> | 否 | Unit-flag 模式条件 | 用于循环依赖 |
| `quant_scale` | AnyFloat | 否 | 量化缩放因子 | 仅在量化模式下使用 |

### 结果说明

| 结果 | 类型 | 说明 |
|------|------|------|
| `result_tensor` | Optional<AnyRankedTensor> | Tensor 语义下的结果张量 |

### 属性说明

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dma_mode` | HIVM_FixpipeDMAModeAttr | `NZ2NZ` | DMA 搬运模式 |
| `dual_dst_mode` | HIVM_FixpipeDualDstModeAttr | `NO_DUAL` | 双目标模式 |
| `pre_quant` | HIVM_FixpipePreQuantModeAttr | `NO_QUANT` | 随路量化模式 |
| `pre_relu` | HIVM_FixpipePreReluModeAttr | `NO_RELU` | 随路激活模式 |
| `channel_split` | BoolAttr | `false` | 通道拆分标志 |
| `unit_flag_mode` | UnitFlagArrayAttr | 无 | Unit-flag 模式数组 |

## 枚举详解

### FixpipeDMAMode（DMA 搬运模式）

定义于 [HIVMAttrs.td:L847-L859](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L847-L859)

| 枚举值 | 数值 | IR 字面量 | 说明 |
|--------|------|----------|------|
| `NZ2ND` | 0 | `nz2nd` | NZ 格式转换为 ND 格式后搬运 |
| `NZ2DN` | 1 | `nz2dn` | NZ 格式转换为 DN 格式后搬运（仅 Ascend950） |
| `NZ2NZ` | 2 | `normal` | 保持 NZ 格式直接搬运（默认） |

### FixpipeDualDstMode（双目标模式）

定义于 [HIVMAttrs.td:L821-L845](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L821-L845)

| 枚举值 | 数值 | IR 字面量 | 说明 |
|--------|------|----------|------|
| `NO_DUAL` | 0 | `NO_DUAL` | 单目标模式（默认） |
| `ROW_SPLIT` | 1 | `ROW_SPLIT` | 按 M 维度拆分，M/2 x N 写入每个 UB，M 须为 2 的倍数 |
| `COLUMN_SPLIT` | 2 | `COLUMN_SPLIT` | 按 N 维度拆分，M x N/2 写入每个 UB，N 须为 32 的倍数 |

双目标模式的约束：
- 仅在 `dma_mode` 为 NZ2ND 或 NZ2NZ 时可用
- 仅在数据从 L0C 搬运到 UB 时可用
- 仅在 Ascend950 系列上支持

### FixpipePreQuantMode（随路量化模式）

定义于 [HIVMAttrs.td:L783-L801](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L783-L801)

| 枚举值 | 数值 | IR 字面量 | 说明 |
|--------|------|----------|------|
| `NO_QUANT` | 0 | `NO_QUANT` | 不执行量化（默认） |
| `F322F16` | 1 | `F322F16` | F32 转 F16 量化 |
| `S322I8` | 9 | `S322I8` | F32 转 I8 量化 |
| `QF322F32_PRE` | 15 | `QF322F32_PRE` | 带 scale 的 F32 到 F32 预量化 |
| `F322BF16` | 16 | `F322BF16` | F32 转 BF16 量化 |

### FixpipePreReluMode（随路激活模式）

定义于 [HIVMAttrs.td:L803-L819](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L803-L819)

| 枚举值 | 数值 | IR 字面量 | 说明 |
|--------|------|----------|------|
| `NO_RELU` | 0 | `NO_RELU` | 不执行激活（默认） |
| `NORMAL_RELU` | 1 | `NORMAL_RELU` | 标准 ReLU：max(0, x) |
| `LEAKY_RELU` | 2 | `LEAKY_RELU` | Leaky ReLU |
| `P_RELU` | 3 | `P_RELU` | Parametric ReLU |

## IR 示例

### 基础搬运（NZ2NZ 模式）

```mlir
func.func @test_fixpipe() {
  %gmC = memref.alloc() : memref<1024x2048xf16>
  %gmCSubview = memref.subview %gmC[0, 0][256, 128][1, 1]
                      : memref<1024x2048xf16> to
                        memref<256x128xf16, strided<[2048, 1], offset: 0>>
  %l0c = memref.alloc() : memref<256x128xf16>
  hivm.hir.fixpipe ins(%l0c : memref<256x128xf16>)
                    outs(%gmCSubview : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
  return
}
```

### NZ2ND 布局转换

```mlir
hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
                  ins(%l0c : memref<256x128xf16>)
                  outs(%gmCSubview : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
```

### NZ2DN 布局转换（仅 Ascend950）

```mlir
hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>}
                  ins(%l0c : memref<256x128xf16>)
                  outs(%gmCSubview : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
```

### L0C 到 UB 搬运（Ascend950）

```mlir
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_fixpipe_l0c_to_ub() {
    %alloc = memref.alloc() : memref<1024x2048xf16, #hivm.address_space<ub>>
    %subview = memref.subview %alloc[0, 0] [256, 128] [1, 1]
               : memref<1024x2048xf16, #hivm.address_space<ub>>
                 to memref<256x128xf16, strided<[2048, 1]>, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<256x128xf16, #hivm.address_space<cc>>
    hivm.hir.fixpipe ins(%alloc_0 : memref<256x128xf16, #hivm.address_space<cc>>)
                     outs(%subview : memref<256x128xf16, strided<[2048, 1]>, #hivm.address_space<ub>>)
    return
  }
}
```

### 随路量化（F322F16）

```mlir
%l0c1 = tensor.empty() : tensor<256x128xf32>
%ret2 = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
                         ins(%l0c1 : tensor<256x128xf32>)
                         outs(%gmCSubview : tensor<256x128xf16>)
                         -> tensor<256x128xf16>
```

### 随路激活（Leaky ReLU）

```mlir
%ret3 = hivm.hir.fixpipe {pre_relu = #hivm.fixpipe_pre_relu_mode<LEAKY_RELU>}
                         ins(%l0c : tensor<256x128xf16>)
                         outs(%gmCSubview : tensor<256x128xf16>)
                         -> tensor<256x128xf16>
```

### 双目标模式

```mlir
%l0c1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cc>>
%ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
hivm.hir.fixpipe ins(%l0c1 : memref<16x16xf16, #hivm.address_space<cc>>)
                 outs(%ub : memref<16x16xf16, #hivm.address_space<ub>>)
                 dual_dst_mode = #hivm.fixpipe_dual_dst_mode<NO_DUAL>
```

## IR 层约束与验证

来源：[HIVMDMAOps.cpp:L848-L891](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L848-L891)

1. **NZ2DN 模式限制**：NZ2DN 仅在 Ascend950 上支持
2. **dst=UB 限制**：目标为 UB 仅在 Ascend950 上支持
3. **双目标模式约束**：
   - `dma_mode` 不能为 NZ2DN
   - 仅在 Ascend950 上支持
   - 数据搬运必须从 L0C 到 UB
4. **Core Type**：属于 Cube 核心类型（`CubeCoreTypeTrait`）
5. **Pipeline**：固定为 FIX（`OpPipeTrait<"PIPE::PIPE_FIX">`）

### FixpipeState

来源：[HIVMDMAOps.cpp:L808-L842](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L808-L842)

Fixpipe 的状态由 `getFixpipeState()` 返回：

| 状态 | 值 | 含义 |
|------|---|------|
| `Init` | -1 | 初始状态 |
| `QuantOrActivation` | 0 | 需要随路量化或激活 |
| `End` | 1 | 包含存储操作，已完成 |

### 库函数命名

```
fixpipe_{dma_mode}_{dual?}{src_type}_to_{dst_type}_{src_rank}d_to_{dst_rank}d_{dst_space}
```

## 与其他 IR 操作的关系

### 在矩阵乘法流水线中的位置

```
hir.nd2nz (GM -> L1)    hir.nd2nz (GM -> L1)
      |                        |
      v                        v
hir.mmadL1 (L1 -> L0A/L0B -> L0C)
      |
      v
hir.fixpipe (L0C -> GM/UB/L1)
```

### 与 nz2nd 的关系

`hir.fixpipe` 的 NZ2ND 模式与 `hir.nz2nd` 的区别：
- `hir.fixpipe` 从 L0C 搬运，支持随路量化/激活
- `hir.nz2nd` 从 L1 搬运，仅执行布局转换

## 常见问题

### Q: 什么时候使用 fixpipe 而不是 nz2nd？

A: 当数据源是 L0C（矩阵乘法结果）时，应使用 `hir.fixpipe`，因为它可以利用随路量化和激活功能，减少额外的计算操作。当数据源是 L1 中的 NZ 格式数据时，使用 `hir.nz2nd`。

### Q: pre_quant 和 pre_relu 可以同时使用吗？

A: 可以。Fixpipe 支持同时执行随路量化和随路激活，这是其核心优势之一。

### Q: dual_dst_mode 的实际用途是什么？

A: 双目标模式用于将 L0C 中的矩阵拆分到两个 UB 缓冲区，常用于混合核心（Mix）模式下的 Cube-Vector 协同计算。ROW_SPLIT 按行拆分，COLUMN_SPLIT 按列拆分。

### Q: quant_scale 的作用是什么？

A: `quant_scale` 用于 `QF322F32_PRE` 量化模式，提供量化缩放因子。在其他量化模式下不需要此参数。

## 相关文档

- Python API：[docs_triton_ascend/03-Ascend-Extensions/03-fixpipe.md](../../../docs_triton_ascend/03-Ascend-Extensions/03-fixpipe.md)
- DMA 操作总览：[00-overview.md](00-overview.md)
- 随路功能详解：[10-padding-quantization.md](10-padding-quantization.md)
- 源码参考：
  - [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td) - TableGen 定义
  - [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td) - 枚举定义
  - [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp) - 验证逻辑实现
  - [dma-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/dma-ops.mlir) - IR 测试用例
