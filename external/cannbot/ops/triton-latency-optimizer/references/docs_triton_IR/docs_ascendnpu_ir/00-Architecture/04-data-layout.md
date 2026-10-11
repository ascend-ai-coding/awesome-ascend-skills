# 数据布局详解

> 关键词：DataLayout, ND, NZ, zN, nZ, Fractal, DOTA_ND, DOTB_ND, DOTC_ND, nd2nz, nz2nd, DataLayoutAttr, fractalSizes

## 概述

Ascend NPU 的 Cube 单元（矩阵乘法引擎）对输入数据的布局有严格要求：矩阵 A 和矩阵 B 必须以特定的 Fractal（分形）格式存储在 L0A/L0B 中，计算结果 L0C 也以 Fractal 格式存储。而用户数据通常以 ND（N-Dimensional，行优先）格式存储在全局内存中。因此，数据在 GM 和 L1 之间搬运时需要进行布局转换（ND <-> NZ），这是 NPU 编程中一个核心且独特的概念。

HIVM IR 通过 `DataLayout` 枚举和 `DataLayoutAttr` 参数化属性精确描述数据布局信息，通过 `nd2nz`/`nz2nd` 操作显式表示布局转换。理解数据布局对于正确编写和优化 HIVM IR 至关重要，因为错误的布局会导致计算结果错误或硬件异常。

本文档从 [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td) 和 [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td) 中精确提取所有布局定义和转换操作。

## DataLayout 枚举完整列表

源文件：[HIVMAttrs.td:84-101](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L84-L101)

| 枚举值 | C++ 符号 | 数值 | IR 标识符 | 物理含义 |
|--------|---------|------|-----------|---------|
| DOTA_ND | `DataLayout::DOTA_ND` | 1 | `dotA_ND` | 矩阵 A 的 ND 布局，用于 Cube 乘法的左操作数 |
| DOTB_ND | `DataLayout::DOTB_ND` | 2 | `dotB_ND` | 矩阵 B 的 ND 布局，用于 Cube 乘法的右操作数 |
| DOTC_ND | `DataLayout::DOTC_ND` | 3 | `dotC_ND` | 矩阵 C 的 ND 布局，用于 Cube 乘法的结果 |
| nZ | `DataLayout::nZ` | 4 | `nZ` | nZ 布局（列优先 Fractal），N 维在低地址连续 |
| zN | `DataLayout::zN` | 5 | `zN` | zN 布局（行优先 Fractal），Z 维在低地址连续 |
| ND | `DataLayout::ND` | 6 | `ND` | ND 布局（标准 N 维行优先），用户数据的标准格式 |
| Fractal | `DataLayout::Fractal` | 7 | `Fractal` | 通用 Fractal 布局，通过 fractalSizes 参数描述具体分形 |

### 布局物理含义详解

#### ND 布局

ND（N-Dimensional）是标准的行优先多维数组布局，也是用户数据在全局内存中的自然存储格式。在 ND 布局中，最后一个维度（最右维度）在内存中连续存储。

```
ND 布局示例（2x4 矩阵）:
[a00, a01, a02, a03, a10, a11, a12, a13]
```

#### zN 布局

zN（Z-major N-minor）是 Cube 单元使用的 Fractal 布局之一。在 zN 布局中，数据按 Fractal 块组织，每个块内 Z 维度（行方向）在低地址连续。zN 布局是矩阵 A 输入到 L0A 时所需的格式。

```
zN 布局示意（Fractal 块大小 16x16）:
按 16x16 分块，块内行优先
[block(0,0), block(1,0), block(0,1), block(1,1), ...]
每个 block 内: [row0, row1, ..., row15]
```

#### nZ 布局

nZ（N-major Z-minor）是另一种 Fractal 布局。在 nZ 布局中，数据按 Fractal 块组织，每个块内 N 维度（列方向）在低地址连续。nZ 布局是矩阵 B 输入到 L0B 时所需的格式。

```
nZ 布局示意（Fractal 块大小 16x16）:
按 16x16 分块，块内列优先
[block(0,0), block(0,1), block(1,0), block(1,1), ...]
每个 block 内: [col0, col1, ..., col15]
```

#### DOTA_ND / DOTB_ND / DOTC_ND

这三种布局是 Cube 矩阵乘法中各操作数的 ND 格式标记，它们在语义上区分了矩阵乘法中 A、B、C 三个操作数的角色。`transpose` 属性仅对 DOTA_ND 和 DOTB_ND 有效且必须提供。

| 布局 | 角色 | transpose 属性 | 说明 |
|------|------|---------------|------|
| DOTA_ND | 矩阵 A | 必须提供 | 标记矩阵 A 是否需要转置 |
| DOTB_ND | 矩阵 B | 必须提供 | 标记矩阵 B 是否需要转置 |
| DOTC_ND | 矩阵 C | 不适用 | 矩阵乘法结果的 ND 格式 |

#### Fractal 布局

通用 Fractal 布局，通过 `fractalSizes` 参数描述具体的分形块大小。`fractalSizes` 是一个包含两个 int64 值的数组，分别表示 Fractal 块的行方向和列方向的大小。

## DataLayoutAttr 参数化属性定义

源文件：[HIVMAttrs.td:103-165](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L103-L165)

### TableGen 定义

```tablegen
def HIVM_DataLayoutAttr : HIVM_Attr<"DataLayout", "data_layout"> {
  let parameters = (ins
    EnumParameter<HIVM_DataLayoutEnum>:$data_layout,
    OptionalParameter<"BoolAttr">:$transpose,
    OptionalParameter<"DenseI64ArrayAttr">:$fractalSizes
  );
  let description = [{
    HIVM data layout mapping attribute. Maps to DOTA_ND, DOTB_ND, DOTC_ND, zN, nZ and ND.
      - `transpose`: Indicates that the layout is transposed.
                     Only valid and must be present for DOTA_ND and DOTB_ND layout.
  }];
  let assemblyFormat = "`<` $data_layout (`,` struct($transpose, $fractalSizes)^)? `>`";
}
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `data_layout` | `DataLayout` 枚举 | 是 | 布局类型 |
| `transpose` | `BoolAttr` | 否（DOTA_ND/DOTB_ND 时必须） | 是否转置 |
| `fractalSizes` | `DenseI64ArrayAttr` | 否 | Fractal 块大小 [row_size, col_size] |

### 额外方法

源码中定义了以下辅助方法：

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getFractalSizesArray()` | `std::optional<SmallVector<int64_t>>` | 获取 fractalSizes 为 SmallVector |
| `getTransposeValue()` | `std::optional<bool>` | 获取 transpose 值 |
| `isNDLayout()` | `bool` | 判断是否为 ND 类布局（DOTA_ND/DOTB_ND/DOTC_ND/ND） |
| `getFractalBlockSizes()` | `FailureOr<FractalSize>` | 提取恰好 2 个 fractal 块大小 |

### IR 语法示例

```mlir
#hivm.data_layout<ND>
#hivm.data_layout<zN>
#hivm.data_layout<nZ>
#hivm.data_layout<dotA_ND, transpose = true>
#hivm.data_layout<dotB_ND, transpose = false>
#hivm.data_layout<dotC_ND>
#hivm.data_layout<Fractal, fractalSizes = [16, 16]>
#hivm.data_layout<dotA_ND, transpose = true, fractalSizes = [16, 32]>
```

## 各布局的使用场景

| 布局 | 使用场景 | 存储位置 | 说明 |
|------|---------|---------|------|
| ND | 用户数据在 GM 中的标准格式 | GM | 所有用户数据默认格式 |
| zN | Cube 矩阵 A 输入 | L1, L0A | 矩阵 A 在 L1 中以 zN 格式存储 |
| nZ | Cube 矩阵 B 输入 | L1, L0B | 矩阵 B 在 L1 中以 nZ 格式存储 |
| DOTA_ND | 标记矩阵 A 的 ND 格式及转置信息 | - | 用于 `mmadL1` 的布局接口 |
| DOTB_ND | 标记矩阵 B 的 ND 格式及转置信息 | - | 用于 `mmadL1` 的布局接口 |
| DOTC_ND | 标记矩阵 C 的 ND 格式 | - | 用于 `mmadL1` 的布局接口 |
| Fractal | 通用分形布局描述 | - | 通过 fractalSizes 参数化描述 |

### 操作与布局的对应关系

| 操作 | 输入布局 | 输出布局 | 说明 |
|------|---------|---------|------|
| `hivm.nd2nz` | ND | zN/nZ | GM -> L1 时将 ND 转为 Fractal 格式 |
| `hivm.nz2nd` | zN/nZ | ND | L1 -> GM 时将 Fractal 转为 ND 格式 |
| `hivm.mmadL1` | zN (A), nZ (B) | Fractal (C) | Cube 矩阵乘法要求特定输入布局 |
| `hivm.fixpipe` | Fractal (L0C) | ND/NZ/DN | 输出时可进行布局转换 |

## ND <-> NZ 转换在 IR 中的表示

### nd2nz 操作

源文件：[HIVMDMAOps.td:328-368](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L328-L368)

将数据从 ND 格式转换为 NZ 格式，同时从 GM 搬运到 L1。属于 PIPE_MTE2 Pipeline。

```mlir
hivm.nd2nz ins(%src : memref<256x256xf16, #hivm.address_space<gm>>)
            outs(%dst : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>)
```

属性：
- `dst_continuous`：可选 UnitAttr，标记目标数据是否连续存储
- `init_out_buffer`：是否初始化输出缓冲区
- `pad_value`：填充值

### nz2nd 操作

源文件：[HIVMDMAOps.td:370-386](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L370-L386)

将数据从 NZ 格式转换为 ND 格式，同时从 L1 搬运到 GM。属于 PIPE_MTE3 Pipeline。

```mlir
hivm.nz2nd ins(%src : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>)
            outs(%dst : memref<256x256xf16, #hivm.address_space<gm>>)
```

### l12ub 操作

源文件：[HIVMDMAOps.td:388-404](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L388-L404)

将数据从 L1 搬运到 UB，同时进行 NZ 到 ND 的布局转换。属于 PIPE_MTE1 Pipeline。

```mlir
hivm.l12ub ins(%src : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>)
            outs(%dst : memref<?x?xf16, #hivm.address_space<ub>>)
```

## Fractal 布局详解

### Cube 矩阵乘法的输入输出布局要求

Cube 单元执行矩阵乘法 `C = A * B` 时，对数据的布局有严格要求：

| 操作数 | 存储位置 | 要求布局 | 说明 |
|--------|---------|---------|------|
| 矩阵 A | L0A | zN | 行优先 Fractal 格式 |
| 矩阵 B | L0B | nZ | 列优先 Fractal 格式 |
| 矩阵 C | L0C | Fractal | 结果以 Fractal 格式存储 |
| Bias | BT Buffer | 特定格式 | 通过 `copy_cbuf_to_bt` 加载 |

### fractalSizes 参数含义

`fractalSizes` 参数是一个包含两个 int64 值的数组 `[blockM, blockN]`，描述 Fractal 块的大小：

- `blockM`：Fractal 块在 M 维度（行方向）的大小
- `blockN`：Fractal 块在 N 维度（列方向）的大小

典型的 Fractal 块大小为 16x16（对于 FP16/FP32 数据类型）。

### mmadL1 的布局接口

源文件：[HIVMMacroOps.td:152-172](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L152-L172)

`MmadL1Op` 实现了 `OpLayoutInterface`，提供以下布局查询方法：

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getOperandALayout()` | `FailureOr<DataLayoutAttr>` | 获取矩阵 A 的目标布局 |
| `getOperandBLayout()` | `FailureOr<DataLayoutAttr>` | 获取矩阵 B 的目标布局 |
| `getOperandCLayout()` | `FailureOr<DataLayoutAttr>` | 获取矩阵 C 的目标布局 |
| `getOperandBiasLayout()` | `FailureOr<DataLayoutAttr>` | 获取 Bias 的目标布局 |
| `getOperandsTargetFractalLayout()` | 接口方法 | 获取所有操作数的目标 Fractal 布局 |

### Fixpipe 的布局转换

源文件：[HIVMAttrs.td:847-859](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L847-L859)

Fixpipe 支持三种布局转换模式（`dma_mode` 属性）：

| 模式 | IR 标识符 | 数值 | 说明 |
|------|-----------|------|------|
| NZ2ND | `nz2nd` | 0 | 将 NZ 格式转为 ND 格式 |
| NZ2DN | `nz2dn` | 1 | 将 NZ 格式转为 DN 格式（仅 950 系列） |
| NZ2NZ | `normal` | 2 | 保持 NZ 格式不变（默认值） |

## 910B vs 950 架构的数据流

### Ascend910B 架构数据流

```
GM (ND格式)
  |
  | [MTE2 + nd2nz]
  v
L1 (NZ格式)
  |
  | [MTE1]
  v
L0A (zN格式)  L0B (nZ格式)
  |              |
  +------+-------+
         |
         | [M: Cube矩阵乘法]
         v
      L0C (Fractal格式)
         |
         | [FIX + dma_mode=NZ2ND]
         v
      GM (ND格式)
```

### Ascend950 架构数据流

```
GM (ND格式)
  |
  | [MTE2 + nd2nz]
  v
L1 (NZ格式)
  |
  | [MTE1]
  v
L0A (zN格式)  L0B (nZ格式)
  |              |
  +------+-------+
         |
         | [M: Cube矩阵乘法]
         v
      L0C (Fractal格式, 256KB)
         |
    +----+----+----+
    |         |         |
    | [FIX]   | [FIX]   | [FIX]
    | NZ2ND   | NZ2ND   | NZ2ND/NZ2DN
    v         v         v
  GM (ND)  L1 (NZ)   UB (ND) ----[V]---- UB ----[MTE3]---- GM
                       ^
                       |
              紧耦合缓冲区/InsertCVDataMovement
              (950特有: L0C直通UB)
```

950 架构的关键差异：
1. L0C 容量增大到 256KB（910B 为 128KB）
2. 支持 L0C -> UB 直通通路，Fixpipe 可直接将结果输出到 UB
3. 支持 NZ2DN 布局转换模式
4. 支持紧耦合缓冲区，Cube 和 Vector 之间可高效传递数据
5. 支持 Dual Dst 模式（ROW_SPLIT / COLUMN_SPLIT），将 Cube 结果切分给两个 Vector 单元

## 常见问题

**Q: 为什么 Cube 单元需要 Fractal 布局而不是 ND 布局？**
A: Fractal 布局将矩阵按固定大小的块（如 16x16）重新排列，使得 Cube 单元可以高效地按块加载和计算。这种布局优化了 L0A/L0B 缓存的数据局部性，减少了缓存未命中。

**Q: DOTA_ND 和 ND 有什么区别？**
A: ND 是通用的行优先布局标记，DOTA_ND 是专门用于矩阵乘法中矩阵 A 的布局标记，它额外携带 `transpose` 属性来标记是否需要转置。`isNDLayout()` 方法对两者都返回 true。

**Q: nd2nz 和 fixpipe 的 NZ2ND 有什么区别？**
A: `nd2nz` 是 GM -> L1 方向的布局转换（ND 转 NZ），属于 PIPE_MTE2。`fixpipe` 的 NZ2ND 是 L0C -> GM/L1/UB 方向的布局转换（NZ 转 ND），属于 PIPE_FIX。两者方向相反，且在不同的 Pipeline 上执行。

**Q: fractalSizes 参数在什么情况下需要设置？**
A: 当使用 `Fractal` 布局类型时，需要通过 `fractalSizes` 指定具体的分形块大小。对于 `zN`、`nZ`、`ND` 等标准布局，fractalSizes 通常不需要设置。

**Q: 950 系列的 NZ2DN 模式是什么？**
A: NZ2DN 是 950 架构新增的布局转换模式，将 NZ 格式转为 DN（Diagonal-N）格式。源码中标注此模式仅在 Ascend950 上支持。

## 相关文档

- 源码参考：[HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- 源码参考：[HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- 源码参考：[HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td)
- 上一节：[03-pipeline-execution-model.md](./03-pipeline-execution-model.md) — Pipeline 执行模型
