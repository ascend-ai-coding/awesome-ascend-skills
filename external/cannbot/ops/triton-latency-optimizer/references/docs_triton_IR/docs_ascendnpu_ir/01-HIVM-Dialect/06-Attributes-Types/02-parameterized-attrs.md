# HIVM 参数化属性

> 关键词：DataLayoutAttr, AddressSpaceAttr, PipeAttr, BlockMappingAttr, SubBlockMappingAttr, TightlyCoupledBufferAttr

## 概述

参数化属性是携带额外参数的复合属性，用于描述 HIVM 中需要多个参数才能完整表达的信息。与简单枚举属性不同，参数化属性可以包含可选参数、数组参数等。

## DataLayoutAttr

源码：[HIVMAttrs.td#L103-L165](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L103-L165)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$data_layout` | EnumParameter\<DataLayoutEnum\> | 是 | 数据布局枚举值 |
| `$transpose` | OptionalParameter\<BoolAttr\> | 否 | 是否转置（DOTA_ND/DOTB_ND 时必须提供） |
| `$fractalSizes` | OptionalParameter\<DenseI64ArrayAttr\> | 否 | Fractal 块大小 |

### IR 格式

```mlir
#hivm.data_layout<ND>
#hivm.data_layout<dotA_ND, transpose = true>
#hivm.data_layout<dotA_ND, transpose = false>
#hivm.data_layout<dotB_ND, transpose = true>
#hivm.data_layout<dotC_ND>
#hivm.data_layout<nZ>
#hivm.data_layout<zN>
```

### 约束

- `transpose` 仅对 DOTA_ND 和 DOTB_ND 布局有效且必须提供
- `fractalSizes` 为可选的 I64 数组，指定 Fractal 块大小

### 额外方法

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getFractalSizesArray()` | `optional<SmallVector<int64_t>>` | 获取 Fractal 大小数组 |
| `getTransposeValue()` | `optional<bool>` | 获取 transpose 值 |
| `isNDLayout()` | bool | 判断是否为 ND 类布局（DOTA_ND/DOTB_ND/DOTC_ND/ND） |
| `getFractalBlockSizes()` | `FailureOr<FractalSize>` | 获取 2 个 Fractal 块大小 |

## AddressSpaceAttr

源码：[HIVMAttrs.td#L190-L197](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L190-L197)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$address_space` | EnumParameter\<AddressSpaceEnum\> | 是 | 地址空间枚举值 |

### IR 格式

```mlir
#hivm.address_space<gm>
#hivm.address_space<cbuf>
#hivm.address_space<ca>
#hivm.address_space<cb>
#hivm.address_space<cc>
#hivm.address_space<ub>
#hivm.address_space<zero>
```

### 接口

实现了 `DeviceMappingAttrInterface`，用于设备映射。

## PipeAttr

源码：[HIVMAttrs.td#L238-L244](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L238-L244)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$pipe` | EnumParameter\<PipeEnum\> | 是 | Pipeline 枚举值 |

### IR 格式

```mlir
#hivm.pipe<PIPE_S>
#hivm.pipe<PIPE_V>
#hivm.pipe<PIPE_M>
#hivm.pipe<PIPE_MTE1>
#hivm.pipe<PIPE_MTE2>
#hivm.pipe<PIPE_MTE3>
#hivm.pipe<PIPE_FIX>
#hivm.pipe<PIPE_UNASSIGNED>
```

## TCoreTypeAttr

源码：[HIVMAttrs.td#L311-L317](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L311-L317)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$tcoretype` | EnumParameter\<TCoreTypeEnum\> | 是 | Core 类型枚举值 |

### IR 格式

```mlir
#hivm.tcore_type<CUBE>
#hivm.tcore_type<VECTOR>
#hivm.tcore_type<CUBE_OR_VECTOR>
#hivm.tcore_type<CUBE_AND_VECTOR>
```

## TFuncCoreTypeAttr

源码：[HIVMAttrs.td#L263-L269](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L263-L269)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$funcCoreType` | EnumParameter\<TFuncCoreTypeEnum\> | 是 | 函数 Core 类型 |

### IR 格式

```mlir
#hivm.func_core_type<AIC>
#hivm.func_core_type<AIV>
#hivm.func_core_type<MIX>
```

## TModuleCoreTypeAttr

源码：[HIVMAttrs.td#L278-L292](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L278-L292)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$moduleCoreType` | EnumParameter\<TModuleCoreTypeEnum\> | 是 | 模块 Core 类型 |

### 推断规则

- 所有函数为 AIV → 模块 Core 类型为 AIV
- 所有函数为 AIC → 模块 Core 类型为 AIC
- 否则 → 模块 Core 类型为 MIX

## EventAttr

源码：[HIVMAttrs.td#L500-L506](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L500-L506)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$event` | EnumParameter\<EventEnum\> | 是 | Event ID |

### IR 格式

```mlir
#hivm.event<EVENT_ID0>
#hivm.event<EVENT_ID7>
```

## UnitFlagAttr

源码：[HIVMAttrs.td#L525-L531](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L525-L531)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$unit_flag` | EnumParameter\<UnitFlagEnum\> | 是 | UnitFlag 模式 |

### 数组类型

`UnitFlagArrayAttr` 是 `UnitFlagAttr` 的数组类型，用于宏操作中多个输出 Tensor 的 UnitFlag 模式。

## SyncBlockModeAttr

源码：[HIVMAttrs.td#L557-L563](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L557-L563)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$sync_mode` | EnumParameter\<SyncBlockModeEnum\> | 是 | 同步 Block 模式 |

## SyncBlockInstrModeAttr

源码：[HIVMAttrs.td#L580-L590](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L580-L590)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$sync_instr_mode` | EnumParameter\<SyncBlockInstrModeEnum\> | 是 | 同步指令模式 |

### 默认值

默认值为 `INTRA_BLOCK_SYNCHRONIZATION`。

## BlockMappingAttr

源码：[HIVMAttrs.td#L712-L720](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L712-L720)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$order` | OptionalParameter\<optional\<int32_t\>\> | 否 | 线性维度序号 |

### IR 格式

```mlir
#hivm.block<linear_dim = 0>
#hivm.block
```

### 接口

实现了 `DeviceMappingAttrInterface`。

## SubBlockMappingAttr

源码：[HIVMAttrs.td#L722-L730](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L722-L730)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$sub_block` | EnumParameter\<MappingIdEnum\> | 是 | Sub-Block 映射 ID |

### 说明

用于 Mix Kernel 中 Cube/Vector Block 维度比例的映射。

## TightlyCoupledBufferAttr

源码：[HIVMAttrs.td#L1010-L1017](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1010-L1017)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$id` | OptionalParameter\<optional\<int32_t\>\> | 否 | 紧耦合缓冲区 ID |

### 说明

用于 Cube-Vector 紧耦合缓冲区，允许 Cube 和 Vector Core 共享数据。

## MemoryEffectAttr

源码：[HIVMAttrs.td#L1057-L1063](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1057-L1063)

### 参数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$effect` | EnumParameter\<MemoryEffectEnum\> | 是 | 内存效果类型 |

### 说明

用于 SIMT VF 模式下的内存效果标记。

## 相关文档

- 源码参考：[HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- 测试用例：[attribute.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/attribute.mlir)
