# NPU 设备规格参数

## 1. 概述

HACC 方言通过 `DeviceSpecEnum` 枚举和 `TargetDeviceSpecAttr` 属性描述 NPU 设备的硬件规格。编译器可基于这些规格进行 Tiling、内存分配等优化决策。

> 源码参考：[HACCAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td#L241-L289)、[Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Transforms/Passes.td#L51-L110)

## 2. DeviceSpecEnum 完整列表

| 枚举值 | 整数值 | 说明 |
|--------|--------|------|
| `AI_CORE_COUNT` | 0 | AI Core 数量 |
| `CUBE_CORE_COUNT` | 1 | Cube Core 数量（矩阵乘单元） |
| `VECTOR_CORE_COUNT` | 2 | Vector Core 数量（向量计算单元） |
| `UB_SIZE` | 3 | Unified Buffer 大小（字节） |
| `L1_SIZE` | 4 | L1 缓存大小（字节） |
| `L0A_SIZE` | 5 | L0A 缓存大小（字节，矩阵乘 A 矩阵缓冲） |
| `L0B_SIZE` | 6 | L0B 缓存大小（字节，矩阵乘 B 矩阵缓冲） |
| `L0C_SIZE` | 7 | L0C 缓存大小（字节，矩阵乘 C 结果缓冲） |
| `UB_ALIGN_SIZE` | 8 | Unified Buffer 对齐大小（字节） |
| `L1_ALIGN_SIZE` | 9 | L1 缓存对齐大小（字节） |
| `L0C_ALIGN_SIZE` | 10 | L0C 缓存对齐大小（字节） |
| `MINIMAL_D_CACHE_SIZE` | 11 | 最小 D Cache 大小 |
| `MAXIMUM_D_CACHE_SIZE` | 12 | 最大 D Cache 大小 |
| `ARCH` | 13 | 架构版本标识 |

### TableGen 定义

```tablegen
def HACC_DeviceSpecEnum :
  HACC_I32Enum<"DeviceSpec", "HACC device spec", [
  I32EnumAttrCase<"AI_CORE_COUNT", 0>,
  I32EnumAttrCase<"CUBE_CORE_COUNT", 1>,
  I32EnumAttrCase<"VECTOR_CORE_COUNT", 2>,
  I32EnumAttrCase<"UB_SIZE", 3>,
  I32EnumAttrCase<"L1_SIZE", 4>,
  I32EnumAttrCase<"L0A_SIZE", 5>,
  I32EnumAttrCase<"L0B_SIZE", 6>,
  I32EnumAttrCase<"L0C_SIZE", 7>,
  I32EnumAttrCase<"UB_ALIGN_SIZE", 8>,
  I32EnumAttrCase<"L1_ALIGN_SIZE", 9>,
  I32EnumAttrCase<"L0C_ALIGN_SIZE", 10>,
  I32EnumAttrCase<"MINIMAL_D_CACHE_SIZE", 11>,
  I32EnumAttrCase<"MAXIMUM_D_CACHE_SIZE", 12>,
  I32EnumAttrCase<"ARCH", 13>
]>
```

## 3. TargetDeviceSpecAttr

`TargetDeviceSpecAttr` 用于表示具体 NPU 设备的规格参数集合，基于 DLTI（Data Layout Target Information）机制存储。

### 3.1 属性定义

```tablegen
def HACC_TargetDeviceSpecAttr :
    HACC_Attr<"TargetDeviceSpec", "target_device_spec",
              [TargetDeviceSpecTrait, HACCTargetDeviceSpecTrait]> {
  let parameters = (ins
    ArrayRefParameter<"DataLayoutEntryInterface", "single spec entry">:$entries
  );
  let assemblyFormat = "`<` $entries `>`";
}
```

### 3.2 MLIR 表示

```mlir
#hacc.target_device_spec<
  #dlti.dl_entry<"UB_SIZE", 196608 : i32>>
```

### 3.3 HACCTargetDeviceSpecInterface

该接口继承自 `TargetDeviceSpecInterface`，提供按枚举值查询规格的方法：

| 方法名 | 返回类型 | 参数 | 说明 |
|--------|----------|------|------|
| `getSpecForIdentifierEnum(DeviceSpec)` | `::mlir::DataLayoutEntryInterface` | `identifier` | 根据枚举值返回对应的规格条目 |

## 4. TargetAttr

`TargetAttr` 用于指示目标设备名称。

```tablegen
def HACC_TargetAttr : HACC_Attr<"Target", "target"> {
  let parameters = (ins AttrParameter<"StringAttr", "target device">:$target);
  let assemblyFormat = "`<` $target `>`";
}
```

## 5. NPU 型号映射

`hacc-append-device-spec` Pass 通过 `--target` 选项指定 NPU 型号，自动附加设备规格信息。支持的型号列表如下：

### 5.1 Ascend 910B 系列

| 型号标识 |
|----------|
| `Ascend910B1` |
| `Ascend910B2` |
| `Ascend910B3` |
| `Ascend910B4` |

### 5.2 Ascend 910_93 系列

| 型号标识 |
|----------|
| `Ascend910_9362` |
| `Ascend910_9372` |
| `Ascend910_9381` |
| `Ascend910_9382` |
| `Ascend910_9391` |
| `Ascend910_9392` |

### 5.3 Ascend 310B 系列

| 型号标识 |
|----------|
| `Ascend310B1` |
| `Ascend310B2` |
| `Ascend310B3` |
| `Ascend310B4` |

### 5.4 Ascend 950 系列

| 型号标识 | 型号标识 |
|----------|----------|
| `Ascend910_950z` | `Ascend950PR_950z` |
| `Ascend910_9579` | `Ascend950PR_9579` |
| `Ascend910_957b` | `Ascend950PR_957a` |
| `Ascend910_957d` | `Ascend950PR_957b` |
| `Ascend910_9581` | `Ascend950PR_957c` |
| `Ascend910_9589` | `Ascend950PR_957d` |
| `Ascend910_958a` | `Ascend950PR_9589` |
| `Ascend910_958b` | `Ascend950PR_958a` |
| `Ascend910_9599` | `Ascend950PR_958b` |
| | `Ascend950PR_958c` |
| | `Ascend950PR_958d` |
| | `Ascend950PR_9599` |
| | `Ascend950PR_959a` |
| | `Ascend950PR_959b` |

### 5.5 Ascend 950DT 系列

| 型号标识 |
|----------|
| `Ascend950DT_950x` |
| `Ascend950DT_950y` |
| `Ascend950DT_9571` ~ `Ascend950DT_9578` |
| `Ascend950DT_9581` ~ `Ascend950DT_9588` |
| `Ascend950DT_9591`, `Ascend950DT_9592` |
| `Ascend950DT_9595`, `Ascend950DT_9596` |
| `Ascend950DT_95A1`, `Ascend950DT_95A2` |

## 6. 内存层次示意

```
+-------------------------------------------+
|               GM (Global Memory)          |
+-------------------------------------------+
        |                    |
+-------v--------+  +-------v--------+
|  L1 Cache      |  |  L1 Cache      |
+-------+--------+  +-------+--------+
        |                    |
+-------v--------+  +-------v--------+
| L0A | L0B | L0C|  | L0A | L0B | L0C|
+-----+-----+----+  +-----+-----+----+
        |                    |
+-------v--------+  +-------v--------+
|  UB (Unified   |  |  UB (Unified   |
|   Buffer)      |  |   Buffer)      |
+----------------+  +----------------+
```

- **GM**：全局内存，Host 与 Device 共享
- **L1**：AI Core 内部一级缓存
- **L0A/L0B/L0C**：Cube 单元专用缓冲
- **UB**：统一缓冲区，Vector 计算单元的工作空间
