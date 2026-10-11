# 硬件规格 IR 映射

本文档将 NPU 型号映射到 HACC 方言的 `TargetDeviceSpecAttr` 字段值，所有数据从源码精确提取。

## DeviceSpecEnum 字段定义

`DeviceSpec` 枚举定义了硬件规格的所有字段，每个字段对应 `TargetDeviceSpecAttr` 中的一个 `dl_entry`：

| 枚举值 | 整数值 | 说明 | 对应 TargetSpec 字段 |
|--------|--------|------|---------------------|
| AI_CORE_COUNT | 0 | AI Core 数量 | AiCoreCount |
| CUBE_CORE_COUNT | 1 | Cube Core 数量 | CubeCoreCount |
| VECTOR_CORE_COUNT | 2 | Vector Core 数量 | VectorCoreCount |
| UB_SIZE | 3 | 统一缓冲区大小（bits） | UbSize |
| L1_SIZE | 4 | L1 缓冲区大小（bits） | L1Size |
| L0A_SIZE | 5 | L0A 缓冲区大小（bits） | L0aSize |
| L0B_SIZE | 6 | L0B 缓冲区大小（bits） | L0bSize |
| L0C_SIZE | 7 | L0C 缓冲区大小（bits） | L0cSize |
| UB_ALIGN_SIZE | 8 | UB 对齐大小（bits） | UbAlignSize |
| L1_ALIGN_SIZE | 9 | L1 对齐大小（bits） | L1AlignSize |
| L0C_ALIGN_SIZE | 10 | L0C 对齐大小（bits） | L0cAlignSize |
| MINIMAL_D_CACHE_SIZE | 11 | 最小 DCache 大小（bits） | MinimalDCacheSize |
| MAXIMUM_D_CACHE_SIZE | 12 | 最大 DCache 大小（bits） | MaximumDCacheSize |
| ARCH | 13 | 架构标识 | Arch |

源码参考：[HACCAttrs.td:L241-L262](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td#L241-L262)

## Ascend910B 系列

### 共享规格（Ascend910B_BaseSpec）

| 字段 | 值 | 换算 |
|------|-----|------|
| UbSize | 1572864 bits | 192 KB |
| L1Size | 4194304 bits | 512 KB |
| L0aSize | 524288 bits | 64 KB |
| L0bSize | 524288 bits | 64 KB |
| L0cSize | 1048576 bits | 128 KB |
| UbAlignSize | 256 bits | 32 B |
| L1AlignSize | 256 bits | 32 B |
| L0cAlignSize | 4096 bits | 512 B |
| Arch | "dav-c220" | — |

### 型号差异

| 型号 | AiCoreCount | CubeCoreCount | VectorCoreCount |
|------|-------------|---------------|-----------------|
| Ascend910B1 | 24 | 24 | 48 |
| Ascend910B2 | 24 | 24 | 48 |
| Ascend910B3 | 20 | 20 | 40 |
| Ascend910B4 | 20 | 20 | 40 |

## Ascend910_93 系列

### 共享规格（Ascend910_93_BaseSpec）

| 字段 | 值 | 换算 |
|------|-----|------|
| UbSize | 1572864 bits | 192 KB |
| L1Size | 4194304 bits | 512 KB |
| L0aSize | 524288 bits | 64 KB |
| L0bSize | 524288 bits | 64 KB |
| L0cSize | 1048576 bits | 128 KB |
| UbAlignSize | 256 bits | 32 B |
| L1AlignSize | 256 bits | 32 B |
| L0cAlignSize | 4096 bits | 512 B |
| Arch | "dav-c220" | — |

### 型号差异

| 型号 | AiCoreCount | CubeCoreCount | VectorCoreCount |
|------|-------------|---------------|-----------------|
| Ascend910_9362 | 20 | 20 | 40 |
| Ascend910_9372 | 20 | 20 | 40 |
| Ascend910_9381 | 24 | 24 | 48 |
| Ascend910_9382 | 24 | 24 | 48 |
| Ascend910_9391 | 24 | 24 | 48 |
| Ascend910_9392 | 24 | 24 | 48 |

## Ascend310B 系列

### 共享规格（Ascend310B_BaseSpec）

| 字段 | 值 | 换算 |
|------|-----|------|
| UbSize | 2097152 bits | 256 KB |
| L1Size | 8388608 bits | 1024 KB |
| L0aSize | 524288 bits | 64 KB |
| L0bSize | 524288 bits | 64 KB |
| L0cSize | 1048576 bits | 128 KB |
| UbAlignSize | 256 bits | 32 B |
| L1AlignSize | 256 bits | 32 B |
| L0cAlignSize | 4096 bits | 512 B |
| Arch | "dav-m300" | — |

### 型号差异

| 型号 | AiCoreCount | CubeCoreCount | VectorCoreCount |
|------|-------------|---------------|-----------------|
| Ascend310B1 | 1 | 1 | 1 |
| Ascend310B2 | 1 | 1 | 1 |
| Ascend310B3 | 1 | 1 | 1 |
| Ascend310B4 | 1 | 1 | 1 |

## Ascend950 系列

### 共享规格（Ascend950_BaseSpec）

| 字段 | 值 | 换算 |
|------|-----|------|
| UbSize | 2031616 bits | 248 KB（reserve 8KB for compiler） |
| MinimalDCacheSize | 262144 bits | 32 KB |
| MaximumDCacheSize | 983040 bits | 120 KB |
| L1Size | 4194304 bits | 512 KB |
| L0aSize | 524288 bits | 64 KB |
| L0bSize | 524288 bits | 64 KB |
| L0cSize | 2097152 bits | 256 KB |
| UbAlignSize | 256 bits | 32 B |
| L1AlignSize | 256 bits | 32 B |
| L0cAlignSize | 4096 bits | 512 B |
| Arch | "dav-c310" | — |

### 型号差异 — Ascend910_95xx

| 型号 | AiCoreCount | CubeCoreCount | VectorCoreCount |
|------|-------------|---------------|-----------------|
| Ascend910_950z | 4 | 4 | 8 |
| Ascend910_9579 | 28 | 28 | 56 |
| Ascend910_957b | 28 | 28 | 56 |
| Ascend910_957d | 28 | 28 | 56 |
| Ascend910_9581 | 32 | 32 | 64 |
| Ascend910_9589 | 32 | 32 | 64 |
| Ascend910_958a | 32 | 32 | 64 |
| Ascend910_958b | 32 | 32 | 64 |
| Ascend910_9599 | 36 | 36 | 72 |

### 型号差异 — Ascend950PR_95xx

| 型号 | AiCoreCount | CubeCoreCount | VectorCoreCount |
|------|-------------|---------------|-----------------|
| Ascend950PR_950z | 4 | 4 | 8 |
| Ascend950PR_9579 | 28 | 28 | 56 |
| Ascend950PR_957a | 28 | 28 | 56 |
| Ascend950PR_957b | 28 | 28 | 56 |
| Ascend950PR_957c | 28 | 28 | 56 |
| Ascend950PR_957d | 28 | 28 | 56 |
| Ascend950PR_9589 | 32 | 32 | 64 |
| Ascend950PR_958a | 32 | 32 | 64 |
| Ascend950PR_958b | 32 | 32 | 64 |
| Ascend950PR_958c | 32 | 32 | 64 |
| Ascend950PR_958d | 32 | 32 | 64 |
| Ascend950PR_9599 | 36 | 36 | 72 |
| Ascend950PR_959a | 36 | 36 | 72 |
| Ascend950PR_959b | 36 | 36 | 72 |

### 型号差异 — Ascend950DT_95xx

| 型号 | AiCoreCount | CubeCoreCount | VectorCoreCount |
|------|-------------|---------------|-----------------|
| Ascend950DT_950x | 8 | 8 | 16 |
| Ascend950DT_950y | 8 | 8 | 16 |
| Ascend950DT_9571 | 28 | 28 | 56 |
| Ascend950DT_9572 | 28 | 28 | 56 |
| Ascend950DT_9573 | 28 | 28 | 56 |
| Ascend950DT_9574 | 28 | 28 | 56 |
| Ascend950DT_9575 | 28 | 28 | 56 |
| Ascend950DT_9576 | 28 | 28 | 56 |
| Ascend950DT_9577 | 28 | 28 | 56 |
| Ascend950DT_9578 | 28 | 28 | 56 |
| Ascend950DT_9581 | 32 | 32 | 64 |
| Ascend950DT_9582 | 32 | 32 | 64 |
| Ascend950DT_9583 | 32 | 32 | 64 |
| Ascend950DT_9584 | 32 | 32 | 64 |
| Ascend950DT_9585 | 32 | 32 | 64 |
| Ascend950DT_9586 | 32 | 32 | 64 |
| Ascend950DT_9587 | 32 | 32 | 64 |
| Ascend950DT_9588 | 32 | 32 | 64 |
| Ascend950DT_9591 | 36 | 36 | 72 |
| Ascend950DT_9592 | 36 | 36 | 72 |
| Ascend950DT_9595 | 36 | 36 | 72 |
| Ascend950DT_9596 | 36 | 36 | 72 |
| Ascend950DT_95A1 | 36 | 36 | 72 |
| Ascend950DT_95A2 | 36 | 36 | 72 |

## IR 表示示例

```mlir
#hacc.target_device_spec<
  #dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>,
  #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>,
  #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>,
  #dlti.dl_entry<"UB_SIZE", 1572864 : i32>,
  #dlti.dl_entry<"L1_SIZE", 4194304 : i32>,
  #dlti.dl_entry<"L0A_SIZE", 524288 : i32>,
  #dlti.dl_entry<"L0B_SIZE", 524288 : i32>,
  #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>,
  #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>,
  #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>,
  #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>,
  #dlti.dl_entry<"ARCH", "dav-c220">>
```

## 架构系列对比

| 架构 | Arch 标识 | UB 大小 | L1 大小 | L0C 大小 | DCache | 代表型号 |
|------|-----------|---------|---------|----------|--------|----------|
| dav-c220 | "dav-c220" | 192 KB | 512 KB | 128 KB | — | Ascend910B, Ascend910_93 |
| dav-m300 | "dav-m300" | 256 KB | 1024 KB | 128 KB | — | Ascend310B |
| dav-c310 | "dav-c310" | 248 KB | 512 KB | 256 KB | 32-120 KB | Ascend950 |

## 源码参考

- NPUTargetSpec 定义：[NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td)
- DeviceSpecEnum 定义：[HACCAttrs.td:L241-L262](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td#L241-L262)
- TargetDeviceSpecAttr 定义：[HACCAttrs.td:L269-L289](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td#L269-L289)
