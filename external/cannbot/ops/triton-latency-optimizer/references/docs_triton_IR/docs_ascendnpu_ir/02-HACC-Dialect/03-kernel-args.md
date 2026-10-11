# Kernel 参数类型

## 1. 概述

HACC 方言通过 `KernelArgType` 枚举对 Kernel 函数的参数进行语义分类。每个参数可附加 `hacc.arg_type` 属性，标识其在 Kernel 执行中的角色。

> 源码参考：[HACCAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td#L77-L104)

## 2. KernelArgType 完整列表

| 枚举值 | 整数值 | 助记符 | 说明 |
|--------|--------|--------|------|
| `kFFTSBaseAddr` | 0 | `ffts_base_address` | FFTS（Fast Function Task Schedule）基地址 |
| `kInput` | 1 | `input` | Kernel 输入数据 |
| `kOutput` | 2 | `output` | Kernel 输出数据 |
| `kInputAndOutput` | 3 | `input_and_output` | 同时作为输入和输出的数据（原地操作） |
| `kWorkspace` | 4 | `workspace` | 工作空间内存 |
| `kSyncBlockLock` | 5 | `sync_block_lock` | 同步 Block 锁 |
| `kTilingKey` | 6 | `tiling_key` | Tiling 键值（用于选择 Tiling 策略） |
| `kTilingData` | 7 | `tiling_data` | Tiling 数据（动态 Tiling 参数） |
| `kTilingStruct` | 8 | `tiling_struct` | Tiling 结构体（打包的 Tiling 参数） |
| `kMeshArg` | 9 | `mesh_arg` | Mesh 参数（多卡通信） |
| `kSanitizerAddr` | 10 | `sanitizer_addr` | Sanitizer 地址（用于调试） |
| `kGMAddr` | 11 | `gm_addr` | 全局内存地址 |

### TableGen 定义

```tablegen
def HACC_KernelArgTypeEnum
    : HACC_I32Enum<"KernelArgType", "HACC Kernel Arg Category",
                   [HACC_kFFTSBaseAddr, HACC_kInput, HACC_kOutput,
                    HACC_kInputAndOutput, HACC_kWorkspace, HACC_kSyncBlockLock,
                    HACC_kTilingKey, HACC_kTilingData, HACC_kTilingStruct,
                    HACC_kMeshArg, HACC_kSanitizerAddr, HACC_kGMAddr]>
```

## 3. KernelArgTypeAttr 属性

```tablegen
def HACC_KernelArgTypeAttr : HACC_Attr<"KernelArgType", "arg_type"> {
  let parameters = (ins EnumParameter<HACC_KernelArgTypeEnum>:$arg_type);
  let assemblyFormat = "`<` params `>`";
}
```

## 4. 各参数类型详解

### 4.1 kFFTSBaseAddr（ffts_base_address）

FFTS 基地址参数，用于 Kernel 启动时的快速函数任务调度。FFTS 机制允许 Host 端预先配置 Kernel 的执行参数，减少启动延迟。

### 4.2 kInput / kOutput / kInputAndOutput

| 类型 | 说明 |
|------|------|
| `input` | 标记参数为只读输入，Kernel 从中读取数据 |
| `output` | 标记参数为只写输出，Kernel 向其写入结果 |
| `input_and_output` | 标记参数为读写，Kernel 既读取又写入（原地操作） |

NPU Kernel 的调用约定中，输出参数也作为输入参数传入（out-param 模式），`hacc.output_idx` 属性用于标记哪个输入参数对应哪个输出值。

### 4.3 kWorkspace

工作空间内存参数，用于 Kernel 执行过程中需要的临时存储。工作空间大小通过 `hacc.infer_workspace_shape_function` 在 Host 端计算。

### 4.4 kSyncBlockLock

同步 Block 锁参数，用于多 Block 间的原子同步。每个原子操作需要一个 `<1xi64>` 类型的 memref 作为锁，锁的数量通过 `hacc.infer_sync_block_lock_num_function` 推断。

### 4.5 kTilingKey / kTilingData / kTilingStruct

| 类型 | 说明 |
|------|------|
| `tiling_key` | Tiling 策略选择键，Host 端根据输入形状计算 |
| `tiling_data` | 单个 Tiling 参数（如 tile 大小、步长等） |
| `tiling_struct` | 打包的 Tiling 结构体，包含多个 Tiling 参数 |

`hacc-pack-tiling-data` Pass 可将多个 `tiling_data` 参数打包为单个 `tiling_struct`。

### 4.6 kMeshArg

Mesh 参数，用于多卡通信场景，与 HMAP 方言的集合通信操作配合使用。

### 4.7 kSanitizerAddr

Sanitizer 地址参数，用于 Device 端内存调试和越界检测。

### 4.8 kGMAddr

全局内存地址参数，用于直接传递 GM 地址给 Kernel。

## 5. MLIR 示例

### 5.1 带 Tiling 数据的 Kernel

```mlir
func.func @device_kernel(
  %input: memref<?x?xf16> {hacc.arg_type = #hacc.arg_type<input>},
  %output: memref<?x?xf16> {hacc.arg_type = #hacc.arg_type<output>},
  %tiling_key: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
  %tiling_data: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}
) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  return
}
```

### 5.2 带 Workspace 和 SyncBlockLock 的 Kernel

```mlir
func.func @kernel_with_workspace(
  %input: memref<?x?xf16> {hacc.arg_type = #hacc.arg_type<input>},
  %output: memref<?x?xf16> {hacc.arg_type = #hacc.arg_type<output>},
  %workspace: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
  %lock: memref<1xi64> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}
) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  return
}
```

### 5.3 使用 isKernelArg 查询

```cpp
// 在 C++ 中查询参数类型
if (funcOp.isKernelArg(argIdx, hacc::KernelArgType::kInput)) {
  // 处理输入参数
}
```
