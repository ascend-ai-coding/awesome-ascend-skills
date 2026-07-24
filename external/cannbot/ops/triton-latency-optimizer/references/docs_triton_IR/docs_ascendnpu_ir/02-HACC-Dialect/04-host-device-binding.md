# Host-Device 函数绑定关系

## 1. 概述

在 HACC 异构模型中，Device Kernel 的执行需要 Host 端提供多种辅助函数（如 Tiling 计算、形状推断等）。HACC 方言通过 `HostFuncType` 枚举和一系列 `FuncRefAttr` 属性建立 Host-Device 函数间的绑定关系。

> 源码参考：[HACCAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td#L110-L225)

## 2. HostFuncType 完整列表

| 枚举值 | 整数值 | 助记符 | 说明 |
|--------|--------|--------|------|
| `kEntry` | 1 | `host_entry` | Host 入口函数 |
| `kTilingFunction` | 2 | `tiling_function` | Tiling 计算函数 |
| `kInferOutputShapeFunction` | 3 | `infer_output_shape_function` | 输出形状推断函数 |
| `kInferWorkspaceShapeFunction` | 4 | `infer_workspace_shape_function` | 工作空间大小推断函数 |
| `kInferSyncBlockLockNumFunction` | 5 | `infer_sync_block_lock_num_function` | 同步锁数量推断函数 |
| `kInferSyncBlockLockInitFunction` | 6 | `infer_sync_block_lock_init_function` | 同步锁初始值推断函数 |
| `kInferVFModeFunction` | 7 | `infer_vf_mode_function` | VF 模式推断函数 |
| `kGetTilingStructSizeFunction` | 8 | `get_tiling_struct_size_function` | Tiling 结构体大小获取函数 |

### TableGen 定义

```tablegen
def HACC_HostFuncTypeEnum
    : HACC_I32Enum<
          "HostFuncType", "HACC Host function type",
          [HACC_kEntry, HACC_kTilingFunction, HACC_kInferOutputShapeFunction,
           HACC_kInferWorkspaceShapeFunction,
           HACC_kInferSyncBlockLockNumFunction,
           HACC_kInferSyncBlockLockInitFunction, HACC_kInferVFModeFunction,
           HACC_kGetTilingStructSizeFunction]>
```

## 3. HostFuncTypeAttr 属性

```tablegen
def HACC_HostFuncTypeAttr : HACC_Attr<"HostFuncType", "host_func_type"> {
  let parameters = (ins EnumParameter<HACC_HostFuncTypeEnum>:$host_func_type);
  let assemblyFormat = "`<` params `>`";
}
```

## 4. FuncRefAttr 绑定属性

Host 端辅助函数通过 `FuncRefAttr` 系列属性绑定到 Device 函数上。每个属性包含一个 `FlatSymbolRefAttr`，指向对应的 Host 函数符号名。

### 4.1 基类定义

```tablegen
class HACC_FuncRefAttr<string attrName, string attrMnemonic>
    : HACC_Attr<attrName, attrMnemonic> {
  let parameters = (ins AttrParameter<"::mlir::FlatSymbolRefAttr",
                                      "function symbol name">:$funcName);
  let assemblyFormat = "`<` $funcName `>`";
}
```

### 4.2 绑定属性列表

| 属性名 | 助记符 | 说明 |
|--------|--------|------|
| `TilingFunctionAttr` | `tiling_function` | 指向 Host 端 Tiling 计算函数 |
| `InferOutputShapeFunctionAttr` | `infer_output_shape_function` | 指向 Host 端输出形状推断函数 |
| `InferWorkspaceShapeFunctionAttr` | `infer_workspace_shape_function` | 指向 Host 端工作空间大小推断函数 |
| `InferSyncBlockLockNumFunctionAttr` | `infer_sync_block_lock_num_function` | 指向 Host 端同步锁数量推断函数 |
| `InferSyncBlockLockInitFunctionAttr` | `infer_sync_block_lock_init_function` | 指向 Host 端同步锁初始值推断函数 |
| `InferVFModeFunctionAttr` | `infer_vf_mode_function` | 指向 Host 端 VF 模式推断函数 |
| `GetTilingStructSizeFunctionAttr` | `get_tiling_struct_size_function` | 指向 Host 端 Tiling 结构体大小获取函数 |

## 5. 各绑定函数详解

### 5.1 TilingFunction

Tiling 函数在 Host 端执行，根据输入张量的形状和设备规格计算 Tiling 参数（如分块大小、迭代次数等），并将结果传递给 Device Kernel。

```
Host: tiling_func(input_shape, device_spec) -> (tile_m, tile_n, tile_k, ...)
Device: kernel(input, output, tile_m, tile_n, tile_k, ...)
```

### 5.2 InferOutputShapeFunction

推断 Device Kernel 的输出张量形状。对于动态形状的 Kernel，Host 端需要预先知道输出大小以分配内存。

### 5.3 InferWorkspaceShapeFunction

推断 Device Kernel 所需的工作空间大小。工作空间用于存储中间计算结果。

### 5.4 InferSyncBlockLockNumFunction

推断 Kernel 所需的同步锁数量。每个原子操作需要所有 Block 共享一个 `<1xi64>` 类型的 memref 作为锁。

### 5.5 InferSyncBlockLockInitFunction

推断同步锁的初始值。每个锁在 Kernel 执行前需要初始化。

### 5.6 InferVFModeFunction

推断 VF（Vector Function）模式，用于控制向量化执行策略。

### 5.7 GetTilingStructSizeFunction

获取 Tiling 结构体的大小（以 i64 为单位），用于 `hacc-pack-tiling-data` Pass 将多个 Tiling 参数打包为结构体。

## 6. MLIR 示例

### 6.1 完整的 Host-Device 绑定

```mlir
module {
  func.func @tiling_func(%arg0: tensor<?x?xf16>) -> (i64, i64)
    attributes {hacc.function_kind = #hacc.function_kind<HOST>,
               hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    %c1 = arith.constant 128 : i64
    %c2 = arith.constant 256 : i64
    return %c1, %c2 : i64, i64
  }

  func.func @infer_shape(%arg0: tensor<?x?xf16>) -> tensor<?x?xf16>
    attributes {hacc.function_kind = #hacc.function_kind<HOST>,
               hacc.host_func_type = #hacc.host_func_type<infer_output_shape_function>} {
    return
  }

  func.func @device_kernel(
    %input: tensor<?x?xf16>,
    %output: tensor<?x?xf16>,
    %t1: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
    %t2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}
  ) attributes {
    hacc.function_kind = #hacc.function_kind<DEVICE>,
    hacc.tiling_function = #hacc.tiling_function<@tiling_func>,
    hacc.infer_output_shape_function = #hacc.infer_output_shape_function<@infer_shape>
  } {
    return
  }
}
```

### 6.2 Host 入口函数

```mlir
func.func @host_entry(%arg0: tensor<?x?xf16>) -> tensor<?x?xf16>
  attributes {hacc.function_kind = #hacc.function_kind<HOST>,
             hacc.host_func_type = #hacc.host_func_type<host_entry>} {
  %result = func.call @device_kernel(%arg0, ...) : (tensor<?x?xf16>, ...) -> tensor<?x?xf16>
  return %result : tensor<?x?xf16>
}
```
