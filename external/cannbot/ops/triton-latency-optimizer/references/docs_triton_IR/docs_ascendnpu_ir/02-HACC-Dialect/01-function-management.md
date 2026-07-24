# HACC 函数管理

## 1. 概述

HACC 方言通过 `HACCFuncType` 枚举和 `HACCFunctionInterface` 接口管理异构函数的归属与属性。每个函数被标记为 HOST 或 DEVICE 类型，并通过接口方法查询和设置函数属性。

## 2. HACCFuncType 枚举

`HACCFuncType` 枚举定义了函数的异构归属类型。

| 枚举值 | 整数值 | 助记符 | 说明 |
|--------|--------|--------|------|
| `HOST` | 1 | `HOST` | Host 端函数，运行在 CPU 上 |
| `DEVICE` | 2 | `DEVICE` | Device 端函数，运行在 NPU 上 |

### 属性定义

```tablegen
def HACC_FuncTypeAttr : HACC_Attr<"HACCFuncType", "function_kind"> {
  let parameters = (ins EnumParameter<HACC_FuncTypeEnum>:$function_kind);
  let assemblyFormat = "`<` params `>`";
}
```

### MLIR 表示

```mlir
hacc.function_kind = #hacc.function_kind<HOST>
hacc.function_kind = #hacc.function_kind<DEVICE>
```

## 3. HACCFunctionInterface 接口

`HACCFunctionInterface` 继承自 `FunctionOpInterface`，为异构函数提供统一的查询和设置方法。

> 源码参考：[HACCInterfaces.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/IR/HACCInterfaces.td)

### 3.1 查询方法

| 方法名 | 返回类型 | 参数 | 说明 |
|--------|----------|------|------|
| `getHACCFuncType()` | `std::optional<hacc::HACCFuncType>` | 无 | 返回函数的 HACC 类型，未知时返回 `std::nullopt` |
| `isHost()` | `bool` | 无 | 判断是否为 Host 函数 |
| `isDevice()` | `bool` | 无 | 判断是否为 Device 函数 |
| `isDeviceEntry()` | `bool` | 无 | 判断是否为 Device 入口函数 |
| `getHostFuncType()` | `std::optional<hacc::HostFuncType>` | 无 | 返回 Host 函数类型，非 Host 函数返回 `std::nullopt` |

### 3.2 设置方法

| 方法名 | 返回类型 | 参数 | 说明 |
|--------|----------|------|------|
| `setDevice()` | `void` | 无 | 将函数标记为 Device 函数，自动移除不允许的属性 |
| `setDeviceEntry()` | `void` | 无 | 将函数标记为 Device 入口函数，自动移除不允许的属性 |
| `setHost()` | `void` | 无 | 将函数标记为 Host 函数，自动移除不允许的属性 |
| `setHostFuncType(::mlir::hacc::HostFuncType)` | `void` | `funcType` | 设置 Host 函数的具体角色类型 |

### 3.3 参数查询方法

| 方法名 | 返回类型 | 参数 | 说明 |
|--------|----------|------|------|
| `isKernelArg(int, ::mlir::hacc::KernelArgType)` | `bool` | `argIdx`, `argType` | 判断第 `argIdx` 个参数是否具有指定的 `hacc.arg_type` |

## 4. HACC To LLVM Translation 属性

`HACCToLLVMIRTranslateAttr` 枚举用于标记函数在 LLVM IR 翻译阶段的行为。

| 枚举值 | 整数值 | 助记符 | 说明 |
|--------|--------|--------|------|
| `ENTRY` | 0 | `hacc.entry` | Device 入口函数 |
| `MIX_ENTRY` | 1 | `hacc.mix_entry` | 混合 Device 入口函数 |
| `ALWAYS_INLINE` | 2 | `hacc.always_inline` | 始终内联函数 |

## 5. 辅助属性

### 5.1 RenameFuncAttr

```tablegen
def HACC_RenameFuncAttr : HACC_Attr<"RenameFunc", "rename_func"> {
  let parameters = (ins AttrParameter<"::mlir::FlatSymbolRefAttr">:$targetName);
}
```

指示当前函数应重命名为目标函数名。`hacc-rename-func` Pass 会据此执行重命名。

### 5.2 InputIdxAttr / OutputIdxAttr

| 属性 | 参数 | 说明 |
|------|------|------|
| `hacc.input_idx` | `unsigned:$argIdx` | 标记函数参数为输入索引 |
| `hacc.output_idx` | `unsigned:$argIdx` | 标记函数参数为输出索引（NPU Kernel 的输出通过输入参数传递） |

### 5.3 其他辅助属性

| 属性名 | 助记符 | 说明 |
|--------|--------|------|
| `ExportAsDAG` | `export_as_dag` | 将函数导出为 DAG |
| `DummyFunc` | `dummy_func` | 虚拟函数标记 |
| `ExternalFunctionPath` | `external_function_path` | 外部函数路径 |
| `CachedIO` | `cached_io` | 标记值已被缓存 IO |
| `NoIOAlias` | `no_io_alias` | 标记函数输入输出严格不别名 |
| `BlockDim` | `block_dim` | 函数的 Block 维度属性 |
| `SIMTModule` | `simt_module` | 标记 SIMT 模块 |

## 6. 典型使用模式

### 6.1 Host 函数

```mlir
func.func @tiling_func(%arg0: tensor<?x?xf16>) -> (i64, i64)
  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c = arith.constant 42 : i64
  return %c, %c : i64, i64
}
```

### 6.2 Device 入口函数

```mlir
func.func @kernel_entry(%arg0: tensor<?x?xf16>,
                        %arg1: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>,
              hacc.tiling_function = #hacc.tiling_function<@tiling_func>} {
  return
}
```

### 6.3 函数重命名

```mlir
func.func @bar() attributes {hacc.rename_func = #hacc.rename_func<@foo>} {
  return
}
```

经过 `hacc-rename-func` Pass 后，`@bar` 将被重命名为 `@foo`。
