# HACC 变换 Pass

## 1. 概述

HACC 方言提供了两个变换 Pass，用于函数重命名和设备规格附加。这些 Pass 在编译流程中为后续的代码生成和运行时调度提供必要的信息。

> 源码参考：[Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Transforms/Passes.td)

## 2. Pass 列表

| Pass 名称 | 作用域 | 构造函数 |
|-----------|--------|----------|
| `hacc-rename-func` | `FuncOp` | `mlir::hacc::createRenameFuncPass()` |
| `hacc-append-device-spec` | `ModuleOp` | `mlir::hacc::createAppendDeviceSpecPass()` |

## 3. hacc-rename-func

### 3.1 功能

根据 `hacc.rename_func` 属性重命名函数，并更新模块内所有对该函数的引用。

### 3.2 变换示例

输入：

```mlir
func.func @bar() attributes {hacc.rename_func = #hacc.rename_func<@foo>} {
  return
}

func.func @caller() {
  func.call @bar() : () -> ()
  return
}
```

输出：

```mlir
func.func @foo() {
  return
}

func.func @caller() {
  func.call @foo() : () -> ()
  return
}
```

### 3.3 约束

- 目标函数名不能与模块中已有函数名冲突

## 4. hacc-append-device-spec

### 4.1 功能

根据指定的 NPU 型号，向模块附加设备规格信息（`#hacc.target_device_spec`）。规格信息基于 DLTI 机制存储，供后续编译 Pass 查询使用。

### 4.2 选项

| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `target` | `::mlir::hacc::TargetDevice` | `Unknown` | 目标设备名称 |

### 4.3 依赖方言

- `hacc::HACCDialect`
- `mlir::DLTIDialect`

### 4.4 支持的设备型号

通过 `--target` 选项指定，完整列表参见 [02-device-specification.md](02-device-specification.md) 第 5 节。主要系列包括：

- Ascend 910B 系列（910B1/910B2/910B3/910B4）
- Ascend 910_93 系列（910_9362 ~ 910_9392）
- Ascend 310B 系列（310B1/310B2/310B3/310B4）
- Ascend 950 系列（910_950z ~ Ascend950DT_95A2）

### 4.5 使用示例

```bash
bishengir-opt --hacc-append-device-spec="target=Ascend910B4" input.mlir
```

### 4.6 变换效果

该 Pass 会在模块上附加 `#hacc.target_device_spec` 属性，包含指定设备型号的硬件规格参数：

```mlir
module attributes {
  #dlti.target_device_spec = #hacc.target_device_spec<
    #dlti.dl_entry<"AI_CORE_COUNT", 30 : i32>,
    #dlti.dl_entry<"UB_SIZE", 196608 : i32>,
    #dlti.dl_entry<"L1_SIZE", 1048576 : i32>,
    ...
  >
} {
  ...
}
```

## 5. Pass 在编译流程中的位置

```
前端 IR
  │
  ├── hacc-append-device-spec ── 附加设备规格
  │
  ├── [HFusion 变换流程]
  │     ├── hfusion-fuse-ops
  │     ├── hfusion-auto-schedule
  │     └── hfusion-auto-vectorize
  │
  ├── hacc-rename-func ── 函数重命名
  │
  └── 后端 Lowering
```

- `hacc-append-device-spec` 通常在编译流程早期执行，为后续 Pass 提供设备规格信息
- `hacc-rename-func` 通常在编译流程后期执行，确保函数名称符合运行时要求
