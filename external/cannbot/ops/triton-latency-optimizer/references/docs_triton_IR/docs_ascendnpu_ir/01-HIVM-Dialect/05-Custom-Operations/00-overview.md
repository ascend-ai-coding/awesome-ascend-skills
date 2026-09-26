# Custom 操作机制总览

> 关键词：Custom Op, BuiltinInfo, __builtin_gather_load, __builtin_index_select, 自定义操作

## 概述

HIVM Custom 操作机制提供了一种通用的扩展接口，允许用户在现有操作无法满足需求时编写自定义实现。Custom 操作支持两种粒度：

1. **hir.custom**：单 Pipe 自定义操作，在单个 Pipeline 上执行
2. **hir.custom_macro**：跨 Pipe 自定义宏操作，涉及多个 Pipeline

Custom 操作通过 `name` 属性标识操作类型，编译器根据 name 查找对应的实现。部分 name 以 `__builtin` 开头的是内置操作，编译器自带实现；其他 name 需要用户提供实现。

## Builtin 机制

内置操作（Builtin）是编译器自带的 Custom 操作实现，name 以 `__builtin` 开头。当前支持的内置操作：

| Builtin Name | 说明 | Core Type | Pipe |
|-------------|------|-----------|------|
| `__builtin_gather_load` | Gather Load 操作 | VECTOR | PIPE_V |
| `__builtin_index_select` | Index Select 操作 | VECTOR | PIPE_V |

### BuiltinInfo 结构

每个 Builtin 操作都有对应的 `BuiltinInfo` 结构，包含：

| 字段 | CustomOp | CustomMacroOp |
|------|----------|---------------|
| coreType | TCoreType | TCoreType |
| pipe / inPipe | PIPE | PIPE |
| - | - | outPipe (PIPE) |
| vfMode | VFMode | VFMode |
| getOpLibraryCallName | function\<string(Op)\> | function\<string(Op)\> |
| gmAddrArgsIndices | SmallVector\<size_t\> | SmallVector\<size_t\> |

## Custom 操作属性

Custom 操作通过属性传递必要信息：

### 必需属性

| 属性 | 说明 |
|------|------|
| `hivm.tcore_type` | 执行的 Core 类型，参见 TCoreTypeAttr |
| `hivm.vf_mode` | Vector 单元运行模式，参见 VFModeAttr（Cube Core 时忽略） |

### CustomOp 特有属性

| 属性 | 说明 |
|------|------|
| `hivm.pipe` | 执行的 Pipe，参见 PipeAttr |

### CustomMacroOp 特有属性

| 属性 | 说明 |
|------|------|
| `hivm.pipe_in` | 输入 Pipe |
| `hivm.pipe_out` | 输出 Pipe |

### 可选属性

| 属性 | 说明 |
|------|------|
| `gm_addr_args_indices` | I32 数组，指示哪些参数包含 GM 纯地址，lowering 时保留地址值 |

## 操作列表

| 操作 | 助记符 | 详细文档 |
|------|--------|---------|
| CustomOp | `hir.custom` | [01-custom-op.md](01-custom-op.md) |
| CustomMacroOp | `hir.custom_macro` | [02-custom-macro-op.md](02-custom-macro-op.md) |

## 相关文档

- 源码参考：[HIVMOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L904-L1167)
- 测试用例：[custom-op.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/custom-op.mlir)
