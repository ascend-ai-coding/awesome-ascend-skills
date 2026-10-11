# hir.custom_macro — 自定义跨 Pipe 宏操作

> 关键词：custom_macro, CustomMacroOp, MacroOpTrait, InPipe, OutPipe, Builtin

## 概述

`hir.custom_macro` 是 HIVM 方言中的自定义跨 Pipe 宏操作，与 `hir.custom` 类似，但涉及多个 Pipeline。该操作通过 `MacroOpTrait` 标记为宏操作，使用 `hivm.pipe_in` 和 `hivm.pipe_out` 属性分别指定输入和输出 Pipeline。

CustomMacroOp 适用于需要跨 Pipeline 协作的场景，如数据加载+计算+后处理的融合操作。

## IR 操作定义

从 [HIVMOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L1105-L1167) 提取：

```
def CustomMacroOp : HIVM_CustomOp<"custom_macro", [MacroOpTrait]> {
  let arguments = (ins StrAttr:$name,
                       Variadic<AnyType>:$inputs,
                       Variadic<AnyType>:$outputs);
  let results = (outs Variadic<AnyType>:$results);
}
```

## 参数说明

### 输入操作数（ins）

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$name` | StrAttr | 是 | 操作名称 |
| `$inputs` | Variadic\<AnyType\> | 是 | 输入参数 |
| `$outputs` | Variadic\<AnyType\> | 是 | 输出/初始化参数 |

### 输出结果（outs）

| 参数 | 类型 | 说明 |
|------|------|------|
| `$results` | Variadic\<AnyType\> | 结果值 |

### 必需属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `hivm.tcore_type` | TCoreTypeAttr | 执行的 Core 类型 |
| `hivm.pipe_in` | PipeAttr | 输入 Pipeline |
| `hivm.pipe_out` | PipeAttr | 输出 Pipeline |
| `hivm.vf_mode` | VFModeAttr | Vector 运行模式 |

### 可选属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `gm_addr_args_indices` | DenseI32ArrayAttr | GM 地址参数索引 |

### 与 CustomOp 的差异

| 特性 | CustomOp | CustomMacroOp |
|------|----------|---------------|
| Trait | SinglePipeOpTrait | MacroOpTrait |
| Pipe 属性 | `hivm.pipe`（单个） | `hivm.pipe_in` + `hivm.pipe_out`（两个） |
| Pipeline 数量 | 单个 | 跨多个 |
| 同步需求 | Pipe 内同步 | 跨 Pipe 同步 |
| BuiltinInfo 字段 | coreType, pipe, vfMode | coreType, inPipe, outPipe, vfMode |

### 额外类方法

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getInPipe()` | PIPE | 获取输入 Pipe |
| `setInPipe(PIPE)` | void | 设置输入 Pipe |
| `getOutPipe()` | PIPE | 获取输出 Pipe |
| `setOutPipe(PIPE)` | void | 设置输出 Pipe |
| `getCoreType()` | `optional<TCoreType>` | 获取 Core 类型 |
| `setCoreType(TCoreType)` | void | 设置 Core 类型 |
| `getVFMode()` | `optional<VFMode>` | 获取 VF 模式 |
| `setVFMode(VFMode)` | void | 设置 VF 模式 |
| `isBuiltin()` | bool | 判断是否为内置操作 |

### Pipe 属性名称常量

```cpp
static constexpr StringLiteral inPipeName = "hivm.pipe_in";
static constexpr StringLiteral outPipeName = "hivm.pipe_out";
```

## IR 示例

### 自定义宏操作

```mlir
%empty = tensor.empty() : tensor<3x3xf32>
%0 = hivm.hir.custom_macro
     { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V> }
     "my_custom_op"
     ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
         : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
     outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
```

### 内置宏操作

```mlir
%empty = tensor.empty() : tensor<3x3xf32>
%0 = hivm.hir.custom_macro
     "__builtin_gather_load"
     ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
         : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
     outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
```

## IR 层约束与验证

1. **MacroOpTrait**：操作被标记为宏操作，编译器在同步分析中特殊处理。
2. **跨 Pipe 同步**：由于涉及多个 Pipeline，InjectSync/GraphSyncSolver Pass 会在操作前后插入跨 Pipe 同步操作。
3. **pipe_in / pipe_out**：必须指定输入和输出 Pipeline，编译器据此生成同步操作。
4. **DestinationStyleOpInterface**：outputs 作为 DPS init 操作数。
5. **Builtin 验证**：内置操作编译器会检查参数正确性。

## 常见问题

**Q: 什么时候用 custom_macro 而不是 custom？**
A: 当操作涉及多个 Pipeline（如数据加载+计算+后处理）时使用 custom_macro。单 Pipeline 操作使用 custom。

**Q: pipe_in 和 pipe_out 如何选择？**
A: pipe_in 是数据进入的 Pipeline（如 PIPE_MTE2 表示从 GM 加载），pipe_out 是数据输出的 Pipeline（如 PIPE_V 表示在 Vector Core 处理）。

**Q: custom_macro 的同步如何处理？**
A: InjectSync/GraphSyncSolver Pass 会根据 pipe_in 和 pipe_out 自动在操作前后插入 set_flag/wait_flag 同步操作。

## 相关文档

- 源码参考：[HIVMOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L1105-L1167)
- 测试用例：[custom-op.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/custom-op.mlir)
- CustomOp：[01-custom-op.md](01-custom-op.md)
