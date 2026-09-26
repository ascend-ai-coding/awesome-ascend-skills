# Symbol 方言

## 1. 概述

Symbol 方言处理动态形状表示，通过符号化整数和仿射映射描述张量的动态维度关系。它为编译器提供了在编译时推理动态形状的能力。

- **方言名称**：`symbol`
- **C++ 命名空间**：`::mlir::symbol`

> 源码参考：[SymbolOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolOps.td)、[Passes.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/Symbol/Transforms/Passes.h)

## 2. 操作定义

### 2.1 symbol.symbolic_int

#### 功能

表示一个具有范围约束的符号化整数值，返回 `index` 类型。支持通过仿射映射表达符号间的约束关系。

#### 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `symbol_name` | `FlatSymbolRefAttr` | 符号名称 |
| `min_val` | `I64Attr` | 最小值（含） |
| `max_val` | `I64Attr` | 最大值（含） |
| `int_symbols` | `Variadic<Index>` | 引用的其他符号整数 |
| `int_expressions` | `OptionalAttr<Builtin_AffineMapAttr>` | 仿射映射表达式 |
| `result` | `Index` | 结果 |

#### Traits

- `Pure`
- `OpAsmOpInterface`（自定义结果名称）

#### Builders

```tablegen
OpBuilder<(ins "TypeRange":$result, "FlatSymbolRefAttr":$symbol_name,
               "int64_t":$min_val, "int64_t":$max_val)>

OpBuilder<(ins "FlatSymbolRefAttr":$symbol_name)>

OpBuilder<(ins "FlatSymbolRefAttr":$symbol_name, "ValueRange":$int_symbols,
               "AffineMapAttr":$int_expressions)>
```

#### MLIR 示例

简单符号整数：

```mlir
%0 = symbol.symbolic_int @s0 {min_val = 5, max_val = 10} : index
%1 = symbol.symbolic_int @s1 {min_val = 2, max_val = 20} : index
```

带仿射映射的符号整数：

```mlir
%2 = symbol.symbolic_int @s2 [%0, %1],
  affine_map<()[s1, s2] -> (s1 * s2)> {min_val = 2, max_val = 20} : index
```

### 2.2 symbol.bind_symbolic_shape

#### 功能

将形状表达式绑定到张量，使用仿射映射描述动态维度与符号的关系。

#### 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `operand` | `AnyShaped` | 目标张量 |
| `shape_symbols` | `Variadic<Index>` | 形状符号（与仿射映射的局部符号 1:1 对应） |
| `shape_expressions` | `Builtin_AffineMapAttr` | 形状表达式仿射映射 |

#### MLIR 示例

```mlir
symbol.bind_symbolic_shape %arg0, [%0, %1],
  affine_map<()[s0, s1] -> (s0, s1, 3)> : tensor<?x?x3xf32>

symbol.bind_symbolic_shape %out0, [%0, %1, %2],
  affine_map<()[s0, s1, s2] -> (s0, s1 * 2 + s2, 3)> : tensor<?x?x3xf32>
```

## 3. 变换 Pass

### 3.1 Pass 列表

| Pass 名称 | 构造函数 | 功能简述 |
|-----------|----------|----------|
| `symbol-propagate` | `createPropagateSymbolPass()` | 传播符号信息 |
| `symbol-erase` | `createEraseSymbolPass()` | 擦除符号信息 |
| `symbol-to-encoding` | `createSymbolToEncodingPass()` | 将 bind_symbolic_shape 转换为 tensor encoding |
| `encoding-to-symbol` | `createEncodingToSymbolPass()` | 将 tensor encoding 转换为 bind_symbolic_shape |
| `symbol-unfold-symbolic-int` | `createUnfoldSymbolicIntPass()` | 将 symbol.symbolic_int 替换为 tensor.dim |

## 4. 语义说明

### 4.1 symbolic_int 的仿射映射

当 `int_expressions` 存在时，`int_symbols` 中的值按顺序对应仿射映射的局部符号。结果值由仿射映射计算得出：

```
result = affine_map(int_symbols[0], int_symbols[1], ...)
```

### 4.2 bind_symbolic_shape 的仿射映射

仿射映射的每个结果对应张量的一个维度。局部符号与 `shape_symbols` 一一对应：

```
dim[i] = affine_map(shape_symbols[0], shape_symbols[1], ...)[i]
```

## 5. 与其他方言的交互

- **HFusion 方言**：`hfusion-fold-symbolic-dim` 和 `hfusion-unfold-symbolic-dim` Pass 在 Symbol 和 HFusion 之间转换
- **Tensor Encoding**：`symbol-to-encoding` 和 `encoding-to-symbol` Pass 在 Symbol 和 Tensor Encoding 之间转换
