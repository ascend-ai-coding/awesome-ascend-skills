# 逐元操作

## 1. 概述

HFusion 方言的逐元操作基于 Linalg 结构化操作范式，通过 OpDSL（YAML）声明式定义。每个操作包含一个 Region，描述逐元计算的标量逻辑。操作支持类型转换（cast）和多种函数选择。

> 源码参考：[HFusionNamedStructuredOps.yaml](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionNamedStructuredOps.yaml)

## 2. 操作列表

### 2.1 load

从输入张量逐元加载数据到输出张量，不执行数值类型转换。

| 操作数 | 类型 | 角色 |
|--------|------|------|
| `I` | input_tensor (type_var: T1) | 输入 |
| `O` | output_tensor (type_var: U) | 输出 |

赋值：`O = I`

### 2.2 store

将输入张量逐元存储到输出张量，支持原子操作语义。

| 操作数 | 类型 | 角色 |
|--------|------|------|
| `I` | input_tensor (type_var: T1) | 输入 |
| `O` | output_tensor (type_var: U) | 输出 |
| `atomic_kind` | atomic_kind_attr (默认: NONE) | 原子操作类型 |

赋值：`O = atomic_kind(I)`

### 2.3 elemwise_unary

对输入张量逐元应用一元函数。

| 操作数 | 类型 | 角色 |
|--------|------|------|
| `I` | input_tensor (type_var: T1) | 输入 |
| `O` | output_tensor (type_var: U) | 输出 |
| `fun` | unary_fn_attr (默认: sqrt) | 一元函数选择 |
| `cast` | type_fn_attr (默认: cast_signed) | 类型转换方式 |

赋值：`O = fun(cast(I))`

支持的 `unary_fn` 值：

| 函数名 | 说明 |
|--------|------|
| `relu` | ReLU 激活 |
| `sqrt` | 平方根 |
| `rsqrt` | 平方根倒数 |
| `rec` | 倒数 |
| `vnot` | 按位取反 |
| `tanh` | 双曲正切 |
| `sin` | 正弦 |
| `cos` | 余弦 |
| `atan` | 反正切 |
| `tan` | 正切 |
| `absi` | 绝对值 |
| `erf` | 误差函数 |
| `log2` | 以 2 为底对数 |
| `log10` | 以 10 为底对数 |
| `log1p` | log(1+x) |
| `exp2` | 2^x |
| `expm1` | e^x - 1 |
| `ilogb` | 指数部分 |

### 2.4 elemwise_binary

对两个输入张量逐元应用二元函数。

| 操作数 | 类型 | 角色 |
|--------|------|------|
| `lhs` | input_tensor (type_var: T1) | 左操作数 |
| `rhs` | input_tensor (type_var: T2) | 右操作数 |
| `O` | output_tensor (type_var: U) | 输出 |
| `fun` | binary_fn_attr (默认: vand) | 二元函数选择 |
| `cast` | type_fn_attr (默认: cast_signed) | 类型转换方式 |

赋值：`O = fun(cast(lhs), cast(rhs))`

支持的 `binary_fn` 值：

| 函数名 | 说明 |
|--------|------|
| `vor` | 按位或 |
| `vand` | 按位与 |
| `vxor` | 按位异或 |
| `minf` | 浮点最小值 |
| `maxf` | 浮点最大值 |
| `powf` | 浮点幂 |
| `mod` | 取模 |
| `shli` | 左移 |
| `shrsi` | 算术右移 |
| `shrui` | 逻辑右移 |
| `ldexp` | ldexp |
| `ceildivsi` | 有符号向上取整除法 |
| `ceildivui` | 无符号向上取整除法 |
| `floordivsi` | 有符号向下取整除法 |
| `powi` | 整数幂 |
| `minnumf` | IEEE 浮点最小值 |
| `maxnumf` | IEEE 浮点最大值 |
| `modui` | 无符号取模 |
| `divfhp` | 高精度除法 |

### 2.5 compare

对两个输入张量逐元执行比较操作。

| 操作数 | 类型 | 角色 |
|--------|------|------|
| `lhs` | input_tensor (type_var: T1) | 左操作数 |
| `rhs` | input_tensor (type_var: T1) | 右操作数 |
| `O` | output_tensor (type_var: U) | 输出 |
| `compare_fn` | compare_fn_attr (默认: veq) | 比较函数选择 |

赋值：`O = compare_fn(lhs, rhs)`

支持的 `compare_fn` 值：

| 函数名 | 说明 |
|--------|------|
| `veq` | 等于 |
| `vne` | 不等于 |
| `vle` | 有符号小于等于 |
| `vlt` | 有符号小于 |
| `vge` | 有符号大于等于 |
| `vgt` | 有符号大于 |
| `vule` | 无符号小于等于 |
| `vuge` | 无符号大于等于 |
| `vugt` | 无符号大于 |
| `vult` | 无符号小于 |

### 2.6 select

根据条件张量逐元选择值。

| 操作数 | 类型 | 角色 |
|--------|------|------|
| `cond` | input_tensor (type_var: U) | 条件 |
| `lhs` | input_tensor (type_var: T1) | 真值 |
| `rhs` | input_tensor (type_var: T1) | 假值 |
| `O` | output_tensor (type_var: T1) | 输出 |

赋值：`O = select(cond, lhs, rhs)`

### 2.7 cast

逐元类型转换，支持舍入模式控制。

| 操作数 | 类型 | 角色 |
|--------|------|------|
| `I` | input_tensor (type_var: T1) | 输入 |
| `O` | output_tensor (type_var: U) | 输出 |
| `round_mode` | round_mode_attr (默认: RINT) | 舍入模式 |
| `enable_overflow` | enable_overflow_attr (默认: true) | 溢出检测 |
| `enable_saturate` | enable_saturate_attr (默认: false) | 饱和截断 |
| `cast` | type_fn_attr (默认: cast_signed) | 类型转换方式 |
| `unsigned_mode` | unsigned_mode_attr (默认: SI2SI) | 无符号转换模式 |

赋值：`O = round(cast(I), round_mode, unsigned_mode)`

### 2.8 bitcast

逐元位转换，不改变底层位模式。

| 操作数 | 类型 | 角色 |
|--------|------|------|
| `I` | input_tensor (type_var: T1) | 输入 |
| `O` | output_tensor (type_var: U) | 输出 |

赋值：`O = bitcast(I)`

## 3. MLIR 示例

### 3.1 逐元一元操作

```mlir
%result = hfusion.elemwise_unary ins(%input : tensor<128x256xf16>)
  outs(%init : tensor<128x256xf32>)
  unary_fn = <sqrt> cast = <cast_signed>
```

### 3.2 逐元二元操作

```mlir
%result = hfusion.elemwise_binary ins(%lhs, %rhs : tensor<128x256xf16>, tensor<128x256xf16>)
  outs(%init : tensor<128x256xf16>)
  binary_fn = <vadd> cast = <cast_signed>
```

### 3.3 比较操作

```mlir
%result = hfusion.compare ins(%lhs, %rhs : tensor<128x256xf32>, tensor<128x256xf32>)
  outs(%init : tensor<128x256xi1>)
  compare_fn = <vgt>
```

### 3.4 类型转换

```mlir
%result = hfusion.cast ins(%input : tensor<128x256xf32>)
  outs(%init : tensor<128x256xi8>)
  round_mode = <FLOOR> cast = <cast_signed>
```
