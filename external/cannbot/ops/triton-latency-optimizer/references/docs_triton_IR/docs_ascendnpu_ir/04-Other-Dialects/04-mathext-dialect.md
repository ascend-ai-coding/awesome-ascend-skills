# MathExt 方言

## 1. 概述

MathExt 方言扩展了 MLIR 标准的 Math 方言，提供 NPU 设备特有的数学操作，包括 ilogb、ldexp 和高精度除法。

- **方言名称**：`mathExt`
- **C++ 命名空间**：`::mlir::mathExt`

> 源码参考：[MathExtOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOps.td)、[MathExtBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtBase.td)

## 2. 方言定义

```tablegen
def MathExt_Dialect : Dialect {
  let name = "mathExt";
  let cppNamespace = "::mlir::mathExt";
  let description = [{
    Extended Math dialect.
  }];
  let hasConstantMaterializer = 1;
}
```

## 3. 操作基类

### 3.1 MathExt_Op

所有 MathExt 操作的基类，自动附加 `Pure`、`VectorUnrollOpInterface` 和 `ElementwiseMappable` traits。

### 3.2 MathExt_FloatUnaryOp

浮点一元操作基类，操作数和结果类型相同（`SameOperandsAndResultType`），支持 `ArithFastMathInterface`。

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `operand` | `FloatLike` | 输入 |
| `fastmath` | `Arith_FastMathAttr` (默认: none) | 快速数学标志 |
| `result` | `FloatLike` | 输出 |

### 3.3 MathExt_FloatBinaryOp

浮点二元操作基类，操作数和结果类型相同，支持 `ArithFastMathInterface`。

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `lhs` | `FloatLike` | 左操作数 |
| `rhs` | `FloatLike` | 右操作数 |
| `fastmath` | `Arith_FastMathAttr` (默认: none) | 快速数学标志 |
| `result` | `FloatLike` | 输出 |

## 4. 操作定义

### 4.1 math_ext.ilogb

#### 功能

计算浮点数的指数部分：`ilogb(x) = floor(log2(abs(x)))`

#### 操作签名

继承自 `MathExt_FloatUnaryOp<"ilogb">`。

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `operand` | `FloatLike` | 输入 |
| `result` | `FloatLike` | 输出 |

#### 特性

- `hasFolder = 1`（支持常量折叠）

#### MLIR 示例

```mlir
%result = math_ext.ilogb %val : f32
%result = math_ext.ilogb %val fastmath <fast> : f32
```

### 4.2 math_ext.ldep

#### 功能

计算浮点数的分数部分：`ldep(x) = x * (ilogb(x) + 1)^(-1)`

> 注意：TableGen 中的 mnemonic 为 `ldep`，但描述中提及 `ldexp` 语义。

#### 操作签名

继承自 `MathExt_FloatBinaryOp<"ldep">`。

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `lhs` | `FloatLike` | 左操作数 |
| `rhs` | `FloatLike` | 右操作数 |
| `result` | `FloatLike` | 输出 |

#### 特性

- `hasFolder = 1`（支持常量折叠）

#### MLIR 示例

```mlir
%result = math_ext.ldep %x, %y : f32
%result = math_ext.ldep %x, %y fastmath <fast> : f32
```

### 4.3 math_ext.divfhp

#### 功能

高精度浮点除法，恢复最后 ULP（Unit in the Last Place）的精度。

#### 操作签名

继承自 `MathExt_FloatBinaryOp<"divfhp">`。

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `lhs` | `FloatLike` | 被除数 |
| `rhs` | `FloatLike` | 除数 |
| `result` | `FloatLike` | 商 |

#### 特性

- `hasFolder = 1`（支持常量折叠）

#### MLIR 示例

```mlir
%result = math_ext.divfhp %x, %y : f32
%result = math_ext.divfhp %x, %y fastmath <fast> : f32
```

## 5. 与 HFusion 方言的关系

MathExt 操作与 HFusion 方言的枚举函数有对应关系：

| MathExt 操作 | HFusion BinaryFn | 说明 |
|-------------|-------------------|------|
| `math_ext.ilogb` | `ilogb` (UnaryFn) | 指数部分 |
| `math_ext.ldep` | `ldexp` (BinaryFn) | ldexp 函数 |
| `math_ext.divfhp` | `divfhp` (BinaryFn) | 高精度除法 |

HFusion 方言依赖 `mathExt::MathExtDialect`，在 `hfusion-legalize-*` Pass 中可能将 HFusion 的枚举函数调用转换为 MathExt 操作。
