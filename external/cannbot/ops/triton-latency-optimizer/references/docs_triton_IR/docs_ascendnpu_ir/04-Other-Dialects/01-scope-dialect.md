# Scope 方言

## 1. 概述

Scope 方言提供区域作用域抽象，用于将操作分组到逻辑作用域中。Scope 操作可携带计算核心类型（如 CUBE/VECTOR）等属性，并支持 inline 和 outline 两种变换策略。

- **方言名称**：`scope`
- **C++ 命名空间**：`::mlir::scope`

> 源码参考：[ScopeOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/Scope/IR/ScopeOps.td)、[Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/Scope/Transforms/Passes.td)

## 2. 操作定义

### 2.1 scope.scope

#### 功能

表示一个区域作用域，内部操作作为一个整体执行。可携带属性（如 `tcore_type`）标记计算核心类型。

#### 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `no_inline` | `UnitAttr` | 标记该作用域不可内联 |
| `results` | `Variadic<AnyType>` | 作用域返回值 |
| `$region` | `SizedRegion<1>` | 作用域区域 |

#### Traits

- `RegionBranchOpInterface`
- `NoRegionArguments`
- `SingleBlockImplicitTerminator<"scope::ReturnOp">`
- `SingleBlock`
- `RecursiveMemoryEffects`

#### MLIR 示例

```mlir
scope.scope : () -> () {
  scope.return
} {tcore_type = #hivm.tcore_type<CUBE>}

scope.scope : () -> () {
  scope.return
}
```

带有 `no_inline` 属性的作用域不会被 `inline-scope` Pass 内联：

```mlir
scope.scope : () -> () {
  scope.return
} {no_inline}
```

### 2.2 scope.return

#### 功能

作为 `scope.scope` 区域的终止操作，返回作用域的计算结果。

#### 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `results` | `Variadic<AnyType>` | 返回值 |

#### Traits

- `HasParent<"ScopeOp">`
- `Pure`
- `ReturnLike`
- `Terminator`

#### MLIR 示例

```mlir
scope.return
scope.return %val1, %val2 : f32, i64
```

## 3. 变换 Pass

### 3.1 outline-scope

| 属性 | 值 |
|------|-----|
| Pass 名称 | `outline-scope` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::scope::createOutlineScopePass()` |
| 依赖方言 | `mlir::func::FuncDialect` |

#### 功能

将 `scope.scope` 转换为独立的 `func.func`，并将作用域属性（如 `tcore_type`）转移到新函数上。

#### 变换示例

输入：

```mlir
module {
  func.func @test() {
    scope.scope : () -> () {
      ...
      scope.return
    } {tcore_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      ...
      scope.return
    } {tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
```

输出：

```mlir
module {
  func.func @test_scope_0() attributes {tcore_type = #hivm.tcore_type<CUBE>} {
    ...
    return
  }
  func.func @test_scope_1() attributes {tcore_type = #hivm.tcore_type<VECTOR>} {
    ...
    return
  }
  func.func @test() {
    call @test_scope_scope_scope_0() : () -> ()
    call @test_scope_scope_scope_1() : () -> ()
    return
  }
}
```

### 3.2 inline-scope

| 属性 | 值 |
|------|-----|
| Pass 名称 | `inline-scope` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::scope::createInlineScopePass()` |

#### 选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `force-inline` | bool | false | 强制内联，忽略 `no_inline` 属性 |

#### 功能

将 `scope.scope` 区域内的操作提升到父区域，除非作用域标记了 `no_inline` 属性。

#### 变换示例

输入：

```mlir
func.func @test() {
  scope.scope : () -> () {
    ...
    scope.return
  } {no_inline}
  scope.scope : () -> () {
    <inlinable_operations>
    scope.return
  }
  return
}
```

输出：

```mlir
func.func @test() {
  scope.scope : () -> () {
    ...
    scope.return
  } {no_inline}
  <inlinable_operations>
  return
}
```

## 4. 典型使用场景

- **计算核心类型标注**：通过 `tcore_type` 属性标记作用域在 CUBE 或 VECTOR 核心上执行
- **VF（Vector Function）划分**：将计算图划分为多个 Scope，每个 Scope 对应一个向量化函数
- **编译流程控制**：`no_inline` 属性确保某些作用域在编译流程中保持独立
