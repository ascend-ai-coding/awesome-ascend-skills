# Gluon 实验性方言

本文档描述 Gluon 实验性方言的定义和语义。

源码参考：[GluonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Gluon/IR/GluonOps.td)

## 1. 方言概述

Gluon 方言（`gluon`）是 Triton 编译器中的实验性方言，命名空间为 `::mlir::triton::gluon`。其核心目的是提供布局推断的基础设施，允许编译器在后续 Pass 中自动将 `auto` 编码替换为最优的具体编码类型。

Gluon 方言当前仅包含一个操作：`gluon.set_auto_layout`。

## 2. 操作详解

### 2.1 gluon.set_auto_layout

将张量的 `auto` 布局编码替换为具体的编码类型。

| 项目 | 内容 |
|------|------|
| 操作名 | `gluon.set_auto_layout` |
| 输入 | `$src`: `TT_Tensor` |
| 输出 | `$result`: `TT_Tensor` |
| Traits | `SameOperandsAndResultShape`, `SameOperandsAndResultElementType` |
| 验证器 | `hasVerifier = 1` |

构建器：

```cpp
OpBuilder<(ins "Attribute":$encoding, "Value":$value)>
```

```mlir
%result = gluon.set_auto_layout %src : tensor<128xf32, #auto> -> tensor<128xf32, #blocked>
```

## 3. 语义说明

### 3.1 Auto 编码

在 Triton 编译流程中，某些张量可能被赋予 `auto` 布局编码，表示其具体布局尚未确定。`gluon.set_auto_layout` 操作用于在布局推断阶段将 `auto` 编码替换为具体的编码属性。

### 3.2 与其他方言的关系

```
Triton IR (tt)
    │
    │  布局推断
    ▼
Gluon IR (gluon)
    │  set_auto_layout: auto → concrete encoding
    ▼
TritonGPU IR (ttg)
    │  具体布局编码已确定
    ▼
LLVM IR
```

Gluon 方言在 Triton 到 TritonGPU 的转换过程中发挥作用，作为布局推断的桥梁：

1. Triton IR 中的张量可能带有 `auto` 编码
2. Gluon 的 `set_auto_layout` 操作将 `auto` 替换为具体编码
3. 转换后的张量进入 TritonGPU IR 时已具有确定的布局

### 3.3 设计动机

Gluon 方言的设计动机包括：

- **延迟布局决策**：允许在编译流程的更晚阶段确定布局，使优化 Pass 有更多信息
- **布局推断分离**：将布局推断逻辑从 TritonToTritonGPU 转换中解耦
- **实验性探索**：为新的布局推断算法提供实验平台

## 4. 验证规则

`gluon.set_auto_layout` 的验证器检查：

1. 输入张量的布局编码必须是 `auto` 或可推断的类型
2. 输出张量的布局编码必须是具体的编码属性
3. 输入和输出的形状必须相同
4. 输入和输出的元素类型必须相同

## 5. 与昇腾适配的关系

在昇腾适配场景下，Gluon 方言可用于：

- 将 `auto` 编码替换为 `FractalSharedEncodingAttr`（Cube 操作数）
- 将 `auto` 编码替换为 `BlockedEncodingAttr`（Vector 操作数）
- 在 SIMD/SIMT/MIX 三种执行模式下选择不同的布局策略
