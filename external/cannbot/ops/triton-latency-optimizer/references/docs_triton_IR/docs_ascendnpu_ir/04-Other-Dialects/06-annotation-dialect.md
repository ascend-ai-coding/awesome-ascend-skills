# Annotation 方言

## 1. 概述

Annotation 方言提供 IR 值标注机制，允许为任意 IR 值附加键值对属性。标注信息可用于指导后续编译 Pass 的行为（如 Bufferization 决策、缓存策略等）。

- **方言名称**：`annotation`
- **C++ 命名空间**：`::mlir::annotation`

> 源码参考：[AnnotationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/Annotation/IR/AnnotationOps.td)、[AnnotationAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/Annotation/IR/AnnotationAttrs.td)、[AnnotationBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/Annotation/IR/AnnotationBase.td)

## 2. 方言定义

```tablegen
def Annotation_Dialect : Dialect {
  let name = "annotation";
  let description = [{
    Annotation dialects for mark operations to
    define some extra attrs for a certain op:

    ```mlir
    annotation.mark %a { attr-dict } : f64
    ```
  }];
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::annotation";
}
```

## 3. 操作定义

### 3.1 annotation.mark

#### 功能

为 IR 值附加键值对属性标注。值可以是静态属性或动态 IR 值。

#### 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `src` | `AnyType` | 被标注的值 |
| `effects` | `StrArrayAttr` (默认: ["write"]) | 内存效果模式 |
| `values` | `Variadic<AnyType>` | 动态属性值 |
| `keys` | `OptionalAttr<StrArrayAttr>` | 属性键名 |

#### Traits

- `MemoryEffectsOpInterface`（自定义 getEffects）

#### MLIR 示例

基本标注：

```mlir
annotation.mark %target keys = ["key"] values = [%val] : f64
```

字典属性标注：

```mlir
annotation.mark %target {key : val} : f64
```

#### 额外类方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `isAnnotatedBy(StringRef key)` | `bool` | 判断是否被指定 key 标注 |
| `isAnnotatedByStaticAttr(StringRef key)` | `bool` | 判断是否有指定 key 的静态属性 |
| `isAnnotatedByDynamicAttr(StringRef key)` | `bool` | 判断是否有指定 key 的动态属性 |
| `getMixedAttrValue(StringRef key)` | `OpFoldResult` | 获取指定 key 的混合属性值 |
| `getStaticAttrValue(StringRef key)` | `Attribute` | 获取指定 key 的静态属性值 |
| `getDynamicAttrValue(StringRef key)` | `Value` | 获取指定 key 的动态属性值 |
| `getAttrNum()` | `int64_t` | 获取属性数量（不含 effects） |
| `isAttrEmpty()` | `bool` | 判断属性是否为空（不含 effects） |

## 4. EffectMode 枚举

| 枚举值 | 整数值 | 助记符 | 说明 |
|--------|--------|--------|------|
| `Write` | 0 | `write` | 写效果 |
| `Read` | 1 | `read` | 读效果 |

`effects` 属性默认为 `["write"]`，表示标注操作对被标注值有写效果。当标注仅用于信息传递而不影响内存行为时，可设置为 `["read"]`。

## 5. 与 Bufferization 的交互

Annotation 方言在 Bufferization 流程中发挥重要作用：

1. **缓存标注**：通过 `annotation.mark` 标记需要缓存的输入/输出
2. **别名信息**：通过 `effects` 属性告知 Bufferization 别名关系
3. **Buffer 决策**：标注信息影响 Buffer 分配和布局决策

### 5.1 缓存 IO 流程

```
hfusion-cache-io Pass
  │
  ├── 识别需要缓存的 annotation.mark 标注
  │
  ├── 插入数据搬移操作（GM <-> UB）
  │
  └── 更新 annotation.mark 的 effects 属性
```

## 6. 典型使用场景

### 6.1 标记缓存需求

```mlir
%cached = annotation.mark %input keys = ["cache"] values = [%input] : tensor<128x256xf32>
```

### 6.2 标记 Bufferization 约束

```mlir
%marked = annotation.mark %val keys = ["no_alias"] : memref<?xf32>
```

### 6.3 标记计算核心类型

```mlir
%marked = annotation.mark %result keys = ["tcore_type"] values = [%cube_val] : tensor<128x256xf32>
```
