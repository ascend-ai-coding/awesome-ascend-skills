# HIVM 接口方法列表

> 关键词：HIVMStructuredOpInterface, HIVMCoreTypeInterface, HIVMInferCoreTypeInterface, HIVMUnitFlagEnabledInterface, OpLayoutInterface

## 概述

HIVM 定义了多个 OpInterface 和 AttrInterface，用于描述操作的通用行为。这些接口是 HIVM 编译器 Pass（如同步注入、Lowering、Bufferization 等）与具体操作之间的契约。

## HIVMStructuredOpInterface

源码：[HIVMInterfaces.td#L67-L678](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td#L67-L678)

HIVM 结构化操作接口，继承自以下接口：
- `DestinationStyleOpInterface`：DPS 目标风格操作
- `OpPipeInterface`：Pipe 信息
- `HIVMCoreTypeInterface`：Core 类型查询
- `FlattenInterface`：维度展平
- `LibraryFunctionOpInterface`：库函数调用

### 方法列表

#### 操作属性查询

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `isElemwiseNaryOp()` | bool | 是否为逐元素 N-ary 操作 |
| `isInlineBroadcastable()` | bool | 是否支持内联广播 |
| `isInlineTransposable()` | bool | 是否支持内联转置 |
| `existInlineBroadcastLoopDims()` | bool | 是否存在内联广播维度 |
| `existInlineTransposeLoopDims()` | bool | 是否存在内联转置维度 |

#### 循环类型处理

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getIteratorTypesArray()` | `SmallVector<IteratorType>` | 获取迭代器类型数组 |
| `setIteratorTypesArray(IteratorType, DenseI64ArrayAttr&)` | LogicalResult | 设置迭代器类型数组 |
| `getNumLoops()` | unsigned | 获取循环总数 |
| `getNumParallelLoops()` | unsigned | 获取并行循环数 |
| `getParallelLoopDims(SmallVectorImpl<int64_t>&)` | void | 获取并行循环维度 |
| `getReductionLoopDims(SmallVectorImpl<int64_t>&)` | void | 获取归约循环维度 |
| `getBroadcastLoopDims(SmallVectorImpl<int64_t>&)` | void | 获取广播循环维度 |
| `getTransposeLoopDims(SmallVectorImpl<int64_t>&)` | void | 获取转置循环维度 |
| `getPadLoopDims(SmallVectorImpl<int64_t>&)` | void | 获取填充循环维度 |
| `getConcatLoopDims(SmallVectorImpl<int64_t>&)` | void | 获取拼接循环维度 |
| `getGatherLoopDims(SmallVectorImpl<int64_t>&)` | void | 获取收集循环维度 |
| `getPermutationArray()` | `ArrayRef<int64_t>` | 获取转置排列数组 |
| `getBroadcastArray()` | `ArrayRef<int64_t>` | 获取广播数组 |
| `getInlinedBroadcastableAxes(OpOperand*)` | `SmallVector<int64_t>` | 获取内联广播轴 |

#### Indexing Maps

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getIndexingMaps()` | ArrayAttr | 获取 indexing maps 属性 |
| `getIndexingMapsArray()` | `SmallVector<AffineMap>` | 获取 indexing maps 数组 |
| `getLoopsToShapesMap()` | AffineMap | 获取循环到形状的映射 |
| `getShapesToLoopsMap()` | AffineMap | 获取形状到循环的映射 |
| `getMatchingIndexingMap(OpOperand*)` | AffineMap | 获取操作数对应的 indexing map |
| `getIndexingMapMatchingResult(OpResult)` | AffineMap | 获取结果对应的 indexing map |

#### 形状与 Rank

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getRank(OpOperand*)` | int64_t | 获取操作数 rank |
| `getShape(OpOperand*)` | `ArrayRef<int64_t>` | 获取操作数形状 |
| `getStaticShape()` | `SmallVector<int64_t>` | 获取静态形状 |
| `hasDynamicShape()` | bool | 是否有动态形状 |
| `hasIndexSemantics()` | bool | 是否有索引语义 |

#### 内存效果与操作数

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getEffects(SmallVectorImpl<...>&)` | void | 获取内存效果 |
| `getTargetSpaceOperands(AddressSpace, bool)` | `SmallVector<Value>` | 获取目标地址空间操作数 |
| `getHIVMOperands(bool)` | `SmallVector<OpOperand*>` | 获取 HIVM 操作数 |
| `getHIVMOperandTypes(bool)` | `SmallVector<Type>` | 获取 HIVM 操作数类型 |
| `getHIVMInputOperands(bool)` | `SmallVector<OpOperand*>` | 获取 HIVM 输入操作数 |
| `isVectorOnlyOperand(size_t)` | bool | 检查操作数是否仅支持 Vector |
| `getContiguousAxes()` | BitVector | 获取连续轴掩码 |
| `getUnitAxesMask()` | BitVector | 获取单位轴掩码 |
| `getPermutedAxesMask()` | BitVector | 获取置换轴掩码 |

#### 额外声明

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `createFlatListOfOperandDims(OpBuilder&, Location)` | `SmallVector<OpFoldResult>` | 创建操作数维度平面列表 |
| `createFlatListOfOperandStaticDims()` | `SmallVector<int64_t, 4>` | 创建静态维度平面列表 |
| `createLoopRanges(OpBuilder&, Location)` | `SmallVector<Range, 4>` | 创建循环范围 |
| `computeStaticLoopSizes()` | `SmallVector<int64_t, 4>` | 计算静态循环大小 |
| `reifyResultShapes(OpBuilder&, ReifiedRankedShapedTypeDims&)` | LogicalResult | 具体化结果形状 |
| `getIndexingMapIndex(OpOperand*)` | int64_t | 获取 indexing map 索引 |

## HIVMCoreTypeInterface

源码：[HIVMInterfaces.td#L27-L46](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td#L27-L46)

Core 类型查询接口，用于确定操作在哪种 Core 上执行。

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getCoreType()` | `optional<TCoreType>` | 获取操作的 Core 类型 |

Core 类型有两种确定方式：
1. **静态**：通过 `CoreTypeTrait` 附加到操作上
2. **动态**：通过 `InferCoreTypeInterface` 推断

## HIVMInferCoreTypeInterface

源码：[HIVMInterfaces.td#L48-L65](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td#L48-L65)

Core 类型推断接口，用于动态推断操作的 Core 类型。

| 方法 | 返回类型 | 默认实现 | 说明 |
|------|---------|---------|------|
| `inferCoreType()` | `optional<TCoreType>` | 返回 `std::nullopt` | 推断操作的 Core 类型 |

## HIVMUnitFlagEnabledInterface

源码：[HIVMInterfaces.td#L680-L761](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td#L680-L761)

UnitFlag 启用接口，用于支持 UnitFlag 同步模式的操作。

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getUnitFlagModes()` | `optional<SmallVector<UnitFlagAttr>>` | 获取 UnitFlag 模式数组 |
| `setUnitFlagModes(SmallVector<UNIT_FLAG>)` | void | 设置 UnitFlag 模式 |
| `getUnitFlagConditions()` | `optional<SmallVector<Value>>` | 获取 UnitFlag 条件值 |
| `setUnitFlagConditions(SmallVector<Value>)` | void | 设置 UnitFlag 条件值 |
| `getUnitFlagModeLibValue(PatternRewriter&)` | Value | 获取传递给库调用的 UnitFlag 值 |

### 验证

该接口包含验证方法 `verifyUnitFlagEnabledInterface`，确保操作正确声明了 `unit_flag_mode` 和 `unit_flag_cond` 参数。

## OpLayoutInterface

源码：[OpLayoutInterface.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.td)

布局接口，用于确定操作数的目标 Fractal Layout。

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getOperandsTargetFractalLayout()` | 需要操作自行实现 | 获取操作数的目标 Fractal Layout |

mmadL1 实现了此接口，提供以下方法：

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getOperandALayout()` | `FailureOr<DataLayoutAttr>` | 获取 A 矩阵布局 |
| `getOperandBLayout()` | `FailureOr<DataLayoutAttr>` | 获取 B 矩阵布局 |
| `getOperandCLayout()` | `FailureOr<DataLayoutAttr>` | 获取 C 矩阵布局 |
| `getOperandBiasLayout()` | `FailureOr<DataLayoutAttr>` | 获取 Bias 布局 |

## 常见问题

**Q: HIVMStructuredOpInterface 和 LinalgOp 的关系？**
A: HIVMStructuredOpInterface 借鉴了 LinalgOp 的设计，但增加了 Pipe、CoreType、Flatten 等 NPU 特有的接口。两者都基于 DestinationStyleOpInterface。

**Q: 什么时候需要实现 HIVMInferCoreTypeInterface？**
A: 当操作的 Core 类型无法静态确定时（如同步操作可能在 Cube 或 Vector 上执行），需要实现此接口动态推断。

**Q: OpLayoutInterface 的用途？**
A: 用于确定操作数在 NPU 存储层次中的数据布局（如 Fractal 格式），编译器据此生成正确的数据搬运指令。

## 相关文档

- 源码参考：[HIVMInterfaces.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td)
- 源码参考：[OpLayoutInterface.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.td)
