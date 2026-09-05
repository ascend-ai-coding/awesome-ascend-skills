# HIVM 向量操作标量降级

> 关键词：HIVM, ImplByScalarOpInterface, shouldLowerToScalarLoops, lowerToLoops, scalar, i64, SIMT VF

## 概述

HIVM 向量操作在编译过程中，部分操作在特定条件下会被**降级为标量循环**（Scalar Lowering）执行，而非使用硬件向量指令。这是因为 AscendNPU 的向量计算单元对某些数据类型和操作组合缺乏硬件级支持，编译器通过 `ImplByScalarOpInterface` 接口自动将这类操作退化为逐元素的标量循环。

标量降级会显著影响性能：向量指令可以一次处理多个数据元素，而标量循环逐元素执行，吞吐量大幅降低。因此，在编写 Triton 算子时，理解哪些操作在什么条件下会被标量降级，对于性能优化至关重要。

> 本文档面向辅助人类编写和优化 Triton 算子的 AI agent，帮助识别和避免标量降级导致的性能瓶颈。

## 标量降级机制

### 接口定义

`ImplByScalarOpInterface` 定义在 [ImplByScalarOpInterface.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Interfaces/ImplByScalarOpInterface.td)，声明了两个关键方法：

| 方法 | 说明 |
|------|------|
| `shouldLowerToScalarLoops()` | 判断当前操作是否应该降级为标量循环 |
| `lowerToLoops(RewriterBase &b)` | 执行实际的标量循环降级 |

### 降级流程

```
HIVM 向量操作
    │
    ├── shouldLowerToScalarLoops() 返回 true？
    │     │
    │     ├── 是 → HIVMLowerToLoopsPass 调用 lowerToLoops()
    │     │         │
    │     │         └── 创建 scf.for 嵌套循环
    │     │             循环体内逐元素执行 arith 标量操作
    │     │
    │     └── 否 → 保持向量操作，后续由硬件向量指令执行
    │               │
    │               ├── Execution Engine 路径：→ hfusion 操作
    │               └── TritonGPU 路径：→ arith 向量操作
```

### 共同前提条件

所有标量降级的共同前提是 **`hasPureBufferSemantics()` 为 true**，即操作必须具有纯 buffer 语义（memref 语义，而非 tensor 语义）。在 bufferization 阶段之后，大部分操作都会满足此条件。

### 降级实现

当 `shouldLowerToScalarLoops()` 返回 true 时，[LowerToLoops.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/LowerToLoops.cpp) 会将向量操作分解为嵌套 `scf.for` 循环，循环体内逐元素执行对应的 `arith` 标量操作：

| HIVM 操作 | 标量降级后的 arith 操作 |
|-----------|----------------------|
| VAddOp | arith.AddIOp / arith.AddFOp |
| VSubOp | arith.SubIOp |
| VMulOp | arith.MulIOp / arith.MulFOp |
| VMinOp | arith.MinSIOp |
| VMaxOp | arith.MaxSIOp |
| VAbsOp | math.AbsIOp |
| VShLOp | arith.ShLIOp |
| VShROp | arith.ShRSIOp |
| VCmpOp | arith.CmpIOp + arith.ExtUIOp (i1→i8) |
| VMulExtOp | arith.MulUIExtendedOp |
| VCumsumOp | arith.AddIOp / arith.AddFOp（含前一轮累积值） |
| VCumprodOp | arith.MulIOp / arith.MulFOp（含前一轮累积值） |
| VReduceOp | 依 reduceOp 不同：MinSI/MaxSI/AddI/MulI/XOrI 及 argmin/argmax 复合逻辑 |

### 对其他编译 Pass 的影响

标量降级判断不仅影响 `HIVMLowerToLoopsPass`，还会影响以下 Pass 的行为：

| Pass | 影响 |
|------|------|
| [OptMemPlanForPipeline](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/Transforms/OptMemPlanForPipeline.cpp#L18-L23) | 标量降级的操作使用不同的 buffer 规划策略 |
| [AdjustAlignUtil](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/Transforms/AlignBuffer/AdjustAlignUtil.cpp#L414-L419) | 标量降级的操作跳过 stride 对齐调整（标量操作不需要向量对齐） |
| [HIVMDecomposeOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/Transforms/HIVMDecomposeOp.cpp#L655-L712) | 对 VCmpOp 做特殊的 i1→i8 输出转换预处理 |

## 第 1 组：通用算术操作

**适用操作**：VAddOp, VSubOp, VMulOp, VMinOp, VMaxOp, VAbsOp, VShLOp, VShROp, VInterleaveOp, VDeinterleaveOp

这 10 个操作共享相同的 `shouldLowerToScalarLoops` 逻辑，通过 `ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL` 宏生成。

源码参考：[ShouldLowerToScalarLoops.cpp:56-64](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp#L56-L64)

### 降级条件

```
hasPureBufferSemantics() == true
AND (isSIMTVF() == true OR elemType == i64)
```

| 条件 | 说明 |
|------|------|
| hasPureBufferSemantics() | 必须具有纯 buffer 语义 |
| isSIMTVF() | 操作处于 SIMT VF 模式（带有 `VFMode::SIMT` 属性） |
| elemType.isInteger(64) | 第一个操作数的元素类型为 i64 |

### 降级判断矩阵

| 元素类型 | SIMD VF 模式 | SIMT VF 模式 |
|---------|-------------|-------------|
| f16 | 不降级 ✅ | **降级** ⚠️ |
| f32 | 不降级 ✅ | **降级** ⚠️ |
| i8 | 不降级 ✅ | **降级** ⚠️ |
| i16 | 不降级 ✅ | **降级** ⚠️ |
| i32 | 不降级 ✅ | **降级** ⚠️ |
| i64 | **降级** ⚠️ | **降级** ⚠️ |

### 各操作受影响的数据类型

| 操作 | IR 支持的元素类型 | 会被标量降级的类型 |
|------|-----------------|------------------|
| VAddOp | I8, I16, I32, F16, F32, I64 | I64（SIMD）/ 全部（SIMT） |
| VSubOp | I8, I16, I32, F16, F32, I64 | I64（SIMD）/ 全部（SIMT） |
| VMulOp | I16, I32, F16, F32, I64 | I64（SIMD）/ 全部（SIMT） |
| VMinOp | I16, I32, F16, F32, I64 | I64（SIMD）/ 全部（SIMT） |
| VMaxOp | I16, I32, F16, F32, I64 | I64（SIMD）/ 全部（SIMT） |
| VAbsOp | F16, F32, I8, I16, I32, I64 | I64（SIMD）/ 全部（SIMT） |
| VShLOp | I16, I32, I64 | I64（SIMD）/ 全部（SIMT） |
| VShROp | I16, I32, I64 | I64（SIMD）/ 全部（SIMT） |
| VInterleaveOp | I16, F16, I32, F32, BF16, I64 等 | I64（SIMD）/ 全部（SIMT） |
| VDeinterleaveOp | I8, I16, F16, I32, F32, BF16, I64 等 | I64（SIMD）/ 全部（SIMT） |

### 优化建议

- **避免 i64 类型的算术运算**：i64 是最常见的标量降级触发条件。如果精度允许，优先使用 i32 类型
- **SIMT VF 模式下所有算术操作都会降级**：SIMT 模式下向量操作退化为标量循环是设计如此，因为 SIMT 模式本身就是标量线程模型
- **浮点运算不受影响**（SIMD 模式下）：f16/f32 的加减乘、最大最小值在 SIMD 模式下均走向量路径

## 第 2 组：比较操作（VCmpOp）

源码参考：[ShouldLowerToScalarLoops.cpp:92-114](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp#L92-L114)

### 降级条件

```
hasPureBufferSemantics() == true
AND src[0] 是 MemRefType 或 TensorType
AND elemType 是整数类型
AND (elemType != i32 OR compare_mode ∉ {EQ, NE})
```

### 降级判断矩阵

| 元素类型 | EQ | NE | LT | GT | LE | GE |
|---------|----|----|----|----|----|-----|
| f16 | 不降级 ✅ | 不降级 ✅ | 不降级 ✅ | 不降级 ✅ | 不降级 ✅ | 不降级 ✅ |
| f32 | 不降级 ✅ | 不降级 ✅ | 不降级 ✅ | 不降级 ✅ | 不降级 ✅ | 不降级 ✅ |
| i8 | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ |
| i16 | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ |
| i32 | 不降级 ✅ | 不降级 ✅ | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ |
| i64 | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ | **降级** ⚠️ |

### 特殊处理

VCmpOp 标量降级时有额外的 i1→i8 扩展处理：

1. **HIVMDecomposeOp 预处理**：将 vcmp 的输出从 i1 转为 i8 临时 buffer（因为 store 不支持 i1），之后用 `VCastOp` 转回 i1
2. **LowerToLoops 后处理**：`arith.CmpIOp` 产生 i1 结果，通过 `arith.ExtUIOp` 零扩展为 i8 后再存储

### 优化建议

- **浮点比较始终走向量路径**：f16/f32 的所有比较模式都由硬件向量指令执行，性能最优
- **i32 的相等/不等比较走向量路径**：i32 的 EQ/NE 是唯一能走向量路径的整数比较
- **避免整数大小比较**：i32 的 LT/GT/LE/GE 以及 i8/i16/i64 的所有比较都会标量降级
- **整数比较的替代策略**：如果业务逻辑允许，将整数比较转为浮点比较（先 cast 再 compare），可避免标量降级
- **vcmp + vsel 模式**：vcmp 的典型使用模式是与 vsel 配合，整数比较的标量降级会使整个条件选择链路性能下降

## 第 3 组：扩展乘法（VMulExtOp）

源码参考：[ShouldLowerToScalarLoops.cpp:120-126](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp#L120-L126)

### 降级条件

```
hasPureBufferSemantics() == true
AND (elemType == i32 OR elemType == i64)
```

### 降级判断矩阵

| 元素类型 | 是否降级 |
|---------|---------|
| i32 | **降级** ⚠️ |
| i64 | **降级** ⚠️ |

注意：VMulExtOp 的 IR 定义仅支持 I32 类型，因此实际上 **vmulext 在所有情况下都会被标量降级**。

### 优化建议

- **vmulext 始终走标量路径**：该操作没有向量硬件支持，应尽量避免使用
- **替代方案**：如果需要高 32 位乘法结果，考虑使用 vmulextended（I16 输入）或手动拆分乘法

## 第 4 组：累积操作（VCumsumOp, VCumprodOp）

源码参考：[ShouldLowerToScalarLoops.cpp:22-48](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp#L22-L48)

### 降级条件

```
hasPureBufferSemantics() == true
AND cumDims.size() == 1
AND (elemType == i64 OR flattenedCumDims[0] == flattenedRank - 1)
```

### 降级判断详解

| 条件 | 说明 |
|------|------|
| cumDims.size() == 1 | 累积维度只能有 1 个，多个累积维度不降级 |
| elemType == i64 | 目标元素类型为 i64 时直接降级 |
| flattenedCumDims[0] == flattenedRank - 1 | 累积维度在 flatten 后是最后一个维度时降级 |

### 降级判断矩阵

| 元素类型 | 累积维度 = 最后维度 | 累积维度 ≠ 最后维度 | 多个累积维度 |
|---------|-------------------|-------------------|------------|
| f16/f32/bf16 | **降级** ⚠️ | 不降级 ✅ | 不降级 ✅ |
| i8/i16/i32 | **降级** ⚠️ | 不降级 ✅ | 不降级 ✅ |
| i64 | **降级** ⚠️ | **降级** ⚠️ | 不降级 ✅ |

### 优化建议

- **避免 i64 累积操作**：i64 类型的累积操作无论维度如何都会标量降级
- **注意累积维度位置**：非 i64 类型下，累积维度为最后维度时会触发降级。如果可能，调整数据布局使累积维度不在最后
- **多个累积维度不降级**：当 cumDims > 1 时不会标量降级，但这种情况可能触发其他限制

## 第 5 组：归约操作（VReduceOp）

源码参考：[ShouldLowerToScalarLoops.cpp:132-281](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp#L132-L281)

VReduceOp 的标量降级逻辑是最复杂的，取决于硬件架构类型和归约操作类型。

### 降级条件

```
hasPureBufferSemantics() == true
AND shouldVReduceOpDecomposeToScalarImpl() == true
```

### Reg-based 架构（A5 代：Ascend310B, Ascend950）

Reg-based（寄存器基）架构的核间同步通过寄存器级指令（SetFlag/WaitFlag）实现，在归约操作上有更好的硬件向量支持。

| reduceOp | 降级条件 |
|----------|---------|
| max_with_index | 内存访问对齐不合法时降级 |
| min_with_index | 内存访问对齐不合法时降级 |
| 其他（sum/prod/max/min/xori 等） | **不降级** ✅ |

内存访问对齐合法性由 [isLegalAccessAlignment](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp#L139-L219) 判断：检查操作数和结果的步幅布局是否满足对齐要求。如果缺少 `StridedLayoutAttr`，默认判定为不合法，需要降级。

### Mem-based 架构（A2/A3 代：Ascend910B, Ascend910_93）

Mem-based（内存基）架构的核间同步依赖 FFTS（Fast Flag Transmit Storage）内存机制，归约操作的标量降级条件更多。

| reduceOp | 降级条件 |
|----------|---------|
| sum / prod / max / min / xori | 元素类型为 i64 |
| max_with_index / min_with_index | i64/i32/i16 → 降级；f16/f32/bf16 且 flatten 后 rank > 2 → 降级；其他 → 不降级 |
| any / all / ori / andi / none | **不降级** ✅ |

### 降级判断矩阵（Mem-based 架构）

#### 基本归约（sum/prod/max/min/xori）

| 元素类型 | 是否降级 |
|---------|---------|
| f16 | 不降级 ✅ |
| f32 | 不降级 ✅ |
| i8/i16/i32 | 不降级 ✅ |
| i64 | **降级** ⚠️ |

#### 带索引极值归约（max_with_index/min_with_index）

| 元素类型 | flatten rank ≤ 2 | flatten rank > 2 |
|---------|-----------------|-----------------|
| f16 | 不降级 ✅ | **降级** ⚠️ |
| f32 | 不降级 ✅ | **降级** ⚠️ |
| bf16 | 不降级 ✅ | **降级** ⚠️ |
| i16 | **降级** ⚠️ | **降级** ⚠️ |
| i32 | **降级** ⚠️ | **降级** ⚠️ |
| i64 | **降级** ⚠️ | **降级** ⚠️ |

### 优化建议

- **基本归约优先使用 f32**：sum/prod/max/min/xori 在 f32 下始终走向量路径
- **避免 i64 归约**：i64 的基本归约和带索引极值归约都会标量降级
- **argmax/argmin 注意维度**：f16/f32/bf16 的 max_with_index/min_with_index 在高维（flatten rank > 2）时会降级，尽量保持低维
- **整数 argmax/argmin 在 Mem-based 架构上始终降级**：i16/i32/i64 的带索引极值归约无法走向量路径
- **Reg-based 架构（A5 代）优势**：Ascend310B/950 等芯片对基本归约有更好的硬件支持，但 argmax/argmin 仍需关注内存对齐

## 完整汇总表

| 序号 | 操作 | 助记符 | 标量降级触发条件 | 最常见触发因素 |
|------|------|--------|----------------|-------------|
| 1 | VAddOp | hir.vadd | i64 或 SIMT VF | i64 类型 |
| 2 | VSubOp | hir.vsub | i64 或 SIMT VF | i64 类型 |
| 3 | VMulOp | hir.vmul | i64 或 SIMT VF | i64 类型 |
| 4 | VMinOp | hir.vmin | i64 或 SIMT VF | i64 类型 |
| 5 | VMaxOp | hir.vmax | i64 或 SIMT VF | i64 类型 |
| 6 | VAbsOp | hir.vabs | i64 或 SIMT VF | i64 类型 |
| 7 | VShLOp | hir.vshl | i64 或 SIMT VF | i64 类型 |
| 8 | VShROp | hir.vshr | i64 或 SIMT VF | i64 类型 |
| 9 | VInterleaveOp | hir.vinterleave | i64 或 SIMT VF | i64 类型 |
| 10 | VDeinterleaveOp | hir.vdeinterleave | i64 或 SIMT VF | i64 类型 |
| 11 | VCmpOp | hir.vcmp | 整数类型 且 (非 i32 或 非 EQ/NE) | 整数大小比较 |
| 12 | VMulExtOp | hir.vmulext | i32 或 i64（即始终降级） | 无向量硬件支持 |
| 13 | VCumsumOp | hir.vcumsum | i64 或 累积维度为最后维度 | 累积维度位置 |
| 14 | VCumprodOp | hir.vcumprod | i64 或 累积维度为最后维度 | 累积维度位置 |
| 15 | VReduceOp | hir.vreduce | 依架构和 reduceOp 类型（见上方详解） | i64 / argmax/argmin |

## Triton 算子优化速查

以下是从 Triton 算子编写角度的快速优化参考，帮助避免标量降级导致的性能问题。

### 数据类型选择

| 场景 | 推荐类型 | 避免类型 | 原因 |
|------|---------|---------|------|
| 算术运算（加减乘、最大最小） | f32, i32 | i64 | i64 触发标量降级 |
| 比较运算 | f32 | i8, i16, i64 | 整数比较大部分会标量降级 |
| 整数相等/不等比较 | i32 | i8, i16, i64 | 仅 i32 的 EQ/NE 走向量路径 |
| 归约运算 | f32 | i64 | i64 归约标量降级 |
| argmax/argmin | f32（低维） | i16, i32, i64 | 整数 argmax/argmin 始终降级 |

### 操作模式选择

| 场景 | 推荐做法 | 避免做法 | 原因 |
|------|---------|---------|------|
| 整数大小比较 | 先 cast 为浮点再比较 | 直接整数 LT/GT/LE/GE | 整数大小比较标量降级 |
| 累积操作 | 累积维度不在最后维度 | 累积维度为最后维度 | 最后维度累积会标量降级 |
| 高精度乘法 | vmulextended (I16) | vmulext (I32) | vmulext 始终标量降级 |

### 常见性能陷阱

1. **i64 陷阱**：i64 是最常见的标量降级触发因素。几乎所有向量操作在 i64 类型下都会降级为标量循环。如果业务逻辑允许，应尽量避免使用 i64
2. **整数比较陷阱**：除 i32 的 EQ/NE 外，所有整数比较都会标量降级。这在 `where` 条件选择模式（vcmp + vsel）中尤其影响性能
3. **vmulext 陷阱**：vmulext 在 IR 层面仅支持 I32，而 I32 恰好触发标量降级，导致该操作实际上始终走标量路径
4. **累积维度陷阱**：非 i64 类型的累积操作在累积维度为最后维度时仍会标量降级，需要关注数据布局

## 常见问题

**Q: 标量降级对性能的影响有多大？**
A: 标量降级将向量操作退化为逐元素的标量循环，性能损失通常在 10x-100x 量级，具体取决于向量宽度和操作复杂度。对于计算密集型算子，避免标量降级是最重要的优化手段之一。

**Q: 如何判断我的 Triton 算子是否触发了标量降级？**
A: 可以通过查看编译后的 HIVM IR，检查是否存在 `scf.for` 循环包裹 `arith` 标量操作的模式。也可以在编译时启用调试日志，观察 `shouldLowerToScalarLoops` 的判断结果。

**Q: SIMT VF 模式下为什么所有算术操作都降级？**
A: SIMT（Single Instruction Multiple Threads）模式本身就是标量线程模型，每个线程独立执行标量操作。在这种模式下，向量操作退化为标量循环是设计如此，不存在向量指令可用。

**Q: 为什么 i64 类型总是触发标量降级？**
A: AscendNPU 的向量计算单元对 i64 类型的硬件支持有限。大部分向量指令不支持 i64 数据类型，编译器只能退化为标量循环逐元素处理。

**Q: VCmpOp 的整数比较为什么只有 i32 的 EQ/NE 能走向量路径？**
A: 硬件向量比较指令仅支持 i32 的相等/不等判断和浮点类型的全部比较模式。i32 的大小比较（LT/GT/LE/GE）以及其他整数宽度的比较没有对应的向量指令。

**Q: 标量降级后结果是否正确？**
A: 是的。标量降级是功能等价的变换，仅影响性能，不影响计算结果的正确性。标量循环逐元素执行与向量指令批量执行在数学上产生相同的结果。

**Q: 可以通过编译选项禁用标量降级吗？**
A: 不建议这样做。标量降级是因为硬件不支持对应的向量操作，如果强制禁用，编译会在后续阶段失败。正确的做法是调整数据类型或操作模式，避免触发标量降级条件。

## 相关文档

- 各操作的详细文档：
  - [01-unary-ops.md](../../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md)（VAbsOp）
  - [02-binary-ops.md](../../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md)（VAddOp, VSubOp, VMulOp, VMinOp, VMaxOp）
  - [05-compare-ops.md](../../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/05-compare-ops.md)（VCmpOp）
  - [06-shift-ops.md](../../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/06-shift-ops.md)（VShLOp, VShROp）
  - [07-reduction-ops.md](../../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/07-reduction-ops.md)（VReduceOp）
  - [08-data-movement.md](../../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/08-data-movement.md)（VInterleaveOp, VDeinterleaveOp）
  - [09-cumulative-sort.md](../../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/09-cumulative-sort.md)（VCumsumOp, VCumprodOp）
  - [10-special-ops.md](../../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md)（VMulExtOp）
- 编译流水线文档：[04-hivm-transforms.md](../../../docs_ascendnpu_ir/06-Compilation-Pipeline/04-hivm-transforms.md)（HIVMLowerToLoopsPass）
- 源码参考：
  - [ImplByScalarOpInterface.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Interfaces/ImplByScalarOpInterface.td) — 接口定义
  - [ShouldLowerToScalarLoops.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp) — 降级判断逻辑
  - [LowerToLoops.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/LowerToLoops.cpp) — 降级实现
  - [HIVMLowerToLoops.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/Transforms/HIVMLowerToLoops.cpp) — 降级 Pass 入口
