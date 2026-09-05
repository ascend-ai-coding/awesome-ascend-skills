# 属性与枚举速查

## 1. 概述

HFusion 方言定义了丰富的枚举和属性，用于控制逐元操作的函数选择、类型转换、舍入模式、原子操作类型等。本文档提供完整的速查表。

> 源码参考：[HFusionEnums.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.td)、[HFusionBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionBase.td)、[HFusionAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td)

## 2. UnaryFn（一元函数枚举）

| 枚举值 | 整数值 | 说明 |
|--------|--------|------|
| `relu` | 0 | ReLU 激活 |
| `sqrt` | 1 | 平方根 |
| `rsqrt` | 2 | 平方根倒数 |
| `rec` | 3 | 倒数 |
| `vnot` | 4 | 按位取反 |
| `tanh` | 5 | 双曲正切 |
| `sin` | 6 | 正弦 |
| `cos` | 7 | 余弦 |
| `atan` | 8 | 反正切 |
| `tan` | 9 | 正切 |
| `absi` | 10 | 绝对值 |
| `erf` | 11 | 误差函数 |
| `log2` | 12 | 以 2 为底对数 |
| `log10` | 13 | 以 10 为底对数 |
| `log1p` | 14 | log(1+x) |
| `exp2` | 15 | 2^x |
| `expm1` | 16 | e^x - 1 |
| `ilogb` | 17 | 指数部分 |

属性语法：`unary_fn = <sqrt>`

## 3. BinaryFn（二元函数枚举）

| 枚举值 | 整数值 | 说明 |
|--------|--------|------|
| `vor` | 0 | 按位或 |
| `vand` | 1 | 按位与 |
| `vxor` | 2 | 按位异或 |
| `minf` | 3 | 浮点最小值 |
| `maxf` | 4 | 浮点最大值 |
| `powf` | 5 | 浮点幂 |
| `mod` | 6 | 取模 |
| `shli` | 7 | 左移 |
| `shrsi` | 8 | 算术右移 |
| `shrui` | 9 | 逻辑右移 |
| `ldexp` | 10 | ldexp |
| `ceildivsi` | 11 | 有符号向上取整除法 |
| `ceildivui` | 12 | 无符号向上取整除法 |
| `floordivsi` | 13 | 有符号向下取整除法 |
| `powi` | 14 | 整数幂 |
| `minnumf` | 15 | IEEE 浮点最小值 |
| `maxnumf` | 16 | IEEE 浮点最大值 |
| `modui` | 17 | 无符号取模 |
| `divfhp` | 18 | 高精度除法 |

属性语法：`binary_fn = <vand>`

## 4. CompareFn（比较函数枚举）

| 枚举值 | 整数值 | 说明 |
|--------|--------|------|
| `veq` | 0 | 等于 |
| `vne` | 1 | 不等于 |
| `vle` | 2 | 有符号小于等于 |
| `vlt` | 3 | 有符号小于 |
| `vge` | 4 | 有符号大于等于 |
| `vgt` | 5 | 有符号大于 |
| `vule` | 6 | 无符号小于等于 |
| `vuge` | 7 | 无符号大于等于 |
| `vugt` | 8 | 无符号大于 |
| `vult` | 9 | 无符号小于 |

属性语法：`compare_fn = <veq>`

## 5. TernaryFn（三元函数枚举）

| 枚举值 | 整数值 | 说明 |
|--------|--------|------|
| `select` | 0 | 条件选择 |

属性语法：`ternary_fn = <select>`

## 6. TypeFn（类型转换函数枚举）

| 枚举值 | 整数值 | 说明 |
|--------|--------|------|
| `cast_signed` | 0 | 有符号类型转换 |
| `cast_unsigned` | 1 | 无符号类型转换 |
| `bitcast` | 2 | 位转换 |

属性语法：`type_fn = <cast_signed>`

## 7. RoundMode（舍入模式枚举）

| 枚举值 | 整数值 | 助记符 | 说明 |
|--------|--------|--------|------|
| `RINT` | 0 | `rint` | 四舍五入到偶数（C 语言 rint） |
| `ROUND` | 1 | `round` | 四舍五入远离零（C 语言 round） |
| `FLOOR` | 2 | `floor` | 向负无穷取整（C 语言 floor） |
| `CEIL` | 3 | `ceil` | 向正无穷取整（C 语言 ceil） |
| `TRUNC` | 4 | `trunc` | 向零取整（C 语言 trunc） |
| `ODD` | 5 | `odd` | 向奇数取整（Von Neumann 舍入） |
| `TRUNCWITHOVERFLOW` | 6 | `truncwithoverflow` | 截断并检测溢出 |

属性语法：`round_mode = <RINT>`

## 8. UnsignedMode（无符号转换模式枚举）

| 枚举值 | 整数值 | 助记符 | 说明 |
|--------|--------|--------|------|
| `SI2SI` | 0 | `si2si` | 有符号整数到有符号整数 |
| `SI2UI` | 1 | `si2ui` | 有符号整数到无符号整数 |
| `UI2SI` | 2 | `ui2si` | 无符号整数到有符号整数 |
| `UI2UI` | 3 | `ui2ui` | 无符号整数到无符号整数 |

属性语法：`unsigned_mode = <SI2SI>`

## 9. AtomicKind（原子操作类型枚举）

| 枚举值 | 整数值 | 助记符 | 说明 |
|--------|--------|--------|------|
| `NONE` | 0 | `none` | 非原子操作 |
| `ADD` | 1 | `add` | 原子加 |
| `MAX` | 2 | `max` | 原子有符号最大值 |
| `MIN` | 3 | `min` | 原子有符号最小值 |
| `AND` | 4 | `and` | 原子按位与 |
| `OR` | 5 | `or` | 原子按位或 |
| `XOR` | 6 | `xor` | 原子按位异或 |
| `CAS` | 7 | `cas` | 原子比较并交换 |
| `XCHG` | 8 | `xchg` | 原子交换 |
| `UMAX` | 9 | `umax` | 原子无符号最大值 |
| `UMIN` | 10 | `umin` | 原子无符号最小值 |

属性语法：`atomic_kind = <add>`

## 10. FusionKind（融合类型枚举）

| 枚举值 | 整数值 | 助记符 | 说明 |
|--------|--------|--------|------|
| `PureElemwise` | 1 | `PURE_ELEMWISE` | 纯逐元融合 |
| `AnyPB` | 2 | `ANY_PB` | 任意 Producer-Consumer 融合 |
| `LastAxisPBR` | 3 | `LAST_AXIS_PBR` | 末轴 PBR 融合 |
| `AnyPBR` | 4 | `ANY_PBR` | 任意 PBR 融合 |
| `SingleCube` | 5 | `SINGLE_CUBE` | 单 Cube 融合 |
| `ShallowCV` | 6 | `SHALLOW_CV` | 浅层 Cube-Vector 融合 |
| `ShallowVV` | 7 | `SHALLOW_VV` | 浅层 Vector-Vector 融合 |
| `MixCV` | 8 | `MIX_CV` | 混合 Cube-Vector 融合 |
| `MixC2` | 9 | `MIX_C2` | 双 Cube 混合融合 |
| `Unknown` | 10 | `UNKNOWN` | 未知融合类型 |

## 11. 其他枚举

### 11.1 FlattenMode

| 枚举值 | 整数值 | 说明 |
|--------|--------|------|
| `Greedy` | 1 | 贪心展平模式 |
| `Tidy` | 2 | 整洁展平模式（带分析） |

### 11.2 OutputMode

| 枚举值 | 整数值 | 说明 |
|--------|--------|------|
| `Multiple` | 1 | 多输出模式 |
| `Single` | 2 | 单输出模式 |
| `SingleAggressive` | 3 | 激进单输出模式 |

### 11.3 ReduceWithIndexKind

| 枚举值 | 助记符 | 说明 |
|--------|--------|------|
| `MIN` | `min` | 最小值归约 |
| `MAX` | `max` | 最大值归约 |

### 11.4 CastMode

| 枚举值 | 助记符 | 说明 |
|--------|--------|------|
| `F32TOI8` | `F32TOI8` | FP32 转 INT8 |
| `F32TOI16` | `F32TOI16` | FP32 转 INT16 |
| `F16TOI8` | `F16TOI8` | FP16 转 INT8 |
| `I64TOI32` | `I64TOI32` | INT64 转 INT32 |
| `I64TOI16` | `I64TOI16` | INT64 转 INT16 |
| `I64TOI8` | `I64TOI8` | INT64 转 INT8 |
| `I32TOI16` | `I32TOI16` | INT32 转 INT16 |
| `I32TOI8` | `I32TOI8` | INT32 转 INT8 |
| `I16TOI8` | `I16TOI8` | INT16 转 INT8 |

### 11.5 TaylerMode

| 枚举值 | 助记符 | 说明 |
|--------|--------|------|
| `SIN` | `sin` | 正弦泰勒展开 |
| `ATAN` | `atan` | 反正切泰勒展开 |

### 11.6 MmMapMode

| 枚举值 | 助记符 | 说明 |
|--------|--------|------|
| `CoreOp` | `core_op` | 核心操作模式 |
| `MacroInstr` | `macro_instr` | 宏指令模式 |

## 12. HFusion 属性列表

| 属性名 | 助记符 | 说明 |
|--------|--------|------|
| `FusionKindAttr` | `fusion_kind` | 融合类型 |
| `StrideAlignDimsAttr` | `stride_align_dims` | 步长对齐维度 |
| `StrideAlignValueInByteAttr` | `stride_align_value_in_byte` | 步长对齐字节数 |
| `MultiBufferAttr` | `multi_buffer` | 多缓冲属性 |
| `BindSubBlockAttr` | `bind_sub_block` | 绑定子 Block |
| `ReduceComposeAttr` | `reduce_composed` | 归约组合 |
| `ReduceWithIndexOpAttr` | `reduce_with_index_kind` | 带索引归约类型 |
| `ReturnOperandNumAttr` | `return_operand_num` | 返回操作数编号 |
| `PaddingOptionAttr` | - | 填充选项（zero/nan） |
| `CacheModifierAttr` | - | 缓存修饰符 |
| `EvictionPolicyAttr` | - | 驱逐策略 |

### 12.1 PaddingOption

| 枚举值 | 整数值 | 助记符 | 说明 |
|--------|--------|--------|------|
| `PAD_ZERO` | 1 | `zero` | 零填充 |
| `PAD_NAN` | 2 | `nan` | NaN 填充 |

### 12.2 CacheModifier

| 枚举值 | 整数值 | 助记符 | 说明 |
|--------|--------|--------|------|
| `NONE` | 1 | `none` | 无缓存修饰 |
| `CA` | 2 | `ca` | 缓存所有 |
| `CG` | 3 | `cg` | 缓存全局 |
| `WB` | 4 | `wb` | 写回 |
| `CS` | 5 | `cs` | 缓存流 |
| `WT` | 6 | `wt` | 写穿透 |
| `CV` | 7 | `cv` | 缓存易失 |

### 12.3 EvictionPolicy

| 枚举值 | 整数值 | 助记符 | 说明 |
|--------|--------|--------|------|
| `NORMAL` | 1 | `evict_normal` | 正常驱逐 |
| `EVICT_FIRST` | 2 | `evict_first` | 优先驱逐 |
| `EVICT_LAST` | 3 | `evict_last` | 最后驱逐 |
