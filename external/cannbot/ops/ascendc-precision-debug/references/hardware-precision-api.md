# 硬件运算指令精度模式 API 汇总

> 按 DAV_ARCH 维度汇总各运算指令的精度模式选项和默认舍入行为。
> 来源：asc-devkit `include/basic_api/reg_compute/kernel_reg_compute_utils.h`。
> 精度调试时，若发现 NPU 与 golden 的中间值存在 1-ULP 差异，先查本表确认该运算指令是否有精度模式可切换。
> 新增 ARCH 时加一节，旧 ARCH 不改。

## ARCH 3510 / 5102 / 3003 / 3113（`__NPU_ARCH__ == 3510` 等）

所有运算指令的默认模式均为 `INTRINSIC`（硬件指令直通），可能不是 IEEE 754 correctly rounded。
`SpecificMode` 结构体带 `precisionMode` 字段（Div/Sqrt）或 `algo` 枚举（Exp/Ln/Log/Div/Sqrt/Rsqrt）。

### 指令总览

| 指令 | Algo 枚举 | 最高精度选项 | SpecificMode 结构体 | 命名空间 |
|------|-----------|-------------|---------------------|----------|
| **Exp** | `ExpAlgo` | `PRECISION_1ULP_FTZ_FALSE` | `ExpSpecificMode{ mrgMode, algo }` | `AscendC::Reg` |
| **Ln** | `LnAlgo` | `PRECISION_1ULP_FTZ_FALSE` | `LnSpecificMode{ mrgMode, algo }` | `AscendC::Reg` |
| **Log** | `LogAlgo` | `PRECISION_1ULP_FTZ_FALSE` | `LogSpecificMode{ mrgMode, algo }` | `AscendC::Reg` |
| **Log2** | `Log2Algo` | `PRECISION_1ULP_FTZ_FALSE` | `Log2SpecificMode{ mrgMode, algo }` | `AscendC::Reg` |
| **Log10** | `Log10Algo` | `PRECISION_1ULP_FTZ_FALSE` | `Log10SpecificMode{ mrgMode, algo }` | `AscendC::Reg` |
| **Reciprocal** | `ReciprocalAlgo` | `PRECISION_1ULP_FTZ_FALSE` | — (无 SpecificMode) | `AscendC` |
| **Div** | `DivAlgo` | `PRECISION_0ULP_FTZ_FALSE` | `DivSpecificMode{ mrgMode, precisionMode, algo }` | `AscendC::Reg` |
| **Sqrt** | `SqrtAlgo` | `PRECISION_0ULP_FTZ_FALSE` | `SqrtSpecificMode{ mrgMode, precisionMode, algo }` | `AscendC::Reg` |
| **Rsqrt** | `RsqrtAlgo` | `PRECISION_0ULP_FTZ_FALSE` | — (无 SpecificMode) | `AscendC` |

### Algo 枚举值汇总

| Algo | 可选值 | 最高精度 | 0-ULP？ |
|------|--------|----------|---------|
| `ExpAlgo` | `INTRINSIC`, `PRECISION_1ULP_FTZ_TRUE`, `PRECISION_1ULP_FTZ_FALSE` | 1-ULP | 否 |
| `LnAlgo` | `INTRINSIC`, `PRECISION_1ULP_FTZ_TRUE`, `PRECISION_1ULP_FTZ_FALSE` | 1-ULP | 否 |
| `LogAlgo` | `INTRINSIC`, `PRECISION_1ULP_FTZ_TRUE`, `PRECISION_1ULP_FTZ_FALSE` | 1-ULP | 否 |
| `Log2Algo` | `INTRINSIC`, `PRECISION_1ULP_FTZ_TRUE`, `PRECISION_1ULP_FTZ_FALSE` | 1-ULP | 否 |
| `Log10Algo` | `INTRINSIC`, `PRECISION_1ULP_FTZ_TRUE`, `PRECISION_1ULP_FTZ_FALSE` | 1-ULP | 否 |
| `ReciprocalAlgo` | `INTRINSIC`, `PRECISION_1ULP_FTZ_TRUE`, `PRECISION_1ULP_FTZ_FALSE` | 1-ULP | 否 |
| `DivAlgo` | `INTRINSIC`, `DIFF_COMPENSATION`, `PRECISION_1ULP_FTZ_TRUE`, `PRECISION_0ULP_FTZ_TRUE`, `PRECISION_0ULP_FTZ_FALSE`, `PRECISION_1ULP_FTZ_FALSE` | **0-ULP** | 是 |
| `SqrtAlgo` | `INTRINSIC`, `FAST_INVERSE`, `PRECISION_1ULP_FTZ_TRUE`, `PRECISION_0ULP_FTZ_FALSE`, `PRECISION_1ULP_FTZ_FALSE` | **0-ULP** | 是 |
| `RsqrtAlgo` | `INTRINSIC`, `FAST_INVERSE`, `PRECISION_1ULP_FTZ_TRUE`, `PRECISION_0ULP_FTZ_FALSE`, `PRECISION_1ULP_FTZ_FALSE` | **0-ULP** | 是 |

> **注意**：Sqrt/Rsqrt 的 `PRECISION_0ULP` 只有 `FTZ_FALSE` 变体（无 `FTZ_TRUE`）；Div 的 `PRECISION_0ULP` 有 `FTZ_TRUE` 和 `FTZ_FALSE` 两个变体。这是各指令实现差异，不是文档遗漏。

### ULP 级别说明

| 级别 | 含义 | 适用容差 |
|------|------|----------|
| `INTRINSIC` | 硬件指令直通，舍入模式未指定（可能截断） | max_relative 容差可吸收 |
| `1ULP` | 误差 ≤ 1 ULP | max_relative 容差可吸收；bitwise_equal 不可 |
| `0ULP` | correctly rounded（IEEE 754 round-to-nearest-even） | bitwise_equal 可能成立（前提：输入也精确） |
| `DIFF_COMPENSATION`（仅 Div） | 差分补偿算法 | 精度介于 INTRINSIC 和 1ULP 之间 |
| `FAST_INVERSE`（仅 Sqrt/Rsqrt） | 快速反平方根算法 | 精度介于 INTRINSIC 和 1ULP 之间 |

### FTZ（Flush To Zero）说明

FTZ 不是独立配置项，而是嵌在每个精度级别的枚举名中。每个精度级别（1ULP/0ULP）通常有 `FTZ_TRUE` 和 `FTZ_FALSE` 两个变体（Sqrt 的 0ULP 例外，只有 `FTZ_FALSE`）。

| FTZ 值 | 含义 | 精度影响 | 适用场景 |
|--------|------|----------|----------|
| `FTZ_TRUE` | Flush To Zero 开启——非正规数（denormal/subnormal）视为 0 | 非正规数区间（fp32: |x| < 1.18e-38）结果为 0 而非精确值 | 性能优先；输入不会出现非正规数时 |
| `FTZ_FALSE` | Flush To Zero 关闭——非正规数正常处理 | 非正规数区间结果精确 | 精度优先；输入可能包含极小值（如梯度裁剪、归一化）时 |

**调试要点**：如果 golden（numpy/CPU）正常处理非正规数而 NPU 开了 FTZ，在极小值区间会产生差异。排查时检查输入值域是否落入非正规数区间（fp32: |x| < 1.18e-38，fp16: |x| < 5.96e-8）。详细排查流程见 `patterns/ftz-subnormal-boundary.md`。

### 关键结构体定义（ARCH 3510 等）

```cpp
// 命名空间：AscendC::Reg
struct ExpSpecificMode  { MaskMergeMode mrgMode; ExpAlgo algo; };
struct LnSpecificMode   { MaskMergeMode mrgMode; LnAlgo algo; };
struct LogSpecificMode  { MaskMergeMode mrgMode; LogAlgo algo; };
struct Log2SpecificMode { MaskMergeMode mrgMode; Log2Algo algo; };
struct Log10SpecificMode{ MaskMergeMode mrgMode; Log10Algo algo; };
struct DivSpecificMode  { MaskMergeMode mrgMode; bool precisionMode; DivAlgo algo; };
struct SqrtSpecificMode { MaskMergeMode mrgMode; bool precisionMode; SqrtAlgo algo; };
```

### Cast（类型转换）

| 项目 | 说明 |
|------|------|
| 结构体 | `struct CastTrait { RegLayout layoutMode; SatMode satMode; MaskMergeMode mrgMode; RoundMode roundMode; }` |
| 命名空间 | `CastTrait` 在 `AscendC::Reg`；`RoundMode`/`SatMode` 在 `AscendC` |
| RoundMode | `CAST_NONE, CAST_RINT（就近舍入到偶数）, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC（向零截断）` |
| SatMode | `NO_SAT（不饱和）, SAT（饱和到范围边界）` |
| 关键行为差异 | fp32→int32 溢出时：`SAT` 饱和到 INT32_MAX；`NO_SAT` 行为未定义；numpy `astype(int32)` 统一返回 INT32_MIN |

### 使用示例

```cpp
// Sqrt 0-ULP（匹配 numpy np.sqrt）
constexpr AscendC::Reg::SqrtSpecificMode kHiPrecSqrt{
    AscendC::Reg::MaskMergeMode::ZEROING, true, AscendC::SqrtAlgo::INTRINSIC};
AscendC::Reg::Sqrt<float, &kHiPrecSqrt>(dst, src, mask);

// Div 0-ULP
constexpr AscendC::Reg::DivSpecificMode kHiPrecDiv{
    AscendC::Reg::MaskMergeMode::ZEROING, true, AscendC::DivAlgo::INTRINSIC};
AscendC::Reg::Div<float, float, &kHiPrecDiv>(dst, src1, src2, mask);

// Cast with SAT + CAST_TRUNC
constexpr AscendC::Reg::CastTrait kCastTraitFromFp32Int32{
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
AscendC::Reg::Cast<int32_t, float, &kCastTraitFromFp32Int32>(dst, src, mask);
```

## 其他 ARCH

> 待补充。遇到新 ARCH 时按同样格式追加。

## 排查方法

1. 用 PRINTF/DumpTensor 在运算前后打印 fp32 中间值（见 [ascendc-dumptensor.md](ascendc-dumptensor.md) 与 [printf-debug.md](printf-debug.md)）
2. 与 golden 逐元素对比，确认差异是否为 1-ULP 级别
3. 查本表确认该运算指令是否有精度模式选项
4. 若有 0-ULP 选项，切换后验证；若最高只有 1-ULP，按容差豁免流程处理
5. 若无精度模式选项，按"剩余失败定性"流程处理
