# 跨代际迁移 API 差异最佳实践（Subnormal 与超越函数）

> **适用 API**：Exp / Ln / Sqrt / Rsqrt / Div / Reciprocal
> **适用平台**：全平台可用；Subnormal（次正规数）处理行为在 DAV_3510（arch35，Ascend 950 系列）与 DAV_2201（arch22，Ascend 910B/910_93 系列）间存在代际差异，详见下文。
> **典型场景**：跨代迁移（DAV_2201 → DAV_3510）时的精度对齐；Softmax、LayerNorm、RMSNorm、RoPE 等含超越函数/除法的算子。

---

## 1. 背景：DAV_3510 的 Subnormal 裁剪

DAV_3510（arch35）裁剪了 subnormal（次正规数）硬件能力：当输入或计算结果为 subnormal 时会被直接置 0（FTZ, Flush-To-Zero）。DAV_2201（arch22）默认支持 subnormal 计算。

> SubNormal 浮点数：指数位全为 0、尾数不为 0，用于表示比最小正常数更小的值，避免"下溢为 0"。

受影响的计算模式：`Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal` 的输入为 subnormal、或计算中间结果/最终结果落入 subnormal 区间。

## 2. 受影响 API 清单

| API | Config 结构体 | algo 参数选项 |
|-----|-------------|-------------|
| Exp | ExpConfig | `ExpAlgo::INTRINSIC` / `PRECISION_1ULP_FTZ_TRUE` / `PRECISION_1ULP_FTZ_FALSE` |
| Ln | LnConfig | `LnAlgo::INTRINSIC` / `PRECISION_1ULP_FTZ_TRUE` / `PRECISION_1ULP_FTZ_FALSE` |
| Sqrt | SqrtConfig | `SqrtAlgo::INTRINSIC` / `FAST_INVERSE` / `PRECISION_1ULP_FTZ_TRUE` / `PRECISION_0ULP_FTZ_FALSE` / `PRECISION_1ULP_FTZ_FALSE` |
| Rsqrt | RsqrtConfig | `RsqrtAlgo::INTRINSIC` / `FAST_INVERSE` / `PRECISION_1ULP_FTZ_TRUE` / `PRECISION_0ULP_FTZ_FALSE` / `PRECISION_1ULP_FTZ_FALSE` |
| Div | DivConfig | `DivAlgo::INTRINSIC` / `DIFF_COMPENSATION` / `PRECISION_1ULP_FTZ_TRUE` / `PRECISION_0ULP_FTZ_TRUE` / `PRECISION_0ULP_FTZ_FALSE` / `PRECISION_1ULP_FTZ_FALSE` |
| Reciprocal | ReciprocalConfig | `ReciprocalAlgo::INTRINSIC` / `PRECISION_1ULP_FTZ_TRUE` / `PRECISION_1ULP_FTZ_FALSE` |

## 3. algo 参数含义

| algo 值 | 含义 | Subnormal 处理 | 性能 |
|---------|------|---------------|------|
| `INTRINSIC` | 单指令计算（**默认值**） | Subnormal 被近似为 0 | 最快 |
| `PRECISION_1ULP_FTZ_TRUE` | 单指令计算 | Subnormal 被近似为 0 | 快 |
| `PRECISION_1ULP_FTZ_FALSE` | 软件模拟，精度扩展 | 支持 Subnormal 计算，避免下溢为 0 | **慢（性能影响大）** |

## 4. 使用示例

### 4.1 高性能模式（默认，subnormal → 0）

```cpp
constexpr AscendC::LnConfig LN_CONFIG_FAST = { AscendC::LnAlgo::INTRINSIC };
AscendC::Ln<T, LN_CONFIG_FAST>(dstLocal, srcLocal, count);
```

### 4.2 高精度模式（支持 subnormal，软件模拟）

```cpp
constexpr AscendC::LnConfig LN_CONFIG_PRECISE = { AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE };
AscendC::Ln<T, LN_CONFIG_PRECISE>(dstLocal, srcLocal, count);
```

### 4.3 MicroAPI（Register-based）模式

通过 `XxxSpecificMode` 结构体配置：

```cpp
__VEC_SCOPE__
{
    constexpr MicroAPI::LnSpecificMode LN_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE
    };
    constexpr MicroAPI::ExpSpecificMode EXP_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        AscendC::ExpAlgo::PRECISION_1ULP_FTZ_FALSE
    };
    constexpr MicroAPI::SqrtSpecificMode SQRT_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        false,
        AscendC::SqrtAlgo::PRECISION_1ULP_FTZ_FALSE
    };
    constexpr MicroAPI::DivSpecificMode DIV_SUBNORMAL_MODE = {
        MicroAPI::MaskMergeMode::ZEROING,
        false,
        AscendC::DivAlgo::PRECISION_1ULP_FTZ_FALSE
    };

    RegTensor<float> regDst, regSrc;
    MaskReg pMask = CreateMask<float, MaskPattern::ALL>();
    MicroAPI::Ln<float, &LN_SUBNORMAL_MODE>(regDst, regSrc, pMask);
    MicroAPI::Exp<float, &EXP_SUBNORMAL_MODE>(regDst, regSrc, pMask);
    MicroAPI::Sqrt<float, &SQRT_SUBNORMAL_MODE>(regDst, regSrc, pMask);
    MicroAPI::Div<float, &DIV_SUBNORMAL_MODE>(regDst, regSrc, pMask);
}
```

## 5. 注意事项

1. **`PRECISION_1ULP_FTZ_FALSE` 性能影响大**：仅在对 subnormal 精度有明确要求的路径启用；性能敏感算子建议按 tiling 参数或属性开 high_precision 分支，仅在需要时切换。
2. **替代方案——eps 规避**：输入数据范围已知时，可在计算路径引入小 eps 常数（大于该 dtype 最小正常数）避免输入落入 subnormal 区间，保持默认 `INTRINSIC` 高性能。注意 eps 值本身须在算子支持的所有 dtype 下可表示（不落到 subnormal/下溢为 0）。
3. **结构性排除**：部分算子的计算结构可证明分母不会落入 subnormal（如 softmax 行内 max 归一使分母 ≥ 1），此时保持默认 `INTRINSIC` 即为正确设计，无需高精度模式。
