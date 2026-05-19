---
title: 精度指南
purpose: 设计 RegBase 精度策略时使用，覆盖 dtype route、cast、mask、tail 和 accumulation。
read_when:
  - 设计涉及 fp16/bf16/fp32 转换。
  - 需要判断 atol/rtol 或数值稳定策略。
keywords:
  - precision
  - cast
  - accumulation
next_reads:
  - precision_failures.md
  - ../api/regbase_api_reference.md
depth: foundation
topic_type: pitfall
---

# 精度指南

## 1. 设计必须显式说明

- 输入 dtype、输出 dtype、VF 内部计算 dtype。
- cast 发生在 load 后、计算中还是 store 前。
- mask 是否覆盖所有可能越界或 padding 通道。
- reduce / norm 的 accumulation dtype。
- 比较、select 和近似公式的误差来源。

## 2. 常见策略

| 场景 | 策略 |
|---|---|
| activation | fp16/bf16 输入可升 fp32 做近似，再 cast 回输出 |
| RMSNorm / LayerNorm | sum/mean/variance 使用更高精度 accumulation |
| compare/select | mask 与两个分支 dtype 保持一致 |
| quant | 记录 scale、rounding、saturate 和输出范围 |
| tail | 无效通道不参与数学，也不写回有效输出 |

## 3. Cast 放置

数值敏感链路尽量在高精度中保持到最终 cast：

- reduce、`exp`、`log`、reciprocal-style、normalization intermediates 优先使用 fp32。
- 只有输出边界要求 fp16 或 bf16 时才 cast 回窄类型。
- 避免多个 nonlinear operation 串在低精度中间结果上。
- 安全规则是尽早升精度，尽晚降精度。

常见 cast 方向：

| 方向 | 常见 round mode | 说明 |
|---|---|---|
| `half -> float` | `CAST_NONE` | 升精度，不应引入舍入策略 |
| `float -> half` | `CAST_ROUND` | 输出边界常见窄化路径 |
| `int32_t -> float` | `CAST_NONE` | 语义通常是类型展开 |
| `half -> int32_t` | 按 quantization rule 选择 | 必须记录 rounding、saturate 和输出范围 |

VF 侧 `Cast` / `Truncate` 的代表签名在 [[../api/regbase_api_reference]] 中维护。写设计或代码前必须回查当前 SDK header 中的 `CastTrait`、`RegLayout`、`SatMode`、`MaskMergeMode`、`RoundMode`，不要只按 API 名称推断参数。

## 4. Reduction 精度和 padding

包含 sum、max、min、mean、norm 或类似聚合时，精度策略还必须说明：

- partial result 的保存位置：`RegTensor`、UB 还是 GM。
- accumulation dtype，以及最终 cast 的位置。
- reduce 维度是否跨 tile，跨 tile 时 partial 如何合并。
- tail lanes 是否参与 count、sum、max/min 或 variance。

padding 不能改变归约语义：

- max padding 不能大于有效数据。
- min padding 不能小于有效数据。
- sum / mean 的无效通道不能参与有效 count。
- variance / norm 必须说明 epsilon、sum 和 count 的 dtype。

## 5. 验收建议

- 构造含 tail 的 shape。
- 覆盖正负值、零、极值、NaN/Inf 相关输入（如算子语义需要）。
- 对 fp16/bf16 明确 atol/rtol。
- 对 fusion 链路比较拆分参考实现。

## 相关文档

- [[precision_failures]]
- [[../api/regbase_api_reference]]
- [[../api/arithmetic_and_reduce]]
