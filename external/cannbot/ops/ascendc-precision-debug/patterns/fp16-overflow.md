# fp16 中间计算溢出 / 精度不足

## 形态指纹

- 输入绝对值越大误差越大（magnitude_correlation 高），或出现 `inf`（fp16 最大 65504）。
- 小输入（|x|<1）比对通过，大输入才错。
- 多发于：乘加链、累加/reduce、exp、除法、中间结果动态范围大的算子；输入/输出是 fp16 但中间计算也用 fp16。

## 根因机理

fp16 只有 10 位尾数（约 3 位十进制精度）且动态范围窄。中间结果用 fp16 累加/连乘时，舍入误差逐级放大，或中间值超过 65504 变 inf。正确做法：**输入输出 fp16，中间计算升 fp32**。

## 验证方法

1. 把中间计算临时改为 fp32（Cast 升精度 → 计算 → Cast 回 fp16），重跑原用例。
2. 误差消失或降到容差内 → 确认。
3. 也可用机读比对看 bad 元素的 |golden| 是否显著大于 good 元素。

## 修复 pattern

```cpp
// 反例：全程 fp16
LocalTensor<half> tmp = ...;  Mul(tmp, x, y, len);  // 中间精度损失

// 正例：中间升 fp32
LocalTensor<float> xf = ..., yf = ..., zf = ...;
Cast(xf, x, RoundMode::CAST_NONE, len);
Cast(yf, y, RoundMode::CAST_NONE, len);
Mul(zf, xf, yf, len);                 // fp32 计算
Cast(z, zf, RoundMode::CAST_RINT, len); // 降回 fp16，注意舍入模式
```

注意：升 fp32 会多占 UB，tile 大小要相应减半重算。

## 反例（像但不是）

- 误差均匀分布且不随输入幅度变化 → 更像 reduce-accumulation-order 或 broadcast 问题。
- golden 与输出都是 fp16 直接比的：先确认所用容差是否与被测 dtype 匹配（容差来源规约见 references/tolerance.md），别拿 fp32 的口径卡 fp16。
