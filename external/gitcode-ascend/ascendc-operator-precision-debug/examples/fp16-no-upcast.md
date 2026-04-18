# 案例：FP16 未升精度到 FP32

## 现象

- **FP16 失败，FP32 通过**
- 全部元素均有偏差，CosineSim ≈ 0.99x
- MaxAbsErr 量级 ~1e-1（远超 FP16 阈值 1e-3）

## 误差分析输出

```
shape=(4, 128, 256)  dtype=torch.float16
MaxAbsErr : 1.23e-01
MeanAbsErr: 3.45e-03
CosineSim : 0.99812
错误元素: 98304/131072 (75.00%)
```

## 根因

kernel Compute 中直接用 FP16 做复杂运算（Mul → Adds → Sqrt → Ln），半精度累积误差严重。

```cpp
// ❌ 错误：FP16 直接计算
AscendC::Mul(tmp, xLocal, xLocal, len);
AscendC::Adds(tmp, tmp, -1.0f, len);
AscendC::Sqrt(tmp, tmp, len);
AscendC::Ln(yLocal, tmp, len);
```

## 修复

添加 FP16→FP32 升精度分支：

```cpp
if constexpr (sizeof(T) == sizeof(half)) {
    AscendC::Cast(xFloat, xLocal, RoundMode::CAST_NONE, len);
    AscendC::Mul(tmp, xFloat, xFloat, len);
    AscendC::Adds(tmp, tmp, -1.0f, len);
    AscendC::Sqrt(tmp, tmp, len);
    AscendC::Ln(resultFloat, tmp, len);
    AscendC::Cast(yLocal, resultFloat, RoundMode::CAST_ROUND, len);
} else {
    AscendC::Mul(tmp, xLocal, xLocal, len);
    AscendC::Adds(tmp, tmp, -1.0f, len);
    AscendC::Sqrt(tmp, tmp, len);
    AscendC::Ln(yLocal, tmp, len);
}
```

## 定位关键

Phase 1 即可判断：「FP16 挂 + FP32 过 + 全部偏差 + CosineSim 高」→ 系统性精度损失 → 直接查 Cast。
