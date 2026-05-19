# 精度转换与混合精度指南

Cast API 使用规范和混合精度计算模式。

---

## 目录

1. [Cast RoundMode 选择](#cast-roundmode-选择)
2. [混合精度计算模式（FP16 输入）](#混合精度计算模式fp16-输入)

---

## Cast RoundMode 选择

### 选择规则

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| **half → float** | `CAST_NONE` | 低精度→高精度，无精度损失 |
| **float → half** | `CAST_ROUND` | 高精度→低精度，有精度损失 |
| half → int32_t | `CAST_ROUND` / `CAST_CEIL` | 量化场景，根据需求选择 |
| int32_t → float | `CAST_NONE` | 整数→浮点，无精度损失 |

### 正确用法

```cpp
// ✅ half → float：低精度到高精度
AscendC::LocalTensor<float> xFloat = workBuf.Get<float>();
AscendC::Cast<float, half>(xFloat, xHalf, AscendC::RoundMode::CAST_NONE, count);

// ✅ float → half：高精度到低精度
AscendC::LocalTensor<half> yHalf = outQueue.AllocTensor<half>();
AscendC::Cast<half, float>(yHalf, xFloat, AscendC::RoundMode::CAST_ROUND, count);
```

---

## 混合精度计算模式（FP16 输入）

### 适用场景

当输入输出为 FP16，但需要 FP32 精度进行中间计算时（如 Softmax、LayerNorm）。

### 计算流程

```
half 输入 → Cast(FP32) → 中间计算(FP32) → Cast(half) → half 输出
```

### 为什么需要 FP32 中间计算？

1. **ReduceMax/Exp/ReduceSum** 在 FP32 上精度更稳定
2. **避免 FP16 数值溢出**：Exp 结果可能超出 FP16 表示范围
3. **累积误差控制**：多次运算的累积误差在 FP32 下更小

### 加减法场景示例

半精度加减法默认升 FP32；仅当 spec 明确"输入同量级"（如 mask 叠加、已归一化概率相加）时才允许直接 `Add/Sub<half>`。BF16 与 FP16 适用同一规则，仅临界比值不同（BF16=128，FP16=1024）。

> 完整示例、决策表与 Kernel 集成要点见 [api-arithmetic.md → 场景3](api-arithmetic.md#场景3半精度加减法精度优化)。
