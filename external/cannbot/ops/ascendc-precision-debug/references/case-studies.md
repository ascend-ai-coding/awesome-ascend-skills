# 实战调试案例

## 案例1：SoftmaxV5 精度调试

### 问题描述

Softmax 算子在特定输入规模下输出精度不符合预期，验证脚本报告精度失败。

### 调试过程

#### Step 1: 最小可复现测试

```bash
# 从最小形状开始，32字节对齐
./softmaxv5 16 16 8 fp32

# 如果通过，测试 FP16
./softmaxv5 16 16 8 fp16
```

**结果**：小规模测试通过，大规模测试失败

#### Step 2: 误差分析

```python
# 分析误差分布
pred = np.load('output.npy')
truth = np.load('expected.npy')
error = np.abs(pred - truth)

print(f"Max error: {error.max():.2e}")
print(f"Mean error: {error.mean():.2e}")

# 找出最差样本
worst_idx = error.argmax()
print(f"Worst @{worst_idx}: pred={pred.flat[worst_idx]}, truth={truth.flat[worst_idx]}")
```

**发现**：误差主要集中在特定列

#### Step 3: 二分拆解验证

```cpp
// Softmax 公式: softmax(x) = exp(x-max) / sum(exp(x-max))

// 第1步：验证 ReduceMax
half max_val = ReduceMax(input);
printf("Step1 - max: %.6f\n", static_cast<float>(max_val));

// 第2步：验证广播后的 Sub
for (int i = 0; i < size; ++i) {
    half shifted = input[i] - max_val;
    if (i < 3) {
        printf("Step2 - shifted[%d]: %.6f (input=%.6f, max=%.6f)\n",
               i, static_cast<float>(shifted),
               static_cast<float>(input[i]), static_cast<float>(max_val));
    }
}

// 第3步：验证 Exp
for (int i = 0; i < size; ++i) {
    half exp_val = Exp(input[i] - max_val);
    if (i < 3) {
        printf("Step3 - exp[%d]: %.6f\n", i, static_cast<float>(exp_val));
    }
}

// 第4步：验证 ReduceSum
half exp_sum = ReduceSum(exp_values);
printf("Step4 - exp_sum: %.6f\n", static_cast<float>(exp_sum));

// 第5步：验证归一化
for (int i = 0; i < size; ++i) {
    output[i] = exp_values[i] / exp_sum;
    if (i < 3) {
        printf("Step5 - output[%d]: %.6f\n", i, static_cast<float>(output[i]));
    }
}

// 验证：输出之和应该接近1
half output_sum = ReduceSum(output);
printf("Verification - output_sum: %.6f (should be 1.0)\n",
       static_cast<float>(output_sum));
```

**Step 4: 边界条件测试**

```bash
# 测试硬件约束边界
./softmaxv5 8 8 8 fp32    # 最小列数=8
./softmaxv5 16 8 8 fp16   # FP16 最小列数

# 测试大规模
./softmaxv5 1024 256 8 fp32

# 测试非方形
./softmaxv5 256 512 4 fp32
```

**关键发现**：
- 列数必须 ≥ 8，否则 ReduceMax/ReduceSum 计算不正确
- 这是硬件 reduce 操作的约束

#### Step 5: 解决方案

1. 添加输入验证：
```cpp
if (cols < 8) {
    printf("Error: cols must be >= 8 (got %d)\n", cols);
    return;
}
```

2. 在文档中明确说明这一约束：
```markdown
## 已知限制
- 列数必须 ≥ 8（硬件 Reduce 操作约束）
```

### 经验总结

| 问题 | 根因 | 解决方案 |
|-----|------|---------|
| 特定规模精度异常 | 硬件约束不满足 | 添加输入验证 + 文档说明 |
| Reduce 结果错误 | 列数 < 8 | 确保列数 ≥ 8 |

---

## 案例2：Sinh 算子 FP16 精度不足

### 问题描述

Sinh 算子在 FP16 下精度明显不足，相对误差超过 1%。

### 数学公式

```
sinh(x) = (exp(x) - exp(-x)) / 2
```

### 调试过程

#### Step 1: FP32 vs FP16 对比

```python
# FP32 测试
result_fp32 = sinh_fp32(test_input)
error_fp32 = np.abs(result_fp32 - expected)
print(f"FP32 max error: {error_fp32.max():.2e}")  # ~1e-6

# FP16 测试
result_fp16 = sinh_fp16(test_input)
error_fp16 = np.abs(result_fp16 - expected)
print(f"FP16 max error: {error_fp16.max():.2e}")  # ~1e-2
```

**发现**：FP16 误差明显大于 FP32

#### Step 2: 二分调试

```cpp
// 拆解计算步骤
half x = 1.5h;

// 第1步：exp(x)
half exp_x = Exp(x);
printf("exp(%.2f) = %.6f\n", static_cast<float>(x), static_cast<float>(exp_x));

// 第2步：exp(-x)
half exp_neg_x = Exp(-x);
printf("exp(%.2f) = %.6f\n", static_cast<float>(-x), static_cast<float>(exp_neg_x));

// 第3步：减法
half numerator = exp_x - exp_neg_x;
printf("numerator = %.6f - %.6f = %.6f\n",
       static_cast<float>(exp_x),
       static_cast<float>(exp_neg_x),
       static_cast<float>(numerator));

// 第4步：除法
half result = numerator / 2.0h;
printf("result = %.6f / 2 = %.6f\n",
       static_cast<float>(numerator),
       static_cast<float>(result));
```

**发现**：减法步骤 `exp_x - exp_neg_x` 在 FP16 下精度损失较大

#### Step 3: 解决方案

使用 FP32 中间精度：

```cpp
half SinhStable(half x) {
    // 使用 FP32 进行中间计算
    float x_f32 = static_cast<float>(x);

    float exp_x = exp(x_f32);
    float exp_neg_x = exp(-x_f32);
    float numerator_f32 = exp_x - exp_neg_x;

    return static_cast<half>(numerator_f32 / 2.0f);
}
```

### 经验总结

| 问题 | 根因 | 解决方案 |
|-----|------|---------|
| FP16 精度不足 | 减法抵消导致精度损失 | 关键步骤使用 FP32 |
| sinh(x) FP16 误差大 | exp(x)-exp(-x) 精度损失 | 中间计算用 FP32 |

---

## 案例3：ReduceSum 累加精度损失

### 问题描述

ReduceSum 在大量元素累加时精度不足，特别是 FP16。

### 调试过程

#### Step 1: 问题复现

```python
# 生成测试数据
size = 1000
input_data = np.ones(size, dtype=np.float16) * 0.1

# 期望结果
expected = 100.0  # 1000 * 0.1

# 实际结果
result = reducesum_fp16(input_data)
print(f"Expected: {expected}, Got: {result}, Error: {abs(result - expected)}")
```

**结果**：误差达到 1.5（相对误差 1.5%）

#### Step 2: 分析原因

FP16 精度有限，多次累加导致误差累积：
- FP16 约 3-4 位有效数字
- 1000 次累加，每次累积误差

#### Step 3: 解决方案

使用 FP32 累加器：

```cpp
half ReduceSumAccurate(half* input, int size) {
    // 使用 FP32 累加器
    float sum_fp32 = 0.0f;

    for (int i = 0; i < size; ++i) {
        sum_fp32 += static_cast<float>(input[i]);
    }

    return static_cast<half>(sum_fp32);
}
```

**验证**：误差降至 1e-4 以下

### 经验总结

| 问题 | 根因 | 解决方案 |
|-----|------|---------|
| Reduce 精度损失 | FP16 累加误差累积 | 使用 FP32 累加器 |

---

## 案例4：exp 溢出导致 Inf

### 问题描述

Softmax 算子在输入值较大时输出 Inf。

### 调试过程

#### Step 1: 问题定位

```python
# 测试大输入
large_input = np.array([[100.0, 101.0, 102.0]], dtype=np.float16)
result = softmax(large_input)
print(result)  # [nan, nan, nan]
```

#### Step 2: Printf 调试

```cpp
// 直接 exp 会溢出
half x = 100.0h;
half exp_x = Exp(x);
printf("exp(%.1f) = %f\n", static_cast<float>(x), static_cast<float>(exp_x));
// 输出: exp(100.0) = inf
```

#### Step 3: 解决方案

数值稳定的 Softmax：

```cpp
half SoftmaxStable(half* input, int size) {
    // 先求最大值
    half max_val = input[0];
    for (int i = 1; i < size; ++i) {
        max_val = max(max_val, input[i]);
    }

    // 计算 exp(x - max)，避免溢出
    float exp_sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        half shifted = input[i] - max_val;  // 最大输入变为 0
        exp_sum += static_cast<float>(Exp(shifted));
    }

    // 归一化
    for (int i = 0; i < size; ++i) {
        half shifted = input[i] - max_val;
        output[i] = Exp(shifted) / static_cast<half>(exp_sum);
    }
}
```

### 经验总结

| 问题 | 根因 | 解决方案 |
|-----|------|---------|
| exp 溢出 | 输入值过大 | 先减 max，再 exp |
| Softmax 输出 Inf | exp(x) 当 x>88 时溢出 | 数值稳定算法 |

---

## 调试经验总结

### 调试效率排序

| 方法 | 适用场景 | 效率 |
|-----|---------|------|
| 误差分析 | 初步诊断 | ⭐⭐⭐ |
| Printf 调试 | 快速定位 | ⭐⭐⭐ |
| 常见陷阱排查 | 典型问题 | ⭐⭐ |
| 二分调试 | 复杂问题 | ⭐⭐ |
| 数据对比 | 系统验证 | ⭐ |

### 快速诊断提示

| 症状 | 可能原因 | 快速检查 |
|-----|---------|---------|
| 所有结果都差 | 公式/API 问题 | 检查公式实现 |
| FP16 特别差 | 精度不足 | 尝试 FP32 中间值 |
| 出现 Inf/NaN | 溢出/除零 | 检查边界值 |
| 特定规模错误 | 硬件约束 | 检查对齐/最小元素数 |
| 减法误差大 | 减法抵消 | 重排公式 |

### 调试黄金法则

1. **先 FP32，后 FP16**：排除算法问题
2. **先对齐，后非对齐**：排除对齐问题
3. **先简单，后复杂**：从最小测试用例开始
4. **先快速，后深入**：快速方法无效再用二分调试
5. **记录一切**：每次修改都要记录误差变化
