# 浮点计算类算子精度验证

## 1. 适用场景

**适用算子类型：** 浮点计算类算子（所有使用浮点数进行数值计算的算子）

**常见数据类型：** FLOAT16, BFLOAT16, FLOAT32, HiFLOAT32, FLOAT8 E4M3, FLOAT8 E5M2

> 整型计算类、搬运类算子不在本标准讨论范围内，需按照各算子的实际业务场景单独制定标准。

## 2. 误差指标

本标准采用**混合容差**（Mixed Tolerance）指标，即结合绝对容差（atol）和相对容差（rtol）的逐元素比对方式。

### 2.1 逐元素通过条件

对于输出张量中的每个元素，当满足以下条件时判定该元素通过：

$$
|actual - golden| \leq atol + rtol \times |golden|
$$

其中：
- **atol**（Absolute Tolerance，绝对容差）：保证小值（golden 接近 0）场景下的合理误差范围，天然避免除零问题。
- **rtol**（Relative Tolerance，相对容差）：保证大值场景下的相对精度。

### 2.2 整体通过条件

定义**通过率**（Matched Ratio）为通过元素数占总元素数的比例：

$$
\text{matched\_ratio} = \frac{\text{通过元素数}}{\text{总元素数}}
$$

当同时满足以下两个条件时，判定该用例通过：

1. `matched_ratio ≥ required_matched_ratio`
2. `max_abs_error ≤ max_abs_error_limit`

其中 `max_abs_error` 为用例中任意元素的最大绝对误差，`max_abs_error_limit` 为绝对误差硬上限。

## 3. 混合容差阈值表

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|---------|---------|----------|---------|-----------|-------------|-------------|
| **rtol** | 2^-9 (1.95e-3) | 2^-6 (1.56e-2) | 2^-10 (9.77e-4) | 2^-9 (1.95e-3) | 2^-2 (0.25) | 2^-1 (0.5) |
| **atol** | 2^-9 (1.95e-3) | 2^-6 (1.56e-2) | 2^-16 (1.53e-5) | 2^-10 (9.77e-4) | 2^-4 (0.0625) | 2^-3 (0.125) |
| **required_matched_ratio** | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| **max_abs_error_limit** | 1e-1 or 32*ULP | 1e-0 or 32*ULP | 1e-2 or 32*ULP | 1e-1 or 32*ULP | 1e-0 or 32*ULP | 1e-1 or 32*ULP |

> `max_abs_error_limit` 中 "A or B" 表示取两者中较宽松（较大）者作为上限。
>
> **ULP**（Unit in the Last Place）取各 dtype 在数值 1.0 处的末位单位精度，各 dtype 取值如下：
> - FLOAT16: 2^-10 (≈9.77e-4); BFLOAT16: 2^-7 (≈7.81e-3); FLOAT32: 2^-23 (≈1.19e-7)
> - HiFLOAT32: 2^-24 (≈5.96e-8); FLOAT8 E4M3: 2^-3 (0.125); FLOAT8 E5M2: 2^-2 (0.25)
>
> 以 FLOAT8 E5M2 为例：`max_abs_error_limit = max(1e-1, 32 × 2^-2) = max(0.1, 8.0) = 8.0`。

## 4. 通过判定

**比对方法：** 单标杆比对 —— 与更高精度的实现（CPU 或昇腾小算子拼接）的单一精度标杆直接比较。

**判定代码：**
```python
from scripts.mixed_tolerance_check import check_mixed_tolerance

result = check_mixed_tolerance(npu_output, golden_output)
assert result['is_pass'], f"精度不达标: matched_ratio={result['matched_ratio']}"
```

当用例同时满足 `matched_ratio ≥ required_matched_ratio` 且 `max_abs_error ≤ max_abs_error_limit` 时，判定该用例精度通过。

## 5. 使用示例

### 5.1 单用例检查

```python
from scripts.mixed_tolerance_check import check_mixed_tolerance

# 执行算子
npu_output = run_operator_on_npu()      # NPU 实现
golden_output = run_reference_on_cpu()  # 高精度 CPU 实现

# 验证精度
result = check_mixed_tolerance(npu_output, golden_output)

assert result['is_pass'], f"精度不达标: matched_ratio={result['matched_ratio']}"
```

返回结果：
```python
# {
#   'is_pass': True/False,
#   'matched_ratio': 0.9995,
#   'required_matched_ratio': 0.99,
#   'max_abs_error': 0.0123,
#   'max_abs_error_limit': 0.1,
#   'rtol': 0.001953125,
#   'atol': 0.001953125,
#   'npu_dtype': 'float16',
#   'golden_dtype': 'float16',
#   'shape': (128, 256)
# }
```

### 5.2 批量检查

```python
from scripts.mixed_tolerance_check import check_mixed_tolerance_batch

outputs_list = [
    (npu_output1, golden_output1),
    (npu_output2, golden_output2),
    ...
]

summary = check_mixed_tolerance_batch(outputs_list)

print(f"通过率: {summary['pass_rate']:.2%}")
print(f"平均 matched_ratio: {summary['matched_ratio_mean']:.6f}")
```

## 6. 参考文档

- **标杆构造：** 见 `benchmark_construction.md`
- **测试用例生成：** 见 `test_case_generation.md`
