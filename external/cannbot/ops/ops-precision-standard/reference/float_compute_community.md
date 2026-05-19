# 浮点计算类算子精度验证（社区标准）

## 1. 适用场景

**适用算子类型：** 浮点计算类算子（所有使用浮点数进行数值计算的算子）

**精度指标：** MERE/MARE Threshold

## 2. 验证方法

**比对方法：** 单标杆比对（Threshold标准）

**验证脚本：** `scripts/mare_mere_threshold.py`

## 3. 使用示例

```python
from scripts.mare_mere_threshold import check_precision_threshold

# 执行算子
npu_output = run_operator_on_npu()
golden_output = run_reference_on_cpu()

# 验证精度
result = check_precision_threshold(npu_output, golden_output)

assert result['is_pass'], f"精度不达标: MERE={result['mere']}"
```

## 4. 通过标准

### 4.1 各数据类型阈值

| 数据类型 | Threshold | 数值 |
|---------|-----------|------|
| **FLOAT16** | 2^-10 | 约0.000977 |
| **BFLOAT16** | 2^-7 | 约0.00781 |
| **FLOAT32** | 2^-13 | 约0.000122 |
| **HiFLOAT32** | 2^-11 | 约0.000488 |
| **FLOAT8 E4M3** | 2^-3 | 约0.125 |
| **FLOAT8 E5M2** | 2^-2 | 约0.25 |

### 4.2 判定条件

**通过标准：**
1. MERE < Threshold
2. MARE < 10 * Threshold

**判定代码：**
```python
threshold = get_threshold_by_dtype(npu_output.dtype)
mare_threshold = 10 * threshold

is_pass = (mere < threshold) and (mare < mare_threshold)
```

## 5. 误差指标定义

**平均相对误差（MERE）：**
```
MERE = avg(|actual - golden| / (|golden| + 1e-7))
```

**最大相对误差（MARE）：**
```
MARE = max(|actual - golden| / (|golden| + 1e-7))
```

## 6. 脚本详细使用

### 6.1 单用例检查

```python
from scripts.mare_mere_threshold import check_precision_threshold

result = check_precision_threshold(npu_output, golden_output)

# 返回结果
# {
#   'is_pass': True/False,
#   'mare': 0.001,
#   'mere': 0.0005,
#   'threshold': 0.000977,
#   'mare_threshold': 0.00977,
#   'mare_pass': True,
#   'mere_pass': True,
#   'npu_dtype': 'float16'
# }
```

### 6.2 批量检查

```python
from scripts.mare_mere_threshold import check_precision_threshold_batch

outputs_list = [
    (npu_output1, golden_output1),
    (npu_output2, golden_output2),
    ...
]

summary = check_precision_threshold_batch(outputs_list)

print(f"通过率: {summary['pass_rate']:.2%}")
print(f"平均MERE: {summary['mere_mean']:.6f}")
```

## 7. 参考文档

- **商用标准：** 见 `golden/COMMERCIAL_OPS_PRECISION_DOCS.md`
- **特殊场景：** 见 `special_cases.md`
- **标杆构造：** 见 `benchmark_construction.md`