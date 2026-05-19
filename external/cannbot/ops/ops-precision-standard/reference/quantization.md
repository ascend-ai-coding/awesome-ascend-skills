# 量化计算类算子精度验证

## 1. 算子定义

**定义特征：** 低精度（整型）与高精度（浮点）之间的转换与计算

**常见数据类型：** INT4/INT8/FLOAT4/FLOAT8

## 2. 验证方法

**根据输出类型选择比对方法：**

| 输出类型 | 比对方法 | 验证脚本 |
|---------|---------|---------|
| 整型输出（量化结果） | 单标杆比对 | `quantization_check.py` |
| 浮点输出（反量化结果） | 双标杆比对 | `quantization_check.py` |

## 3. 使用示例

### 3.1 整型输出（量化结果）

```python
from scripts.quantization_check import check_quantization_with_level

# 执行量化算子（FP16 → INT8）
npu_output = run_quantize_on_npu()  # dtype: int8
golden_output = run_quantize_on_cpu()  # dtype: int8

result = check_quantization_with_level(
    npu_output, golden_output,
    input_dtype='float16',
    output_dtype='int8'
)

assert result['is_pass'], f"量化精度不达标: max_abs_error={result['max_abs_error']}"
```

### 3.2 浮点输出（反量化结果）

```python
from scripts.quantization_check import check_quantization_with_level

# 执行反量化算子（INT8 → FP16）
npu_output = run_dequantize_on_npu()  # dtype: float16
golden_output = run_dequantize_on_cpu()  # dtype: float16 (高精度实现)
third_party_output = run_dequantize_on_gpu()  # dtype: float16

result = check_quantization_with_level(
    npu_output, golden_output, third_party_output,
    precision_level='L1',
    input_dtype='int8',
    output_dtype='float16'
)

assert result['is_pass'], f"反量化精度不达标"
```

## 4. 通过标准

### 4.1 整型输出标准

**核心标准：** 绝对误差 ≤ 1

**判定代码：**
```python
abs_error = np.abs(npu_output - golden_output)
max_abs_error = np.max(abs_error)
is_pass = (max_abs_error <= 1)
```

### 4.2 浮点输出标准

**参考浮点计算类标准：** 使用MARE/MERE/RMSE Ratio

**精度等级阈值：** 根据precision_level确定
- L0: MARE ratio ≤ 10, MERE ratio ≤ 2, RMSE ratio ≤ 2
- L1: MARE ratio ≤ 5, MERE ratio ≤ 1.5, RMSE ratio ≤ 1.5
- L2: MARE ratio ≤ 2, MERE ratio ≤ 1.2, RMSE ratio ≤ 1.2

## 5. 完整通过标准表

| 输入类型 | 输出类型 | 通过标准 | 验证方法 |
|---------|---------|---------|---------|
| 整型（INT4/INT8/INT16等） | 整型（INT4/INT8/INT16等） | N/A（不常见场景） | - |
| 整型（INT4/INT8/INT16等） | 浮点（FLOAT4/FLOAT8/FLOAT16/FLOAT32等） | 参考浮点类标准 | 双标杆比对 |
| 浮点（FLOAT4/FLOAT8/FLOAT16/FLOAT32等） | 整型（INT4/INT8/INT16等） | 绝对误差 ≤ 1 | 单标杆比对 |
| 浮点（FLOAT4/FLOAT8/FLOAT16/FLOAT32等） | 浮点（FLOAT4/FLOAT8/FLOAT16/FLOAT32等） | 参考浮点类标准 | 双标杆比对 |

## 6. 脚本详细使用

### 6.1 check_quantization函数

```python
from scripts.quantization_check import check_quantization

# 基础验证（自动判断输出类型）
result = check_quantization(
    npu_output, 
    golden_output,
    third_party_output=None,  # 浮点输出时需要
    input_dtype='float16',
    output_dtype='int8'
)

# 返回结果（整型输出）
# {
#   'is_pass': True/False,
#   'max_abs_error': 1,
#   'mean_abs_error': 0.5,
#   'threshold': 1,
#   'comparison_method': 'single_benchmark'
# }

# 返回结果（浮点输出）
# {
#   'comparison_method': 'dual_benchmark',
#   'mare_npu': 0.001,
#   'mere_npu': 0.0005,
#   'rmse_npu': 0.001,
#   'mare_third': 0.0008,
#   'mere_third': 0.0004,
#   'rmse_third': 0.0008,
#   'mare_ratio': 1.25,
#   'mere_ratio': 1.25,
#   'rmse_ratio': 1.25,
#   'is_pass': None  # 需要外部指定精度等级后判断
# }
```

### 6.2 check_quantization_with_level函数

```python
from scripts.quantization_check import check_quantization_with_level

# 完整验证（包含精度等级判定）
result = check_quantization_with_level(
    npu_output, golden_output, third_party_output,
    precision_level='L1',
    input_dtype='int8',
    output_dtype='float16'
)

# 返回结果
# {
#   'is_pass': True/False,
#   'precision_level': 'L1',
#   'mare_ratio': 1.25,
#   'mere_ratio': 1.25,
#   'rmse_ratio': 1.25,
#   'mare_pass': True,
#   'mere_pass': True,
#   'rmse_pass': True,
#   'thresholds_used': {'mare_ratio': 5, 'mere_ratio': 1.5, 'rmse_ratio': 1.5}
# }
```

## 7. 参考文档

- 浮点计算类标准：见 `float_compute.md`
- 详细精度标准：见 `golden/COMMERCIAL_OPS_PRECISION_DOCS.md`
- 标杆构造方法：见 `benchmark_construction.md`