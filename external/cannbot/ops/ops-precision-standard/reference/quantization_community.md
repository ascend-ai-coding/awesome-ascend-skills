# 量化计算类算子精度验证（社区标准）

> **注意：** 本文为 skill 扩展内容，golden 标准文档（COMMERCIAL_OPS_PRECISION_DOCS.md）中仅定义了商用量化标准。
> 本社区标准复用生态浮点 Threshold 标准（OPS_PRECISION_STANDARDS.md）作为量化算子在无竞品对标场景下的简化验收方案。

## 1. 适用场景

**适用算子类型：** 量化/反量化类算子（整型↔浮点转换）

**精度指标：** MERE/MARE Threshold（适用于所有浮点比对场景）

## 2. 验证方法

**比对方法：** 单标杆比对（Threshold标准）

**验证脚本：** `scripts/mare_mere_threshold.py`

## 3. 使用示例

```python
from scripts.mare_mere_threshold import check_precision_threshold

# 执行算子
npu_output = run_operator_on_npu()  # dtype: float16/float32等
golden_output = run_reference_on_cpu()  # 高精度CPU实现

# 验证精度
result = check_precision_threshold(npu_output, golden_output)

assert result['is_pass'], f"精度不达标: MERE={result['mere']}"
```

## 4. 通过标准

### 4.1 各数据类型阈值

根据输出dtype选择阈值：

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

## 6. 参考文档

- **商用标准完整版：** 见 `quantization.md`
- **社区标准原文：** 见 `golden/OPS_PRECISION_STANDARDS.md`
- **浮点计算类标准：** 见 `float_compute.md`