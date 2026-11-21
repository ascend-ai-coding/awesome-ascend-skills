# 量化计算类算子精度验证

## 1. 适用场景

**适用算子类型：** 量化/反量化类算子（整型 ↔ 浮点转换）

## 2. 验证方法

根据**输出 dtype** 选择比对方法：

| 输出类型 | 比对方法 | 验证脚本 | 标准 |
|---------|---------|---------|------|
| 浮点输出（反量化结果） | 混合容差（atol/rtol） | `scripts/mixed_tolerance_check.py` | 见 `float_compute.md` |
| 整型输出（量化结果） | 二进制一致 / 绝对误差为 0 | `scripts/integer_compute_check.py` | 见 `integer_compute.md` |

> 生态算子开源精度标准仅覆盖浮点计算类算子。量化算子的整型输出按整型计算类规则单独制定，不在浮点标准讨论范围内。

## 3. 使用示例

### 3.1 浮点输出（反量化，INT8 → FP16）

```python
from scripts.mixed_tolerance_check import check_mixed_tolerance

# 执行反量化算子
npu_output = run_dequantize_on_npu()      # dtype: float16
golden_output = run_dequantize_on_cpu()   # dtype: float16 (高精度实现)

result = check_mixed_tolerance(npu_output, golden_output)
assert result['is_pass'], f"反量化精度不达标: matched_ratio={result['matched_ratio']}"
```

### 3.2 整型输出（量化，FP16 → INT8）

```python
from scripts.integer_compute_check import check_integer_compute

# 执行量化算子
npu_output = run_quantize_on_npu()      # dtype: int8
golden_output = run_quantize_on_cpu()   # dtype: int8

result = check_integer_compute(npu_output, golden_output)
assert result['is_pass'], f"量化精度不达标"
```

## 4. 参考文档

- **浮点输出标准：** 见 `float_compute.md`（混合容差 atol/rtol 阈值表）
- **整型输出标准：** 见 `integer_compute.md`（二进制一致）
