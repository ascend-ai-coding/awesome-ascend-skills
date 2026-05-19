# 整数计算类算子精度验证

## 1. 算子定义

**定义特征：** 输入输出均为整型，包含算术运算

## 2. 验证方法

**比对方法：** 单标杆比对（CPU实现）

**验证脚本：** `scripts/integer_compute_check.py`

## 3. 使用示例

```python
from scripts.integer_compute_check import check_integer_compute

# 执行算子
npu_output = run_operator_on_npu()  # dtype: int32
golden_output = run_reference_on_cpu()  # dtype: int32

# 验证精度
result = check_integer_compute(npu_output, golden_output)
assert result['is_pass'], f"整数计算精度不达标"
```

## 4. 通过标准

**核心标准：** 二进制一致 或 绝对误差为0

### 4.1 判定条件（满足其一即可）

**条件1：二进制一致**
```python
is_bitwise_match = np.array_equal(npu_output, golden_output)
```

**条件2：绝对误差为0**
```python
is_abs_zero = np.all(np.abs(npu_output - golden_output) == 0)
```

**最终判定：**
```python
is_pass = is_bitwise_match or is_abs_zero
```

### 4.2 为什么允许绝对误差为0但不二进制一致？

整数计算中，可能存在以下情况：
- 不同位宽整数转换后数值相同但二进制表示不同（如int8的10和int64的10）
- 符号扩展方式不同但数值结果一致

只要数值结果一致（绝对误差为0），即视为通过。

## 5. 脚本详细使用

### 5.1 单用例检查

```python
from scripts.integer_compute_check import check_integer_compute

result = check_integer_compute(npu_output, golden_output)

# 返回结果
# {
#   'is_pass': True/False,
#   'bitwise_match': True/False,
#   'abs_error_zero': True/False,
#   'max_abs_error': 0,
#   'mean_abs_error': 0.0,
#   'npu_dtype': 'int32',
#   'golden_dtype': 'int32',
#   'shape': (128, 256)
# }
```

### 5.2 批量检查

```python
from scripts.integer_compute_check import check_integer_compute_batch

outputs_list = [
    (npu_output1, golden_output1),
    (npu_output2, golden_output2),
    ...
]

summary = check_integer_compute_batch(outputs_list)

# 返回汇总信息
# {
#   'total_cases': 100,
#   'pass_count': 95,
#   'fail_count': 5,
#   'pass_rate': 0.95,
#   'bitwise_match_count': 90,  # 二进制一致的用例数
#   'abs_zero_count': 5,         # 绝对误差为0的用例数
#   'detail_results': [...]
# }
```

## 6. 参考文档

- 详细精度标准：见 `golden/COMMERCIAL_OPS_PRECISION_DOCS.md`
- 测试用例生成：见 `test_case_generation.md`