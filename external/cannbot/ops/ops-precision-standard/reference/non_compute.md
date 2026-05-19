# 非计算类算子精度验证

## 1. 算子定义

**定义特征：** 搬移/Cast，无数值计算

## 2. 验证方法

**比对方法：** 单标杆比对（CPU实现）

**验证脚本：** `scripts/bitwise_match.py`

## 3. 使用示例

```python
from scripts.bitwise_match import check_bitwise_match

# 执行算子
npu_output = run_operator_on_npu()
golden_output = run_reference_on_cpu()

# 验证精度
result = check_bitwise_match(npu_output, golden_output)
assert result['is_pass'], "二进制不一致"
```

## 4. 通过标准

**核心标准：** 二进制一致（Bitwise Match）

**判定代码：**
```python
is_pass = np.array_equal(npu_output, golden_output)
```

## 5. 脚本详细使用

### 5.1 单用例检查

```python
from scripts.bitwise_match import check_bitwise_match

result = check_bitwise_match(npu_output, golden_output)

# 返回结果
# {
#   'is_pass': True/False,
#   'bitwise_match': True/False,
#   'npu_dtype': 'float16',
#   'golden_dtype': 'float16',
#   'npu_shape': (128, 256),
#   'golden_shape': (128, 256)
# }
```

### 5.2 批量检查

```python
from scripts.bitwise_match import check_bitwise_match_batch

outputs_list = [
    (npu_output1, golden_output1),
    (npu_output2, golden_output2),
    ...
]

summary = check_bitwise_match_batch(outputs_list)

# 返回汇总信息
# {
#   'total_cases': 100,
#   'pass_count': 98,
#   'fail_count': 2,
#   'pass_rate': 0.98,
#   'detail_results': [...]
# }
```

## 6. 参考文档

- 详细精度标准：见 `golden/COMMERCIAL_OPS_PRECISION_DOCS.md`
- 特殊场景处理：见 `special_cases.md`