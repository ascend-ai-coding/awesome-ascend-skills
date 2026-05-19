# 随机数生成类算子精度验证

## 1. 算子定义

**定义特征：** 按指定分布生成随机数

## 2. 验证方法

**比对方法：** KS检验（Kolmogorov-Smirnov test）

**验证脚本：** `scripts/random_distribution_check.py`

## 3. 使用示例

### 3.1 单次KS检验

```python
from scripts.random_distribution_check import ks_test_distribution

# 执行算子
npu_output = run_rand_on_npu()

# 验证分布
result = ks_test_distribution(
    npu_output, 
    expected_distribution='uniform',  # 或 'normal'
    params={'low': 0, 'high': 1}  # 均匀分布参数
)

assert result['is_pass'], f"分布检验失败: p_value={result['p_value']}"
```

### 3.2 批量检验

```python
from scripts.random_distribution_check import check_random_distribution_batch

# 生成多个随机数输出
outputs_list = [run_rand_on_npu() for _ in range(100)]

# 批量检验
summary = check_random_distribution_batch(
    outputs_list, 
    expected_distribution='uniform',
    params={'low': 0, 'high': 1},
    N=100
)

print(f"通过率: {summary['actual_pass_rate_percent']:.2f}%")
print(f"理论阈值: {summary['theoretical_pass_rate_percent']:.2f}%")
```

## 4. 通过标准

### 4.1 p值检验

**显著性水平：** α = 0.01

**判定标准：** p值 > α

**判定代码：**
```python
from scipy import stats

statistic, p_value = stats.kstest(npu_output, 'uniform', args=(0, 1))
is_pass = (p_value > 0.01)
```

### 4.2 批量检验通过标准

**通过条件：**
```
至少 ((1-α) + z×√(α(1-α)/N)) × 100% 的测试用例满足 p > α
```

**参数说明：**
- α = 0.01（显著性水平）
- z = -3.0902（正态分布99.9%截尾点）
- N = 测试次数（推荐100）

**计算示例：**
```python
alpha = 0.01
z = -3.0902
N = 100

theoretical_pass_rate = (1 - alpha) + z * np.sqrt(alpha * (1 - alpha) / N)
theoretical_pass_rate_percent = theoretical_pass_rate * 100
# 约98.07%
```

## 5. 支持的分布类型

### 5.1 均匀分布（uniform）

**参数：**
- low: 下界（默认0）
- high: 上界（默认1）

**示例：**
```python
result = ks_test_distribution(
    npu_output,
    expected_distribution='uniform',
    params={'low': -1, 'high': 1}
)
```

### 5.2 正态分布（normal）

**参数：**
- mean: 均值（默认0）
- std: 标准差（默认1）

**示例：**
```python
result = ks_test_distribution(
    npu_output,
    expected_distribution='normal',
    params={'mean': 0, 'std': 1}
)
```

### 5.3 指数分布（exponential）

**参数：**
- scale: 尺度参数（默认1）

**示例：**
```python
result = ks_test_distribution(
    npu_output,
    expected_distribution='exponential',
    params={'scale': 1.0}
)
```

## 6. 脚本详细使用

### 6.1 ks_test_distribution函数

```python
from scripts.random_distribution_check import ks_test_distribution

result = ks_test_distribution(
    npu_output,
    expected_distribution='uniform',
    params={'low': 0, 'high': 1}
)

# 返回结果
# {
#   'is_valid': True,
#   'ks_statistic': 0.05,
#   'p_value': 0.85,
#   'alpha': 0.01,
#   'is_pass': True,
#   'expected_distribution': 'uniform',
#   'params': {'low': 0, 'high': 1},
#   'sample_size': 1000,
#   'interpretation': 'p_value 0.85 > alpha 0.01,分布与期望一致'
# }
```

### 6.2 check_random_distribution_batch函数

```python
from scripts.random_distribution_check import check_random_distribution_batch

outputs_list = [run_rand_on_npu() for _ in range(100)]

summary = check_random_distribution_batch(
    outputs_list,
    expected_distribution='uniform',
    params={'low': 0, 'high': 1},
    N=100,
    alpha=0.01
)

# 返回汇总信息
# {
#   'total_tests': 100,
#   'pass_count': 99,
#   'fail_count': 1,
#   'actual_pass_rate': 0.99,
#   'actual_pass_rate_percent': 99.0,
#   'theoretical_pass_rate': 0.9807,
#   'theoretical_pass_rate_percent': 98.07,
#   'is_pass': True,  # 99% > 98.07%
#   'p_value_mean': 0.65,
#   'p_value_std': 0.15,
#   'detail_results': [...]
# }
```

## 7. 参考文档

- 详细精度标准：见 `golden/COMMERCIAL_OPS_PRECISION_DOCS.md`
- KS检验理论：scipy.stats.kstest文档