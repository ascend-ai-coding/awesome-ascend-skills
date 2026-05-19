# 浮点计算类算子精度验证（重点）

## 1. 算子定义

**定义特征：** 使用浮点数进行数值计算

**常见数据类型：** FLOAT16, BFLOAT16, FLOAT32, HiFLOAT32, FLOAT8

## 2. 验证方法选择

根据业务场景选择标准：

| 业务场景 | 精度标准 | 比对方法 | 验证脚本 |
|---------|---------|---------|---------|
| 商用算子（内部交付） | L0/L1/L2分级 | 双标杆Ratio | `mare_mere_rmse_ratio.py` |
| 生态贡献算子 | 简化标准 | 单标杆Threshold | `mare_mere_threshold.py` |

## 3. 商用标准（双标杆Ratio）

### 3.1 使用示例

```python
from scripts.mare_mere_rmse_ratio import check_precision_ratio

# 执行算子
npu_output = run_operator_on_npu()  # NPU实现
golden_output = run_reference_on_cpu()  # 高精度CPU实现
third_party_output = run_on_gpu()  # 三方芯片实现

# 验证精度
result = check_precision_ratio(
    npu_output, golden_output, third_party_output, 
    precision_level='L1'  # L0/L1/L2
)

assert result['is_pass'], f"精度不达标: MARE ratio={result['mare_ratio']}"
```

### 3.2 精度等级与阈值

| 精度等级 | 适用场景 | 用例规模 | MARE Ratio | MERE Ratio | RMSE Ratio |
|---------|---------|---------|-----------|-----------|-----------|
| **L0** | 常规算子，非敏感业务 | ≥5,000 | ≤10 | ≤2 | ≤2 |
| **L1** | 重要算子，LLM/多模态/推荐系统 | ≥10,000 | ≤5 | ≤1.5 | ≤1.5 |
| **L2** | 关键算子，核心业务 | ≥30,000 | ≤2 | ≤1.2 | ≤1.2 |

### 3.3 误差指标定义

**最大相对误差（MARE）：**
```
MARE = max(|actual - golden| / (|golden| + 1e-7))
```

**平均相对误差（MERE）：**
```
MERE = avg(|actual - golden| / (|golden| + 1e-7))
```

**均方根误差（RMSE）：**
```
RMSE = sqrt(mean((actual - golden)^2))
```

**Ratio定义：**
```
Ratio = NPU误差指标 / 三方芯片误差指标
```

### 3.4 批量验证

```python
from scripts.mare_mere_rmse_ratio import check_precision_ratio_batch

outputs_list = [
    (npu_output1, golden_output1, third_party_output1),
    (npu_output2, golden_output2, third_party_output2),
    ...
]

summary = check_precision_ratio_batch(outputs_list, precision_level='L1')

print(f"通过率: {summary['pass_rate']:.2%}")
print(f"平均MARE ratio: {summary['mare_ratio_mean']:.3f}")
```

## 4. 生态标准（单标杆Threshold）

### 4.1 使用示例

```python
from scripts.mare_mere_threshold import check_precision_threshold

# 执行算子
npu_output = run_operator_on_npu()
golden_output = run_reference_on_cpu()

# 验证精度
result = check_precision_threshold(npu_output, golden_output)

assert result['is_pass'], f"精度不达标: MERE={result['mere']}"
```

### 4.2 各数据类型阈值

| 数据类型 | Threshold | 数值 |
|---------|-----------|------|
| **FLOAT16** | 2^-10 | 约0.000977 |
| **BFLOAT16** | 2^-7 | 约0.00781 |
| **FLOAT32** | 2^-13 | 约0.000122 |
| **HiFLOAT32** | 2^-11 | 约0.000488 |
| **FLOAT8 E4M3** | 2^-3 | 约0.125 |
| **FLOAT8 E5M2** | 2^-2 | 约0.25 |

### 4.3 通过标准

**判定条件：**
1. MERE < Threshold
2. MARE < 10 * Threshold

**判定代码：**
```python
threshold = get_threshold_by_dtype(npu_output.dtype)
mare_threshold = 10 * threshold

is_pass = (mere < threshold) and (mare < mare_threshold)
```

## 5. 特殊场景

### 5.1 小值域场景

**触发条件：** 当golden值小于Small Value Threshold时

**使用脚本：**
```python
from scripts.small_value_check import check_small_value_precision, should_use_small_value_standard

# 判断是否启用
if should_use_small_value_standard(npu_output, golden_output):
    # 注意: 需要三方芯片输出进行双标杆比对
    result = check_small_value_precision(npu_output, golden_output, third_party_output)
```

**详细标准：** 见 `special_cases.md`

### 5.2 INF/NAN场景

**使用脚本：**
```python
from scripts.inf_nan_check import check_inf_nan_consistency

result = check_inf_nan_consistency(npu_output, golden_output, third_party_output)
```

**详细判定规则：** 见 `special_cases.md`

### 5.3 精度复检

**触发条件：** 单用例不满足通过标准时

**使用脚本：**
```python
from scripts.confidence_interval import analyze_recheck_ratios

# 收集N次运行数据,计算每次的Ratio
ratios = []
for i in range(1000):
    npu_error = calculate_error(run_npu(seed=i), golden)
    benchmark_error = calculate_error(run_benchmark(seed=i), golden)
    ratio = npu_error / max(benchmark_error, 1e-10)
    ratios.append(ratio)

# 分析复检结果
result = analyze_recheck_ratios(np.array(ratios))
assert result['recheck_pass'], f"复检失败: CI=[{result['ci_lower']}, {result['ci_upper']}]"
```

**复检流程：** 见 `special_cases.md`

## 6. 标杆构造

### 6.1 标杆选择原则

**竞品对标优先，逐级降级备选**

| 优先级 | 标杆类型 | 适用场景 |
|-------|---------|---------|
| **1** | 业界CPU/三方芯片同等功能算子 | 标准算子 |
| **2** | 小算子拼接组合实现 | 融合算子 |
| **3** | 自行构造的CPU实现 | 非标准数据类型 |

### 6.2 双标杆实现

```python
# Golden: 使用高精度CPU实现
golden = cpu_implementation_high_precision(input)

# Benchmark: 三方芯片实现
benchmark = gpu_implementation(input)

# NPU实现
npu_output = npu_implementation(input)

# 计算Ratio
mare_ratio = mare(npu, golden) / mare(benchmark, golden)
```

## 7. 参考文档

- 详细精度标准：见 `golden/COMMERCIAL_OPS_PRECISION_DOCS.md`
- 特殊场景处理：见 `special_cases.md`
- 标杆构造方法：见 `benchmark_construction.md`
- 测试用例生成：见 `test_case_generation.md`