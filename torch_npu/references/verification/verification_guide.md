# PyTorch NPU 运算精度验证指南

## 概述

本文档提供 PyTorch NPU 运算的精度验证流程、标准和方法。

## 验证流程（5 步）

### 1. 准备验证环境

```bash
# 安装依赖
pip install torch-npu

# 检查 NPU 可用性
python3 -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

### 2. 选择验证模板

使用 `templates/basic_operator.py` 作为基础模板：

```python
from templates.basic_operator import verify_operator
import torch

# 定义运算函数
def add_func(a, b):
    return a + b

# 定义输入规格
inputs_spec = [
    {'shape': (2, 3, 4), 'dtype': torch.float32},
    {'shape': (2, 3, 4), 'dtype': torch.float32}
]

# 执行验证
passed = verify_operator(add_func, inputs_spec)
```

### 3. 运行验证脚本

```bash
# 使用验证脚本
python3 scripts/verify_operator_accuracy.py --op add --shapes 2x3x4 --dtypes fp32

# 或在 Python 中直接运行
python3 basic_operator.py
```

### 4. 分析验证结果

验证脚本输出以下信息：
- 算子名称
- 最大绝对误差（Max abs diff）
- 最大相对误差（Max rel diff）
- 验证结果（PASS/FAIL）

### 5. 处理验证失败

如果验证失败：
1. 检查 NPU 驱动和版本
2. 检查输入数据范围和分布
3. 调整容差参数（rtol, atol）
4. 联系技术支持

## 精度标准

### FP32 验证标准

| 指标 | 数值 |
|------|------|
| 最大绝对误差 | < 1e-5 |
| 最大相对误差 | < 1e-4 |
| 通过阈值 | abs_diff < 1e-5 且 rel_diff < 1e-4 |

### FP16 验证标准

| 指标 | 数值 |
|------|------|
| 最大绝对误差 | < 1e-3 |
| 最大相对误差 | < 1e-2 |
| 通过阈值 | abs_diff < 1e-3 且 rel_diff < 1e-2 |

### BF16 验证标准

| 指标 | 数值 |
|------|------|
| 最大绝对误差 | < 1e-2 |
| 最大相对误差 | < 5e-2 |
| 通过阈值 | abs_diff < 1e-2 且 rel_diff < 5e-2 |

### 调整容差参数

```python
# 自定义容差参数
verify_operator(
    op_func,
    inputs_spec,
    rtol=1e-4,  # 相对误差容差
    atol=1e-5   # 绝对误差容差
)
```

## 常见问题

### Q1: 验证失败但误差很小（< 1e-4）

**原因：** 浮点数精度差异或 NPU 驱动版本问题

**解决方案：**
1. 检查 NPU 驱动版本是否匹配 PyTorch 版本
2. 使用相同版本在不同硬件上验证
3. 调整容差参数

### Q2: FP16 验证结果不一致

**原因：** FP16 下梯度累积可能导致累积误差

**解决方案：**
1. 增加数据点数量验证稳定性
2. 使用混合精度训练时重新验证
3. 考虑使用 bfloat16 替代 fp16

### Q3: 不同算子验证结果差异大

**原因：** 算子实现复杂度和精度要求不同

**解决方案：**
1. 查阅官方文档了解各算子的精度要求
2. 参考已知验证通过的案例
3. 针对特定算子调整验证标准

### Q4: 内存不足错误

**原因：** 输入数据量过大

**解决方案：**
1. 减小输入 shape
2. 使用分批次验证
3. 监控 NPU 内存使用情况

### Q5: 驱动报错

**原因：** NPU 驱动或 CUDA 版本不兼容

**解决方案：**
```bash
# 检查驱动版本
npu-smi info

# 检查 PyTorch 版本
python3 -c "import torch; print(torch.__version__)"

# 更新驱动和 PyTorch
pip install --upgrade torch-npu
```

## 参考模板

- **基础模板**: `references/verification/templates/basic_operator.py`
- **验证脚本**: `scripts/verify_operator_accuracy.py`

## 示例代码

详细示例请参考 `basic_operator.py` 和 `verify_operator_accuracy.py`。

## 技术支持

如遇到验证问题，请联系技术支持并提供：
- PyTorch 版本
- NPU 驱动版本
- 验证错误日志
- 输入数据和算子信息
