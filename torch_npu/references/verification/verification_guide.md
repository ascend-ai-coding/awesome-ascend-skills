# PyTorch NPU 运算精度验证指南

## 概述

本文档提供 PyTorch NPU 运算的精度验证流程、标准和方法。

## 验证前置条件

### 1. 环境检查清单

在执行精度验证前，必须确认以下环境配置正确。

#### 使用自动检测脚本（推荐）

运行 CANN 环境检测脚本自动识别版本和配置：```bash
bash references/verification/cann_environment_check.sh
```

#### 手动配置步骤

```bash
# 1. 设置 CANN 环境
# 对于 CANN 8.5.0+:
source /usr/local/Ascend/cann/set_env.sh

# 对于 CANN 8.3.RC1 或更早版本:
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ⚠️ 注意：不要使用 nnae/set_env.sh，这可能导致算子找不到错误

# 2. 检查 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | head -20

# 3. 检查 torch 和 torch_npu 版本
python3 -c "import torch; import torch_npu; print(f'PyTorch: {torch.__version__}'); print(f'torch_npu: {torch_npu.__version__}')"

# 4. 验证 NPU 可用
python3 -c "import torch; import torch_npu; print(f'NPU available: {torch.npu.is_available()}'); print(f'Device count: {torch.npu.device_count()}')"
```

**重要**: 使用正确的 `set_env.sh` 路径！参考：
- CANN 8.5.0+: `/usr/local/Ascend/cann/set_env.sh`
- CANN 8.3.RC1: `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- ❌ 错误路径: `/usr/local/Ascend/nnae/set_env.sh`
### 2. 版本配套检查

**重要**: torch_npu 版本必须与 CANN 版本配套，否则会出现算子执行失败。

| torch_npu 版本 | CANN 版本 | PyTorch 版本 |
|---------------|-----------|-------------|
| 2.8.0 | 8.0.RCx / 8.2.RC1 | 2.8.0 |
| 2.7.1 | 8.0.RC1 | 2.7.1 |
| 2.6.0 | 7.0.0 | 2.6.0 |

详细配套表参见：[version_compatibility.md](../installation/version_compatibility.md)

### 3. 常见错误码参考

| 错误码 | 含义 | 常见原因 | 解决方案 |
|--------|------|----------|----------|
| **561103** | 内核解析失败 | CANN/torch_npu 版本不配套 | 检查版本配套表，重新安装匹配版本 |
| **561000** | 算子不存在 | NPU 缺少该算子实现 | 升级 CANN 或使用 op-plugin 扩展 |
| **501000** | 内存不足 | 输入数据过大 | 减小 batch size 或输入维度 |
| **501013** | 设备繁忙 | NPU 正在执行其他任务 | 等待或使用其他 NPU 设备 |


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
### Q6: 所有算子执行失败，错误码 561103

**原因：** CANN 和 torch_npu 版本不配套

**诊断步骤：**
```bash
# 检查版本
python3 -c "import torch; import torch_npu; print(f'torch: {torch.__version__}, torch_npu: {torch_npu.__version__}')"
ls -la /usr/local/Ascend/nnae/
```

**解决方案：**
1. 确认 torch_npu 版本与 CANN 版本配套（参见版本配套表）
2. 重新安装正确版本的 torch_npu:
   ```bash
   pip uninstall torch-npu
   pip install torch-npu==<compatible_version>
   ```
3. 如问题持续，联系华为技术支持

### Q7: "Op does not has any binary" 错误

**原因：** NPU 缺少对应算子的二进制实现

**解决方案：**
1. 检查 CANN 版本是否支持该算子
2. 使用 op-plugin 扩展算子支持
3. 升级到更新版本的 CANN



## 参考模板

- **基础模板**: `references/verification/templates/basic_operator.py`
- **验证脚本**: `scripts/verify_operator_accuracy.py`
- **CANN 环境检测脚本**: `references/verification/cann_environment_check.sh`

## 示例代码

## CANN 环境检测

使用 `cann_environment_check.sh` 脚本自动检测 CANN 版本：```bash
bash references/verification/cann_environment_check.sh
```

该脚本会：
1. 自动检测 CANN 安装路径和版本
2. 识别 CANN 8.5.0+ 或 8.3.RC1 结构
3. 提供正确的环境配置命令
4. 检查 opp 包是否安装
5. 提供故障排查建议

## 技术支持

如遇到验证问题，请联系技术支持并提供：
- PyTorch 版本
- NPU 驱动版本
- 验证错误日志
- 输入数据和算子信息
- CANN 版本（运行 cann_environment_check.sh 获取)
## 技术支持

如遇到验证问题，请联系技术支持并提供：
- PyTorch 版本
- NPU 驱动版本
- 验证错误日志
- 输入数据和算子信息
- CANN 版本（运行 cann_environment_check.sh 获取）
