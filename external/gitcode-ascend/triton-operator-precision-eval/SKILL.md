---
name: external-gitcode-ascend-triton-operator-precision-eval
description: 接收Triton算子实现，自动调用Torch小算子实现（CPU或NPU）进行精度比对，并生成精度报告。当用户需要验证Triton算子实现的正确性和精度、与PyTorch实现进行精度比对、生成标准化精度报告时使用。
original-name: triton-operator-precision-eval
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-18'
synced-commit: 9f4c6c19a042f03239a07ac2f3196fb590d0a114
license: UNKNOWN
---

# Triton算子精度评估技能

## 核心原则

**精度是算子正确性的底线，任何优化都不能突破这条底线。**

## 功能概述

该技能用于自动化评估Triton算子实现的精度，通过与PyTorch（CPU或NPU）的对应算子实现进行比对，生成详细的精度验证报告。

### 核心功能
- 自动接收Triton算子实现
- 支持与CPU或NPU上的Torch小算子进行比对
- 支持多种数据类型（float16、float32、int8、uint8等）
- 自动生成精度验证报告
- 支持批量测试不同参数配置

## 工作流程

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Triton算子实现  │────▶│ 生成测试数据    │────▶│ 执行Torch对比实现 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
          ▲                     │                     │
          │                     ▼                     ▼
          │              ┌─────────────────┐     ┌─────────────────┐
          │              │ 执行Triton实现  │     │ 计算误差指标    │
          │              └─────────────────┘     └─────────────────┘
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │ 生成精度报告    │
                        └─────────────────┘
```

### 核心组件

1. **测试数据生成**：使用 `test_common.generate_numpy()` 生成随机测试数据
2. **Torch对比实现**：用户提供的Torch算子实现
3. **Triton算子执行**：使用Triton JIT编译并执行用户提供的Triton kernel
4. **精度验证**：使用 `test_common.validate_cmp()` 进行精度比对，支持不同数据类型的误差阈值
5. **报告生成**：生成包含误差指标的精度验证报告

## 使用方法

### 前置条件

- 已安装Triton和PyTorch环境
- 已安装 `torch_npu`（如果使用NPU进行测试）
- 已准备Triton算子实现代码

### 编写测试用例

创建测试文件（如 `test_abs.py`），包含以下内容：

1. **导入必要模块**：
   ```python
   import triton
   import triton.language as tl
   import numpy as np
   import torch
   import pytest
   import test_common
   ```

2. **实现Torch对比函数**：
   ```python
   def torch_pointwise(x0):
       # 实现与Triton算子对应的Torch功能
       return torch.abs(x0)
   ```

3. **实现Triton算子**：
   ```python
   @triton.jit
   def triton_abs(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
       # Triton kernel实现
       offset = tl.program_id(0) * XBLOCK
       base1 = tl.arange(0, XBLOCK_SUB)
       loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
       for loop1 in range(loops1):
           x0_prime = offset + (loop1 * XBLOCK_SUB) + base1
           x0 = offset + (loop1 * XBLOCK_SUB) + base1
           tmp0 = tl.load(in_ptr0 + (x0), None)
           tmp2 = tl.abs(tmp0)
           tl.store(out_ptr0 + (x0), tmp2, None)
   ```

4. **编写测试用例**：
   ```python
   @pytest.mark.parametrize('param_list',
                          [
                              ['float16', (2, 4096, 8), 32, 2048, 64],
                              ['float32', (2, 4096, 8), 32, 2048, 64],
                              ['int8', (2, 4096, 8), 32, 2048, 64],
                              ['uint8', (2, 4096, 8), 32, 2048, 64],
                          ]
                          )

   def test_case(param_list):
       dtype, shape, ncore, xblock, xblock_sub = param_list
       np_x0 = test_common.generate_numpy(shape, dtype)
       x0 = torch.from_numpy(np_x0).to(eval('torch.' + dtype)).npu()
       y_ref = torch_pointwise(x0)
       y_cal = torch.zeros(shape, dtype = eval('torch.' + dtype)).npu()
       triton_abs[ncore, 1, 1](x0, y_cal, xblock, xblock_sub)
       test_common.validate_cmp(dtype, y_cal, y_ref)
   ```

### 运行测试

```bash
# 运行单个测试文件
pytest test_abs.py -v

# 运行所有测试文件
pytest ./examples/ -v
```

## 精度验证规则

### 不同数据类型的验证规则

| 数据类型 | 验证方式 | 误差阈值 |
|---------|---------|---------|
| float16 | 相对误差 | rtol=1e-03, atol=1e-03 |
| float32 | 相对误差 | rtol=1e-04, atol=1e-04 |
| bfloat16 | 相对误差 | rtol=1e-02, atol=1e-02 |
| int32/int64/int16/int8 | 完全相等 | - |
| uint32/uint64/uint16/uint8 | 完全相等 | - |
| bool | 完全相等 | - |

### 误差指标

- **平均相对误差(MERE)**：所有元素相对误差的平均值
- **最大相对误差(MARE)**：所有元素相对误差的最大值
- **绝对误差**：元素值之差的绝对值

## 精度报告格式

生成的精度报告（如 `eco_report.txt`）包含以下内容：

```
================================================================================
                              Triton算子精度验证报告                               
--------------------------------------------------------------------------------
[验证配置]:
  数据类型: float32 (Single Precision)
  MERE阈值: 1.220703e-04
  MARE阈值: 1.220703e-03 (10×MERE阈值)
  小值域阈值: 1.000000e-07
--------------------------------------------------------------------------------
[精度标准]:
  float16: 相对误差 rtol=1e-03, atol=1e-03
  float32: 相对误差 rtol=1e-04, atol=1e-04
  bfloat16: 相对误差 rtol=1e-02, atol=1e-02
  int32/int64/int16/int8: 完全相等
  uint32/uint64/uint16/uint8: 完全相等
  bool: 完全相等
--------------------------------------------------------------------------------
[验证结果]:
  验证结果: FAIL
  样本总数: 4096
--------------------------------------------------------------------------------
[误差指标]:
  平均相对误差(MERE): 6.642197e-03
    阈值要求: MERE < 1.220703e-04
  最大相对误差(MARE): 3.458786e+00
    阈值要求: MARE < 1.220703e-03
--------------------------------------------------------------------------------
[判定条件]:
  ✓ MERE < 阈值: False
  ✓ MARE < 10×阈值: False
  ✓ 总体结果: False
================================================================================
```

### 报告内容要求

精度报告必须包含以下内容：

1. **验证配置**：算子名称、测试形状、数据类型、NPU核心数等
2. **精度标准**：每个数据类型的具体精度要求（误差阈值或完全相等）
3. **验证结果**：测试总数、通过数量、失败数量、总体结果
4. **详细误差指标**：每个数据类型的平均相对误差、最大相对误差、最大绝对误差
5. **判定条件**：所有数据类型测试通过的状态

其中，**精度标准**部分必须列出所有支持的数据类型及其对应的精度要求，确保报告的可读性和可追溯性。

## 反模式清单（NEVER）

- ❌ 不提供 Torch 对比实现就进行精度验证
- ❌ 使用错误的误差阈值（FP16 用 FP32 的阈值）
- ❌ 归约操作不升精度到 FP32
- ❌ 只测试一种数据类型就断言精度正确
- ❌ 跳过边界情况测试（如非对齐维度）
- ❌ 不生成标准化精度报告
- ❌ 在验证通过前就进行性能优化

## 检查清单

### 测试用例完整性
- [ ] 提供了 Torch 对比实现？
- [ ] 测试用例覆盖多种形状？
- [ ] 测试用例覆盖多种数据类型？
- [ ] 测试用例包含边界情况？

### 精度验证
- [ ] 归约操作使用 FP32 精度？
- [ ] 使用了正确的误差阈值？
- [ ] 在 NPU 上进行测试？
- [ ] 验证了测试标杆的准确性？

### 报告生成
- [ ] 生成了标准化精度报告？
- [ ] 报告包含所有必要信息？
- [ ] 误差指标计算正确？

## 故障处理

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| Triton kernel编译失败 | Triton语法错误或版本不兼容 | 检查Triton语法，确保Triton版本与代码兼容 |
| 精度验证失败 | 算子实现逻辑错误或精度损失 | 检查算子实现，调整算法以提高精度 |
| NPU设备不可用 | 未安装torch_npu或设备未正确配置 | 安装torch_npu，检查NPU设备状态 |
| 内存不足 | 测试数据过大 | 减小测试数据规模或调整参数配置 |
