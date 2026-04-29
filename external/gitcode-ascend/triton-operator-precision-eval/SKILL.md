---
name: external-gitcode-ascend-triton-operator-precision-eval
description: Triton 算子精度评估。与 PyTorch 参考实现对比，自动计算误差指标，生成标准化精度报告。关键词：精度测试、precision
  evaluation、精度报告、accuracy verification。
original-name: triton-operator-precision-eval
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-29'
synced-commit: 8faee0275e457955c8f50989aef8972c0838db31
license: UNKNOWN
---

# Triton 算子精度评估

## 核心原则

**精度是算子正确性的底线。验证通过前不做性能优化。**

## 工作流

1. 编写 Torch 参考实现（标杆）
2. 编写测试用例（覆盖多种 shape × dtype × 边界情况）
3. 在 NPU 上执行 Triton 算子和 Torch 参考实现
4. 用 `test_common.validate_cmp()` 比对，生成精度报告

**MANDATORY - READ ENTIRE FILE**：编写测试前，完整阅读 [`test_common.py`](references/test_common.py)。

## 误差阈值

| 数据类型 | rtol | atol |
|---------|------|------|
| float16 | 1e-03 | 1e-03 |
| float32 | 1e-04 | 1e-04 |
| bfloat16 | 1e-02 | 1e-02 |
| int8/uint8/int16/uint16/int32/uint32/int64/uint64 | 完全相等 | — |
| bool | 完全相等 | — |

## 报告必须包含

- 验证配置：算子名称、测试形状、dtype、核心数
- 精度标准：所有 dtype 的误差阈值
- 验证结果：通过/失败总数
- 误差指标：MERE（平均相对误差）、MARE（最大相对误差）
- 判定条件

## 反模式清单（NEVER）

- ❌ 不提供 Torch 参考实现就做精度验证
- ❌ 用错误阈值（如 FP16 用 FP32 的阈值）
- ❌ 归约操作不升精度到 FP32
- ❌ 只测一种 dtype/shape 就断言正确
- ❌ 跳过边界情况（非对齐维度）
- ❌ 精度通过前做性能优化

## 检查清单

- [ ] Torch 参考实现正确且在 NPU 上验证
- [ ] 覆盖多种 shape × dtype
- [ ] 归约操作用 FP32
- [ ] 生成标准化精度报告
