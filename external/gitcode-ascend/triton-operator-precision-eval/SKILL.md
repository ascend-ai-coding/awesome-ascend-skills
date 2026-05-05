---
name: external-gitcode-ascend-triton-operator-precision-eval
description: Triton 算子精度评估。与 PyTorch 参考实现对比，自动计算误差指标，生成标准化精度报告。关键词：精度测试、precision
  evaluation、精度报告、accuracy verification。
original-name: triton-operator-precision-eval
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
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

**MANDATORY - READ ENTIRE FILE**：编写测试前，完整阅读 [`scripts/test_common.py`](scripts/test_common.py)。

## 误差阈值

判定条件：**MERE < 阈值 且 MARE < 10 × 阈值**

| 数据类型 | 阈值 | MERE 上限 | MARE 上限 |
|---------|------|----------|----------|
| float16 | 2⁻¹⁰ ≈ 9.77e-4 | 9.77e-4 | 9.77e-3 |
| float32 | 2⁻¹³ ≈ 1.22e-4 | 1.22e-4 | 1.22e-3 |
| bfloat16 | 2⁻⁷ ≈ 7.81e-3 | 7.81e-3 | 7.81e-2 |
| int8/uint8/int16/uint16/int32/uint32/int64/uint64 | 完全相等 | — | — |
| bool | 完全相等 | — | — |

其中：
- MERE = mean(|y_cal - y_ref| / |y_ref|)，即平均相对误差
- MARE = max(|y_cal - y_ref| / |y_ref|)，即最大相对误差

## 报告必须包含

- 验证配置：算子名称、测试形状、dtype、核心数
- 精度标准：所有 dtype 的阈值及判定条件（MERE < 阈值 且 MARE < 10 × 阈值）
- 验证结果：通过/失败总数
- 误差指标：MERE（平均相对误差）、MARE（最大相对误差）
- 判定条件

## 输出约定

- 精度报告写入算子目录下 `precision_report.md`（Markdown 格式）
- 精度评估脚本写入算子目录下 `precision_eval.py`
- 参考 `examples/` 目录下的示例脚本和报告格式

## 反模式清单（NEVER）

- ❌ 不提供 Torch 参考实现就做精度验证
- ❌ 用错误阈值（如 FP16 用 FP32 的阈值）
- ❌ 不用 MERE/MARE 判定条件（仅用 rtol/atol 不够严格）
- ❌ 归约操作不升精度到 FP32
- ❌ 只测一种 dtype/shape 就断言正确
- ❌ 跳过边界情况（非对齐维度）
- ❌ 精度通过前做性能优化

## 检查清单

- [ ] Torch 参考实现正确且在 NPU 上验证
- [ ] 覆盖多种 shape × dtype
- [ ] 归约操作用 FP32
- [ ] 生成标准化精度报告
