# {{OP_NAME}} 算子精度验证报告

**算子名称**: {{OP_NAME}}
**公式**: {{OP_FORMULA}}
**测试平台**: Ascend 910B
**参考基线**: PyTorch CPU `{{CPU_REF_EXPR}}` (float32 计算后转目标 dtype)
**支持的 dtype**: {{SUPPORTED_DTYPES_STR}}
**精度标准**: 生态算子开源精度标准（MERE/MARE）
**测试时间**: {{DATE}}

## 总览

| 指标 | 值 |
|------|-----|
| 总用例数 | {{TOTAL}} |
| 通过数 | {{PASSED}} |
| 失败数 | {{FAILED}} |
| 通过率 | {{PASS_RATE}}% |

## 精度阈值标准

通过条件：MERE < Threshold **且** MARE < 10 × Threshold

相对误差计算：`abs(actual - golden) / (abs(golden) + 1e-7)`

| dtype | Threshold | MERE 上限 | MARE 上限 (10×) |
|-------|-----------|----------|----------------|
<!-- 根据算子实际支持的 dtype 填写，如: -->
<!-- | float16 | 2⁻¹⁰ ≈ 9.77e-4 | 9.77e-4 | 9.77e-3 | -->
<!-- | float32 | 2⁻¹³ ≈ 1.22e-4 | 1.22e-4 | 1.22e-3 | -->

## 常规 Shape 测试结果

<!-- 按 TEST_SHAPES 中的 category 分组，每组一个表 -->

### {{CATEGORY_NAME}}

| # | 描述 | Shape | dtype | 元素数 | MERE | MARE | MaxAbsErr | CosSim | 结果 |
|---|------|-------|-------|--------|------|------|-----------|--------|------|

## 边界值测试结果

| # | 描述 | 值 | dtype | MERE | MARE | MaxAbsErr | CosSim | 结果 |
|---|------|-----|-------|------|------|-----------|--------|------|

## 按 dtype 汇总统计

| dtype | 用例数 | Threshold | MERE 范围 | MARE 范围 | CosSim 范围 |
|-------|--------|-----------|----------|----------|-------------|

## 关键发现

1. **各 dtype 精度特征**: ...
2. **规模稳定性**: ...
3. **边界值表现**: ...
4. **生产可用性**: ...
