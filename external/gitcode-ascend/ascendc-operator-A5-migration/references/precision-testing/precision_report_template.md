# {{OP_NAME}} 算子精度验证报告（ascend950）

**算子名称**: {{OP_NAME}}
**公式**: {{OP_FORMULA}}
**测试平台**: Ascend 950
**调用接口**: torch.ops.npu（{{NPU_CALL_EXPR}}）
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

## 常规 Shape 测试结果

### {{CATEGORY_NAME}}

| # | 描述 | Shape | dtype | 元素数 | MERE | MARE | MaxAbsErr | CosSim | 结果 |
|---|------|-------|-------|--------|------|------|-----------|--------|------|

## 边界值测试结果

| # | 描述 | 值 | dtype | MERE | MARE | MaxAbsErr | CosSim | 结果 |
|---|------|-----|-------|------|------|-----------|--------|------|

## 按 dtype 汇总统计

| dtype | 用例数 | Threshold | MERE 范围 | MARE 范围 | CosSim 范围 |
|-------|--------|-----------|----------|----------|-------------|

## 950 vs 910b 精度对比（如有基线）

| dtype | 910b MERE | 950 MERE | 910b MARE | 950 MARE | 结论 |
|-------|-----------|----------|-----------|----------|------|

## 关键发现

1. **各 dtype 精度特征**: ...
2. **规模稳定性**: ...
3. **边界值表现**: ...
4. **950 vs 910b 对比**: ...
5. **生产可用性**: ...
