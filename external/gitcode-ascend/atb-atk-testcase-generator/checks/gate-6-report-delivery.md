# Gate 6 结果交付

## 输入

- Gate 5 通过结果
- 精度/性能/CSV（可选）结果数据
- 日志与报表产物路径

## 命令

```bash
# 汇总命令按项目实际脚本执行：
# 1) 汇总精度/性能/CSV 结果
# 2) 生成交付报告
# 3) 校验关键产物路径存在
```

## 通过标准

### 交付产物

- `ATB_<OpName>_gen.yaml`
- `generator_<op>.py`
- `execute_<op>.py`
- `node.yaml`
- `node_perf.yaml`
- `atk_output/.../log/atk.log`
- `atk_output/.../report/...xlsx`

- [ ] 汇总表已输出
- [ ] 关键产物路径可访问
- [ ] 风险与失败项已标注
- [ ] 用户确认交付完成

## 失败回流

- 产物缺失：回流 Gate 5 重新执行或补齐产物
- 汇总缺字段：修复模板并重生成报告
- 用户未确认：保持在 Gate 6，补充说明后再提交
