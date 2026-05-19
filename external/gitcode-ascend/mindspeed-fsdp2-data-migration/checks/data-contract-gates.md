# 数据契约门禁

## G1 样本层

- [ ] 抽样 N 条样本可解析
- [ ] 必需字段完整（至少 `input_ids`、`labels`、`attention_mask`）
- [ ] 多模态场景字段完整（如 `pixel_values`、`image_flags`）

## G2 路径层

- [ ] 支持 `dataset` 为 `str`
- [ ] 支持 `dataset` 为 `list[str]`
- [ ] 对不存在路径、空路径、权限错误给出可读报错

## G3 batch 层

- [ ] `collate_fn` 对变长样本稳定
- [ ] dtype/shape 与模型 forward 契约一致

## G4 证据层

- [ ] 每项检查有命令证据
- [ ] 失败项有最小复现输入与日志摘要
