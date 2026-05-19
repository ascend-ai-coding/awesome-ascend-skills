# 数据迁移验收清单

- 注册存在且与目标 `dataset_type` 一致。
- 可通过 `build_mm_dataset` 完成数据集构建。
- `__getitem__` 字段满足模型 forward 契约。
- `collate_fn` 能安全支持变长 batch。
- 源预处理核心逻辑已复用，或适配理由充分。
- 支持路径输入变体（`str` 与 `list[str]`）。
- 对无效 schema 假设会给出清晰错误。
