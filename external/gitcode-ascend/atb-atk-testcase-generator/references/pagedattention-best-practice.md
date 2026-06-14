# PagedAttention 最佳实践摘要

**详细案例复盘（精度不达标、输入规范化、与 data_generation 对齐、内联解耦）**：见同目录 **[pagedattention-case-study.md](pagedattention-case-study.md)**。

---

- 节点配置：`node.yaml` / `node_perf.yaml` 必须包含至少一条 `backend: atb`，否则不会进入 ATB 推理与性能路径（可与 `backend: cpu` 等对照节点共存）
- 关键约束：`keyCache` 与 `valueCache` 的 `num_blocks`、`block_size` 必须对齐
- 关键约束：`query` 的 `head_size` 必须匹配 cache 对应维度
- 建议做法：`qkScale` 使用概率分布覆盖，不要只写固定值
- 注意事项：`in_formats` 可能是 list，不能假设一定是 string
- 稳定性：优先使用独立 `execute_paged_attention.py`，Golden 对齐 `data_generation` ref 或内联等价子集，避免不必要依赖全局状态
