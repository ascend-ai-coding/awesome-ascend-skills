+# 索引、Scatter 与 Hash

本 archetype 是路由层，覆盖 categorical_embedding_bag、embedding_hash_table_lookup_or_insert、kv_cache_append、scatter_block_update、moe_re_routing。它不是完整实现手册；实施前必须继续读任务合同、DESIGN/PLAN 模板、c-api-primitives、sync-and-lifecycle，以及 source-map 中对应参考实现的路线审计结论。

## 路线选择

| 场景 | 执行单元 | 说明 |
|---|---|---|
| 离散 gather/scatter、每元素索引 | SIMT/AIV | SIMT 对不规则访问天然友好 |
| 大 block 连续搬运 | AIV DMA/Reg | 先按 inner 连续段合并访问 |
| first-insert-wins、flags | AIV + atomic/CAS | 必须冻结 winner 规则 |
| prefix sum + token 重排 | AIV | 两阶段或核内分段归约 |

## 关键设计点

1. 索引合法性：负数、padding_idx、filter_key、slot 越界、重复 slot 都是合同边界，不得试探性启动。
2. gather 输入维度合并为 outer/gather/inner；输出索引分解和输入索引构造必须一一对应。
3. scatter 必须先处理“是否复制整个输入”，再写 update；重复目标 index 的顺序语义要冻结。
4. first-insert-wins 不能用未排序 atomic 直接等价；winner 需要 flat index 最小或等价的确定性 tie-break。
5. KV cache append 要处理 cache 复制与 slot 更新；若 slot 重复，明确最后写者或禁止重复。
6. MoE re-routing 用 prefix sum 计算源/目标偏移，核对 expert_token_num 总和与 token 数一致。

## 验证顺序

1. 单索引、连续索引、重复索引、非法索引、空 bag。
2. 非对齐 embedding_dim、rank 1/2/3、int32/int64 index。
3. hash 表插入覆盖 hit/insert/filtered 三种 flags；并发插入用固定多轮输入验证确定性。
4. scatter 覆盖重复目标、末尾块、全量复制、独立输出地址。

## 事实源

- cann-samples：Samples/0_Introduction/03_simt_api/01_gather/general_gather、Samples/2_Performance/simt_scatter_story
- ops-nn：hash/embedding_hash_table_lookup_or_insert、index/embedding_bag、index/scatter_update、index/scatter_nd_update
- ops-transformer：attention/scatter_pa_kv_cache、moe/moe_re_routing

L3 映射：categorical_embedding_bag、embedding_hash_table_lookup_or_insert、kv_cache_append、moe_re_routing、scatter_block_update。
