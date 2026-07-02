# Attention Pattern Index

生成 attention 算子前先读本文件，用它决定需要看的模式文档。模式可以组合，但不要临时发明新模式。

渐进式披露原则：不要一上来读完所有模式文档。先用本索引定位命中的模式，再只读对应文档顶部的 `先读这个`；只有实现细节不够时，再继续读该文档后面的 `核心规则 / 地址公式 / 边界`。

## 核心模式

| 如果看到 | 读这个文档 |
| --- | --- |
| `actual_q_len` / `actual_kv_len` / `(T, H, D)` 拼接布局 | `@references/attention-patterns/references/tnd_pattern.md` |
| `block_table` / `page_table` / `k_cache` / `v_cache` / paged KV cache | `@references/attention-patterns/references/page_attention_pattern.md` |
| `mask` / causal / padding / visibleCols / `-inf` score 注入 | `@references/attention-patterns/references/mask_causal_pattern.md` |
| `Hq != Hkv` / GQA / MQA / `n_heads` vs `n_kv_heads` | `@references/attention-patterns/references/head_sharing_pattern.md` |
| `q_nope` / `q_rope` / `k_nope` / `k_rope` / `headdim_qk != headdim_v` / latent KV | `@references/attention-patterns/references/mla_pattern.md` |
| `indices` / `topk` / selected token / sparse gather | `@references/attention-patterns/references/topk_kv_sparse_pattern.md` |
| `sink_k` / `sink_v` / sink token / fixed prefix KV source | `@references/attention-patterns/references/sink_pattern.md` |
| **以上全部不命中** → 标准 Attention `[B, H, S, D]` | `workflows/templates/archive_tasks/flash_attention/` — block/tile 设计模板 |

## 组合顺序

多个模式同时出现时，按这个顺序理解：

```text
1. TND：如果输入是 (T, H, D)，先恢复 batch 边界。
2. Head Sharing：确定 q_head -> kv_head。
3. MLA：如果 Dqk 和 Dv 分离，确定 QK 路和 PV 路各自消费哪些维度。
4. Sink：如果有 sink source，确定 sink/local 的 source dispatch。
5. Sparse TopK：如果有 indices，确定要读哪些 logical token。
6. Paged KV：如果是 paged cache，把 logical token 映射到 physical cache row。
7. Mask/Causal：在 softmax 前处理可见性、padding、causal。
```

组合命中时仍然按需阅读：

- 先读每个命中文档的 `先读这个`，确认该模式在当前算子中承担的唯一职责。
- 优先读主导布局的模式：TND 或 Paged KV。
- `Head Sharing` 只决定 `kv_head`；不要在它里面找 page/sparse 的寻址规则。
- `MLA` 只决定 `Dqk/Dv` 和 latent 行的列消费规则；不要在它里面找 page 本身的映射规则。
- `Mask/Causal` 最后读；它只改 score visibility，不改 source 路径。

## 生成前问题

1. 输入是标准 `[B, H, S, D]`，还是 `(T, H, D)` 拼接布局？
2. K/V 是连续 tensor，还是 paged cache？
3. `Hq` 和 `Hkv` 是否相等？
4. `Dqk` 和 `Dv` 是否相等？
5. 是否有 `sink_k/sink_v`？
6. 是否有 `indices/topk`？
7. 是否有 causal、padding、显式 mask？

根据答案选择上面的文档。若 7 个问题全部否定 → 标准 Attention，读最后一行指向的 `workflows/templates/archive_tasks/flash_attention/` 中的 block/tile 设计实现，作为标准 MHA 的 block 级 Q 分块循环、O 分块循环、softmax rescale、C/V split 流水线设计参考。
