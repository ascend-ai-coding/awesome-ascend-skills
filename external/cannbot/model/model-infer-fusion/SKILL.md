---
name: external-cannbot-model-model-infer-fusion
description: 基于 PyTorch 框架的昇腾 NPU 模型推理融合算子优化技能。分析模型代码，识别可替换为 torch_npu 融合算子的计算模式，生成替换方案。触发场景：torch_npu
  融合算子替换、MoE/Attention/FFN/Norm 等模块的推理算子适配、torch_npu API 使用咨询。基于仓库已有模型的融合算子经验，按计算语义推荐最佳方案。
original-name: model-infer-fusion
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# torch_npu 融合算子优化技能

分析 PyTorch 模型代码中的计算模式，匹配 torch_npu 融合算子进行替换优化。基于 torch_npu 本地 docstring 查询脚本（`scripts/torch_npu_query.py`）和仓库已有模型的融合算子使用经验，按模块逐一分析、匹配、替换、验证。

---

## 工作流程

### 第一步：分析模型代码，拆解模块

分析模型代码，输出模块清单。

**拆解网络结构**：识别顶层结构（Embedding → Transformer Blocks → LM Head），拆解每个 Block 为独立模块：
- **Attention 层**：Norm → QKV 投影 → RoPE → KV Cache → Flash Attention → O 投影（MLA 含 V absorb）
- **MoE / FFN 层**：Norm → Gate → 路由分发 → 专家计算 → 聚合
- **其他模块**：Embedding、LM Head、跨层残差等

**记录运行场景特征**：
- Prefill / Decode 分支差异：同一模块在两个阶段可能走不同算子路径，替换时分别处理
- 量化需求：BF16 / W8A8 / W8A8C8 / W4A16
- 分布式配置：TP / EP / DP，影响 MoE 路由算子选择
- 固定约束：layout（NZ/BSND/TND）、cache 格式、metadata

**关键子链路展开**：对 Attention、MoE 等复杂模块，展开到可替换链路级别：
- **Attention**：RoPE、KV Cache 写入/读取、Attention Core（FA / FA v2 / Sparse FA）
- **MoE**：Gate / TopK、Routing Init / Dispatch、Expert 计算、Finalize / Combine、Shared Expert（若有）
- 其他关键链路（Residual + Norm、QKV Projection、O Projection、MC2 / AllToAll 等）一并纳入


### 第二步：按模块独立匹配仓库参考实现

对第一步拆解的每个模块，**独立**在仓库参考实现中匹配最接近的算子链路。命中后，将该路径作为**候选蓝图**，并对照第一步的关键子链路清单补充分析该路径未覆盖的部分；未命中的模块跳到第三步在算子总表中搜索。

#### Attention 层

**注意**：若 FA 融合算子已在前置阶段替换，仅跳过 FA 调用本身。Attention 的其他子链路（RoPE、KV Cache 写入、Residual+Norm、QKV/O Projection 等）仍需逐一检查是否有可用融合算子。同时注意多步骤整体融合算子（如 `npu_kv_rmsnorm_rope_cache` 融合 RMSNorm+RoPE+Cache写入，`npu_mla_prolog_v3` 融合 Q/KV投影+RMSNorm+RoPE+Cache写入），不要将单个子模块简单归为"标准线性层无融合算子"而跳过评估。

分析 Attention 时，`KV Cache` 需要与 `Attention Core` 结合讨论，不建议完全拆开；除架构外，还应同时确认：
- cache 组织方式：连续非PA / PA
- 写入索引语义：`kv_len / start_pos`，或 `slot_mapping / cache_index`
- 后续 `Attention Core`：非融合 attention、`npu_fused_infer_attention_score`、`npu_fused_infer_attention_score_v2`、`npu_sparse_flash_attention`

常见实现组合包括：
- 连续非PA + 连续写入（如 `scatter_update_` + `kv_len / start_pos`）→ 再评估非融合 attention 或融合 `FA`
- PA + slot-based write（如 `npu_scatter_nd_update_` / `npu_scatter_pa_kv_cache` / `cache_index`）→ 再评估与该 cache 形态匹配的 `FA`
- MLA absorb 特化路径：`npu_mla_prolog_v3(cache_index / slot_mapping)` + 融合 `FA`

先根据当前实现的架构、cache 形态、写入方式和 `Attention Core` 组织，判断它更接近哪条参考链路，再进入对应详情文档。

```
Attention 架构？
  │
  ├─ GQA / MHA（标准多头 / 分组查询注意力）
  │    │
  │    ├─ Prefill: 当前 batch 的 q/k/v → FA(sparse_mode=3 推荐，sparse_mode=2 不推荐)
  │    └─ Decode:  从 KV Cache 读取 → FA(actual_seq_lengths_kv)
  │    → 详情：references/module-attention-gqa.md
  │
  └─ MLA（Multi-head Latent Attention，低秩 KV 压缩）
       │
       ├─ 无 Indexer
       │    ├─ Prefill: 分步投影 → 展开 K/V → FA v1 或 v2
       │    └─ Decode:  absorb（手动或 npu_mla_prolog_v3）→ FA v1 或 v2 → V absorb
       │    → 详情：references/module-attention-mla-absorb.md
       │
       └─ 有 Indexer（稀疏 Top-K KV 选择）
            ├─ Prefill/Decode 共路径
            │  npu_mla_prolog_v3 → Indexer → 稀疏 FA → V absorb
            → 详情：references/module-attention-mla-indexer.md
```


#### MoE / FFN 层

先确认当前模块属于 Dense FFN 还是 MoE；若为 MoE，再结合 gate 形式、并行模式和阶段判断更接近哪条参考链路。

```
是否有 MoE？
  │
  ├─ 无（Dense FFN）→ 在算子总表中确认 Dense Linear / Activation / Norm 可用融合算子
  │
  └─ 有 MoE
       │
       ├─ Gate 算子
       │    ├─ softmax 打分 → npu_moe_gating_top_k_softmax（qwen3-moe）
       │    └─ sigmoid/noaux → npu_moe_gating_top_k（deepseek 系列）
       │
       └─ 路由 + 专家计算（按并行模式和阶段区分）
            ├─ Prefill（纯 TP）: init_routing_v2 → grouped_matmul → finalize_routing
            ├─ Prefill（EP）:    init_routing_v2 → AllToAll → re_routing → grouped_matmul → finalize_routing
            └─ Decode（EP+TP）:  MC2 dispatch_v2 → grouped_matmul → MC2 combine_v2
       → 详情：references/moe-fusion-reference.md
```

#### 未匹配模块

其他未被上述判断树覆盖的模块（Embedding、LM Head、跨层残差、Diffusion 特有模块等），跳到第三步在算子总表中搜索。

---

### 第三步：查阅算子接口文档，确认可用性与适配性

对第二步命中的算子或未匹配到模式的模块，按以下优先级查阅 torch_npu 接口文档：

**优先：本地 docstring**

torch_npu wheel 内置 `_op_plugin_docs.py`，含所有算子的完整中文文档。通过 `scripts/torch_npu_query.py` 查询：

```bash
# 单 API 详情
python3 scripts/torch_npu_query.py show <api_name>

# 反向搜索
python3 scripts/torch_npu_query.py search "<keyword>"

# 枚举所有算子名
python3 scripts/torch_npu_query.py list [--prefix npu_]
```

**确认可用性**：

- 对模式命中的算子：查阅详情文档确认参数约束和适配场景
- 对未命中模式的模块：在算子总表中搜索，阅读详情文档分析功能

**适配验证**：

- 检查 shape、dtype、layout、cache 组织及 metadata 是否满足算子要求
- 若差异可通过合理前置改造解决（如格式转换、RoPE 预计算与取值路径整理、KV Cache 静态化/PA 改造等），应标记为“候选 + 需前置改造”，并说明所需改造项；部分流程较复杂的改造可询问用户是否采用
- 仅当差异属于硬约束且无法通过合理前置改造解决时，才可标记为”不适配”。标记时必须注明具体硬约束（如算子报错信息、文档明确的参数限制），不能仅以”需改动较大”为由标记不适配

---

### 第四步：分析阶段审查

在进入代码替换前，审查以下各项。未完成的项须返回对应步骤补齐，不得跳过直接进入实施。

若当前任务仅要求分析，则在本步结束，输出分析结果、候选方案和验证计划，不进入代码实施。

**分析阶段审查项**：
- [ ] 模块拆解完整：已展开到可替换链路级别（含 Prefill/Decode 分支差异），Attention/MoE 关键子链路已覆盖
- [ ] 参考匹配完整：已按模块匹配仓库参考实现（或确认无匹配），已补充分析参考路径未覆盖的子链路
- [ ] 算子约束已确认：已查阅 reference / API 文档确认候选算子的适配约束和量化兼容性
- [ ] 候选清单已形成：每个候选模块明确前置条件、最小验证切口和阻塞点


---

### 第五步：逐模块实施替换

已有全面的算子候选分析后，依照替换流程对候选清单中的所有模块逐个迭代替换，每次只改一处，验证通过再继续下一个；不得跳过任何已进入候选清单的模块。若当前模块无法继续实施，也必须记录其失败证据、阻塞原因和当前结论。

若候选模块依赖以下前置改造且尚未完成，可按需查阅对应资源：

- `RoPE` 预计算、缓存和取值路径整理：[`references/rotary-embedding-pattern.md`](references/rotary-embedding-pattern.md)
- `KV Cache` 静态化 / `PA` 改造：参考 `model-infer-kvcache` skill

**融合算子替换流程**：

1. 完成前置改造（如需要）
2. 替换该模块的算子代码
3. 验证：精度对比（融合前后输出对齐）
4. 验证：性能对比（确认有收益）
5. 若验证通过 → 保留，继续下一个模块
6. 若验证失败 → 回退改动，重新分析该模块：
   - 是否有其他可用算子或替代方案？→ 尝试替代方案
   - 确认当前无适配算子但有融合收益 → 记录为新需求（模块位置、计算语义、不适配原因、期望融合形式）
   - 当前无法继续实施 → 回退后记录失败证据和阻塞原因，继续下一个
7. 记录该模块的优化报告：精度对比结果、性能对比结果、日志或报错路径

注：替换时可参考仓库中最接近的模型实现和 `references/` 下的算子接口文档使用说明。

**实施阶段检查项**：
- [ ] 候选清单中的所有模块均已逐一处理（替换或记录跳过原因），无遗漏
- [ ] 每个已实施模块均记录了精度和性能对比结果
- [ ] 实施失败的模块已记录报错信息、尝试过的处理方式和最终结论

---

## 参考文档索引

> 以下文档按需查阅，避免一次性加载消耗 token

| 主题 | 路径 |
|------|------|
| **torch_npu API 文档查询脚本** | [`scripts/torch_npu_query.py`](scripts/torch_npu_query.py) |
| Attention: GQA 参考链路 | [`references/module-attention-gqa.md`](references/module-attention-gqa.md) |
| Attention: MLA Absorb 参考链路 | [`references/module-attention-mla-absorb.md`](references/module-attention-mla-absorb.md) |
| Attention: MLA+Indexer 参考链路 | [`references/module-attention-mla-indexer.md`](references/module-attention-mla-indexer.md) |
| MoE 算子模式详解 | [`references/moe-fusion-reference.md`](references/moe-fusion-reference.md) |
| RotaryEmbedding 预计算与调用模式 | [`references/rotary-embedding-pattern.md`](references/rotary-embedding-pattern.md) |
| op-plugin 仓库（在线） | [gitcode.com/Ascend/op-plugin](https://gitcode.com/Ascend/op-plugin) |
