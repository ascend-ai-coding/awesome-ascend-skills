# 典型结构量化改造卡

按结构匹配参考方案，不按模型名称匹配。模型名只能帮助定位代码，不能作为选择量化改造方案的依据。

每张结构卡包含「适用结构 / 映射 / 改造要点 / 常见问题 / 已确认模型经验」五段。模型经验只沉淀已正式确认并复现的内容；未确认或仅含量化绝对性能（无 BF16 同条件基线）的结论留在模型 `progress.md` 中，不进本卡。

> 运行对象映射的总表见 `quantization-contract.md` §4；本卡只在每张结构卡内给出该结构特化的映射差异。

---

## 结构卡 A：Dense Decoder

### 适用结构

- Decoder-only Transformer。
- Attention 是 MHA/GQA/MQA，核心投影为 `q_proj/k_proj/v_proj/o_proj` 或 fused QKV。
- FFN 是 dense MLP，核心投影为 `gate_proj/up_proj/down_proj` 或 fused gate-up。
- 无 routed experts。

### 映射

| 结构对象 | infer 对象 |
|----------|------------|
| Attention projections | `Linear` / `ReplicatedLinear` |
| MLP projections | `Linear` / `ReplicatedLinear` |
| fused QKV / gate-up | 只有所有 shard 量化策略一致时才可融合接入 |
| `ignore` 模块 | `UnquantizedLinearMethod` |

### 改造要点

- runner 读取 `quantization_config`，生成 `config.quant_config`。
- 所有可量化投影层构造时传入 `quant_config` 和稳定 prefix。
- prefix 必须能匹配 `targets` 或 regex target。
- `weight_scale` 通常按 output dim 加载；`smooth_scales` 用于动态 activation quant。
- post-load 重点检查 transpose、NZ format、scale dtype。

### 常见问题

| 问题 | 处理 |
|------|------|
| fused QKV 名称与产物中的 q/k/v 名称不一致 | 先拆回原有 q/k/v 或补 fused mapping，不能改量化方案 |
| 某些层在 `ignore` 中 | 显式浮点回退，记录模块 |
| 融合 Linear kernel 不支持量化 layout | 回退原有 Linear 路径继续量化验证 |

### 相关适配经验

#### 经验 A1：post-load 恢复大粒度量化 matmul

- 量化产物可能把原本融合的 `gate/up` 或 `q/k/v` 拆成多个 Linear。若直接逐个 Linear 跑，会增加多次 `npu_dynamic_quant` 和小粒度 `npu_quant_matmul` 开销。
- 可在 post-load 后恢复运行时融合：拼接 int8 weight，拼接 weight scale，共用一份 smooth scale，然后一次 dynamic quant + 一次 quant matmul。
- 融合前必须检查条件：权重 shape 兼容、投影均为量化 Linear、weight dtype 为 int8、`smooth_scales` 完全一致。
- `gate/up` 融合后接 `npu_swiglu`；`q/k/v` 融合后按 q/k/v 输出维度 split，再继续 norm/RoPE/FA。
- 若后续要转 NZ 格式，应优先在融合权重生成后统一处理，避免先 cast 再拼接导致 layout 不一致。

Skill 动作：

- Dense 量化初评估时，显式检查 `gate/up` 与 `q/k/v` 是否被量化产物拆散。
- 改造时优先设计 post-load 融合恢复路径；若 smooth scale 不一致，记录为不能融合的性能风险。
- 验证时同时比较"逐 Linear 量化"和"融合后量化"的 decode 性能，尤其是小模型。

---

## 结构卡 B：MoE Decoder

### 适用结构

- Decoder-only Transformer + routed experts。
- router/topk 通常保持浮点。
- experts 可能是 per-expert Linear，也可能是 packed expert 权重。
- 有 TP/EP 切分时，scale / smooth scale 可能需要按 rank 对齐。

### 映射

| 结构对象 | infer 对象 |
|----------|------------|
| shared attention / dense MLP | `Linear`（按结构卡 A 处理） |
| routed experts gate/up/down | 优先 `MoEGMM`；不满足时回退 per-expert Linear |
| router/topk | 通常浮点，不作为量化 target |
| expert smooth scale | 按 EP/TP 规则加载或聚合 |

### 改造要点

- `config_groups.targets` 若包含 `MoEGMM`，优先使用 `CompressedTensorsMoEGMMMethod`。
- 若只声明 `Linear`，需要确认 experts 是否仍能按 Linear 语义加载。
- W8A8 MoE 关注 `w13_weight`、`w2_weight`、`w13_weight_scale`、`w2_weight_scale`、`smooth_scale_1/2`。
- W4A16 / W4A8 关注 int32 pack/unpack、group_size、offset、scale layout。
- packed expert checkpoint 必须明确切片和 shard 规则。
- 混合位宽 MoE 必须有独立 `targets=["MoEGMM"]` config group；统一 W8A8 时检查 `gmm_quant_mode` 是否继承 `mm_quant_mode`。

### 常见问题

| 问题 | 处理 |
|------|------|
| MoE fused GMM 不支持当前量化 bit/scale | 回退原有 expert 路径继续验证，记录 GMM 量化需求 |
| EP 下 smooth scale 不完整 | 补 infer 侧 gather / rank 对齐能力，不能修改产物语义 |
| packed expert 名称与 `FusedMoEGMM` loader 不匹配 | 补加载映射或回退 per-expert Linear |
| dense / shared expert MLP 仍走浮点 SwigLU | 参考本卡经验 B1 |

### 已确认模型经验

#### 经验 B1：W8A8 用量化友好 SwigLU 链替代浮点 SwigLU

代表路径：`cann-recipes-infer/models/deepseek-v3.2-exp/models/modeling_deepseek.py`、`cann-recipes-infer/models/glm-5/models/modeling_glm.py`、`cann-recipes-infer/models/kimi-k2-thinking/models/modeling_deepseek.py`、`cann-recipes-infer/models/longcat-flash/models/modeling_longcat_flash.py`

- W8A8 下，`gate_up_proj` 常以 `out_dtype=torch.int32` 输出累加结果和 per-token scale。
- 原浮点链路 `gate_up -> npu_swiglu -> down_proj` 需要替换为 `npu_dequant_swiglu_quant`，同时消费 `gate_up_proj.weight_scale`、`down_proj.smooth_scales` 和上一段 matmul 的 `pertoken_scale`。
- `npu_dequant_swiglu_quant` 输出新的 activation 和 per-token scale，必须继续传给 `down_proj(intermediate, pertoken_scale)`。
- 这类改造不是公共 Linear 量化框架自动完成的，而是模型 MLP/SharedExpert forward 中的算子链路改写。
- runner 中经常需要把 `gate_up_proj` 的 weight scale 和 `down_proj` 的 smooth scale cast 到 float，避免融合算子输入 dtype 不匹配。

Skill 动作：

- 命中 `MergedColumnParallelLinear gate_up_proj + RowParallelLinear down_proj` 时，检查是否存在 `forward_w8a8` 分支。
- 若仍使用 `npu_swiglu` 浮点链路，标为 W8A8 性能/正确性风险。
- post-load 检查中增加 scale dtype map：`gate_up_proj` weight scale、`down_proj` smooth scale 通常需要 float。

#### 经验 B2：A8 scale 是路由/通信/专家计算的 side channel

代表路径：`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`、`cann-recipes-infer/models/deepseek-v3.2-exp/models/modeling_deepseek.py`、`cann-recipes-infer/models/glm-5/models/modeling_glm.py`、`cann-recipes-infer/models/kimi-k2-thinking/models/modeling_deepseek.py`、`cann-recipes-infer/models/longcat-flash/models/modeling_longcat_flash.py`

- MoE A8 不只是 expert 内部 `FusedMoEGMM` 量化；routing / dispatch / combine 是否携带 activation scale 也是适配核心。
- EP 路线至少有两条合法实现：一条是在 `npu_moe_init_routing_v2` / `npu_moe_distribute_dispatch_v2` 上显式走 A8 scale side channel；另一条是 dispatch / all-to-all 保持 BF16，在 `npu_moe_re_routing` 或 `dispatch_v2` 之后按 rank 本地 `npu_dynamic_quant`，再把 `pertoken_scale` 传给 experts。
- 因此，Prefill EP 不一定要求 `pertoken_scale` 随 token 一起 all-to-all；Decode 也不一定要求 `dispatch_v2` 直接产出可供 experts 消费的 `dynamic_scale`。先画清楚当前模型真实走的是哪条路，再判断缺项。
- TP 路线中，routing 输出的 token 分组模式（`npu_moe_init_routing_v2` 的 `expert_tokens_num_type`，0=cumsum / 1=count / 2=key_value）必须与 GMM 的 `group_list_type`（0/1/2 为同义编码格式）配套一致，再加 `row_idx_type` / `drop_pad_mode` 对齐，否则 GMM 的 token 分组和 scale 对不上。本仓 A8 MoE GMM 实测走 `expert_tokens_num_type=1` + `group_list_type=1`；`=2`（key_value / [idx,size]）是另一合法编码，非硬性要求。
- 若采用“全局 smooth scale 驱动的量化路由”路径，EP 下每卡只保留本地 expert 的 smooth scale，post-load 后需要 all-gather `smooth_scale_1`。若采用“BF16 dispatch / rerouting 后本地 dynamic quant”路径，则这不是通用必查项，重点改查 per-rank `pertoken_scale` 是否连续传递、dispatch 是否仍保持 BF16、以及额外 dynamic quant 的开销归因。
- LongCat 还存在 zero expert/copy expert 语义。只对 routed experts dispatch，zero expert 的权重贡献在 finalize 侧单独处理，不能把 topk 权重和 token 范围简单套成普通 MoE。

Skill 动作：

- MoE 初评估必须画出 `routing -> dispatch/all_to_all -> rerouting -> experts -> combine/finalize` 中 scale 的传递路径。
- 不要只检查 `FusedMoEGMM` 是否可量化；还要检查 routing/dispatch/combine 是否接入 A8 scale。
- EP 场景先判定真实实现属于“全局 smooth scale 路由”还是“BF16 dispatch 后本地 dynamic quant”；前者把 `smooth_scale_1` all-gather 列入 post-load 必查项，后者把 `pertoken_scale` 传递、dispatch BF16 保持和额外 `npu_dynamic_quant` 开销列入必查项。
- 有 zero/copy expert 时，单独记录 active expert range、topk weight mask 和 finalize 规则。

---

## 结构卡 C：MLA + MoE / KVCache

### 适用结构

- Attention 使用 MLA、latent KV、LoRA rank projection 或 KV cache 特殊 layout。
- 通常同时伴随 MoE。
- 量化对象除 Linear/MoEGMM 外，可能包含 `kv_cache_scheme` 或 k/v output scale。
- 子场景：长序列 + Indexer / LightningIndexer / Sparse Flash Attention（见经验 C2）。

### 映射

| 结构对象 | infer 对象 |
|----------|------------|
| MLA q/k/v/latent projections | `Linear` |
| routed experts | `MoEGMM` 或 per-expert Linear（按结构卡 B 处理） |
| KVCache quant | `kv_cache_scheme` / `get_cache_scale()` |
| MLA fused prolog / sparse attention | 融合不支持量化时回退原有非融合路径 |
| Indexer / LightningIndexer / LI cache | 独立量化闭环，详见经验 C2 |

### 改造要点

- 先拆清 prefill/decode 的 MLA 路径，不要只看类名。
- KVCache 量化必须有明确 scale 来源；`.k_proj.output_scale` / `.v_proj.output_scale` 可映射到 k/v cache scale。
- MLA projection 的 post-load 要核对 cache layout、rope/nope split、rank 切分。
- MoE 部分按结构卡 B 处理。
- 融合 attention/prolog 不支持量化时，回退原有算子继续量化基线。

### 常见问题

| 问题 | 处理 |
|------|------|
| KVCache scale 缺失 | 输出契约补充诉求 |
| MLA fused op 不支持 quantized layout | 回退非融合 MLA 路径，记录融合量化需求 |
| MoE 与 MLA 同时引入 scale dtype 冲突 | 分别记录 Linear/MoEGMM/KVCache 的 scale dtype 要求 |

### 已确认模型经验

#### 经验 C1：MLA/KVCache C8 的量化边界由 prolog、cache 和 FA 共同定义

代表路径：`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`、`cann-recipes-infer/models/deepseek-v3.2-exp/models/modeling_deepseek.py`、`cann-recipes-infer/models/glm-5/models/modeling_glm.py`、`cann-recipes-infer/models/kimi-k2-thinking/models/modeling_deepseek.py`

- C8 不是简单把 KV cache dtype 改成 int8。模型侧需要同时处理 `npu_mla_prolog_v3`、cache layout、query scale、cache scale 和后续 attention kernel。
- DeepSeek-R1 路线会在进入 MLA prolog 前对 hidden states 做动态量化，并通过 `weight_quant_mode=2`、`kv_cache_quant_mode=1`、`query_quant_mode=1` 让 prolog 处理 W8A8C8。
- DeepSeek v3.2/GLM 路线中，`npu_mla_prolog_v3` 常只启用部分权重量化，使用 `kv_cache_quant_mode=3`、`quant_scale_repo_mode=1`、`k_nope_clip_alpha` 等 per-tile/clip 契约。
- C8 下 rope cache 可能通过 fake/empty tensor 占位，nope cache 内同时承载量化 cache 和 scale repo，后续 sparse attention 需要 `key_quant_mode/value_quant_mode` 与 `quant_scale_repo_mode` 对齐。
- 对 sparse attention，C8 路线可能切到 `npu_sparse_flash_attention_antiquant`，而不是普通 `npu_sparse_flash_attention`。
- CP/offload 场景还要同步 cache 与 scale 的 scatter/all-gather/reorder，不能只验证单卡 cache shape。

Skill 动作：

- 遇到 MLA + KVCache C8 时，必须把 `npu_mla_prolog_v3` 参数作为适配核心审查项。
- 记录 `weight_quant_mode`、`kv_cache_quant_mode`、`query_quant_mode`、`cache_mode`、`quant_scale_repo_mode`、clip alpha 和 dequant scale 来源。
- 验证时检查 FA/Sparse FA 是否真实消费 int8 cache 和对应 scale；不能只凭 cache dtype 判定 C8 生效。

#### 经验 C2：Indexer / LI cache C8 有独立量化闭环

代表路径：`cann-recipes-infer/models/deepseek-v3.2-exp/models/indexer.py`、`cann-recipes-infer/models/glm-5/models/indexer.py`

- Indexer 侧通常不是普通 attention 量化，而是独立的 LI cache / sparse indexer 量化闭环。
- q/k 在进入 lightning indexer 前会先做 Hadamard transform，再动态量化，生成 `query_dequant_scale` 和 `key_dequant_scale`。
- `past_key_scales_indexer` 是和 indexer key cache 配套的 scale cache；prefill 用 scatter update，decode 用 indexed update。
- CP prefill 场景下，key scale 要 all-gather 并按 reverse split/reorder 恢复；query scale 可能还要切成 prev/next 两段传给两次 indexer。
- GLM 使用 `li_cache_quant_mode` 区分 LI cache 量化；DeepSeek v3.2 复用 `kv_cache_quant_mode` 并在 decode W8A8C8 下可走 PYTPO 融合 prolog。
- 量化 indexer 调用需要显式传 `key_dequant_scale`、`query_dequant_scale`、`key_quant_mode`、`query_quant_mode`，weights 常转 float16。

Skill 动作：

- 发现 Indexer/LightningIndexer 时，单独建立 Indexer 量化检查项，不要并入普通 KVCache。
- 检查 Hadamard、q/k dynamic quant、scale cache update、CP scale reorder 和 quant lightning indexer 参数。
- 验证 decode 时关注是否进入量化 indexer 分支；prefill 与 decode 可以命中不同算子。

---

## post-load 例外汇总

代表路径：模型 ForCausalLM 的 `process_weights_after_loading` 方法（如 `cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py`、`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`），由 `cann-recipes-infer/executor/core/model_worker/model_worker.py::_process_weights_after_loading` 在权重加载后触发。

同一套公共量化方法在不同模型中，post-load 参数不一定相同。常见差异：

- **跳过模块**：DeepSeek/GLM 常跳过 `kv_b_proj`。
- **scale dtype 适配**：对 `gate_up_proj/q_b_proj/wq_b` 的 scale、`down_proj` 的 smooth scale 做 float 适配。
- **transpose / NZ 例外**：LongCat 对 `classifier` router 权重避免 transpose/NZ，并在 online split weight 关闭时走已有 `scale_dtype_adapter/cast_format`。
- **format 选择**：现有 runner 会按模型和模块设置 `is_nz`、`is_transpose`、scale dtype 等 post-load 例外；新增模型需要基于实际 runtime 和实测收益单独确认 base/NZ format，不能沿用未经验证的默认策略。

Skill 动作：

- 每个模型都要从 runner 的 `_process_weights_after_loading` 或等价函数提取 post-load 例外表。
- 初评估输出中单独列出：skip 模块、scale dtype map、smooth scale dtype map、transpose/NZ 例外、post-load 融合顺序。
- 性能异常时先核对 post-load 顺序和 dtype/layout，再看更深层 kernel profiling。

---

## 新结构补卡规则

只有满足任一条件才新增结构卡：

- 当前模型结构无法归入 Dense Decoder、MoE Decoder、MLA + MoE / KVCache。
- 出现新的 infer runtime object，不是 `Linear`、`MoEGMM`、KVCache。
- 出现新的量化张量语义或 post-load 规则。
- 出现可复用的新融合回退模式。

新增卡必须以结构命名，不以模型名称命名。

## 经验沉淀模板

新增模型量化适配经验时，挂在对应结构卡的「已确认模型经验」下，按以下格式：

```markdown
#### 经验 X：标题

代表路径：

- 经验描述（包含算子链、scale 路径、cache layout 等具体细节）
- 不能融合/不能量化的条件
- Prefill/Decode 差异
- TP/EP/CP/offload 差异
- 真实生效证据
- 性能 profiling 关注点

Skill 动作：

- 量化初评估时的检查项
- 改造要点
- 验证关注点
```

不沉淀以下内容：未确认方案 / 失败试验细节 / 非量化优化（AFD / 多流 / 权重预取 / SuperKernel / graph cache）/ 仅含量化绝对性能没有 BF16 同条件基线的结论。
