---
name: external-cannbot-model-model-infer-parallel-analysis
description: 基于 PyTorch 框架的昇腾 NPU 模型推理并行策略分析技能。分析模型架构参数和昇腾硬件规格，推荐最优的 TP/EP/DP 并行配置（parallel_config）。触发场景：新模型需要确定并行策略、现有配置需要优化、部署卡数或硬件变更后需要重新评估。输出为结构化的
  parallel_config 推荐及定量依据。
original-name: model-infer-parallel-analysis
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# 模型并行策略分析

> **本 skill 仅做分析和推荐，不修改任何代码。** 输出为 parallel_config 推荐和方案文档。

---

## 适用范围

- **模型类型**：Dense Transformer、MoE Transformer
- **硬件平台**：昇腾 Atlas A2 / A3 系列
- **决策范围**：`parallel_config` 中各 `*_tp_size`（TP）、`o_proj_tp_size`、`cp_size`（CP）、`kvp_size`（KVP）的值，以及由此推导的 DP/EP 度（`*_dp_size = world_size // *_tp_size`）

## 分析流程

```
第一步：提取模型参数与模块链路
    ↓
第二步：定性分类 → 候选并行模式
    ↓
第三步：定量估算（先查阅 references/config-index.md 获取起点数值）
    ↓
第四步：方案审查 → 输出推荐
    ↓
（可选）第五步：Profiling 校准
```

**禁止**：跳过前三步直接给配置、不做估算凭经验拍数

---

## 第一步：提取模型参数与模块链路

### 1.1 基础参数

从模型 config（`config.json` 或 `configuration_*.py`）中提取：

| 参数 | 符号 | 获取字段 | 说明 |
|------|------|---------|------|
| 架构类型 | - | 模型文档 | Dense / MoE / MoE+MLA / MoE+DSA |
| 总层数 | L | `num_hidden_layers` | |
| MoE 层数 | L_moe | `num_moe_layers` 或推断 | Dense 模型为 0 |
| Hidden size | H | `hidden_size` | |
| FFN 中间维度 | H_ffn | `intermediate_size` | |
| 注意力头数 | N_h | `num_attention_heads` | |
| KV 头数 | N_kv | `num_key_value_heads` | GQA: N_kv < N_h |
| Head dim | D_h | H / N_h 或 `head_dim` | |
| 专家总数 | E | `num_experts` | Dense 模型为 0 |
| 激活专家数 | E_act | `num_experts_per_tok` | |
| 词表大小 | V | `vocab_size` | |

MLA 模型额外提取：

| 参数 | 获取字段 | 说明 |
|------|---------|------|
| KV 压缩维度 | `kv_lora_rank` | MLA 的 compressed KV dim |
| Q 压缩维度 | `q_lora_rank` | MLA 的 compressed Q dim |
| V head dim | `v_head_dim` | oproj_tp 约束：`N_h * v_head_dim % oproj_tp == 0` |

### 1.2 模块链路识别

不同模块在 Prefill 和 Decode 阶段的计算特性不同，但当前分析目标是在**单套 parallel_config** 下兼顾两阶段。需要识别模型的关键模块链路：

| 模块 | Prefill 特性 | Decode 特性 | 并行维度 |
|------|-------------|-----------|---------|
| Embedding | 大词表矩阵 | 大词表矩阵 | embed_tp |
| Attention QKV | 计算密集（大 S） | 访存密集（S=1） | attn_tp / cp_size |
| Attention Core（FA） | 计算密集 | 访存密集 | attn_tp / cp_size / kvp_size |
| Attention O_proj | 计算密集 | 访存密集 | oproj_tp（MLA 模型独立设） |
| Dense FFN | 计算密集 | 访存密集 | dense_tp |
| MoE Router | 轻量 | 轻量 | 跟随 EP |
| MoE Expert FFN | 计算密集 | 分散访存 | moe_tp（通常=1，走 EP） |
| LM Head | 大词表矩阵 | 大词表矩阵 | lmhead_tp |

> **关键认知**：单机部署下 Prefill 和 Decode 共用一套配置。分析时根据用户目标场景确定侧重：高吞吐侧重 Decode，低时延侧重 Prefill TTFT，均衡则两者兼顾。不要输出两套独立 parallel_config。

> **注意非标准模块**：模型中可能存在非标准的大参数模块（如 N-gram Embedding、多模态 Vision Encoder 等），其切分方式可能与标准 TP 不同。需要单独分析这些模块在不同并行配置下的切分约束和显存占用，避免方案看起来显存可行但实际因某个模块无法切分而 OOM。

### 1.3 部署信息确认

**自动获取**：
- 单卡显存 M：通过 `npu-smi info` 或 `torch.npu.get_device_properties` 确认

**向用户确认**：

| 参数 | 说明 |
|------|------|
| 部署卡数 W | 单节点总 NPU 数 |
| 目标场景 | 高吞吐 / 低时延 / 均衡 |
| 最大序列长度 | 决定是否需要 CP / KVP |

> **当前限制**：本框架仅支持单机 BF16 部署，不支持多机分布式、PD 分离和量化推理。分析时 TP/EP 通信均在节点内，显存按 BF16 估算。

### 完成标志

- [ ] 架构类型已确认（Dense / MoE / MoE+MLA / MoE+DSA）
- [ ] 基础参数完整提取
- [ ] 模块链路已识别，Prefill/Decode 计算特性已标注并纳入单套配置权衡
- [ ] 部署信息已确认（W, M, 目标场景, 序列长度）
- [ ] 已确认仓库中是否有相似模型

---

## 第二步：定性分类

根据模型参数和部署场景，用决策树确定候选并行模式。**优先向仓库已验证的最佳实践靠拢**。

> 决策树表达分析思路，所有判断均为基线候选而非硬约束，需结合定量估算验证。硬约束、强经验、实现特性的分类见第四步"约束检查"。

### 统一决策树

决策树分三层：架构主干（由模型结构决定）→ 场景调整（由目标场景和 batch size 决定）→ 序列长度附加（按需叠加 CP/KVP）。三层共同产出一套 parallel_config。

#### 第一层：架构主干

确定非 Attention 模块的基础并行方式，以及 Attention 的默认起点。第二层再根据场景调整 Attention TP 度。

```
架构类型？
├─ Dense Transformer
│  ├─ 单卡放得下？→ 基线候选：DP
│  └─ 放不下？→ 基线候选：纯 TP，节点内 TP 起步
│
└─ MoE Transformer
   │
   ├─ MoE 层
   │  ├─ EP 和 TP 均需对比（单机场景 EP 度 = 卡数）
   │  ├─ expert_ffn 较小时 TP 会导致碎矩阵，EP 计算效率更高
   │  └─ EP 需确认 experts_per_rank ≤ 24（dispatch_v2 硬件约束）
   │
   ├─ Dense FFN
   │  ├─ 有独立 Dense 层 → TP，度数控制在节点内
   │  └─ 全 MoE 无独立 Dense → dense_tp=1
   │
   ├─ O_proj（MLA 模型）→ 待第二层确定 attn_tp 后决定
   │  attn_tp>1 时 oproj_tp=1，attn_tp=1 时可独立设值
   │
   ├─ Embed/LMHead
   │  ├─ 大词表（V>100K）→ 独立设 TP
   │  └─ 小词表 → 跟随主策略
   │
   └─ 单机部署规模（卡数 W 由 1.3 确认）
      ├─ 单卡 → 无需并行
      ├─ 少卡（W < 节点满配）→ 纯 TP 为主，节点内 HCCS 带宽充足
      └─ 满卡（W = 节点满配）→ 纯 TP 或混合 EP+TP，注意 TP 碎矩阵退化
```

#### 第二层：场景调整

根据目标场景和 batch size 调整 Attention TP 度。

```
目标场景？
├─ 高吞吐（大 batch）
│  ├─ MLA → attn_tp=1，最大化 DP，各卡独立处理不同请求
│  └─ GQA → Decode 默认 attn_tp=1 以最大化 DP；
│     Prefill 需结合 batch 和 seq_len 判断，若单卡 attention 计算成为瓶颈则保留提升 attn_tp 的候选
│
├─ 低时延（小 batch）
│  ├─ MLA → 保留小度数 attn_tp（如 2/4）作为候选，分摊 Prefill 计算降低 TTFT；
│     Decode 阶段仅在 profiling 显示 attention 计算成为明显瓶颈时才提升 attn_tp
│  └─ GQA → attn_tp 适度，分摊 Prefill 计算降低 TTFT
│
└─ 均衡
   → 在高吞吐和低时延之间折中

背景：MLA 压缩 KV 不按 head 存，TP 对 KV 侧无收益，但 Q/O 计算仍可分摊
```

#### 第三层：序列长度附加

CP 和 KVP 作为附加机制，默认在不推翻前两层主干的前提下叠加。若引入后导致 Attention 或 O_proj 的并行约束改变，需回到第二层重新校正。

```
最大序列长度？
├─ ≤4K → 无附加
├─ 4K-64K → 检查 Prefill 计算分摊是否充分
│  ├─ 充分 → 无附加
│  └─ 不充分
│     ├─ MLA → +CP，可能减少对大 attn_tp 的需求
│     └─ GQA → 提升 attn_tp 或 +CP
└─ 64K+ → +CP，超长 Decode 可叠加 KVP
   注意：KVP 约束 oproj_tp = kvp_size，需回到第一层校正 O_proj 配置
```

> **模块间切换代价**：不同 TP 度边界需 AllGather/ReduceScatter 重排，通常远小于统一 TP 度带来的浪费，但切换频繁时需关注。

### 速查表

> 速查表为基线候选，不是最终答案，需结合定量估算验证。

| 场景 | 基线候选 | 关键配置 | 已验证参考 |
|------|---------|---------|-----------|
| Dense 中小规模 | 纯 TP | all tp = W | GPT-OSS 8卡 |
| MoE 大规模高吞吐 | Attn DP + MoE EP + 差异化 TP | attn_tp=1, moe_tp=1 | R1/V3.2/GLM-5/Kimi-K2 128卡 |
| MoE 中规模 | 混合 EP+TP 或 纯 TP | 需对比两种方案 | LongCat 32卡 / R1 16卡 |
| MoE 小规模低时延 | 纯 TP 或 attn_tp>1 + EP | attn_tp=W, moe_tp=1 | 待验证 |
| 长序列 + MLA | 叠加 CP | +cp_size | V3.2/GLM-5/Kimi-K2 64卡 |
| 长序列 + GQA | 大 TP 或 叠加 CP | attn_tp=W 或 +cp_size | R1 32卡 |
| 超长序列 | 叠加 KVP | +kvp_size, oproj_tp=kvp_size | LongCat 131K |
| 进阶高吞吐 | +AFD+EPLB+MTP | enable_afd, perfect_eplb | LongCat/R1 128卡 |

### 已验证的并行模式

仓库中存在以下经过验证的模块组合模式（具体数值见 `references/config-index.md`）：

| 模式 | 特征 | 代表模型 |
|------|------|---------|
| 大规模高吞吐（全 MoE） | Attn DP + MoE EP，所有 tp=1 | R1 128卡 |
| 大规模高吞吐（MLA + Shared Expert） | Attn DP + Dense TP + MoE EP + O_proj TP | V3.2/GLM-5/Kimi-K2 128卡 |
| 大规模高吞吐（AFD） | 同上 + Attention-FFN 分离 | LongCat 128卡 |
| 中规模混合 | Attn DP + Dense TP + MoE EP | LongCat 32卡 |
| 小规模纯 TP | 所有模块统一 TP | R1/Qwen3 16卡 |
| 长序列 + CP（MLA） | Attn 走 CP + MoE EP | V3.2/GLM-5/Kimi-K2 64卡 |
| 长序列 + 大 TP（GQA） | 所有非 MoE 模块统一大 TP | R1 32卡 |

> **使用方式**：从速查表选候选模式，再从此表确认模式特征是否匹配，定量估算时查阅 reference 获取具体数值。

### 完成标志

- [ ] 候选并行模式已确定（一套配置 + 可选 CP/KVP 附加）
- [ ] 各模块的 TP 度方向已确定，参考了已验证配置
- [ ] 关联了仓库中最接近的参考模型

---

## 第三步：定量估算

根据候选模式，查阅参考配置获取起点数值，再做量化评估确定 parallel_config 具体值。

### 3.0 查阅参考配置

从 `{file:./references/config-index.md}` 中找到与候选模式最匹配的已验证配置，作为定量估算的起点：

- 按部署场景（Decode 高吞吐/低时延/混合、Prefill 长序列、超长序列）查找
- 读取匹配模型的实际 YAML 配置文件和设计文档（匹配表见 config-index 末尾）
- 以已验证数值为基准，结合目标模型的参数差异做调整

### 3.1 参数量计算

**标准 GQA Attention**：
```
Attention/层 = H × D_h × (2 × N_h + 2 × N_kv)    # Q, K, V, O_proj
```

**MLA Attention**（DeepSeek-V3/R1、LongCat-Flash 等）：
```
Attention/层 ≈ H × q_lora_rank                      # q_down_proj
              + q_lora_rank × D_h × N_h             # q_up_proj
              + H × kv_lora_rank                     # kv_down_proj
              + kv_lora_rank × (N_kv × D_h × 2)     # kv_up_proj (K + V)
              + N_h × v_head_dim × H                 # o_proj
```
> 注意：MLA 的具体投影结构因模型而异（是否有 rope 分离、absorb 模式等），此公式为近似。精确参数量以模型 config 和代码中的矩阵维度为准。

**FFN / MoE / Embed**：
```
FFN/层       = 3 × H × H_ffn                        # gate + up + down
MoE/层       = E × 3 × H × H_ffn                    # 所有专家
Embed        = V × H
LMHead       = V × H

P_total = (L - L_moe) × (P_attn + P_ffn)
        + L_moe × (P_attn + P_moe)
        + P_embed + P_lmhead
```

### 3.2 单卡显存估算

> 以下公式为 rough sizing，用于判断候选策略是否显存可行。精确值需要实际运行或参考已验证配置的 profiling 数据。

**并行度推导**（各模块独立，不要混用）：
```
attn_dp_size  = world_size // attn_tp_size
dense_dp_size = world_size // dense_tp_size
moe_ep_size   = world_size // moe_tp_size    # EP 度由 moe_tp_size 决定，与 attn_tp_size 无关
```

**参数权重显存**（按模块分别除以对应的 TP/EP 度）：
```
P_per_card = P_attn / attn_tp_size
           + P_ffn / dense_tp_size          # 非 MoE 层
           + P_moe / moe_ep_size            # MoE 层, moe_ep_size = world_size // moe_tp_size
           + P_embed / embed_tp_size
           + P_lmhead / lmhead_tp_size

参数显存 = P_per_card × bytes_per_param
```
bytes_per_param：当前仅支持 BF16 = 2

**KV Cache 显存**（每卡 B_local = B / dp_size）：
```
标准 GQA：B_local × S × L × 2 × N_kv × D_h × bytes_per_kv
MLA 压缩：B_local × S × L × kv_lora_rank × bytes_per_kv
```
> 注意：实际 KV Cache 还受 page/block 分配粒度、prefix cache、chunked prefill 等影响

**其他显存开销**（公式无法精确估算，需关注）：
- 量化附加状态（当前不支持量化，后续扩展时需考虑）
- 通信 buffer（AllReduce / AllToAll 临时缓冲）
- graph 模式 workspace
- 激活临时 buffer（Decode 通常较小；Prefill 与 S、chunk 大小、attention backend 强相关）

**可行性判断**：参数显存 + KV Cache 已接近卡容量 × 0.8 → 不可行，需增大并行度

### 3.3 通信分析

通信对性能的影响不是简单的"字节量大则慢"。按以下三层分析，优先级从高到低。

**层 1：是否跨节点（影响最大）**
- 节点内 HCCS：~56 GB/s（A2），延迟低
- 节点间 RDMA：带宽和延迟都远差于节点内
- 判断：tp_size > 单节点卡数 → 跨节点，通常应避免

**层 2：通信原语类型**

| 原语 | 触发场景 | 特点 |
|-----|---------|------|
| AllReduce | TP 的 RowParallel 输出 | 数据量 = tensor_size × 2(tp-1)/tp |
| AllToAll | EP 的 token dispatch/combine | 数据量取决于激活专家数和 token 分布 |
| AllGather / ReduceScatter | 模块间 TP 度切换边界 | 数据量 = tensor_size |
| Send/Recv | CP 的 KV 分片交换、AFD 通信 | 点对点，可与计算重叠 |

**层 3：是否可被计算重叠**
- 多流并行可以隐藏部分通信延迟（Send/Recv + Compute 并行）
- AllReduce / AllToAll 通常在关键路径上，难以完全重叠
- 仓库实践：LongCat AFD 用 Send/Recv 替代 AllReduce 正是为了利用重叠

> 排序优先级：通信是否跨节点 > 原语类型 > 通信字节量 > 是否可 overlap

### 3.4 根据估算结果调整策略

估算完成后，按以下判据检查和调整：

**显存可行性**：
- 总显存 > 卡容量 × 0.95 → 需要增大并行度或减小 batch
- KV Cache 占比 > 50% 可用显存 → 长序列场景，考虑 KVP / CP / KVCache Offload / moe_chunk 调小
- 仓库实践：R1 长序列 moe_chunk=512，LongCat 131K 用 KVP=8

**通信可行性**：
- tp_size > 单节点卡数 → TP 通信跨节点，性能急剧下降，必须调整
- 节点内 HCCS：~56 GB/s（A2），节点间 RDMA：远低于此
- 仓库实践：所有模型的 TP 通信都控制在节点内
- 估算通信时间 > 计算时间 → TP 度过大，减小 TP 或改用 EP
- 仓库实践：MoE 大规模 Decode 用 EP（AllToAll）替代 TP（AllReduce）正是因为此

**Batch 吞吐**：
- B_local = B / dp_size 太小（如 < 16）→ DP 度不够，吞吐受限
- 增大 DP 的方式：减小 TP 度（dp_size = world_size / tp_size）
- 仓库实践：128 卡 Decode attn_tp=1 使 attn_dp=128，最大化吞吐

**Prefill TTFT**：
- 单卡 Prefill 计算量过大（S > 4K）→ 需要 TP 或 CP 分摊
- MLA 模型优先 CP，标准 GQA 优先 TP（原因见决策树）

> 如果估算显示已验证配置的参考值合理（显存可行 + 通信在节点内），优先直接采用已验证值，不必重新推导。

### 完成标志

- [ ] 各候选策略的单卡显存已计算
- [ ] 通信量已估算
- [ ] 策略已排序（有定量依据）
- [ ] parallel_config 具体值已确定

---

## 第四步：方案审查

输出前检查以下约束：

### 约束检查

**[A] 硬约束**（不满足则配置非法）：
- [ ] 各模块 `world_size % *_tp_size == 0`
- [ ] `num_attention_heads % attn_tp_size == 0`
- [ ] `num_key_value_heads % attn_tp_size == 0`（按 KV head 切分的实现）
- [ ] `num_experts % ep_size == 0`（要求均匀专家划分的实现）

**[B] 强经验检查**（通常应满足，不满足需有充分理由）：
- [ ] 单卡显存估算 ≤ 实际显存 × 0.95
- [ ] tp_size ≤ 单节点卡数（避免跨节点 TP）
- [ ] Decode 阶段 attn_tp 不大于必要值（避免 S=1 时的无效通信）

**[C] 本仓库实现检查**（依赖具体框架/模型实现）：
- [ ] dp_size 由 `world_size // tp_size` 自动推导（本框架特性，见 inference_config.py）
- [ ] KVP 模式下 `oproj_tp_size == kvp_size`（本仓库 KVP 实现约束）
- [ ] `N_h * v_head_dim % oproj_tp_size == 0`（MLA o_proj 切分约束）

### 输出格式

各模块的 TP 度是独立决策，候选方案应充分组合（如 attn_tp=8 + moe_tp=1 + dense_tp=8），不要只考虑"全 TP"或"全 EP"两个极端。输出 2~3 个候选方案并排序，每个候选都说明对高吞吐/低时延/长序列场景的取舍：

```yaml
# 候选 A（推荐）：{模式名称}
# 理由：{为什么推荐}
# 参考：{最接近的已验证配置}
candidate_a:
  parallel_config:
    attn_tp_size: {value}       # attn_dp_size = world_size // attn_tp_size
    dense_tp_size: {value}
    moe_tp_size: {value}        # moe_ep_size = world_size // moe_tp_size
    embed_tp_size: {value}
    lmhead_tp_size: {value}
    o_proj_tp_size: {value}     # MLA 模型需要，非 MLA 可省略
    cp_size: {value}            # 按需设置
    kvp_size: {value}           # 按需设置
  estimation:
    params_per_card_gb: {value}
    kv_cache_per_card_gb: {value}
    memory_feasible: true/false
    cross_node_comm: true/false
  tradeoff:
    throughput: {优势/中性/风险}
    latency: {优势/中性/风险}
    long_context: {优势/中性/风险}

# 候选 B（备选）：{模式名称}
# 理由：{适用场景 / 与 A 的权衡}
candidate_b:
  parallel_config: ...

# 注：dp_size/ep_size 不在 YAML 中配置，由框架自动推导
```

**排序维度**（按优先级）：
1. 显存可行性（不可行直接排除）
2. 跨节点通信风险（有则降级）
3. 预期性能（定量估算显示结构性优势的方案优先）
4. 与已验证配置的距离（无性能差异时，越近越优先）

> 当定量分析明确显示某方案有结构性性能优势（如 EP 的完整矩阵 vs TP 的碎矩阵），即使该方案未经验证，也应标注为"推荐对比测试"，不因已验证距离自动降级。

将推荐结果以结构化格式输出。

### 权重处理提示

配置变更后注意权重是否需要重新处理：
- `enable_online_split_weight: True`：无需重新转换，运行时自动切分
- `enable_online_split_weight: False`：必须重新转换，权重和 parallel_config 绑定
- 具体操作见 model-infer-parallel-impl skill

### 完成标志

- [ ] 硬约束全部通过
- [ ] 推荐配置含定量依据
- [ ] 已关联仓库参考配置

---

## （可选）第五步：Profiling 校准

当有实际 profiling 数据时，校准第三步的估算。

### 需要的三个关键指标

**指标 1：通信占比**（最重要）

从 trace 中提取所有 `Hccl*` 算子耗时之和 / 单步总耗时：
- < 20%：策略合理
- 20-40%：有优化空间，检查 TP 通信是否跨节点
- \> 40%：策略有问题，需调整 TP/EP 比例

> 查看位置：TensorBoard Trace View，筛选 `HcclAllReduce`、`HcclAllToAll` 等算子

**指标 2：显存峰值**

通过 `torch.npu.max_memory_allocated()` 获取各 Rank 峰值：
- 验证第三步的估算偏差
- 确定 batch_size 可以开多大

> 当前仓库 profiler 配置 `profile_memory=False`，需额外插入一行代码获取

**指标 3：Rank 间耗时方差**（MoE EP 专属）

对比各 Rank 的单步总耗时，计算 (max - min) / mean：
- < 5%：均衡良好
- 5-15%：考虑启用 `perfect_eplb: True`
- \> 15%：EP 策略可能不适合当前负载

> 查看位置：各 Rank 独立 trace 文件（`{res_path}/prof/` 目录）

### 偏差处理

| 偏差范围 | 行动 |
|---------|------|
| < 10% | 策略确认，无需调整 |
| 10-30% | 针对偏差模块微调（如调整 tp_size 使通信在节点内） |
| > 30% | 重新评估策略；考虑模块级差异化并行 |

---

## 常见错误

| 错误模式 | 根因 | 预防 |
|---------|------|------|
| 跳过分析直接用默认全 TP | 未评估通信开销 | 必须完成通信量估算再定配置 |
| MoE 模型小规模部署只考虑一种方案 | 未对比 EP 和 TP 的通信效率 | 中小规模需对比 EP 和 TP，参考 R1 16 卡两种配置 |
| 忽略 Embed/LMHead 的 tp_size | 大词表模型中这两层显存占比不低 | V > 100K 时独立设置，参考 V3.2/Kimi-K2 embed=16 |
| 显存估算漏掉 KV Cache | Decode 长序列时 KV Cache 是显存大头 | 必须单独计算 KV Cache，MLA 用 kv_lora_rank 估算 |
| MLA 模型对 Attention 做 TP | MLA KV Cache 是压缩向量不按 head 存，TP 无收益 | MLA 模型 attn_tp=1，Prefill 用 CP 替代 |
| 只看 Decode 或只看 Prefill 单侧指标 | 单机部署下两阶段共用一套 parallel_config，单侧最优可能导致整体退化 | 在单套配置下同时评估吞吐、TTFT 和长序列需求 |
| TP 通信跨节点 | 跨节点 RDMA 带宽远低于节点内 HCCS | tp_size ≤ 单节点卡数 |
| 改了 parallel_config 但没重新转换权重 | offline split 的权重和配置绑定 | 推荐 enable_online_split_weight，或改配置后重跑 weight_convert |

---

完整的仓库参考配置索引见 `{file:./references/config-index.md}`
