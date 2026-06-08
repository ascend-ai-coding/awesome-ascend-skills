# Tensor Tags 参考指南

标准化的 tensor tag 名称，用于 vLLM 和 SGLang 框架的 tensor dump 日志。

> **本文档为 Tag 标准参考**，详细注入代码模板请参阅 [framework-integration.md](./framework-integration.md)

---

## 框架差异速查

| 特性 | vLLM | SGLang |
|------|------|--------|
| 层 ID 属性 | `self.layer_idx` | `self.layer_id` |
| Attention | 单一 forward / **MLA** | forward_prepare/forward_core (分离式) |
| 残差管理 | 内联 `residual + x` | LayerCommunicator |
| MoE 路径 | 单一 forward_normal | forward_normal + forward_deepep |
| 日志前缀 | `[TD_CMP]` | `[TD_CMP]` |

### Attention 类型说明

| 类型 | 模型示例 | 说明 |
|------|---------|------|
| **标准Attention** | Qwen3, LLaMA | 使用 `qkv_proj`, `o_proj` |
| **MLA Attention** | DeepSeek-V2/V3, GLM-4.5/4.6/4.7, GLM-5 | 使用 `MultiHeadLatentAttentionWrapper` |

---

## Tag 格式规范

Tags 遵循层级点分隔格式：`<prefix>.<component>.<detail>`

### 命名前缀

| 前缀 | 框架 | 说明 |
|------|------|------|
| 无 | 通用 | 标准组件 |
| `forward_prepare.*` | SGLang | 分离式 Attention prepare 阶段 |
| `forward_core.*` | SGLang | 分离式 Attention core 阶段 |
| `forward_normal.*` | SGLang | MoE forward_normal 路径 |
| `forward_deepep.*` | SGLang | MoE forward_deepep 路径 |
| `step3p5.*` | SGLang | Step3.5 模型特定 |

### 命名规则

1. 遵循层级点分隔格式
2. 使用小写加下划线
3. 使用组件名作为前缀 (如 `moe.`, `mlp.`, `attn.`)
4. 跨框架保持命名一致性（见下方对照表）

---

## 通用 Tags (两框架必须支持)

| Tag | 说明 | vLLM | SGLang |
|-----|------|------|--------|
| `input_ids` | 输入 token | ✅ | ✅ |
| `positions` | 位置编码 | ✅ | ✅ |
| `hs.in` | Layer 输入 | ✅ | ✅ |
| `hs.out` | Layer 输出 | ✅ | ✅ |
| `attn.in` | Attention 输入 | ✅ | ✅ |
| `attn.out` | Attention 最终输出 | ✅ | ✅ |
| `mlp.input` | MLP 输入 | ✅ | ✅ |
| `mlp.output` | MLP 输出 | ✅ | ✅ |
| `moe.router_logits` | MoE 路由 logits | ✅ | ✅ |
| `moe.output` | MoE 输出 | ✅ | ✅ |
| `final_hs` | 最终 hidden_states | ✅ | ✅ |
| `logits` | 输出 logits | ✅ | ✅ |

---

## 跨框架 Tag 命名差异对照

| 功能 | vLLM | SGLang | 备注 |
|------|------|--------|------|
| Layer 输入 | `hs.in` | `hs.in` | 一致 |
| Layer 输出 | `hs.out` | `hs.out` | 一致 |
| Attention 输出 | `attn.out` | `attn.out` | 一致 |
| MLP 输出 | `mlp.output` | `mlp.output` | 一致 |
| MoE 路由 | `moe.router_logits` | `moe.router_logits` | 一致 |
| MoE 输入 | `moe.hidden_states_input` | `forward_normal.moe.hidden_states_input` | 路径前缀 |
| AllReduce 前 | `moe.output` | `forward_normal.moe.final_hs.pre_allreduce` | 路径前缀 |
| AllReduce 后 | `moe.post_reduce` | `forward_normal.moe.hidden_states_out.post_all_reduce` | 路径前缀 |
| 残差 | 内联计算 | `prepare_mlp.out.residual` | SGLang 单独打点 |

### SGLang 特有路径前缀

| 前缀 | 使用场景 |
|------|---------|
| `forward_prepare.*` | 分离式 Attention prepare 阶段 |
| `forward_core.*` | 分离式 Attention core 阶段 |
| `forward_normal.*` | MoE forward_normal 路径 |
| `forward_deepep.*` | MoE forward_deepep 路径 |
| `step3p5.*` | Step3.5 模型 |

---

## 完整Tags列表（自检对照表）

### 完整 Tags 列表
对照自检表检查，
| Tag | 说明 | 模块 |
|-----|------|------|
| `input_ids` | 输入 token | 模型层面 |
| `positions` | 位置编码 | 模型层面 |
| `hs.in` | Layer 输入 | DecoderLayer |
| `hs.ln1` | input_layernorm 后 | DecoderLayer |
| `hs.post_attn` | Attention 后 | DecoderLayer |
| `hs.post_attn_residual` | 残差相加后 | DecoderLayer |
| `hs.ln2` | post_attention_layernorm 后 | DecoderLayer |
| `hs.mlp.out` | MLP/MoE 输出后 | DecoderLayer |
| `hs.out` | Layer 最终输出 | DecoderLayer |
| `attn.in` | Attention 输入 | Attention |
| `attn.in.q` | Q 输入 attention | Attention |
| `attn.in.k` | K 输入 attention | Attention |
| `attn.in.v` | V 输入 attention | Attention |
| `qkv_proj.out` | QKV 投影输出 | Attention |
| `q_norm.out` | Q 归一化后 | Attention |
| `k_norm.out` | K 归一化后 | Attention |
| `rotary_emb.out` | RoPE 后 | Attention |
| `attn.out.pre_o` | Attention 输出前 o_proj | Attention |
| `attn.out` | Attention 最终输出 | Attention |
| `mlp.input` | MLP 输入 | MLP |
| `mlp.gate_up` | gate_up 输出 | MLP |
| `mlp.act_out` | 激活后 | MLP |
| `mlp.output` | MLP 最终输出 | MLP |
| `moe.hidden_states_input` | MoE 输入 | MoE |
| `moe.router_logits` | 路由 logits | MoE |
| `moe.shared_output` | 共享专家输出 | MoE |
| `moe.post_reduce` | AllReduce 后 | MoE |
| `moe.output` | MoE 最终输出 | MoE |
| `final_hs` | 最终 hidden_states | 模型层面 |
| `logits` | 输出logits | 模型层面 |

---

### SGLang 特有 Tags

| Tag | 说明 | 模块 | 优先级 |
|-----|------|------|---------|
| `forward_prepare.attn.in` | Prepare 输入 | Attention | ⭐⭐⭐ |
| `forward_prepare.qkv_proj.out` | Prepare QKV 输出 | Attention | ⭐⭐ |
| `forward_prepare.rotary_emb.out.q` | Prepare RoPE 后 Q | Attention | ⭐⭐ |
| `forward_prepare.rotary_emb.out.k` | Prepare RoPE 后 K | Attention | ⭐⭐ |
| `forward_core.attn.in.q` | Core Q | Attention | ⭐⭐ |
| `forward_core.attn.in.k` | Core K | Attention | ⭐⭐ |
| `forward_core.attn.in.v` | Core V | Attention | ⭐⭐ |
| `forward_core.attn.out.pre_o` | Core Attention 输出前 | Attention | ⭐ |
| `forward_core.attn.out` | Core 输出 | Attention | ⭐⭐⭐ |
| `hs.prepare_attn.out` | prepare_attn 后 | DecoderLayer | ⭐⭐ |
| `hs.post.attn` | Attention 后 | DecoderLayer | ⭐⭐ |
| `prepare_mlp.out.hs` | prepare_mlp 后 | DecoderLayer | ⭐⭐ |
| `prepare_mlp.out.residual` | 残差 | DecoderLayer | ⭐⭐⭐ |
| `forward_normal.moe.hidden_states_input` | forward_normal MoE 输入 | MoE | ⭐⭐ |
| `forward_normal.moe.router_logits` | forward_normal 路由 | MoE | ⭐⭐ |
| `forward_normal.moe.final_hs.pre_allreduce` | AllReduce 前 | MoE | ⭐ |
| `forward_normal.moe.hidden_states_out.post_all_reduce` | AllReduce 后 | MoE | ⭐ |
| `forward_normal.moe.hidden_states_out` | forward_normal 输出 | MoE | ⭐⭐ |
| `forward_deepep.moe.hidden_states_input` | forward_deepep MoE 输入 | MoE | ⭐⭐ |
| `forward_deepep.moe.router_logits` | forward_deepep 路由 | MoE | ⭐⭐ |
| `forward_deepep.moe.hidden_states_out` | forward_deepep 输出 | MoE | ⭐⭐ |

---

### Step3.5 Tags

| Tag | 说明 | 优先级 |
|-----|------|---------|
| `step3p5.input_ids` | 输入 token | ⭐⭐⭐ |
| `step3p5.positions` | 位置编码 | ⭐⭐ |
| `step3p5.final_hs` | 最终 hidden_states | ⭐⭐ |
| `step3p5.hs.in` | Layer 输入 | ⭐⭐⭐ |
| `step3p5.hs.ln1` | input_layernorm 后 | ⭐⭐ |
| `step3p5.hs.post_attn` | Attention 后 | ⭐⭐ |
| `step3p5.hs.pre_mlp` | MLP 前 | ⭐⭐ |
| `step3p5.hs.share_expert.out` | 共享专家输出 | ⭐⭐ |
| `step3p5.hs.moe.out` | MoE 输出 | ⭐⭐ |
| `step3p5.hs.post_reduce` | 归约后 | ⭐⭐ |
| `step3p5.hs.out` | Layer 输出 | ⭐⭐⭐ |
| `step3p5.moe.input` | MoE 输入 | ⭐⭐ |
| `step3p5.moe.router_logits` | 路由 logits | ⭐⭐⭐ |
| `step3p5.moe.output` | MoE 输出 | ⭐⭐⭐ |

---

## 环境变量过滤

```bash
# 必须环境变量
export TORCHDYNAMO_DISABLE=1     # vLLM 必须
export TENSOR_DUMP_ENABLE=1      # 启用 tensor dump

# 可选过滤
export TENSOR_DUMP_DEVICE=npu:1 # 设备过滤 (npu:1, cuda:0, npu:*, *)
export TENSOR_DUMP_LAYERS=0,1,2  # 层过滤
export TENSOR_DUMP_TAGS="hs.in,hs.out,attn.out"  # Tag 过滤

# 只打关键节点
export TENSOR_DUMP_TAGS="hs.in,hs.out,attn.out,mlp.output,logits"

# 只打前 3 层
export TENSOR_DUMP_LAYERS="0,1,2"
```

### 设备过滤说明

| 环境变量值 | 效果 |
|-----------|------|
| `npu:1` | 只打点 npu:1 |
| `cuda:0` | 只打点 cuda:0 |
| `npu:*` | 打点所有 NPU |
| `cuda:*` | 打点所有 GPU |
| `*` | 不限制设备 |

---

## 添加自定义 Tags

1. 遵循层级点分隔格式 `<prefix>.<component>.<detail>`
2. 使用小写加下划线
3. 使用组件名作为前缀 (如 `moe.`, `mlp.`, `attn.`)
4. 在本文档中记录 Tag
5. 确保在 GPU 和 NPU 运行中保持一致

---

## layer_id 获取指南

### vLLM

| 类 | layer_id 属性 |
|---|---------------|
| Attention | `extract_layer_index(prefix)` |
| DecoderLayer | `extract_layer_index(prefix)` |
| ForCausalLM | `-1` (模型级别) |

### SGLang

| 类 | layer_id 属性 |
|---|---------------|
| Attention | `self.layer_id` |
| DecoderLayer | `self.layer_id` |
| ForCausalLM | `-1` (模型级别) |
| MLP | `-1` (无 layer_id) |

> 注意：某些类接收 `layer_id` 参数但未保存，需要在 `__init__` 中添加 `self.layer_id = layer_id`
