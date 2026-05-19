# MoE 融合算子参考

> 本文档覆盖 MoE 模块的融合算子替换（TP 模式）。EP 模式的完整实施（含通信、权重加载、算子链）见 parallel-implementation skill 的 `references/moe-parallel.md`。

---

## TP 模式 MoE 全流程

### 适用场景

- Prefill 阶段，纯张量并行
- 所有专家在每张卡上都有副本（或按 TP 切分）

### 完整代码流程

```python
# 来源：models/deepseek-v3.2-exp/models/modeling_deepseek.py

# ===== Step 1: 门控 =====
logits = F.linear(hidden_states.view(-1, h), self.gate.weight)
topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k(
    logits, k=self.top_k,
    bias=self.gate.e_score_correction_bias.float(),
    k_group=self.topk_group, group_count=self.n_group,
    group_select_mode=1, renorm=0, norm_type=1,
    routed_scaling_factor=self.routed_scaling_factor,
    eps=float(1e-20)
)
topk_idx = topk_idx.to(torch.int32)

# ===== Step 2: 路由初始化 =====
expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = \
    torch_npu.npu_moe_init_routing_v2(
        hidden_states.view(-1, h),
        expert_idx=topk_idx,
        active_num=topk_idx.shape[0] * topk_idx.shape[1],
        expert_num=num_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, num_experts],
        quant_mode=-1  # BF16: -1, W8A8: 1
    )

# ===== Step 3: 专家计算（GMM）=====
expert_output = experts(expanded_x, tokens_per_expert, pertoken_scale=pertoken_scale)

# ===== Step 4: 路由结果聚合 =====
hidden_states = torch_npu.npu_moe_finalize_routing(
    expert_output,
    skip1=shared_expert_output,
    skip2=None, bias=None,
    scales=topk_weight.to(expert_output.dtype),
    expanded_src_to_dst_row=expanded_row_idx,
    export_for_source_row=None,
    drop_pad_mode=2
)
```

### 关键参数说明

**`expert_tokens_num_type`**：
- `0`：cumsum 模式 — tokens_per_expert 是累积和
- `1`：count 模式 — tokens_per_expert 是每个专家的 token 数量

**`quant_mode`**：
- `-1`：无量化 → expanded_x 保持 BF16
- `1`：动态量化 → expanded_x 为 INT8，返回 pertoken_scale

**`drop_pad_mode`**：
- `2`：标准模式，按 expanded_src_to_dst_row 映射恢复

---

## MoE 门控策略

### `npu_moe_gating_top_k`（DeepSeek 系列）

```python
topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k(
    logits, k=top_k,
    bias=e_score_correction_bias.float(),
    k_group=topk_group, group_count=n_group,
    group_select_mode=1, renorm=0, norm_type=1,
    routed_scaling_factor=routed_scaling_factor,
    eps=1e-20
)
```

### `npu_moe_gating_top_k_softmax`（Qwen3-MoE 等）

带 softmax 归一化的 Top-K 变体，适用于使用 softmax 打分的模型。

### 标准 softmax Top-K（回退方案）

当融合算子不适用时，使用 PyTorch 原生实现：
```python
scores = logits.softmax(dim=-1, dtype=torch.float32)
topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
```

---

## Shared Expert 处理

共享专家（shared expert）独立于路由专家，所有 token 都经过共享专家：

```python
def forward_shared_expert(self, hidden_states, is_prefill):
    if self.n_shared_experts > 0:
        hidden_states_share = self.shared_experts(hidden_states.view(-1, hidden_states.shape[-1]))
    else:
        hidden_states_share = None
    return hidden_states_share
```

聚合方式见 parallel-implementation skill 的 `references/moe-parallel.md` Shared Expert 处理章节。

---

## 量化模式变体

### 不同量化模式的 MoE FFN 算子链

**BF16**：
```
init_routing(quant_mode=-1) → grouped_matmul(BF16→BF16) → npu_swiglu → grouped_matmul(BF16→BF16)
```

**W8A8**：
```
init_routing(quant_mode=1) → grouped_matmul(INT8→INT32) → npu_dequant_swiglu_quant → grouped_matmul(INT8→BF16)
```

**W8A8C8**：
```
init_routing(quant_mode=1) → grouped_matmul(INT8→BF16, with scale) → npu_swiglu_clip_quant → grouped_matmul(INT8→BF16, with scale)
```

---

## 多流并行中的 MoE

MoE 计算和共享专家计算可以并行执行：

```python
use_aclgraph_event = self.enable_multi_streams and self.enable_aclgraph
if use_aclgraph_event:
    tng.ops.npu_tagged_event_wait(moe_npu_events[1])
```

共享专家与路由专家在不同 stream 上并行计算，通过 event 同步。
