# MoE 并行实施指南

> EP 改造同时包含并行基础设施（通信组、权重加载）和 MoE 算子链替换（routing → GMM → finalize）。本文档覆盖 EP 实施的完整内容。

---

## MoE TP 模式（`moe_tp_size > 1`）

适用于小规模部署（如 16 卡纯 TP）。每个专家的 FFN 做 TP 切分。

```python
expert_output = torch_npu.npu_grouped_matmul(
    [hidden_states], [self.expert_weights],
    group_list=expert_tokens, group_type=0
)
dist.all_reduce(expert_output, op=dist.ReduceOp.SUM, group=self.moe_tp_group)
```

通信组：`moe_tp_group`
参考：`cann-recipes-infer/models/qwen3_moe/models/modeling_*.py`

---

## MoE EP 模式（`moe_tp_size = 1`）

适用于大规模部署。专家分布到各卡，通过 AllToAll 交换 token。

### EP Prefill（double-routing 模式）

```python
# 来源：models/deepseek-v3.2-exp/models/modeling_deepseek.py

# ===== Step 1: 门控 + 路由初始化 =====
topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k(
    logits, k=self.top_k, bias=self.gate.e_score_correction_bias.float(), ...)
topk_idx = topk_idx.to(torch.int32)

expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = \
    torch_npu.npu_moe_init_routing_v2(
        hidden_states.view(-1, h), expert_idx=topk_idx,
        active_num=topk_idx.shape[0] * topk_idx.shape[1],
        expert_num=num_experts,
        expert_tokens_num_type=1,  # 1=count 模式
        expert_tokens_num_flag=True,
        active_expert_range=[0, num_experts],
        quant_mode=-1  # BF16: -1, W8A8: 1
    )

# ===== Step 2: AllToAll dispatch =====
expert_input = dist.all_to_all_single(..., group=self.moe_ep_group)

# ===== Step 3: EP 重路由（按本地专家重排）=====
hidden_states, gathered_scale, gathered_ids_unsort, tokens_per_local_expert = \
    torch_npu.npu_moe_re_routing(
        gathered_tokens,
        tokens_per_expert_group.view(self.moe_ep_size, -1),
        per_token_scales=gathered_pertoken_scale
    )

# ===== Step 4: 专家计算（GMM）=====
expert_output = experts(hidden_states, tokens_per_local_expert, ...)

# ===== Step 5: 恢复顺序 + AllToAll combine =====
new_x = torch.index_select(expert_output, 0, gathered_ids_unsort.float().argsort().int())
output = dist.all_to_all_single(..., group=self.moe_ep_group)

# ===== Step 6: 最终聚合 =====
hidden_states = torch_npu.npu_moe_finalize_routing(
    output, skip1=shared_expert_output,
    scales=topk_weight.to(output.dtype),
    expanded_src_to_dst_row=expanded_row_idx,
    drop_pad_mode=2
)
```

### EP Decode（dispatch/combine 模式）

> **硬件约束**：`npu_moe_distribute_dispatch_v2` 每卡最多支持 24 个 expert。当 `experts_per_rank > 24` 时（如 256/EP8=32），需要改用 Prefill 的 double-routing 路径。

```python
# 来源：models/deepseek-v3.2-exp/models/modeling_deepseek.py
# 适用条件：experts_per_rank <= 24

# ===== Step 1: MC2 分发（融合 AllToAll + token 分组）=====
output = torch_npu.npu_moe_distribute_dispatch_v2(
    x=hidden_states.view(-1, h),
    expert_ids=topk_ids,
    **dispatch_kwargs  # 包含 group、moePara 等配置
)
expand_x, dynamic_scale, expand_idx, expert_token_num, ep_recv, tp_recv = output[:6]

# ===== Step 2: 专家计算（GMM）=====
expert_output = experts(expand_x, expert_token_num, ...)

# ===== Step 3: MC2 聚合（融合 AllToAll + 加权聚合）=====
hidden_states = torch_npu.npu_moe_distribute_combine_v2(
    expand_x=expert_output,
    shared_expert_x=shared_expert_output,
    expert_ids=topk_ids,
    assist_info_for_combine=expand_idx,
    expert_scales=topk_weight.float(),
    ep_send_counts=ep_recv,
    tp_send_counts=tp_recv,
    **combine_kwargs
)
```

### EP Decode 回退（experts_per_rank > 24）

当 `experts_per_rank > 24` 无法使用 dispatch_v2 时，Decode 也使用 double-routing 路径（同 Prefill），用 `all_to_all_single` 替代 MC2 融合通信。性能略低但无 expert 数量限制。

通信组：`moe_ep_group`（需要 HCCL group name，因为 NPU dispatch 算子要求）
参考：`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`

---

## 关键算子速查

| 算子 | 功能 | 阶段 | 约束 |
|------|------|------|------|
| `npu_moe_init_routing_v2` | 路由：TopK 选专家 + 权重计算 | Prefill & Decode | |
| `npu_moe_re_routing` | EP 重路由：按 ep_size 重新分配 | Prefill EP | |
| `npu_moe_distribute_dispatch_v2` | MC2 Dispatch：融合 AllToAll + 分组 | Decode EP | 每卡 ≤24 experts |
| `npu_moe_distribute_combine_v2` | MC2 Combine：融合 AllToAll + 聚合 | Decode EP | 每卡 ≤24 experts |
| `npu_moe_finalize_routing` | 最终聚合：含 shared expert skip | Prefill | |
| `npu_grouped_matmul` | 批量专家计算（GMM） | 全部 | |
| `npu_moe_gating_top_k` | 门控：sigmoid/noaux 打分 | DeepSeek 系列 | |
| `npu_moe_gating_top_k_softmax` | 门控：softmax 打分 | Qwen3-MoE 等 | |

---

## Shared Expert 处理

部分 MoE 模型有 Shared Expert（如 DeepSeek），其计算独立于 EP routing，但需要与 MoE 输出做残差加法：

**TP 模式**：通过 `finalize_routing` 的 `skip1` 参数融合
```python
hidden_states = torch_npu.npu_moe_finalize_routing(
    expert_output, skip1=shared_expert_output, ...)
```

**EP+TP 模式（MC2）**：通过 `combine_v2` 的 `shared_expert_x` 参数融合
```python
hidden_states = torch_npu.npu_moe_distribute_combine_v2(
    expand_x=expert_output, shared_expert_x=shared_expert_output, ...)
```

参考：`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py` MoE block 实现

---

## EP 负载均衡（EPLB）

```yaml
model_config:
  perfect_eplb: True
```

开启后框架会重新分配 expert 到各 rank，确保负载均衡。需配合对应 routing 算子参数。

---

## 通信组注意事项

通信组获取方式因仓库版本不同，详见 SKILL.md 第二步。以下注意事项按版本标注。

**通用**：
- `moe_ep_group` 创建时需返回 HCCL group name（`return_name=True`），因为 NPU dispatch 算子要求
- EP 模式下 Prefill 和 Decode **必须使用不同的 routing 路径**（double-routing vs dispatch/combine），需要 `is_prefill` 分支
- AllToAll 的 input/output splits 取决于各卡的 token 分布，是动态的
- Identity expert（无 FFN 权重）的 routing weight 可能是 FP32（来自 router），与 hidden_states 运算前需 cast 到 BF16

**重构版 CommManager 特有**：
- CommManager 对 `moe_ep_group` 使用**延迟初始化**（`_deferred_ep`）：模型 `__init__` 时获取的 group/group_name 是 None 占位符，权重加载后才调用 `initialize_deferred_groups()` 创建真实 HCCL 组。模型代码中需要运行时实时获取 group，不能缓存 `__init__` 时的值

**当前仓库 hccl_comm_dict 模式**：
- `moe_ep_group` 通过 `hccl_comm_dict["moe_ep_group"]` 获取，在模型实例化时已创建完毕，无延迟初始化问题
- group name 通过 `hccl_comm_dict.get("moe_ep_group_name")` 获取

---

## 量化模式下的 EP MoE

不同量化模式影响 routing 和 GMM 的参数选择：

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
