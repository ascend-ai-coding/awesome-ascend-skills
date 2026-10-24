# MoE 并行与融合算子实施指南

> MoE 改造同时包含并行基础设施（通信组、权重加载）和 MoE 算子链替换（gate → routing → GMM → finalize/combine）。本文档作为 MoE 并行与融合算子链路的单一参考。

## MoE 门控策略

| Gate 形式 | 推荐算子 | 代表模型 |
|-----------|----------|----------|
| sigmoid / noaux 打分 | `npu_moe_gating_top_k` | DeepSeek 系列 |
| softmax 打分 | `npu_moe_gating_top_k_softmax` | Qwen3-MoE 等 |
| 算子不适配 | PyTorch `softmax + topk` 回退 | 通用 |

门控输出的 `topk_idx` 需转为 `int32`，供后续 routing / dispatch 算子使用。

---

## MoE EP 模式（`moe_tp_size = 1`）

全 EP 是 MoE 大规模部署最常见的主路径。每张卡只保存一部分 expert，`moe_tp_size=1`，`moe_ep_size=world_size`，通过 AllToAll / MC2 dispatch-combine 把 token 送到 expert 所在 rank。

### EP Prefill（double-routing 模式）

```python
# 来源：cann-recipes-infer/models/deepseek-v3.2-exp/models/modeling_deepseek.py

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

# ===== Step 2: AllToAll dispatch（token count + token 数据）=====
tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=self.moe_ep_group)

combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
combine_tokens = combine_tokens.view(2, self.moe_ep_size, -1).sum(2)
all_tokens = combine_tokens[0].sum()
combine_tokens_cpu = combine_tokens.cpu().tolist()

input_splits = combine_tokens_cpu[1]
output_splits = combine_tokens_cpu[0]

gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
dist.all_to_all_single(
    gathered_tokens,
    expanded_x,
    output_splits,
    input_splits,
    group=self.moe_ep_group,
)

gathered_pertoken_scale = None
if pertoken_scale is not None:
    gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
    dist.all_to_all_single(
        gathered_pertoken_scale,
        pertoken_scale,
        output_splits,
        input_splits,
        group=self.moe_ep_group,
    )

# ===== Step 3: EP 重路由（按本地专家重排）=====
hidden_states, gathered_scale, gathered_ids_unsort, tokens_per_local_expert = \
    torch_npu.npu_moe_re_routing(
        gathered_tokens,
        tokens_per_expert_group.view(self.moe_ep_size, -1),
        per_token_scales=gathered_pertoken_scale,
    )

# ===== Step 4: 专家计算（GMM）=====
expert_output = experts(hidden_states, tokens_per_local_expert, ...)

# ===== Step 5: 恢复顺序 + AllToAll combine =====
new_x = torch.index_select(expert_output, 0, gathered_ids_unsort.float().argsort().int())
gathered_tokens = new_x.new_empty(*expanded_x.shape)
dist.all_to_all_single(
    gathered_tokens,
    new_x,
    input_splits,
    output_splits,
    group=self.moe_ep_group,
)

# ===== Step 6: 最终聚合 =====
hidden_states = torch_npu.npu_moe_finalize_routing(
    gathered_tokens, skip1=shared_expert_output,
    scales=topk_weight.to(gathered_tokens.dtype),
    expanded_src_to_dst_row=expanded_row_idx,
    drop_pad_mode=2
)
```

### EP Decode（dispatch/combine 模式）

> **硬件约束**：A2 常规 MC2 路径下，`npu_moe_distribute_dispatch_v2` 每 rank 最多支持 24 个 MoE expert，即 `moe_expert_num / (ep_world_size - shared_expert_rank_num) <= 24`。当 `experts_per_rank > 24` 时（如 256/EP8=32）或 MC2 不适配时，需要进入 Decode 回退方案选择。

```python
# 来源：cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py::set_mc2_kwargs / forward
# 适用条件：experts_per_rank <= 24

# 一次性构造 dispatch_kwargs / combine_kwargs（在 set_mc2_kwargs 内）
moe_ep_group_name = comm_manager.get_group_name("moe_ep_group_mc2")  # MC2 fullmesh_v2 独立物理组
global_rank = dist.get_rank()
dispatch_kwargs = {
    "group_ep": moe_ep_group_name,
    "ep_world_size": self.moe_ep_size,
    "ep_rank_id": global_rank // self.moe_tp_size,
    "group_tp": moe_ep_group_name,            # moe_tp=1 时复用 ep group_name
    "tp_world_size": self.moe_tp_size,
    "tp_rank_id": global_rank % self.moe_tp_size,
    "moe_expert_num": self.n_routed_experts,
    "shared_expert_rank_num": self.shared_expert_rank_num,
    "global_bs": 0,
    "expert_shard_type": 0,
    "x_active_mask": None,
    "scales": ...,         # 量化 scale，BF16 时为 None
    "quant_mode": 2 if "a8" in gmm_quant_mode else 0,
}
combine_kwargs = {**dispatch_kwargs}
combine_kwargs.pop("scales", None)
combine_kwargs.pop("quant_mode", None)

# ===== Step 1: MC2 分发（融合 AllToAll + token 分组）=====
output = torch_npu.npu_moe_distribute_dispatch_v2(
    x=hidden_states.view(-1, h),
    expert_ids=topk_ids,
    **dispatch_kwargs,
)
expand_x, dynamic_scale, expand_idx, expert_token_num, ep_recv, tp_recv = output[:6]

# ===== Step 2: 专家计算（GMM）=====
expert_output = experts(expand_x, expert_token_num, group_list_type=1)

# ===== Step 3: MC2 聚合（融合 AllToAll + 加权聚合）=====
hidden_states = torch_npu.npu_moe_distribute_combine_v2(
    expand_x=expert_output,
    shared_expert_x=shared_expert_output,
    expert_ids=topk_ids,
    assist_info_for_combine=expand_idx,
    expert_scales=topk_weight.float(),
    ep_send_counts=ep_recv,
    tp_send_counts=tp_recv,
    **combine_kwargs,
)
```

> 本文只覆盖仓库已验证的全 EP Decode 主路径（按 `moe_tp_size=1` 使用，`group_tp` 复用 `moe_ep_group_name`）；`moe_tp_size>1` 时参考仓内 TP 路径，不要从接口字段推导未验证的组合路径。

### EP Decode 回退方案选择（experts_per_rank > 24 或 MC2 不适配）

当 `experts_per_rank > 24` 无法使用 dispatch_v2，或 MC2 因通信域、图模式捕获等原因不适配时，不能默认只有一种回退路径，需按目标场景选择：

- **double-routing**：沿用 Prefill 路径，用 `all_to_all_single` 替代 MC2 融合通信，语义直接、无每 rank expert 数限制，但 Decode 图模式捕获和性能可能受 AllToAll 影响。
- **local-expert Decode**：Decode 阶段只计算本 rank experts，非本地 expert 权重置零，最后通过固定 shape 的 AllReduce 聚合；这是 MC2 / double-routing 不适配 fullgraph 时的妥协路径。Prefill 通常仍保持 double-routing，图模式断点处理思路见 `model-infer-graph-mode/references/llm-model-guide.md`。

通信组：MC2 Decode 主路径用 `moe_ep_group_mc2`（独立 fullmesh_v2 物理组，需 HCCL group name）；回退到 double-routing / local-expert 时复用 `moe_ep_group`（AllToAll 通用组）。
参考：`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py` 中搜索 `init_parallel_comm_group` 和 `set_mc2_kwargs`

---

## MoE TP 模式（`moe_tp_size > 1`）

适用于小规模部署（如 16 卡纯 TP）或策略要求 expert FFN 做 TP 切分的场景。所有专家在每张卡上都有副本或按 TP 切分，专家计算后通过 `moe_tp_group` 聚合。

```python
# ===== Step 1: 门控 =====
topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k(...)
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
        quant_mode=-1  # BF16: -1, A8/W8A8: 1
    )

# ===== Step 3: 专家计算（GMM）=====
expert_output = experts(expanded_x, tokens_per_expert, group_list_type=1)

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

# ===== Step 5: TP 聚合 =====
dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, group=self.moe_tp_group)
```

通信组：`moe_tp_group`
参考：`cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py`、`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`

---

## 关键算子与参数速查

| 算子 | 功能 | 阶段 | 约束 |
|------|------|------|------|
| `npu_moe_init_routing_v2` | 按 `topk_idx` 展开/重排 token，并统计 expert token count | Prefill & Decode | |
| `npu_moe_re_routing` | EP 重路由：按 ep_size 重新分配 | Prefill EP | |
| `npu_moe_distribute_dispatch_v2` | MC2 Dispatch：融合 AllToAll + 分组 | Decode EP | 每卡 ≤24 experts |
| `npu_moe_distribute_combine_v2` | MC2 Combine：融合 AllToAll + 聚合 | Decode EP | 每卡 ≤24 experts |
| `npu_moe_finalize_routing` | 最终聚合：含 shared expert skip | Prefill | |
| `npu_grouped_matmul` | 批量专家计算（GMM） | 全部 | |
| `npu_moe_gating_top_k` | 门控：sigmoid/noaux 打分 | DeepSeek 系列 | |
| `npu_moe_gating_top_k_softmax` | 门控：softmax 打分 | Qwen3-MoE 等 | |

**常用参数**：
- `expert_tokens_num_type=1`：count 模式，输出每个 expert 的 token 数量，通常与 `group_list_type=1` 配套。
- `expert_tokens_num_type=2`：key-value 模式，输出 `[[expert_id, token_count], ...]`，仅记录非 0-token expert，通常与 INT8 GMM 的 `group_list_type=2` 配套。
- `group_list_type=1`：`group_list` 表示每个 group 的 token count，BF16 和常规 GMM 路径常用。
- `group_list_type=2`：`group_list` 表示 `[[group_id, group_size], ...]`；当前 `npu_grouped_matmul` 文档中受 INT8 + `group_type=0` 等条件限制，不是 BF16 通用配置。
- `quant_mode=-1`：非量化，`expanded_x` 保持原 dtype。
- `quant_mode=1`：动态量化，`expanded_x` 为 INT8，并返回 per-token scale。
- `drop_pad_mode=2`：`npu_moe_finalize_routing` 常用标准聚合模式，按 `expanded_src_to_dst_row` 恢复。

---

## Shared Expert 处理

部分 MoE 模型有 Shared Expert（如 DeepSeek），其计算独立于路由专家，通常所有 token 都经过 shared expert，最后与 MoE 输出融合。

**TP 模式**：通过 `finalize_routing` 的 `skip1` 参数融合
```python
hidden_states = torch_npu.npu_moe_finalize_routing(
    expert_output, skip1=shared_expert_output, ...)
```

**EP Decode MC2 模式**：通过 `combine_v2` 的 `shared_expert_x` 参数融合
```python
hidden_states = torch_npu.npu_moe_distribute_combine_v2(
    expand_x=expert_output, shared_expert_x=shared_expert_output, ...)
```

参考：`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py` MoE block 实现

---

## 专家权重加载

MoE 并行改造必须同步检查 `FusedMoEGMM` 的权重加载：

- `make_expert_params_mapping(...)` 负责把 checkpoint 中每个 expert 的 `gate/up/down` 权重映射到 `FusedMoEGMM.weight_loader`。
- `FusedMoEGMM` 内部按 `tp_size` 切 `intermediate_size_per_partition = intermediate_size // tp_size`。
- EP 场景下需要按 `ep_rank` 保留本 rank 负责的 expert 范围，并将全局 `expert_id` 映射为本地 expert id。
- 改 `moe_tp_size / moe_ep_size` 后，必须确认 `FusedMoEGMM(tp_size, ep_size, tp_rank, ep_rank)`、通信组和权重加载逻辑一致。

---

## EP 负载均衡（EPLB）

```yaml
model_config:
  perfect_eplb: True
```

开启后框架会重新分配 expert 到各 rank，确保负载均衡。需配合对应 routing 算子参数。

---

## 通信组注意事项

**通用**：
- `moe_ep_group` 和 `moe_ep_group_mc2` 创建时需返回 HCCL group name（`return_name=True`），因为 NPU dispatch 算子要求
- EP 模式下 Prefill 和 Decode **必须使用不同的 routing 路径**（double-routing vs dispatch/combine），需要 `is_prefill` 分支
- AllToAll 的 input/output splits 取决于各卡的 token 分布，是动态的
- Identity expert（无 FFN 权重）的 routing weight 可能是 FP32（来自 router），与 hidden_states 运算前需 cast 到 BF16

**框架部署：模型自管 + CommManager.register_group 范式**

CommManager 不再启动时一次性建好所有组，而是提供 `register_group()` API；模型在 `__init__` 调用 `init_parallel_comm_group()` 按需注册自己用到的所有通信组。框架按 (`group_num`, `group_size`, `group_stride`) 等形状自动判断 HCCL 物理通信组是否可以复用；需要独立物理组（如 MC2 fullmesh_v2、multi-stream 副本组）时显式 `allow_physical_reuse=False`。

`init_parallel_comm_group` 模板（按本模型实际 `parallel_config` 裁剪；`group_stride` / `allow_physical_reuse` / `hccl_buffer_size` 因组而异，需理解下方注释后再写，不要直接照抄）：

```python
# Model.__init__ 末尾调用：self.init_parallel_comm_group()

def init_parallel_comm_group(self):
    # 1) 基础 TP 组：world_size / *_size 标准模式，按本模型启用的切分注册
    self.comm_manager.register_group(
        name="attn_tp_group",
        group_num=self.world_size // self.attn_tp_size,
        group_size=self.attn_tp_size,
    )
    # embed_tp_group / lmhead_tp_group / dense_tp_group / moe_tp_group 按同一模式
    # （仅在对应 *_size > 1 时注册）

    # 2) MoE EP 组：注意需传 group_stride（rank 步长），EP 排布跨 rank 而非相邻
    if self.moe_ep_size > 1:
        moe_ep_group_num = self.world_size // self.moe_ep_size
        self.comm_manager.register_group(
            name="moe_ep_group",
            group_num=moe_ep_group_num,
            group_size=self.moe_ep_size,
            group_stride=moe_ep_group_num,
            return_name=True,
        )

    # 3) MC2 独立物理组：必须显式 allow_physical_reuse=False，避免与 moe_ep_group
    #    共享 HCCL 物理组导致 fullmesh_v2 算法语义错乱；buffer 用 calc_moe_hccl_buffer_size 算
    if self.moe_ep_size > 1 and self.moe_tp_size == 1:
        self.comm_manager.register_group(
            name="moe_ep_group_mc2",
            group_num=self.world_size // self.moe_ep_size,
            group_size=self.moe_ep_size,
            group_stride=self.world_size // self.moe_ep_size,
            allow_physical_reuse=False,
            hccl_buffer_size=calc_moe_hccl_buffer_size(self.infer_config, self.config, is_full_mesh_v2=True),
            return_name=True,
        )
```

`register_group` 关键参数：

| 参数 | 含义 |
|------|------|
| `name` | 通信组名，后续 `get_group(name)` / `get_group_name(name)` / `get_rank(name)` 引用 |
| `group_num` / `group_size` | 组数量 / 每组大小，决定 HCCL 通信组拓扑 |
| `group_stride` | rank 步长，EP / 跨卡组需指定，TP 默认相邻可省 |
| `return_name` | 是否返回 HCCL group name（dispatch / combine 算子要求时设 `True`） |
| `allow_physical_reuse` | 默认 `True`，框架按形状自动复用同型 HCCL 物理组；需要独立物理组（MC2 fullmesh_v2、multi-stream 副本组等）时显式 `False` |
| `hccl_buffer_size` | MC2 类组需按 `calc_moe_hccl_buffer_size` 算入，避免默认 buffer 不够 |

**取通信组**：模型内部用 `comm_manager.get_group(name)` 取 ProcessGroup，`comm_manager.get_group_name(name)` 取 HCCL name（dispatch / combine 算子要求）。

完整范例参考 `cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py` 中搜索 `init_parallel_comm_group`；多流 micro-batch（`*_stream1` 副本组）和 `dense_tp_group` 等扩展场景参考 `cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py` 中搜索 `init_parallel_comm_group`。

**独立部署模式**：通信组由 Runner 内调 `executor.utils.hccl_utils.init_comm_group(return_name=True)` 一次性建好，对外通过 `parallel_ctx.get_group("moe_ep_group")` / `parallel_ctx.get_group_name("moe_ep_group")` 取（接口与 CommManager 对齐）。dispatch/combine 算子调用与本节示例 100% 一致，详见 `standalone_parallel.md`。

> 量化路径下 routing / GMM 的 quant_mode、group_list_type 选择见 model-infer-quantization skill；MoE 与 shared expert 的多流并行见 model-infer-multi-stream skill。本文不展开。
