# 常用融合算子详细参考

本文件是 SKILL.md §1.1 表格的展开。**写替换代码前必读对应小节**。

**统一要求：调用任何 `torch_npu` 融合 API 前，必须先查当前环境的 API 说明文档，确认签名、参数类型/语义、shape/layout 约束和返回值格式。首选查询方法：**

```bash
python3 scripts/query_torch_npu_api.py show <api_name>
python3 scripts/query_torch_npu_api.py show <api_name> --full
```

如果 `query_torch_npu_api.py` 因当前环境缺少 `torch_npu`、缺少 `_op_plugin_docs.py` 条目或查询不到 API 而失败，再查看给出的文档链接。不要凭记忆写 API 调用；同名 API 在不同 `torch_npu` 版本可能有签名或约束差异。

## 目录

1. [RMSNorm → npu_rms_norm](#1-rmsnorm--npu_rms_norm)
2. [SwiGLU → npu_swiglu](#2-swiglu--npu_swiglu)
3. [Rotary Position Embedding → npu_rotary_mul](#3-rotary-position-embedding--npu_rotary_mul)
4. [Attention → npu_fusion_attention](#4-attention--npu_fusion_attention)
5. [Grouped Matmul（MoE）](#5-grouped-matmulmoe)

---

## 1. RMSNorm → npu_rms_norm

最简单、收益稳定的替换。直接替换手写 RMSNorm 的 forward。

文档：https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/zh/custom_APIs/torch_npu/（beta）torch_npu-npu_rms_norm.md

### 识别 pattern

手写 RMSNorm 通常长这样（各种变体都算）：

```python
def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight
```

或展开形式：

```python
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
output = (self.weight * hidden_states).to(input_dtype)
```

### 替换后

```python
def forward(self, x):
    if _HAS_TORCH_NPU and str(x.device).startswith("npu"):
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
    output = self._norm(x.float()).type_as(x)
    return output * self.weight
```

### 注意事项

- **返回值是 tuple，取 `[0]`** 才是结果张量。直接用返回值会拿到 tuple 而非 tensor。
- 在混合精度训练中，`hidden_states` 可能以 FP32 格式传递，而模型权重 `weight` 仍保持 BF16。虽然 NPU 的 RMSNorm 算子 `aclnnRmsNorm` 支持 BF16/FP32 类型的 weight，但为了确保计算精度，建议在调用前通过 `.float()` 将 weight 显式提升为 FP32，使整个 RMSNorm 计算在更高精度下完成，例如：

```python
output = torch_npu.npu_rms_norm(hidden_states, self.weight.float(), epsilon=self.variance_epsilon)[0]
```

---

## 2. SwiGLU → npu_swiglu

将 `SiLU + 门控乘法` 替换为单个融合算子。

文档：https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/（beta）torch_npu-npu_swiglu.md

### 识别 pattern

```python
# 原始：a1 * silu(a2)
intermediate = a1 * F.silu(a2)
```

### 替换后

```python
# NPU 优化：concat 顺序 [a2, a1] 使得 SiLU(a2) * a1
gate_up = torch.cat([a2, a1], dim=-1)
intermediate = torch_npu.npu_swiglu(gate_up, dim=-1)
```

### 注意事项

**concat 顺序必须正确。** `npu_swiglu(x, dim=-1)` 的语义是：对前半做 SiLU，乘以后半。所以：

- 原始代码 `a1 * silu(a2)` → 需 concat `[a2, a1]`，让前半是 a2（被 SiLU），后半是 a1。
- 如果原始代码是 `silu(a1) * a2` → concat `[a1, a2]`。

判断方法：**谁被 silu 包，谁放前半。** 反了语义就错了，但不会报错，但精度会出问题。

---

## 3. Rotary Position Embedding → npu_rotary_mul

替换手写的旋转位置编码计算。

函数接口：
```
torch_npu.npu_rotary_mul(x, r1, r2, rotary_mode="half", rotary_matrix=None)
```

文档：https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_rotary_mul.md

### 识别 pattern

```
def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

output = x * cos + rotate_half(x) * sin
```

### 替换后
```
# rotary_mode 默认即为 "half"
output = torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode='half')
```


### 注意事项

- `half` 模式按最后一维前、后两半旋转，最后一维必须能被 2 整除；`r1/r2` 的最后一维应与待旋转的 x 一致，并遵守 PyTorch 广播规则。
- 常规 `cos/sin` 应分别是 `r1/r2`。在 RoPE 场景中通常就是 `cos` 和 `sin`。
- 部分 RoPE 维度不能直接传完整 `x`：先切出旋转前缀，融合后再拼回未旋转尾部。
- 部分模型的 cos/sin 最后一维只有 head_dim / 2。调用前需要扩展为 head_dim，常见方式是 `torch.cat([cos, cos], dim=-1)`。
- 交错布局应使用官方文档所述的 `rotary_mode="interleave"` 和对应的 `rotary_matrix`，或先转换为 `half` 布局；不能把交错排列的张量直接按默认 `half` 模式调用

---

## 4. Attention → npu_fusion_attention（最复杂，收益最大）

替换手写的 `Q@K^T → mask → softmax → @V` 为单个融合算子。

文档：https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fusion_attention.md

### 识别 pattern

任何手写 `scale * Q @ K^T → mask → softmax → @V` 的代码，都是候选。

### 替换后

```python
# query/key/value shape: (B, S, H, D) — 即 BSND 格式
scale = 1.0 / math.sqrt(head_dim)

# 构建 causal mask（bool，True = mask out）
seq_len_q = query.shape[1]
seq_len_k = key.shape[1]
causal_mask = torch.triu(
    torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device),
    diagonal=seq_len_k - seq_len_q + 1,
)
# 合并 padding mask（如果有）
if attention_mask is not None:
    # attention_mask 是 float 加法 mask (0.0=attend, -inf=mask)，转为 bool
    pad_mask = (attention_mask.squeeze(1).squeeze(1) < -1.0)
    causal_mask = causal_mask.unsqueeze(0) | pad_mask.unsqueeze(-2)
    atten_mask = causal_mask.unsqueeze(1)  # (B, 1, S_q, S_kv)
else:
    atten_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S_q, S_kv)

npu_out = torch_npu.npu_fusion_attention(
    query, key, value,
    num_heads,
    input_layout="BSND",
    pse=None,
    atten_mask=atten_mask,
    scale=scale,
    pre_tockens=65536,
    next_tockens=0,
    keep_prob=1.0 if not training else 1.0 - dropout_p,
)[0]
context_layer = npu_out.flatten(2, 3).contiguous()  # (B, S, H*D)
```

### 注意事项（逐条核对）

- **返回值是 tuple，取 `[0]`**。
- **必须使用显式 causal mask。** 仅靠 `pre_tockens/next_tockens=0` 不足以正确实现因果注意力。实测去掉显式 mask 后 logits 余弦相似度从 0.999 暴跌至 0.5~0.8——精度验证能抓出来，但别等验证，直接加。
- **mask 语义**：`atten_mask` 中 `True` 表示"屏蔽"（不参与注意力），`False` 表示"参与"。**与 PyTorch 的 float additive mask 相反**（float mask 里 0.0=attend, -inf=mask）。从 float mask 转 bool 时注意取反：`(attention_mask < -1.0)` 得到 bool mask。
- **padding mask 合并**：Transformer 层传入的 `attention_mask` 通常是 float 格式，需转为 bool 后与 causal mask 做 **OR** 合并（`|`），不是相加。
- **input_layout="BSND"**：B=batch, S=seq_len, N=num_heads, D=head_dim。确认 query/key/value 从 `_split_heads` 出来的 shape 是 `(B, S, H, D)` 而非 `(B, H, S, D)`。layout 填错 shape 解释就错了。
- **decode 阶段（seq_len_q=1）**：`diagonal = S_kv - 1 + 1 = S_kv`，`torch.triu(..., diagonal=S_kv)` 全零，即不屏蔽任何 token，行为正确。不用特别处理。

---

## 5. Grouped Matmul（MoE）

将 MoE 中"逐 expert 取 token，再各做两次 matmul"的 eager 循环替换为单次 `npu_grouped_matmul` 调用。核心思路：先按 expert 重排 token 使其连续，再以 `tokens_per_expert` 为分组依据一次完成所有 expert 的矩阵乘。

文档：https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md

### 识别 pattern

典型 MoE eager 循环——逐 expert 取 token、做 gate_up 和 down 两次 matmul、再散射累加回输出：

```python
# x: [num_tokens, hidden_dim]
# gate_up_weight: [num_experts, hidden_dim, intermediate_dim * 2]  或 [E, N, K]（需转置）
# down_weight:    [num_experts, intermediate_dim, hidden_dim]       或 [E, N, K]（需转置）
# selected_experts: [num_tokens, top_k]
# routing_weights:  [num_tokens, top_k]

output = torch.zeros_like(x)

for expert_id in range(num_experts):
    token_mask = (selected_experts == expert_id)  # [num_tokens, top_k]
    token_ids, top_k_ids = token_mask.nonzero(as_tuple=True)

    if token_ids.numel() == 0:
        continue

    expert_x = x[token_ids]                        # [M_e, hidden_dim]

    # 第一次 matmul：gate_up
    gate_up = expert_x @ gate_up_weight[expert_id] # [M_e, 2 * intermediate_dim]
    gate, up = gate_up.chunk(2, dim=-1)
    h = activation(gate) * up                     # activation 可能是 silu 或其他

    # 第二次 matmul：down
    y = h @ down_weight[expert_id]                 # [M_e, hidden_dim]

    # 应用 routing weight 并散射累加
    y = y * routing_weights[token_ids, top_k_ids].unsqueeze(-1)
    output.index_add_(0, token_ids, y)
```

关键信号：
- 按 expert 维度循环
- `x[token_ids]` 或 `index_select` 按 expert 取 token
- 每个 expert 内两次 matmul（gate_up → activation → down）
- `index_add_` 或 `scatter_add_` 将各 expert 结果合并

### 替换后

三步走：**token_permute → grouped GEMM × 2 → token_unpermute**。

#### 完整流程（无 EP）

```python
# === 1. 按 expert 重排 token（使同一 expert 的 token 连续） ===
permuted_x, row_ids = torch_npu.npu_moe_token_permute(x, selected_experts.int())
# row_ids 记录重排映射，供 unpermute 恢复原顺序

# === 2. 统计每个 expert 的 token 数 ===
tokens_per_expert = torch.histc(
    selected_experts.float(),
    bins=num_experts,
    min=0,
    max=num_experts - 1,
).to(torch.int64)
# 确保：len(tokens_per_expert) == num_experts
#       tokens_per_expert.sum() == permuted_x.shape[0]

# === 3. 第一次 grouped matmul：gate_up ===
gate_up = torch_npu.npu_grouped_matmul(
    [permuted_x],
    [gate_up_weight],          # [E, K, 2*intermediate]
    bias=None,
    group_list=tokens_per_expert,
    split_item=2,
    group_type=0,              # 沿 M/token 轴分组
    group_list_type=1,         # group_list 是每组大小（非 cumsum）
)[0]

gate, up = gate_up.chunk(2, dim=-1)
h = activation(gate) * up     # 保持原模型 activation，不要擅自换成 swiglu

# === 4. 第二次 grouped matmul：down ===
y = torch_npu.npu_grouped_matmul(
    [h],
    [down_weight],             # [E, intermediate, hidden]
    bias=None,
    group_list=tokens_per_expert,
    split_item=2,
    group_type=0,
    group_list_type=1,
)[0]

# === 5. 恢复 token 原顺序，并在此处应用 routing weight ===
output = torch_npu.npu_moe_token_unpermute(y, row_ids, probs=routing_weights)
# unpermute 内部完成：恢复顺序 + 路由概率加权 + 多 expert 结果累加
```

#### 训练反向（`torch.autograd.Function` 包装）

仅替换 forward 不够——训练需要 input-gradient 和 weight-gradient。标准做法是包装为 `torch.autograd.Function`：

```python
class GmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group_list):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        return torch_npu.npu_grouped_matmul(
            [x], [weight],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors

        # input 梯度：grad_y @ weight^T，仍是 grouped matmul
        grad_x = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight.transpose(1, 2)],   # [E, N, K]
            bias=None,
            group_list=ctx.group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

        # weight 梯度：x^T @ grad_y，split_item=3, group_type=2
        grad_weight = torch_npu.npu_grouped_matmul(
            [x.T],
            [grad_output],
            bias=None,
            group_list=ctx.group_list,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]

        return grad_x, grad_weight, None
```

> **`split_item` 与 `group_type` 速查**：正向 `split_item=2, group_type=0`；input 梯度同正向参数；weight 梯度 `split_item=3, group_type=2`。不要混用。

#### EP 场景

Expert Parallel 下 All-to-All 已将 token 按本地 expert 连续分发，因此：

- **不要再次 token_permute**，直接用 All-to-All 后的 token 顺序
- **不要重复应用 routing weight**（已在 unpermute 中处理）
- `group_list` 必须用 **All-to-All 后本地 expert 实际收到的全局 token 数**，而非本 rank 初始的 token 数

```python
# EP 路径：All-to-All → GMM
num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)
gate_up = GmmFunction.apply(recv_x, gate_up_weight, num_global_tokens_per_local_expert)
```

### 注意事项

1. **权重布局是 `[E, K, N]`，不是 `[E, N, K]`**。HF checkpoint 通常是后者，需 `.transpose(1, 2)` 创建转置 view，不能改变 checkpoint 语义。

2. **`tokens_per_expert` 的 sum 不总是 token 数**。`top_k > 1` 时每个 token 被路由到多个 expert，`sum = num_tokens × top_k`（路由副本数），不是原始 token 数。

3. **bias 不会自动生效**。`npu_grouped_matmul` 的 `bias` 参数语义与 eager 循环中的逐 expert bias 不同。若原始代码有 expert bias，需通过 `repeat_interleave(tokens_per_expert, dim=0)` 展开后单独相加。

4. **EP 空 token 场景**：当前 rank 完全没有 token 时，执行零值 matmul 以保留所有 expert 权重的梯度连接，否则训练会静默断连。

5. **MC2 不是简单 GMM 替换**。它把通信与正反向 GMM 组合进 `npu_alltoallv_gmm` / `npu_gmm_alltoallv`，是独立路径，不要试图用普通 `npu_grouped_matmul` 替代。
