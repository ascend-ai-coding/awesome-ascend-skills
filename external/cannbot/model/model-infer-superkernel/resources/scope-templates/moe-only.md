# MoE-Only Scope 模板

本模板提供仅将 MoE 模块纳入 SuperKernel scope 的实现方案。

---

## 适用场景

- MoE 架构模型（如 DeepSeek-V3、Qwen-MoE）
- MoE 是性能瓶颈
- 专家计算占用大量时间
- 需要针对性优化 MoE 部分

---

## 优点

- **针对性强**：专门优化 MoE 瓶颈
- **效果明显**：MoE 计算密集，融合效果好
- **风险可控**：不影响 Attention 等其他模块

---

## 实现方案

```python
# cann-recipes-infer/models/{model_name}/models/modeling_*.py

from executor.utils import superkernel_scope

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enable_superkernel = config.enable_superkernel
        self.num_experts = config.num_experts
        self.gate = nn.Linear(config.hidden_size, config.num_experts)
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config.num_experts)
        ])

    def forward(self, hidden_states, is_prefill=False):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 路由计算（在 Scope 外）
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)

        # SuperKernel scope 仅包含专家计算
        with superkernel_scope(
            self.enable_superkernel and not is_prefill,
            label="moe_experts",
            option="stream-fusion=1"
        ):
            # 专家计算
            final_hidden_states = torch.zeros_like(hidden_states)

            for expert_idx in range(self.num_experts):
                # 选择当前专家的 token
                expert_mask = routing_weights[:, :, expert_idx] > threshold
                expert_input = hidden_states[expert_mask]

                # 专家计算
                expert_output = self.experts[expert_idx](expert_input)

                # 加权累加
                final_hidden_states[expert_mask] += (
                    expert_output * routing_weights[expert_mask, expert_idx].unsqueeze(-1)
                )

        return final_hidden_states


class DecoderLayer(nn.Module):
    def forward(self, hidden_states, is_prefill=False, **kwargs):
        # Attention（在 Scope 外）
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, _ = self.self_attn(hidden_states, is_prefill=is_prefill)
        hidden_states = residual + attn_output

        # MoE（在 SuperKernel Scope 内）
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_output = self.moe(hidden_states, is_prefill=is_prefill)
        hidden_states = residual + moe_output

        return hidden_states
```

---

## 配置文件

```yaml
exe_mode: "ge_graph"
model_config:
  enable_superkernel: True
  # MoE 相关配置
  num_experts: 64
  num_experts_per_tok: 8
  moe_chunk_max_len: 1024  # 可选：MoE chunk 优化
```

---

## 预期性能提升

| 模型类型 | 预期提升 | 说明 |
|---------|---------|------|
| DeepSeek-V3 | 20-30% | MoE 占比大 |
| Qwen-MoE | 15-25% | 专家计算密集 |
| 其他 MoE | 10-20% | 取决于 MoE 占比 |

---

## 参考资源

- Attention-Only 模板：`attention-only.md`
- Full-Model 模板：`full-model.md`
