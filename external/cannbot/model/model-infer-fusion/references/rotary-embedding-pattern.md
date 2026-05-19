# RotaryEmbedding 预计算与调用模式参考

## 目的

本文档用于指导 RoPE 前置改造，收敛出更成熟的 `RotaryEmbedding` 模块化写法和调用方式。

## 适用场景

当目标算子或目标链路需要 RoPE 相关前置改造时，可优先参考本文档。例如：

- 需要将分散的 `freqs_cis` / `cos/sin` 计算收敛成统一模块
- 需要为 `npu_mla_prolog_v3`、`npu_kv_rmsnorm_rope_cache_v2`、`npu_fused_infer_attention_score_v2` 等路径准备稳定的 `cos/sin` 或 `query_rope / key_rope`
- 需要把 `Prefill / Decode` 的 RoPE 取值逻辑从各层局部切片改成模型级统一准备
- 需要为 `PA / 非PA` 路径整理更清晰的 RoPE metadata 组织方式

## 推荐模式

优先参考 `deepseek-v3.2-exp` 和 `deepseek-r1` 等成熟实现，并采用下面的共同思路：

1. 将 RoPE 的预计算、缓存和取值逻辑收敛到统一位置，不再分散到各层或多个辅助函数中；实现位置可以是独立类、公共模块或模型内部统一入口
2. 在模型初始化阶段完成第一次 `cos/sin` cache 预计算，并用 `register_buffer` 保存
3. 在模型级或统一入口按当前阶段和位置语义取出实际需要的 `cos/sin`
4. 下游 Attention / MLA 层只消费已经准备好的 `cos/sin` 或 `cos_sin`，避免每层各自从整表切片
5. 若存在 `Prefill / Decode`、`PA / 非PA`、`BNSD / TND` 等差异，应在统一入口收敛这些分支，而不是散落到每一层
6. 若后续链路需要 `query_rope / key_rope` 或直接传 `rope_sin / rope_cos` 给融合算子，应在统一入口先整理好 RoPE 输入，再传入下游模块

## 仓库成熟参考

- `cann-recipes-infer/models/deepseek-v3.2-exp/models/modules.py` 与 `cann-recipes-infer/models/deepseek-v3.2-exp/models/modeling_deepseek.py`
  - 参考 `RotaryEmbedding` 类、`_init_rope()`、模型级统一取值
- `cann-recipes-infer/models/deepseek_r1/models/modules.py` 与 `cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`
  - 参考 `Prefill / Decode`、`PA / 非PA`、`BNSD / TND` 等分支处理方式

## 最小样例

下面的样例保留成熟模式的核心结构，适合作为前置改造模板：

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, dtype):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, hidden_states, position_ids):
        cos = torch.index_select(self.cos_cached, 0, position_ids.view(-1)).unsqueeze(1).unsqueeze(1)
        sin = torch.index_select(self.sin_cached, 0, position_ids.view(-1)).unsqueeze(1).unsqueeze(1)
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)
```

```python
def _init_rope(self):
    self.rotary_emb = RotaryEmbedding(
        dim=self.config.qk_rope_head_dim,
        max_position_embeddings=self.config.max_position_embeddings,
        base=self.config.rope_theta,
    )
```

```python
def forward(self, hidden_states, position_ids, ...):
    cos_sin = self.rotary_emb(hidden_states, position_ids)
    for layer in self.layers:
        hidden_states = layer(hidden_states, cos_sin=cos_sin, ...)
    return hidden_states
```

## 使用边界

本文档只用于指导 RoPE 预计算与调用路径改造，不负责：

- 选择具体融合算子
- 展开完整 Attention 替换流程
- 定义 `query_rope / key_rope` 的最终算子接口细节

这些内容仍以主 skill 和对应算子文档为准。
