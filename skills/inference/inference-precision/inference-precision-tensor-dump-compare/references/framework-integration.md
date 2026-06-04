# Tensor Dump 框架集成指南

本文档提供 vLLM 和 SGLang 框架下 tensor dump 注入的完整指南。

> **Tag 标准定义请参阅** [tensor-tags-guide.md](./tensor-tags-guide.md)

---

## 快速参考

| 框架 | 必须环境变量 | Tag 列表 |
|------|-------------|---------|
| vLLM | `TORCHDYNAMO_DISABLE=1`, `TENSOR_DUMP_ENABLE=1` | [tensor-tags-guide.md](./tensor-tags-guide.md) |
| SGLang | `TENSOR_DUMP_ENABLE=1` | [tensor-tags-guide.md](./tensor-tags-guide.md) |

---

## 框架架构对比

| 特性 | vLLM | SGLang |
|------|------|--------|
| **层 ID 属性** | `self.layer_idx` | `self.layer_id` |
| **Attention** | 单一 forward / **MLA** | forward_prepare/forward_core (分离式) |
| **残差管理** | 内联 `residual = hidden_states + x` | LayerCommunicator |
| **MoE 路径** | 单一 forward_normal | forward_normal + forward_deepep |
| **forward_batch** | 无 | 有 ForwardBatch 参数 |
| **AllReduce** | `maybe_all_reduce_tensor_model_parallel` | `tensor_model_parallel_all_reduce` |

### Attention 类型说明

| 类型 | 模型示例 | 说明 |
|------|---------|------|
| **标准Attention** | Qwen3, LLaMA | 使用 `qkv_proj`, `o_proj` |
| **MLA Attention** | DeepSeek-V2/V3, GLM-4.5/4.6/4.7, GLM-5 | 使用 `MultiHeadLatentAttentionWrapper` |

### vLLM 架构特点

- 层 ID 属性: `self.layer_idx` (通过 `extract_layer_index(prefix)`)
- Attention: 单一 forward 方法 / **MLA Attention**
- 残差管理: 内联 `residual + x`
- MoE 路径: 单一 forward_normal

### SGLang 特有组件

1. **LayerCommunicator**: 封装残差管理
   - `prepare_attn()`: 准备 Attention (含 input_layernorm)
   - `prepare_mlp()`: 准备 MLP (含 post_attention_layernorm)
   - `postprocess_layer()`: 后处理 (残差相加)

2. **分离式 Attention**: MiMo-v2 等模型使用
   - `forward_prepare()`: QKV 计算 + RoPE
   - `forward_core()`: Attention 计算 + O_proj

3. **DeepEP MoE**: 分布式专家并行
   - `forward_deepep()`: DeepEP 路径

---

## 日志函数模板

```python
import os as _os
from typing import Any, Optional

_TD_ENABLE = _os.environ.get("TENSOR_DUMP_ENABLE", "0") == "1"

_TD_DEVICE_FILTER = _os.environ.get("TENSOR_DUMP_DEVICE", "npu:1")
_TD_LAYERS_FILTER = None
_td_layers_str = _os.environ.get("TENSOR_DUMP_LAYERS", "all")
if _td_layers_str != "all":
    try:
        _TD_LAYERS_FILTER = set(int(x.strip()) for x in _td_layers_str.split(","))
    except ValueError:
        pass

def _td_compare_log(tag: str, obj: Any, *, layer_idx: Optional[int] = None) -> None:
    if not _TD_ENABLE:
        return
    if layer_idx is not None and _TD_LAYERS_FILTER is not None:
        if layer_idx not in _TD_LAYERS_FILTER:
            return
    if obj is None:
        layer_str = str(layer_idx) if layer_idx is not None else "-"
        logger.info("[TD_CMP] layer=%s %s=None", layer_str, tag)
        return
    if not isinstance(obj, torch.Tensor):
        layer_str = str(layer_idx) if layer_idx is not None else "-"
        logger.info("[TD_CMP] layer=%s %s type=%s", layer_str, tag, type(obj))
        return
    device_str = str(obj.device)
    if _TD_DEVICE_FILTER.endswith(":*"):
        device_prefix = _TD_DEVICE_FILTER[:-2]
        if not device_str.startswith(device_prefix):
            return
    elif device_str != _TD_DEVICE_FILTER:
        return
    tensor_l1n = obj.to(torch.float).norm(p=1).item()
    layer_str = str(layer_idx) if layer_idx is not None else "-"
    logger.info(
        "[TD_CMP] layer=%s %s device=%s shape=%s dtype=%s l1n=%.8f",
        layer_str, tag, device_str, tuple(obj.shape), obj.dtype, tensor_l1n,
    )
```

---

## 模型分析命令

### 通用分析

```bash
# 查看模型类
grep -E "class.*ForCausalLM|class.*Model|class.*Attention" /path/to/model.py
```

### vLLM 专用

```bash
# 判断 MoE
grep -E "use_moe|is_moe|num_local_experts|gate\(" /path/to/model.py

# 判断 qk_norm
grep -E "q_norm|k_norm|qk_norm" /path/to/model.py

# 查看 layer_idx 属性
grep -E "self.layer_idx\s*=" /path/to/model.py
```

### SGLang 专用

```bash
# 判断分离式 Attention
grep -E "forward_prepare|forward_core" /path/to/model.py

# 判断 LayerCommunicator
grep -E "layer_communicator|prepare_attn|prepare_mlp" /path/to/model.py

# 判断 MoE 路径
grep -E "forward_deepep|_enable_a2a_moe|is_deepep" /path/to/model.py

# 查看 layer_id 属性
grep -E "self.layer_id\s*=" /path/to/model.py
```

---

# vLLM 详细注入手册

> Tag 标准定义请参阅 [tensor-tags-guide.md](./tensor-tags-guide.md#通用-tags-两框架必须支持)

## vLLM Attention 类型说明

| 类型 | 模型示例 | 说明 |
|------|---------|------|
| **标准Attention** | Qwen3, LLaMA | 使用 `qkv_proj`, `o_proj` |
| **MLA Attention** | DeepSeek-V2/V3, GLM-4.5/4.6/4.7, GLM-5 | 使用 `MultiHeadLatentAttentionWrapper`，是DeepSeek特有的注意力机制 |

### 判断模型使用的Attention类型

```bash
# 检查模型架构
grep -E "GlmMoeDsaForCausalLM|DeepseekV2ForCausalLM|DeepseekV3ForCausalLM" config.json

# 检查是否使用MLA
grep -E "use_mla|model_type.*deepseek" /path/to/model.py

# MLA Attention类名
grep -E "class.*MLAAttention|MultiHeadLatentAttention" /path/to/model.py
```

## 1. ForCausalLM

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    if _TD_ENABLE:
        _td_compare_log("input_ids", input_ids, layer_idx=-1)
        _td_compare_log("positions", positions, layer_idx=-1)
    
    if hidden_states is None:
        hidden_states = self.get_input_embeddings(input_ids)
    
    if _TD_ENABLE:
        _td_compare_log("hs.in", hidden_states, layer_idx=-1)
    
    for layer_idx, layer in enumerate(self.layers):
        layer_output = layer(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = layer_output
    
    if _TD_ENABLE:
        _td_compare_log("final_hs", hidden_states, layer_idx=-1)
    
    hidden_states = self.norm(hidden_states)
    if _TD_ENABLE:
        _td_compare_log("norm.out", hidden_states, layer_idx=-1)
    
    logits = self.compute_logits(hidden_states, embedding_bias=None)
    if _TD_ENABLE:
        _td_compare_log("logits", logits, layer_idx=-1)
    
    return logits
```

## 2. Qwen3Attention

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch = None,
) -> torch.Tensor:
    if _TD_ENABLE:
        _td_compare_log("attn.in", hidden_states, layer_idx=self.layer_idx)
    
    qkv, _ = self.qkv_proj(hidden_states)
    if _TD_ENABLE:
        _td_compare_log("qkv_proj.out", qkv, layer_idx=self.layer_idx)
    
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    
    # QK norm (Qwen3 特有)
    if hasattr(self, 'q_norm'):
        q = self.q_norm(q)
        k = self.k_norm(k)
        if _TD_ENABLE:
            _td_compare_log("q_norm.out", q, layer_idx=self.layer_idx)
            _td_compare_log("k_norm.out", k, layer_idx=self.layer_idx)
    
    q, k = self.rotary_emb(positions, q, k)
    if _TD_ENABLE:
        _td_compare_log("rotary_emb.out", (q, k), layer_idx=self.layer_idx)
    
    attn_output = self.attn(q, k, v, forward_batch)
    if _TD_ENABLE:
        _td_compare_log("attn.out.pre_o", attn_output, layer_idx=self.layer_idx)
    
    output, _ = self.o_proj(attn_output)
    if _TD_ENABLE:
        _td_compare_log("attn.out", output, layer_idx=self.layer_idx)
    
    return output
```

## 3. DeepseekV2MLAAttention (MLA Attention)

MLA (Multi-head Latent Attention) 是 DeepSeek 系列模型特有的注意力机制。GLM-5 模型也使用 MLA Attention。

### MLA 结构特点

- 使用 `MultiHeadLatentAttentionWrapper` 封装注意力计算
- `forward` 方法调用 `self.mla_attn(positions, hidden_states, llama_4_scaling)`
- 需要在 `__init__` 中保存 `self.prefix` 以获取 layer_idx

### MLA Attention 注入代码

```python
# 在 __init__ 中添加 self.prefix
def __init__(
    self,
    vllm_config: VllmConfig,
    config: DeepseekV2Config | DeepseekV3Config,
    # ... 其他参数 ...
    prefix: str = "",
) -> None:
    super().__init__()
    # ... 其他初始化代码 ...
    self.prefix = prefix

def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
) -> torch.Tensor:
    if _TD_ENABLE:
        _td_compare_log("attn.in", hidden_states, 
                        layer_idx=_td_extract_layer_idx(self.prefix) if hasattr(self, 'prefix') else 0)
    output = self.mla_attn(positions, hidden_states, llama_4_scaling)
    if _TD_ENABLE:
        _td_compare_log("attn.out", output, 
                        layer_idx=_td_extract_layer_idx(self.prefix) if hasattr(self, 'prefix') else 0)
    return output
```

### MLA DecoderLayer 注入代码

MLA 模型的 DecoderLayer 需要特殊处理残差克隆：

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    # Self Attention
    if _TD_ENABLE:
        _td_compare_log("hs.in", hidden_states, layer_idx=self.layer_idx)
    if residual is None:
        residual = hidden_states.clone()  # MLA需要clone
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    if _TD_ENABLE:
        _td_compare_log("hs.ln1", hidden_states, layer_idx=self.layer_idx)

    # Attention (MLA)
    if _TD_ENABLE:
        _td_compare_log("attn.in", hidden_states, layer_idx=self.layer_idx)
    hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)
    if _TD_ENABLE:
        _td_compare_log("attn.out", hidden_states, layer_idx=self.layer_idx)

    # ... 其他处理 ...

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    if _TD_ENABLE:
        _td_compare_log("hs.ln2", hidden_states, layer_idx=self.layer_idx)
    hidden_states = self.mlp(hidden_states)
    if _TD_ENABLE:
        _td_compare_log("hs.mlp.out", hidden_states, layer_idx=self.layer_idx)

    if _TD_ENABLE:
        _td_compare_log("hs.out", hidden_states, layer_idx=self.layer_idx)
    return hidden_states, residual
```

## 4. Qwen3DecoderLayer

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    if _TD_ENABLE:
        _td_compare_log("hs.in", hidden_states, layer_idx=self.layer_idx)
    
    # Pre-layernorm
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    
    if _TD_ENABLE:
        _td_compare_log("hs.ln1", hidden_states, layer_idx=self.layer_idx)
    
    # Attention
    hidden_states = self.self_attn(positions, hidden_states, forward_batch)
    if _TD_ENABLE:
        _td_compare_log("hs.post_attn", hidden_states, layer_idx=self.layer_idx)
    
    # Residual add
    hidden_states = residual + hidden_states
    if _TD_ENABLE:
        _td_compare_log("hs.post_attn_residual", hidden_states, layer_idx=self.layer_idx)
    
    # Post-layernorm
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    if _TD_ENABLE:
        _td_compare_log("hs.ln2", hidden_states, layer_idx=self.layer_idx)
    
    # MLP/MoE
    hidden_states = self.mlp(hidden_states, forward_batch)
    if _TD_ENABLE:
        _td_compare_log("hs.mlp.out", hidden_states, layer_idx=self.layer_idx)
    
    # Final residual add
    hidden_states = residual + hidden_states
    if _TD_ENABLE:
        _td_compare_log("hs.out", hidden_states, layer_idx=self.layer_idx)
    
    return hidden_states
```

## 4. MLP

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if _TD_ENABLE:
        _td_compare_log("mlp.input", x, layer_idx=self.layer_idx)
    
    gate_up, _ = self.gate_up_proj(x)
    if _TD_ENABLE:
        _td_compare_log("mlp.gate_up", gate_up, layer_idx=self.layer_idx)
    
    x = self.act_fn(gate_up)
    if _TD_ENABLE:
        _td_compare_log("mlp.act_out", x, layer_idx=self.layer_idx)
    
    x, _ = self.down_proj(x)
    if _TD_ENABLE:
        _td_compare_log("mlp.output", x, layer_idx=self.layer_idx)
    
    return x
```

## 5. MoE

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    router_logits: torch.Tensor = None,
) -> torch.Tensor:
    if _TD_ENABLE:
        _td_compare_log("moe.hidden_states_input", hidden_states, layer_idx=self.layer_idx)
    
    if router_logits is None:
        router_logits, _ = self.gate(hidden_states)
    if _TD_ENABLE:
        _td_compare_log("moe.router_logits", router_logits, layer_idx=self.layer_idx)
    
    final_hidden_states = self.experts(hidden_states, router_logits)
    
    if self.tp_size > 1:
        final_hidden_states = maybe_all_reduce_tensor_model_parallel(final_hidden_states)
        if _TD_ENABLE:
            _td_compare_log("moe.post_reduce", final_hidden_states, layer_idx=self.layer_idx)
    
    if _TD_ENABLE:
        _td_compare_log("moe.output", final_hidden_states, layer_idx=self.layer_idx)
    
    return final_hidden_states
```

## vLLM 启动命令

```bash
# 必须设置 TORCHDYNAMO_DISABLE=1，否则日志不输出!
export TORCHDYNAMO_DISABLE=1
export TENSOR_DUMP_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NPU
export TENSOR_DUMP_DEVICE=npu:1
vllm serve /path/to/model --tensor-parallel-size 8 --device npu --port 8000

# GPU
export TENSOR_DUMP_DEVICE=cuda:0
vllm serve /path/to/model --tensor-parallel-size 8 --device cuda --port 8000
```

---

# SGLang 详细注入手册

> Tag 标准定义请参阅 [tensor-tags-guide.md](./tensor-tags-guide.md#通用-tags-两框架必须支持)

## 1. ForCausalLM

```python
@torch.no_grad()
def forward(self, input_ids, positions, forward_batch, input_embeds=None, ...):
    if _TD_ENABLE:
        _td_compare_log("input_ids", input_ids, layer_idx=-1)
        _td_compare_log("positions", positions, layer_idx=-1)

    hidden_states, hidden_states_before_norm = self.model(...)

    if _TD_ENABLE:
        _td_compare_log("final_hs", hidden_states, layer_idx=-1)

    if self.pp_group.is_last_rank:
        logits = self.logits_processor(...)
        if _TD_ENABLE:
            _td_compare_log("logits", logits.next_token_logits, layer_idx=-1)
        return logits
```

## 2. Attention (合并式 - Qwen3.5 等)

```python
def forward(self, positions, hidden_states, forward_batch):
    if _TD_ENABLE:
        _td_compare_log("attn.in", hidden_states, layer_idx=self.layer_id)
    
    qkv, _ = self.qkv_proj(hidden_states)
    if _TD_ENABLE:
        _td_compare_log("qkv_proj.out", qkv, layer_idx=self.layer_id)
    
    q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    
    if _TD_ENABLE:
        _td_compare_log("rotary_emb.out.q", q, layer_idx=self.layer_id)
        _td_compare_log("rotary_emb.out.k", k, layer_idx=self.layer_id)
    
    if self.v_scale is not None:
        v = v * self.v_scale
    
    if _TD_ENABLE:
        _td_compare_log("attn.in.v", v, layer_idx=self.layer_id)
        if self.attention_sink_bias is not None:
            _td_compare_log("attn.in.sinks", self.attention_sink_bias, layer_idx=self.layer_id)
    
    attn_output = self.attn(q, k, v, forward_batch, sinks=self.attention_sink_bias)
    if _TD_ENABLE:
        _td_compare_log("attn.out.pre_o", attn_output, layer_idx=self.layer_id)
    
    output, _ = self.o_proj(attn_output)
    if _TD_ENABLE:
        _td_compare_log("attn.out", output, layer_idx=self.layer_id)
    return output
```

## 3. Attention (分离式 - MiMo-v2)

```python
def forward_prepare(self, positions, hidden_states, forward_batch):
    if hidden_states.shape[0] == 0:
        return hidden_states, forward_batch, None
    
    if _TD_ENABLE:
        _td_compare_log("forward_prepare.attn.in", hidden_states, layer_idx=self.layer_id)
    
    qkv, _ = self.qkv_proj(hidden_states)
    if _TD_ENABLE:
        _td_compare_log("forward_prepare.qkv_proj.out", qkv, layer_idx=self.layer_id)
    
    q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    
    if _TD_ENABLE:
        _td_compare_log("forward_prepare.rotary_emb.out.q", q, layer_idx=self.layer_id)
        _td_compare_log("forward_prepare.rotary_emb.out.k", k, layer_idx=self.layer_id)
    
    return None, forward_batch, (q, k, v, forward_batch)

def forward_core(self, intermediate_state):
    hidden_states, forward_batch, inner_state = intermediate_state
    if inner_state is None:
        return hidden_states
    
    q, k, v, _ = inner_state
    if _TD_ENABLE:
        _td_compare_log("forward_core.attn.in.q", q, layer_idx=self.layer_id)
        _td_compare_log("forward_core.attn.in.k", k, layer_idx=self.layer_id)
        _td_compare_log("forward_core.attn.in.v", v, layer_idx=self.layer_id)
    
    attn_output = self.attn(*inner_state, sinks=self.attention_sink_bias)
    if _TD_ENABLE:
        _td_compare_log("forward_core.attn.out.pre_o", attn_output, layer_idx=self.layer_id)
    
    output, _ = self.o_proj(attn_output)
    if _TD_ENABLE:
        _td_compare_log("forward_core.attn.out", output, layer_idx=self.layer_id)
    return output
```

## 4. DecoderLayer (LayerCommunicator)

```python
def forward(self, positions, hidden_states, forward_batch, residual, ...):
    if _TD_ENABLE:
        _td_compare_log("hs.in", hidden_states, layer_idx=self.layer_id)
    
    hidden_states, residual = self.layer_communicator.prepare_attn(
        hidden_states, residual, forward_batch
    )
    if _TD_ENABLE:
        _td_compare_log("hs.prepare_attn.out", hidden_states, layer_idx=self.layer_id)
    
    if hidden_states.shape[0] != 0:
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
    
    if _TD_ENABLE:
        _td_compare_log("hs.post.attn", hidden_states, layer_idx=self.layer_id)
    
    hidden_states, residual = self.layer_communicator.prepare_mlp(
        hidden_states, residual, forward_batch
    )
    if _TD_ENABLE:
        _td_compare_log("prepare_mlp.out.hs", hidden_states, layer_idx=self.layer_id)
        _td_compare_log("prepare_mlp.out.residual", residual, layer_idx=self.layer_id)
    
    hidden_states = self.mlp(hidden_states, forward_batch, ...)
    
    if _TD_ENABLE:
        _td_compare_log("hs.mlp.out", hidden_states, layer_idx=self.layer_id)
    
    hidden_states, residual = self.layer_communicator.postprocess_layer(...)
    
    if _TD_ENABLE:
        _td_compare_log("hs.out", hidden_states, layer_idx=self.layer_id)
    
    return hidden_states, residual
```

## 5. MLP

```python
def forward(self, x, forward_batch=None, should_allreduce_fusion=False, use_reduce_scatter=False):
    if (self.tp_size == 1) and x.shape[0] == 0:
        return x
    
    if _TD_ENABLE:
        _td_compare_log("mlp.input", x)
    
    gate_up, _ = self.gate_up_proj(x)
    if _TD_ENABLE:
        _td_compare_log("mlp.gate_up", gate_up)
    
    x = self.act_fn(gate_up)
    if _TD_ENABLE:
        _td_compare_log("mlp.act_out", x)
    
    x, _ = self.down_proj(x, skip_all_reduce=...)
    if _TD_ENABLE:
        _td_compare_log("mlp.output", x)  # 注意: 使用 output 而非 act_output
    return x
```

## 6. MoE (forward_normal)

```python
def forward_normal(self, hidden_states, should_allreduce_fusion=False, use_reduce_scatter=False):
    if _TD_ENABLE:
        _td_compare_log("forward_normal.moe.hidden_states_input", hidden_states, layer_idx=self.layer_id)
    
    if hidden_states.shape[0] > 0:
        router_logits = self.gate(hidden_states)
        if _TD_ENABLE:
            _td_compare_log("forward_normal.moe.router_logits", router_logits, layer_idx=self.layer_id)
        topk_output = self.topk(hidden_states, router_logits)
    else:
        topk_output = self.topk.empty_topk_output(hidden_states.device)
    
    final_hidden_states = self.experts(hidden_states, topk_output)
    
    if _TD_ENABLE:
        _td_compare_log("forward_normal.moe.final_hs.pre_allreduce", final_hidden_states, layer_idx=self.layer_id)
    
    if self.tp_size > 1 and not should_allreduce_fusion and not use_reduce_scatter:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        if _TD_ENABLE:
            _td_compare_log("forward_normal.moe.hidden_states_out.post_all_reduce", 
                          final_hidden_states, layer_idx=self.layer_id)
    
    if _TD_ENABLE:
        _td_compare_log("forward_normal.moe.hidden_states_out", final_hidden_states, layer_idx=self.layer_id)
    
    return final_hidden_states
```

## 7. MoE (forward_deepep)

```python
def forward_deepep(self, hidden_states, forward_batch):
    if _TD_ENABLE:
        _td_compare_log("forward_deepep.moe.hidden_states_input", hidden_states, layer_idx=self.layer_id)
    
    if hidden_states.shape[0] > 0:
        router_logits = self.gate(hidden_states)
        if _TD_ENABLE:
            _td_compare_log("forward_deepep.moe.router_logits", router_logits, layer_idx=self.layer_id)
        topk_output = self.topk(...)
    else:
        topk_output = self.topk.empty_topk_output(hidden_states.device)
    
    final_hidden_states = self.experts(hidden_states, topk_output)
    
    if _TD_ENABLE:
        _td_compare_log("forward_deepep.moe.hidden_states_out", final_hidden_states, layer_idx=self.layer_id)
    
    return final_hidden_states
```

## SGLang 启动命令

```bash
# NPU
export TENSOR_DUMP_ENABLE=1
export TENSOR_DUMP_DEVICE=npu:1

python -m sglang.launch_server \
    --model-path /path/to/model \
    --tp 8 \
    --device npu \
    --port 30000

# GPU
export TENSOR_DUMP_ENABLE=1
export TENSOR_DUMP_DEVICE=cuda:0

python -m sglang.launch_server \
    --model-path /path/to/model \
    --tp 8 \
    --device cuda \
    --port 30000
```

---

# 注入检查清单

## 注入前

- [ ] 分析模型架构 (继承关系、模块分布)
- [ ] 确定需要注入的文件数量
- [ ] 备份原文件

## 注入后

- [ ] 语法验证: `python -m py_compile model.py`
- [ ] Tag 数量检查: `grep -c "_td_compare_log" model.py`
- [ ] layer_idx/layer_id 检查: 确保已定义
- [ ] 关键 Tag 存在性检查

## 必须包含的 Tags (⭐⭐⭐)

| 框架 | ForCausalLM | DecoderLayer | MoE |
|------|------------|-------------|-----|
| vLLM | `logits` | `hs.in`, `hs.out` | `moe.router_logits` |
| SGLang | `logits` | `prepare_mlp.out.residual` | `moe.router_logits` |

---

# 验证命令

```bash
# 语法验证
python3 -m py_compile /path/to/model.py

# Tag 数量统计
grep -c "_td_compare_log" /path/to/model.py

# Tag 列表 (vLLM)
grep -o '"[^"]*"' /path/to/model.py | grep "_td_compare_log" | sort | uniq

# Tag 列表 (SGLang)
grep -o '"[^"]*"' /path/to/model.py | grep -E "input_ids|hs\.|attn\.|mlp\.|moe\." | sort | uniq

# 验证完整性 (SGLang)
python scripts/verify_tags.py /path/to/model.py --model-type mimo_v2
```

**详见**: `references/verification-debugging.md` - 常见问题与解决方案、错误排查
