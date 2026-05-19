# Full-Model Scope 模板

本模板提供将整个 Decoder 层纳入 SuperKernel scope 的实现方案。

---

## 适用场景

- 模型结构简单
- 已有成功案例
- 追求最大性能提升
- Attention-Only 验证成功后

---

## 优点

- **融合范围最大**：整个 Decoder 层融合
- **性能提升潜力最高**：减少最多的调度开销
- **充分利用 SuperKernel**：最大化融合效果

---

## 缺点

- **调试难度大**：问题定位困难
- **编译时间长**：Scope 范围大，编译慢
- **风险较高**：可能引入更多问题

---

## 实现方案

```python
# cann-recipes-infer/models/{model_name}/models/modeling_*.py

from executor.utils import superkernel_scope

class Model(nn.Module):
    def decode(self, input_ids, **kwargs):
        is_prefill = False  # decode 阶段

        hidden_states = self.embed_tokens(input_ids)

        # SuperKernel scope 包含所有 Decoder 层
        with superkernel_scope(
            self.enable_superkernel and not is_prefill,
            label="all_decoder_layers",
            option="stream-fusion=1"
        ):
            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    is_prefill=is_prefill
                )

        logits = self.lm_head(hidden_states)
        return logits
```

---

## 预期性能提升

| 模型类型 | 预期提升 | 说明 |
|---------|---------|------|
| 标准 Transformer | 25-35% | 融合范围最大 |
| MoE 模型 | 30-40% | 整体融合效果好 |

---

## 参考资源

- Attention-Only 模板：`attention-only.md`
- MoE-Only 模板：`moe-only.md`
