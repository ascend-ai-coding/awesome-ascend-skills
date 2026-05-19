# Attention-Only Scope 模板

本模板提供仅将 Attention 模块纳入 SuperKernel scope 的实现方案。

---

## 适用场景

- 首次尝试 SuperKernel
- 模型性能瓶颈主要在 Attention 计算
- 需要最小风险的优化方案
- 标准 Transformer 架构模型

---

## 优点

- **风险最小**：Scope 范围小，问题易定位
- **易于调试**：编译快，调试简单
- **稳定性高**：不影响其他模块
- **适合首次尝试**：验证 SuperKernel 是否有效

---

## 缺点

- **性能提升有限**：仅优化 Attention 部分
- **未充分利用**：其他计算密集型模块未优化

---

## 实现方案

### 方案 A：在 Attention 模块内部标记

适用于 Attention 模块独立实现的情况。

```python
# cann-recipes-infer/models/{model_name}/models/modeling_*.py

from executor.utils import superkernel_scope

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enable_superkernel = config.enable_superkernel
        # ... 其他初始化

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        is_prefill=False,
        **kwargs
    ):
        # SuperKernel scope 仅包含 Attention 计算
        with superkernel_scope(
            self.enable_superkernel and not is_prefill,
            label="attention",
            option="stream-fusion=1"
        ):
            # Q, K, V 投影
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # Attention 计算
            attn_output = self._compute_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                past_key_value
            )

            # O 投影
            attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value
```

---

### 方案 B：在 Decoder 层中标记

适用于 Attention 作为 Decoder 层一部分的情况。

```python
# cann-recipes-infer/models/{model_name}/models/modeling_*.py

from executor.utils import superkernel_scope

class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.enable_superkernel = config.enable_superkernel
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        # ... 其他初始化

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        is_prefill=False,
        **kwargs
    ):
        residual = hidden_states

        # SuperKernel scope 仅包含 Attention
        with superkernel_scope(
            self.enable_superkernel and not is_prefill,
            label=f"attention_layer_{self.layer_idx}",
            option="stream-fusion=1"
        ):
            # Self-Attention
            hidden_states = self.input_layernorm(hidden_states)
            attn_output, past_key_value = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                is_prefill=is_prefill
            )

        # Add & Norm（在 Scope 外）
        hidden_states = residual + attn_output
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP（在 Scope 外）
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value
```

---

## 配置文件

```yaml
# cann-recipes-infer/models/{model_name}/config.yaml

exe_mode: "ge_graph"              # 必须是 ge_graph
model_config:
  enable_superkernel: True        # 启用 SuperKernel
  enable_multi_streams: False     # 可选：多流并行
  enable_cache_compile: False     # 可选：缓存编译
```

---

## 验证步骤

### 1. 编译验证

```bash
cd cann-recipes-infer/models/{model_name}
bash infer.sh 2>&1 | tee compile.log

# 检查编译日志
grep -i "superkernel" compile.log
grep -i "attention" compile.log
```

### 2. 功能验证

```bash
# 对比启用前后的输出
# 1. 禁用 SuperKernel
sed -i 's/enable_superkernel: True/enable_superkernel: False/' config.yaml
bash infer.sh > baseline_output.txt

# 2. 启用 SuperKernel
sed -i 's/enable_superkernel: False/enable_superkernel: True/' config.yaml
bash infer.sh > optimized_output.txt

# 3. 对比输出
diff baseline_output.txt optimized_output.txt
# 应该完全一致
```

### 3. 性能验证

```bash
# 使用性能测试脚本
python scripts/benchmark.py --config config.yaml --output performance.json

# 查看性能提升
cat performance.json | jq '.improvement_percent'
```

---

## 预期性能提升

| 模型类型 | 预期提升 | 说明 |
|---------|---------|------|
| 标准 Transformer | 10-15% | Attention 占比约 30-40% |
| MoE 模型 | 5-10% | Attention 占比较小 |
| 长序列模型 | 15-20% | Attention 计算密集 |

---

## 常见问题

### Q1: 编译失败，提示不支持的算子

**可能原因**：Attention 模块中使用了不支持的算子

**解决方法**：
1. 检查 Attention 实现，确认使用的算子
2. 如果使用了 Tiling 下沉算子，将其移出 Scope
3. 参考调试指南进行排查

### Q2: 性能提升不明显

**可能原因**：
1. Attention 不是性能瓶颈
2. 模型结构特殊，Attention 占比小

**解决方法**：
1. 使用 profiler 分析性能瓶颈
2. 如果 MoE 是瓶颈，尝试 MoE-Only 模板
3. 如果整体都是瓶颈，尝试 Full-Model 模板

### Q3: 精度不一致

**可能原因**：
1. DCache 一致性问题（如果使用自定义算子）
2. 数值计算顺序变化

**解决方法**：
1. 检查是否使用了自定义算子
2. 如果使用了，添加 DCache 刷新
3. 调用 `model-infer-precision-debug` skill 进行排查

---

## 下一步

验证成功后，可以考虑：
1. **扩大范围**：尝试 Full-Model 模板，将整个 Decoder 层纳入 Scope
2. **结合其他优化**：启用多流并行、融合算子等
3. **性能调优**：调整编译选项，进一步优化性能

---

## 参考资源

- Scope 分析指南：`../../references/scope-analysis-guide.md`
- 性能基线指南：`../../references/performance-baseline-guide.md`
- 调试指南：`../../references/debugging-guide.md`
- MoE-Only 模板：`moe-only.md`
- Full-Model 模板：`full-model.md`
