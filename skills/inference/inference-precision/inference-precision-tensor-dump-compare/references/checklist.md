# Tensor Dump 检查清单

本清单提供 tensor dump 注入前后的完整检查流程。

---

## 注入前检查

- [ ] 分析模型架构 (继承关系、模块分布)
- [ ] 确定需要注入的文件数量
- [ ] 备份原文件
- [ ] 分析 Attention 类型 (标准 / 分离式)
- [ ] 分析 FFN 类型 (Dense MLP / MoE)
- [ ] 确定必须 Tags 列表

---

## 注入后检视流程

### Step 1: 语法验证 ⭐

```bash
docker exec <container> python -m py_compile /path/to/model.py
```

- [ ] 无语法错误
- [ ] 无缩进错误

### Step 2: Tag 完整性检查 ⭐⭐

**必须 Tags (必须存在)**:

| 模块 | 必须 Tags |
|------|---------|
| ForCausalLM | `input_ids`, `positions`, `final_hs`, `logits` |
| Attention | `attn.in`, `attn.out` |
| DecoderLayer | `hs.out`, `hs.mlp.out`, `prepare_mlp.out.residual` (SGLang) |
| MoE | `moe.router_logits` |

**检查命令**:
```bash
# 导出所有 Tags
docker exec <container> grep 'tag="' /path/to/model.py | grep -o 'tag="[^"]*"' | sort | uniq

# 检查特定 Tag
docker exec <container> grep '"tag_name"' /path/to/model.py
```

### Step 3: 属性可用性检查 ⭐⭐

- [ ] layer_id 属性已定义 (Attention, DecoderLayer)
- [ ] tp_rank 获取方式正确
- [ ] device 参数正确传递

### Step 4: 打点顺序检查 ⭐⭐

- [ ] 变量在打点前已定义
- [ ] 无 UnboundLocalError

### Step 5: 服务启动验证 ⭐⭐⭐

- [ ] 服务正常启动
- [ ] 日志有 `[TD_CMP]` 输出
- [ ] 所有必须 Tags 都有日志

---

## 常见错误与解决

| 错误 | 原因 | 解决 |
|------|------|------|
| `IndentationError` | 缩进不匹配 | 检查并修正缩进 |
| `AttributeError: layer_id` | 未定义属性 | 添加 `self.layer_id = layer_id` |
| `AttributeError: tp_rank` | 获取位置错误 | 使用 `self.self_attn.tp_rank` |
| `UnboundLocalError` | 打点在变量定义前 | 调整打点代码顺序 |
| 无 TD_CMP 日志 | 环境变量未设置 | 设置 `TENSOR_DUMP_ENABLE=1` |

---

## 快速检视命令

```bash
# 语法验证
docker exec <container> python -m py_compile /path/to/model.py

# Tag 列表
docker exec <container> grep 'tag="' /path/to/model.py | grep -o 'tag="[^"]*"' | sort | uniq

# layer_id 检查
docker exec <container> grep "self.layer_id = layer_id" /path/to/model.py

# 服务日志检查
docker exec <container> grep "TD_CMP" /tmp/sglang_server.log | head -50
```

---

## 参考文档

详细检视流程见: `references/review-guide.md` (必读!)

---

## vLLM 检查清单

### ForCausalLM

- [ ] `input_ids` 打点
- [ ] `positions` 打点
- [ ] `hs.in` 打点
- [ ] `final_hs` 打点
- [ ] `logits` 打点

### Attention

- [ ] `attn.in` 打点
- [ ] `qkv_proj.out` 打点
- [ ] `q.split`, `k.split`, `v.split` 打点 (可选)
- [ ] `q_norm.out`, `k_norm.out` 打点 (如有 qk_norm)
- [ ] `rotary_emb.out` 打点
- [ ] `attn.in.q`, `attn.in.k`, `attn.in.v` 打点
- [ ] `attn.out.pre_o` 打点
- [ ] `attn.out` 打点

### Attention (MLA - DeepSeek/GLM-5)

- [ ] `attn.in` 打点
- [ ] `attn.out` 打点
- [ ] `self.prefix` 属性已定义 (用于 layer_idx 提取)

### DecoderLayer

- [ ] `hs.in` 打点
- [ ] `hs.ln1` 打点
- [ ] `hs.post_attn` 打点
- [ ] `hs.post_attn_residual` 打点
- [ ] `hs.ln2` 打点
- [ ] `hs.mlp.out` 或 `hs.moe.out` 打点
- [ ] `hs.out` 打点

### MLP

- [ ] `mlp.input` 打点
- [ ] `mlp.gate_up` 打点
- [ ] `mlp.act_out` 打点
- [ ] `mlp.output` 打点

### MoE

- [ ] `moe.hidden_states_input` 打点
- [ ] `moe.router_logits` 打点
- [ ] `moe.shared_output` 打点 (如有)
- [ ] `moe.post_reduce` 打点
- [ ] `moe.output` 打点

---

## SGLang 检查清单

### ForCausalLM

- [ ] `input_ids` 打点
- [ ] `positions` 打点
- [ ] `final_hs` 打点
- [ ] `logits` 打点

### Attention (合并式 - 如 Qwen3.5)

- [ ] `attn.in` 打点
- [ ] `qkv_proj.out` 打点
- [ ] `rotary_emb.out.q` 打点
- [ ] `rotary_emb.out.k` 打点
- [ ] `attn.in.q`, `attn.in.k`, `attn.in.v` 打点
- [ ] `attn.out.pre_o` 打点
- [ ] `attn.out` 打点

### Attention (分离式 - 如 MiMo-v2)

- [ ] `forward_prepare.attn.in` 打点
- [ ] `forward_prepare.qkv_proj.out` 打点
- [ ] `forward_prepare.rotary_emb.out.q` 打点
- [ ] `forward_prepare.rotary_emb.out.k` 打点
- [ ] `forward_core.attn.in.q` 打点
- [ ] `forward_core.attn.in.k` 打点
- [ ] `forward_core.attn.in.v` 打点
- [ ] `forward_core.attn.out.pre_o` 打点
- [ ] `forward_core.attn.out` 打点

### DecoderLayer

- [ ] `hs.in` 打点
- [ ] `hs.prepare_attn.out` 打点
- [ ] `hs.post.attn` 打点
- [ ] `prepare_mlp.out.hs` 打点
- [ ] `prepare_mlp.out.residual` 打点 ← **必须！**
- [ ] `hs.mlp.out` 或 `hs.moe.out` 打点
- [ ] `hs.out` 打点

### MLP

- [ ] `mlp.input` 打点
- [ ] `mlp.gate_up` 打点
- [ ] `mlp.act_out` 打点
- [ ] `mlp.act_output` 打点 ← **使用此命名！不是 mlp.output**

### MoE (forward_normal)

- [ ] `forward_normal.moe.hidden_states_input` 打点
- [ ] `forward_normal.moe.router_logits` 打点 ← **必须！**
- [ ] `forward_normal.moe.final_hs.pre_allreduce` 打点
- [ ] `forward_normal.moe.hidden_states_out.post_all_reduce` 打点
- [ ] `forward_normal.moe.hidden_states_out` 打点

### MoE (forward_deepep)

- [ ] `forward_deepep.moe.hidden_states_input` 打点
- [ ] `forward_deepep.moe.router_logits` 打点
- [ ] `forward_deepep.moe.hidden_states_out` 打点

---

## Step3.5 检查清单

Step3.5 使用 `step3p5.*` 前缀：

### ForCausalLM

- [ ] `step3p5.input_ids` 打点
- [ ] `step3p5.positions` 打点
- [ ] `step3p5.final_hs` 打点

### Attention

- [ ] `step3p5.attn.in` 打点
- [ ] `step3p5.attn.prepare.in` 打点
- [ ] `step3p5.attn.prepare.qkv` 打点
- [ ] `step3p5.attn.prepare.q` 打点
- [ ] `step3p5.attn.prepare.k` 打点
- [ ] `step3p5.attn.prepare.rotary.q` 打点
- [ ] `step3p5.attn.prepare.rotary.k` 打点
- [ ] `step3p5.attn.out` 打点

### DecoderLayer

- [ ] `step3p5.hs.in` 打点
- [ ] `step3p5.hs.ln1` 打点
- [ ] `step3p5.hs.post_attn` 打点
- [ ] `step3p5.hs.pre_mlp` 打点
- [ ] `step3p5.hs.share_expert.out` 打点 ← **Step3.5 特有**
- [ ] `step3p5.hs.moe.out` 打点
- [ ] `step3p5.hs.post_reduce` 打点 ← **Step3.5 特有**
- [ ] `step3p5.hs.out` 打点

### MoE

- [ ] `step3p5.moe.input` 打点
- [ ] `step3p5.moe.router_logits` 打点
- [ ] `step3p5.moe.output` 打点

---

## 必须包含的 Tags (⭐⭐⭐)

### vLLM

| 位置 | 必须 Tags |
|------|---------|
| ForCausalLM | `input_ids`, `final_hs`, `logits` |
| Attention (标准) | `attn.in`, `attn.out` |
| Attention (MLA) | `attn.in`, `attn.out` |
| DecoderLayer | `hs.in`, `hs.out` |
| MoE | `moe.router_logits`, `moe.output` |

### SGLang

| 位置 | 必须 Tags |
|------|---------|
| ForCausalLM | `input_ids`, `final_hs`, `logits` |
| DecoderLayer | `hs.in`, `hs.out`, `prepare_mlp.out.residual` |
| Attention | `attn.in`, `attn.out` |
| MoE | `moe.router_logits`, `moe.output` |

---

## 常见遗漏

| 框架 | 遗漏项 | 位置 | 影响 |
|------|--------|------|------|
| vLLM | `logits` | ForCausalLM | 无法验证最终精度 |
| vLLM | `moe.router_logits` | MoE | 无法排查路由问题 |
| vLLM (MLA) | `attn.in`/`attn.out` | DeepseekV2MLAAttention | 无法排查注意力问题 |
| vLLM (MLA) | `self.prefix` | MLA Attention __init__ | layer_idx 无法提取 |
| SGLang | `logits` | ForCausalLM | 无法验证最终精度 |
| SGLang | `prepare_mlp.out.residual` | DecoderLayer | 无法排查残差问题 |
| SGLang | `moe.router_logits` | MoE | 无法排查路由问题 |
| SGLang | `mlp.act_output` | MLP | 命名错误 |
| SGLang | `forward_deepep.*` | MoE | DeepEP 路径无日志 |

---

## Tag 命名错误

| 框架 | 错误命名 | 正确命名 |
|------|---------|---------|
| SGLang | `mlp.output` | `mlp.act_output` |
| SGLang | `moe.input` | `forward_normal.moe.hidden_states_input` |
| SGLang | `hs.ln1` | `hs.prepare_attn.out` |
| SGLang | `hs.pre_mlp` | `prepare_mlp.out.hs` |

---

## 快速验证命令

```bash
# vLLM - 语法检查
python3 -m py_compile /path/to/model.py

# SGLang - 语法检查
python3 -m py_compile /path/to/model.py

# Tag 数量统计
grep -c "_td_compare_log\|_mimo_compare_log" /path/to/model.py

# 必须 Tags 存在性检查
grep -E '"input_ids"|"positions"|"final_hs"|"logits"' /path/to/model.py
grep -E '"hs\.in"|"hs\.out"' /path/to/model.py
grep -E '"moe\.router_logits"|"moe\.output"' /path/to/model.py

# SGLang 特有检查
grep '"prepare_mlp.out.residual"' /path/to/model.py
grep '"mlp.act_output"' /path/to/model.py

# 日志验证
grep "\[TD_CMP\]\|\[MIMO_CMP\]" logs/*.log | head -20
```

---

## 回滚

```bash
# 查看备份
ls -la /path/to/model.py.bak*

# 恢复
cp /path/to/model.py.bak.20240101_120000 /path/to/model.py

# 使用脚本
python scripts/rollback_tensor_dump.py --model-path model.py --restore
```

---

## 清理临时脚本

```bash
rm -f /tmp/inject_*.py
rm -f /tmp/fix_*.py
rm -f /tmp/verify_*.py
rm -f /tmp/compare_*.py
rm -f /tmp/list_tags.py
```

---

## SGLang 注入属性检查清单

### 需要检查的类

| 类 | 需要添加的属性 | 检查方法 |
|----|---------------|---------|
| Attention 子类 | `self.layer_id = layer_id` | `grep "self.layer_id" model.py` |
| DecoderLayer 子类 | `self.layer_id = layer_id` | `grep "self.layer_id" model.py` |

### tp_rank 获取方式

| 类 | 正确写法 |
|----|---------|
| Attention | `self.tp_rank` |
| DecoderLayer | `self.self_attn.tp_rank` |
| ForCausalLM | `self.pp_group.rank_in_group` |

### 打点顺序检查

确保以下 Tag 在变量定义之后打点：

- [ ] `attn.out.pre_o` - 在 `attn_output = self.attn(...)` 之后
- [ ] `attn.out` - 在 `attn_output = self.attn(...)` 之后
- [ ] `qkv_proj.out` - 在 `qkv, _ = self.qkv_proj(...)` 之后

### 语法验证

```bash
docker exec <container> python -m py_compile /path/to/model.py
```

### 通用验证命令

```bash
# 检查 layer_id 属性是否存在
grep "self.layer_id = layer_id" /path/to/model.py

# 检查 tp_rank 使用是否正确
grep "self.tp_rank" /path/to/model.py | head -5

# 检查 tag 是否完整
grep 'tag="' /path/to/model.py | grep -o 'tag="[^"]*"' | sort | uniq
```
