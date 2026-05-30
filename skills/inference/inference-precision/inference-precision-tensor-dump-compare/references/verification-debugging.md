# Tensor Dump 验证与调试指南

本指南提供 tensor dump 注入后的验证命令、错误排查和调试技巧。

> **Tag 标准定义请参阅** [tensor-tags-guide.md](./tensor-tags-guide.md)

---

## 快速验证

### 1. 语法验证

```bash
# 容器内验证
docker exec <container> python -m py_compile /path/to/model.py

# 本地验证
python -m py_compile /path/to/model.py
```

**验证标准**: 无报错输出

---

### 2. Tag 列表导出

```bash
# 导出所有 tag
grep '_td_compare_log' /path/to/model.py | grep '"' | grep -o '"[^"]*"' | sort | uniq

# 统计打点数量
grep -c "_td_compare_log" /path/to/model.py
```

**验证标准**: 输出包含所有关键 tags（见 [tensor-tags-guide.md](./tensor-tags-guide.md#优先级建议)）

---

### 3. 环境变量检查

```bash
# vLLM 必须设置
echo $TORCHDYNAMO_DISABLE  # 应该输出 1
echo $TENSOR_DUMP_ENABLE    # 应该输出 1

# SGLang 必须设置
echo $TENSOR_DUMP_ENABLE    # 应该输出 1
echo $TENSOR_DUMP_DEVICE    # 应该输出 npu:0
```

---

## 启动与验证

### vLLM

```bash
# 设置环境变量
export TORCHDYNAMO_DISABLE=1  # 必须设置!
export TENSOR_DUMP_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 启动服务
vllm serve /path/to/model \
    --tensor-parallel-size 8 \
    --device npu \
    --max-model-len 8192 \
    --max-num-seqs 64

# 检查日志
tail -f /tmp/vllm_server.log | grep "TD_CMP"
```

### SGLang

```bash
# 设置环境变量
export TENSOR_DUMP_ENABLE=1
export TENSOR_DUMP_DEVICE=npu:0

# 启动服务
python -m sglang.launch_server \
    --model-path /path/to/model \
    --tp 8 \
    --device npu

# 检查日志
tail -f logs/sglang.log | grep "TD_CMP"
```

---

## 错误排查表

### 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `AttributeError: layer_id` | 未定义属性 | 添加 `self.layer_idx = _extract_layer_index(prefix)` |
| `AttributeError: tp_rank` | 属性获取位置错误 | 使用 `self.self_attn.tp_rank` |
| `UnboundLocalError` | 打点在变量定义前 | 调整打点代码顺序 |
| `IndentationError` | 缩进不匹配 | 检查周围代码缩进 |
| `NameError` | 变量名拼写错误 | 检查变量名大小写 |
| GPU/NPU 内存不平衡 | 其他进程占用 | 停止其他容器 |
| **MLP layer=-** | **MLP 未定义 layer_idx** | **MLP 也需要添加 `self.layer_idx`** |
| **MLA layer=-** | **MLA 未定义 `self.prefix`** | **MLA 需要 `self.prefix = prefix`** |

### SGLang 特有问题

#### 1. layer_id 属性缺失

**问题**: `AttributeError: 'XXXDecoderLayer' object has no attribute 'layer_id'`

**原因**: `__init__` 接收 `layer_id` 参数但未保存为实例属性

**解决方案**:
```python
# 在类的 __init__ 中添加
self.layer_id = layer_id
```

#### 2. tp_rank 属性获取问题

**问题**: `AttributeError: 'XXXDecoderLayer' object has no attribute 'tp_rank'`

**原因**: DecoderLayer 等类本身没有 tp_rank，需从子模块获取

**解决方案**:
```python
# 错误写法
device=str(self.tp_rank)

# 正确写法 (从子模块获取)
device=str(self.self_attn.tp_rank)    # DecoderLayer
device=str(self.tp_rank)              # Attention (已有)
device=str(self.pp_group.rank_in_group)  # ForCausalLM
```

#### 3. 变量作用域问题 (UnboundLocalError)

**问题**: `UnboundLocalError: cannot access local variable 'xxx'`

**原因**: 打点代码放置在变量定义之前

**解决方案**:
```python
# 错误顺序 - 变量尚未定义
_td_compare_log(tag="xxx.pre", ...)
xxx = self.some_method(...)  # 变量在这里才定义!

# 正确顺序 - 先定义变量后打点
xxx = self.some_method(...)
_td_compare_log(tag="xxx.pre", ...)
```

#### 4. 缩进错误

**问题**: `IndentationError`

**原因**: 注入代码缩进与周围代码不匹配

**解决方案**: 确保打点代码与同级别语句缩进一致

---

## 验证检查清单

### 注入前

- [ ] 分析模型架构 (继承关系、模块分布)
- [ ] 确定需要注入的文件数量
- [ ] 备份原文件

### 注入后

- [ ] 语法验证: `python -m py_compile model.py`
- [ ] Tag 数量检查: `grep -c "_td_compare_log" model.py`
- [ ] 关键 Tag 存在性检查

### 必须包含的 Tags (⭐⭐⭐)

注入完成后，必须按照Tag自检表检查以下所有 Tag 都已正确注入到相应模块中。

#### 1. 模型级别 (ForCausalLM / Model)

| Tag | 必须 | 说明 |
|-----|------|------|
| `input_ids` | ⭐⭐⭐ | 输入 token |
| `positions` | ⭐⭐ | 位置编码 |
| `hs.in` | ⭐⭐⭐ | Embedding 后 hidden_states |
| `final_hs` | ⭐⭐ | 最终 hidden_states (norm 前) |
| `logits` | ⭐⭐⭐ | 输出 logits |

#### 2. Attention

| Tag | 必须 | 说明 |
|-----|------|------|
| `attn.in` | ⭐⭐⭐ | Attention 输入 |
| `qkv_proj.out` | ⭐⭐ | QKV 投影输出 |
| `rotary_emb.out` | ⭐⭐ | RoPE 后 |
| `attn.out` | ⭐⭐⭐ | Attention 最终输出 |

#### 3. DecoderLayer

| Tag | 必须 | 说明 |
|-----|------|------|
| `hs.in` | ⭐⭐⭐ | Layer 输入 |
| `hs.out` | ⭐⭐⭐ | Layer 输出 |

#### 4. MLP

| Tag | 必须 | 说明 |
|-----|------|------|
| `mlp.input` | ⭐⭐ | MLP 输入 |
| `mlp.output` | ⭐⭐⭐ | MLP 输出 |

#### 5. MoE

| Tag | 必须 | 说明 |
|-----|------|------|
| `moe.router_logits` | ⭐⭐⭐ | 路由 logits |
| `moe.output` | ⭐⭐⭐ | MoE 输出 |

---

### Tag 自检表

| 模块 | 数量 | Tags |
|------|------|------|
| 模型级别 | 5 | `input_ids`, `positions`, `hs.in`, `final_hs`, `logits` |
| Attention (标准) | 7 | `attn.in`, `qkv_proj.out`, `rotary_emb.out`, `attn.out`, `q_norm.out`, `k_norm.out`, `attn.out.pre_o` |
| Attention (MLA) | 2 | `attn.in`, `attn.out` |
| DecoderLayer | 6 | `hs.in`, `hs.ln1`, `hs.post_attn`, `hs.ln2`, `hs.mlp.out`, `hs.out` |
| MLP | 4 | `mlp.input`, `mlp.gate_up`, `mlp.act_out`, `mlp.output` |
| MoE | 5 | `moe.hidden_states_input`, `moe.router_logits`, `moe.shared_output`, `moe.post_reduce`, `moe.output` |
| SGLang特有 | 7 | `forward_prepare.attn.in`, `forward_core.attn.in`, `prepare_mlp.out.residual`, `hs.prepare_attn.out`, `hs.post.attn`, `prepare_mlp.out.hs`, `forward_normal.moe.router_logits` |

**总计: 36 个 Tags** (标准Attention + MLA)

### 启动验证

- [ ] 环境变量正确设置
- [ ] 服务正常启动
- [ ] 日志中有 `TD_CMP` 输出
- [ ] 日志中没有异常信息

---

## Script 工具

| 脚本 | 功能 |
|------|------|
| `inject_tensor_dump.py` | vLLM 自动化注入 |
| `inject_tensor_dump_sglang.py` | SGLang 专用注入 |
| `verify_tags.py` | Tag 完整性验证 |
| `analyze_logs.py` | 日志分析 |
| `rollback_tensor_dump.py` | 回滚 |

### 使用示例

```bash
# vLLM 注入
python scripts/inject_tensor_dump.py --framework vllm --model-path /path/to/model.py --backup

# SGLang 注入
python scripts/inject_tensor_dump_sglang.py --model-path /path/to/model.py --backup

# 验证 Tag 完整性
python scripts/verify_tags.py /path/to/model.py --model-type mimo_v2

# 日志分析
python scripts/analyze_logs.py /tmp/sglang_server.log
```

---

## 常见问题 (FAQ)

### 通用问题

**Q: 日志中没有输出？**
- vLLM: **必须设置 `TORCHDYNAMO_DISABLE=1`** ← 这是最常见问题！
- SGLang: 确认 `TENSOR_DUMP_ENABLE=1` 或 `MIMO_COMPARE=1`
- 检查服务是否正常启动
- 检查 `TENSOR_DUMP_LAYERS` 是否正确设置

**Q: 语法错误？**
- 运行 `python -m py_compile model.py` 验证

**Q: 只看到部分层？**
- 检查 `TENSOR_DUMP_LAYERS` 环境变量设置

**Q: SGLang 找不到 layer_id？**
- SGLang 使用 `self.layer_id` 或从 `prefix` 提取
- 详见 [tensor-tags-guide.md](./tensor-tags-guide.md#layer_id-获取指南)

### MLA Attention 常见问题

**Q: MLA Attention 没有日志输出？**
- 原因: `self.prefix` 未定义或 `_td_extract_layer_idx` 函数缺失
- 解决:
  1. 在MLA Attention的`__init__`中添加`self.prefix = prefix`
  2. 确保导入了`_td_extract_layer_idx`函数
  3. 在forward中使用: `layer_idx = _td_extract_layer_idx(self.prefix) if hasattr(self, 'prefix') else 0`

**Q: MLA layer=- 显示？**
- 原因: `self.prefix` 未定义，layer_idx 使用默认值0
- 解决: 确保MLA Attention的`__init__`中有`self.prefix = prefix`

### vLLM 日志无输出排查

**症状**: 服务启动正常，推理正常，但日志中没有 `[TD_CMP]` 输出

**排查步骤**:

1. **检查环境变量是否正确设置**:
   ```bash
   # 在启动命令中确认包含:
   export TORCHDYNAMO_DISABLE=1
   export TENSOR_DUMP_ENABLE=1
   ```

2. **确认环境变量被传递到子进程**:
   ```bash
   # 检查 vLLM 进程的环境变量
   docker exec <container> cat /proc/<pid>/environ | tr '\0' '\n' | grep TORCH
   ```

3. **检查日志文件是否有 TD_CMP**:
   ```bash
   grep "TD_CMP" /tmp/vllm.log
   ```

4. **重启服务**: 修改环境变量后必须重启服务才能生效

**根本原因**: vLLM 使用 `torch.compile` 编译模型，编译过程中会优化掉 tensor dump 代码。即使代码已注入，如果不禁用 torch.compile，打点代码会被优化掉。

### SGLang 常见问题与解决方案

#### 1. layer_id 属性缺失

**问题**: `AttributeError: 'XXXDecoderLayer' object has no attribute 'layer_id'`

**原因**: `__init__` 接收 `layer_id` 参数但未保存为实例属性

**解决方案**:
```python
# 在类的 __init__ 中添加
self.layer_id = layer_id
```

#### 2. tp_rank 属性获取问题

**问题**: `AttributeError: 'XXXDecoderLayer' object has no attribute 'tp_rank'`

**原因**: DecoderLayer 等类本身没有 tp_rank，需从子模块获取

**解决方案**:
```python
# 错误写法
device=str(self.tp_rank)

# 正确写法 (从子模块获取)
device=str(self.self_attn.tp_rank)    # DecoderLayer
device=str(self.tp_rank)              # Attention (已有)
device=str(self.pp_group.rank_in_group)  # ForCausalLM
```

#### 3. 变量作用域问题 (UnboundLocalError)

**问题**: `UnboundLocalError: cannot access local variable 'xxx'`

**原因**: 打点代码放置在变量定义之前

**解决方案**:
```python
# 错误顺序 - 变量尚未定义
_td_compare_log(tag="xxx.pre", ...)
xxx = self.some_method(...)  # 变量在这里才定义!

# 正确顺序 - 先定义变量后打点
xxx = self.some_method(...)
_td_compare_log(tag="xxx.pre", ...)
_td_compare_log(tag="xxx", ...)
```

#### 4. 缩进错误

**问题**: `IndentationError`

**原因**: 注入代码缩进与周围代码不匹配

**解决方案**: 确保打点代码与同级别语句缩进一致

---

## 清理临时文件

```bash
# 清理临时脚本
rm -f /tmp/inject_*.py
rm -f /tmp/fix_*.py
rm -f /tmp/verify_*.py
rm -f /tmp/compare_*.py
rm -f /tmp/list_tags.py

# 清理容器目录
docker exec <container> rm -f /tmp/inject_*.py
```

---

## 参考文档

| 文档 | 职责 |
|------|------|
| **[tensor-tags-guide.md](./tensor-tags-guide.md)** | Tag 标准定义、命名规范、优先级建议 |
| **[framework-integration.md](./framework-integration.md)** | 详细注入手册、代码模板 |
| **[checklist.md](./checklist.md)** | 完整检查清单 |
| **[review-guide.md](./review-guide.md)** | 注入后检视指南 |

---
