# Tensor Dump 检视指南

本文档提供 tensor dump 打点注入完成后的系统化检视流程。

---

## 检视流程概览

```
┌─────────────────────────────────────────────────────────────┐
│                    注入完成 → 检视开始                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 语法验证                                          │
│  - py_compile 语法检查                                      │
│  - 确保无语法错误                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Tag 完整性检查                                    │
│  - 必须 Tags 对照检查                                       │
│  - 可选 Tags 统计                                          │
│  - 缺失 Tags 补充                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 属性可用性检查                                     │
│  - layer_id 属性                                          │
│  - tp_rank/device 获取方式                                  │
│  - 有问题需解决                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: 打点顺序检查                                       │
│  - 变量必须在打点前定义                                     │
│  - 无 UnboundLocalError                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: 服务启动验证                                       │
│  - 服务正常启动                                             │
│  - TD_CMP 日志输出                                          │
│  - 所有 Tags 都有日志                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    检视完成                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: 语法验证

### 检查命令

```bash
# 语法检查
docker exec <container> python -m py_compile /path/to/model.py

# 检查多个文件
docker exec <container> python -m py_compile /path/to/model1.py
docker exec <container> python -m py_compile /path/to/model2.py
```

### 通过标准

- 无输出表示语法正确
- 有 `SyntaxError` 或 `IndentationError` 表示失败

### 失败处理

| 错误类型 | 原因 | 解决 |
|----------|------|------|
| `IndentationError` | 缩进不匹配 | 检查周围代码缩进，调整注入代码缩进 |
| `SyntaxError` | 语法错误 | 检查括号、引号、关键字拼写 |

---

## Step 2: Tag 完整性检查

### 必须 Tags (⭐⭐⭐)

#### ForCausalLM (5个)

| Tag | 说明 | 检查命令 |
|-----|------|---------|
| `input_ids` | 输入 token | `grep '"input_ids"' model.py` |
| `positions` | 位置编码 | `grep '"positions"' model.py` |
| `final_hs` | 最终隐藏状态 | `grep '"final_hs"' model.py` |
| `logits` | 最终 logits | `grep '"logits"' model.py` |

#### Attention (2个)

| Tag | 说明 | 检查命令 |
|-----|------|---------|
| `attn.in` | Attention 输入 | `grep '"attn.in"' model.py` |
| `attn.out` | Attention 输出 | `grep '"attn.out"' model.py` |

#### Attention (MLA - DeepSeek/GLM-5) (2个)

| Tag | 说明 | 检查命令 |
|-----|------|---------|
| `attn.in` | MLA Attention 输入 | `grep '"attn.in"' model.py` |
| `attn.out` | MLA Attention 输出 | `grep '"attn.out"' model.py` |
| `self.prefix` | layer_idx 提取所需 | `grep "self.prefix = prefix" model.py` |

#### DecoderLayer (3个)

| Tag | 说明 | 检查命令 |
|-----|------|---------|
| `hs.out` | Layer 输出 | `grep '"hs.out"' model.py` |
| `prepare_mlp.out.residual` | 残差 (SGLang) | `grep '"prepare_mlp.out.residual"' model.py` |
| `hs.mlp.out` | MLP 输出 | `grep '"hs.mlp.out"' model.py` |

#### MoE (1个)

| Tag | 说明 | 检查命令 |
|-----|------|---------|
| `moe.router_logits` | 路由 logits | `grep '"moe.router_logits"' model.py` |

### 推荐 Tags (⭐⭐)

| Tag | 说明 | 检查命令 |
|-----|------|---------|
| `qkv_proj.out` | QKV 输出 | `grep '"qkv_proj.out"' model.py` |
| `rotary_emb.out` | RoPE 输出 | `grep '"rotary_emb.out"' model.py` |
| `mlp.act_output` | MLP 最终输出 | `grep '"mlp.act_output"' model.py` |
| `mlp.input` | MLP 输入 | `grep '"mlp.input"' model.py` |

### 快速 Tag 列表导出

```bash
# 导出所有 Tag
docker exec <container> grep 'tag="' /path/to/model.py | grep -o 'tag="[^"]*"' | sort | uniq

# 统计 Tag 数量
docker exec <container> grep 'tag="' /path/to/model.py | grep -o 'tag="[^"]*"' | sort | uniq | wc -l
```

### 缺失处理

如果有必须 Tags 缺失：
1. 确认缺失的 Tag 位置
2. 在正确位置添加打点代码
3. 重新进行语法验证

---

## Step 3: 属性可用性检查

### layer_id 检查

```bash
# 检查 layer_id 是否已定义
docker exec <container> grep "self.layer_id = layer_id" /path/to/model.py
```

**检查对象**:
- Attention 类
- DecoderLayer 类
- **MLP 类 (容易被忽略!)**

**问题**: 如果类有 `layer_id` 参数但无 `self.layer_idx` 定义
**解决**: 添加 `self.layer_idx = _extract_layer_index(prefix)`

### MLA Attention layer_idx 提取

MLA Attention 使用 `self.prefix` 获取 layer_idx，需要在 `__init__` 中保存：

```bash
# 检查 MLA Attention 的 self.prefix 是否定义
docker exec <container> grep "self.prefix = prefix" /path/to/model.py
```

**注入模板**:
```python
# 在 MLA Attention __init__ 中
self.prefix = prefix

# 在 MLA Attention forward 中
layer_idx = _td_extract_layer_idx(self.prefix) if hasattr(self, 'prefix') else 0
```

### tp_rank 检查

```bash
# 检查 tp_rank 使用
docker exec <container> grep "self.tp_rank\|self.self_attn.tp_rank" /path/to/model.py
```

**正确获取方式**:

| 类 | 正确写法 |
|----|---------|
| Attention | `self.tp_rank` |
| DecoderLayer | `self.self_attn.tp_rank` |
| ForCausalLM | `self.pp_group.rank_in_group` |

**问题**: 如果 `AttributeError: 'XXX' object has no attribute 'tp_rank'`
**解决**: 使用正确的属性获取方式

### device 参数检查

```bash
# 检查 device 参数是否正确
docker exec <container> grep "_td_compare_log" /path/to/model.py | head -10
```

确保打点调用中 `device` 参数正确传递。

---

## Step 4: 打点顺序检查

### 检查方法

查看源码，确认打点代码在变量定义**之后**：

```python
# 正确顺序 ✓
attn_output = self.attn(q, k, v, forward_batch)
_td_compare_log(tag="attn.out", ...)
_td_compare_log(tag="attn.out.pre_o", ...)

# 错误顺序 ✗
_td_compare_log(tag="attn.out", ...)  # attn_output 未定义!
attn_output = self.attn(q, k, v, forward_batch)
```

### 常见问题位置

| Tag | 依赖变量 | 检查点 |
|-----|---------|-------|
| `attn.out.pre_o` | `attn_output` | 确保在 `attn_output = self.attn(...)` 之后 |
| `attn.out` | `attn_output` | 确保在 `attn_output = self.attn(...)` 之后 |
| `qkv_proj.out` | `qkv` | 确保在 `qkv, _ = self.qkv_proj(...)` 之后 |

### 启动服务验证

如果存在 UnboundLocalError，服务启动时会报错：
```
UnboundLocalError: cannot access local variable 'xxx'
```

**解决**: 调整打点代码顺序

---

## Step 5: 服务启动验证

### 启动命令

```bash
export TENSOR_DUMP_ENABLE=1
export TENSOR_DUMP_DEVICE=npu:0

python -m sglang.launch_server \
    --model-path /path/to/model \
    --tp 4 \
    --device npu
```

### 验证清单

- [ ] 服务正常启动 (无报错)
- [ ] 日志中有 `[TD_CMP]` 输出
- [ ] 每层都有日志输出
- [ ] 所有必须 Tags 都有日志

### 日志检查命令

```bash
# 检查 TD_CMP 日志
docker exec <container> grep "TD_CMP" /tmp/sglang_server.log | head -50

# 统计各 Tag 出现次数
docker exec <container> grep "TD_CMP" /tmp/sglang_server.log | grep -o 'tag=[^ ]*' | sort | uniq -c

# 检查特定 Tag 是否存在
docker exec <container> grep "TD_CMP.*tag=attn.out" /tmp/sglang_server.log
```

### 常见启动问题

| 问题 | 原因 | 解决 |
|------|------|------|
| GPU 内存不平衡 | 其他进程占用 | 停止其他容器 |
| 启动超时 | 模型加载慢 | 增加等待时间 |
| **无 TD_CMP 日志** | **vLLM 未设置 TORCHDYNAMO_DISABLE=1** | **必须设置此环境变量！** |

### ⚠️ vLLM 特别注意: TORCHDYNAMO_DISABLE=1

**问题症状**: 服务启动正常，推理正常，但日志中没有 `[TD_CMP]` 输出

**根本原因**: vLLM 默认启用 `torch.compile`，会优化掉 tensor dump 代码

**解决方案**:
```bash
# 启动 vLLM 服务时必须设置
export TORCHDYNAMO_DISABLE=1
export TENSOR_DUMP_ENABLE=1
vllm serve /path/to/model --tensor-parallel-size 8
```

---

## 完整检视脚本

```bash
#!/bin/bash
# tensor_dump_review.sh - Tensor Dump 检视脚本

CONTAINER=$1
MODEL_PATH=$2

echo "=== Step 1: 语法验证 ==="
docker exec $CONTAINER python -m py_compile $MODEL_PATH
if [ $? -eq 0 ]; then
    echo "✓ 语法验证通过"
else
    echo "✗ 语法验证失败"
    exit 1
fi

echo ""
echo "=== Step 2: Tag 完整性检查 ==="
echo "必须 Tags 检查:"
for tag in "input_ids" "positions" "final_hs" "logits" "attn.in" "attn.out" "hs.out"; do
    count=$(docker exec $CONTAINER grep "\"$tag\"" $MODEL_PATH 2>/dev/null | wc -l)
    if [ $count -gt 0 ]; then
        echo "  ✓ $tag"
    else
        echo "  ✗ $tag (缺失!)"
    fi
done

echo ""
echo "=== Step 3: Tag 列表 ==="
docker exec $CONTAINER grep 'tag="' $MODEL_PATH | grep -o 'tag="[^"]*"' | sort | uniq

echo ""
echo "=== Step 4: layer_id 检查 ==="
count=$(docker exec $CONTAINER grep "self.layer_id = layer_id" $MODEL_PATH 2>/dev/null | wc -l)
if [ $count -gt 0 ]; then
    echo "✓ layer_id 属性已定义"
else
    echo "✗ layer_id 属性缺失"
fi

echo ""
echo "=== 检视完成 ==="
```

### 使用方法

```bash
chmod +x tensor_dump_review.sh
./tensor_dump_review.sh sglang_059 /path/to/model.py
```

---

## 检视记录模板

```
日期: YYYY-MM-DD
模型: <模型名称>
框架: [vLLM | SGLang]
注入文件: <文件路径>

Step 1: 语法验证
  结果: [通过 | 失败]
  错误: [如有]

Step 2: Tag 完整性
  必须 Tags: [N/M 通过]
  缺失 Tags: [列表]
  推荐 Tags: [N/M 通过]

Step 3: 属性可用性
  layer_id: [通过 | 问题]
  tp_rank: [通过 | 问题]

Step 4: 打点顺序
  结果: [通过 | 问题]

Step 5: 服务启动
  启动: [成功 | 失败]
  TD_CMP 日志: [有 | 无]

最终结论: [通过 | 需修复]
```

---

## 快速检视命令

```bash
# 一键检视 (假设容器名为 sglang_059)
echo "=== 语法验证 ===" && \
docker exec sglang_059 python -m py_compile /path/to/model.py && echo "✓ OK" && \
echo "" && \
echo "=== 必须 Tags ===" && \
for tag in input_ids positions final_hs logits attn.in attn.out hs.out; do
    docker exec sglang_059 grep "\"$tag\"" /path/to/model.py > /dev/null 2>&1 && echo "✓ $tag" || echo "✗ $tag"
done && \
echo "" && \
echo "=== Tag 列表 ===" && \
docker exec sglang_059 grep 'tag="' /path/to/model.py | grep -o 'tag="[^"]*"' | sort | uniq && \
echo "" && \
echo "=== layer_id ===" && \
docker exec sglang_059 grep "self.layer_id = layer_id" /path/to/model.py > /dev/null 2>&1 && echo "✓ 已定义" || echo "✗ 缺失"
```
