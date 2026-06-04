---
name: inference-precision-tensor-dump-compare
description: 模型层 Tensor 打点与精度对比工具。用于在模型 forward 过程中捕获模型各层中间 tensor，实现 GPU/NPU 精度对比调试。支持 vLLM、SGLang 推理框架。When to use： When you need to debug precision issues between GPU and NPU,or validate layer-wise tensor outputs during inference.
keywords:
    - tensor dump
    - tensor打点
    - GPU NPU 对比
    - precision debugging
    - forward dump
    - 层tensor
    - vLLM
    - SGLang
    - 精度调试
---

# Tensor Dump Compare

在模型 forward 过程中打点，捕获中间 tensor，用于 GPU/NPU 精度对比。

**支持框架**: vLLM, SGLang

---

## ⚠️ 必须环境变量

### vLLM 框架

| 变量 | 必须 | 说明 |
|------|------|------|
| `TORCHDYNAMO_DISABLE=1` | ✅ **必须** | 禁用 torch.compile，否则 tensor dump 日志无法输出 |
| `TENSOR_DUMP_ENABLE=1` | ✅ 必须 | 启用 tensor dump |
| `ASCEND_RT_VISIBLE_DEVICES` | 推荐 | 指定 NPU 设备，如 `0,1,2,3,4,5,6,7` |

**⚠️ 重要**: vLLM 默认启用 torch.compile，会导致 tensor dump 日志不输出。**必须设置 `TORCHDYNAMO_DISABLE=1`！**

```bash
# vLLM 启动命令 (必须)
export TORCHDYNAMO_DISABLE=1
export TENSOR_DUMP_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve /path/to/model --tensor-parallel-size 8
```

### SGLang 框架

| 变量 | 必须 | 说明 |
|------|------|------|
| `TENSOR_DUMP_ENABLE=1` | ✅ 必须 | 启用 tensor dump |

```bash
# SGLang 启动命令
export TENSOR_DUMP_ENABLE=1
export TENSOR_DUMP_DEVICE=npu:0

python -m sglang.launch_server --model-path /path/to/model --tp 8
```

### 环境变量速查表

| 框架 | 必须设置 | 变量 |
|------|---------|------|
| **vLLM** | ✅ | `TORCHDYNAMO_DISABLE=1` |
| **vLLM** | ✅ | `TENSOR_DUMP_ENABLE=1` |
| **SGLang** | ✅ | `TENSOR_DUMP_ENABLE=1` |
| 通用 | 可选 | `TENSOR_DUMP_DEVICE=npu:0` |
| 通用 | 可选 | `TENSOR_DUMP_LAYERS=0,1,2` |
| 通用 | 可选 | `TENSOR_DUMP_TAGS=hs.in,hs.out` |

---

## 框架选择

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| **vLLM** | 单一 forward，TP 内联 allreduce | DeepSeek, Qwen, LLaMA |
| **SGLang** | 分离式 forward (prepare/core)，LayerCommunicator | MiMo-v2, Step3.5, Qwen3.5 |

---

## ⚠️ 工作流程 (必须遵循)

**注入完成后，必须立即执行自检，不得跳过！**

> **⚠️ 重要提示**: 即使使用自动注入脚本，也必须根据Tag自检表进行自检！自动注入可能因代码模式匹配失败而遗漏部分 Tags。

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. 分析模型  │ -> │  2. 注入代码  │ -> │  3. 自检验证  │ -> │  4. 启动服务  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                               │
                                               ▼
                                    ┌───────────────────────────────┐
                                    │  失败或者缺少Tag则修复并重新注入 │
                                    └───────────────────────────────┘
```

### 自检步骤 (注入后必做)

| 步骤 | 命令 | 通过标准 |
|------|------|---------|
| **1. 语法验证** | `python -m py_compile model.py` | 无错误输出 |
| **2. 打点统计** | `grep -c '_td_compare_log' model.py` | 数量符合预期 |
| **3. Tag 列表** | `grep '_td_compare_log' model.py \| grep '"' \| sort -u` | 必须 Tags 全部存在 |
| **4. layer_idx** | `grep 'self.layer_idx = ' model.py` | 各模块都已定义 |
| **5. 启动验证** | 服务启动后 `grep TD_CMP log` | 日志中有输出 |

### 必须 Tags (⭐⭐⭐)

注入后必须验证以下 Tags 存在：
| 模块 | 必须Tags | layer_idx 检查 |
|------|------|------|
| 模型级别 | `input_ids`, `positions`, `hs.in`, `final_hs`, `logits` | N/A |
| Attention (标准) | `attn.in`, `qkv_proj.out`, `rotary_emb.out`, `attn.out`, `q_norm.out`, `k_norm.out`, `attn.out.pre_o` | `grep "self.layer_idx" model.py` |
| Attention (MLA) | `attn.in`, `attn.out` | 需要 `_td_extract_layer_idx(self.prefix)` |
| DecoderLayer | `hs.in`, `hs.ln1`, `hs.post_attn`, `hs.ln2`, `hs.mlp.out`, `hs.out` | `grep "self.layer_idx" model.py` |
| MLP | `mlp.input`, `mlp.gate_up`, `mlp.act_out`, `mlp.output` | `grep "self.layer_idx" model.py` |
| MoE | `moe.hidden_states_input`, `moe.router_logits`, `moe.shared_output`, `moe.post_reduce`, `moe.output` | `grep "self.layer_idx" model.py` |

> 完整 Tag 列表请参阅 [references/tensor-tags-guide.md](./references/tensor-tags-guide.md)

### 自检验证命令

```bash
# 1. 检查各模块 layer_idx 是否都已定义 (所有模块都需要!)
for cls in Attention DecoderLayer MLP; do
    grep "class $cls" model.py > /dev/null && \
    echo "$cls: $(grep -c 'self.layer_idx = ' model.py) 次"
done

# 2. 检查必须 Tags
for tag in input_ids hs.in final_hs logits attn.in attn.out hs.out mlp.input mlp.output; do
    grep "\"$tag\"" model.py > /dev/null && echo "✓ $tag" || echo "✗ $tag 缺失"
done
```

### 自检失败处理

| 问题 | 原因 | 解决 |
|------|------|------|
| 语法错误 | 缩进/引号/括号错误 | 检查注入代码格式 |
| Tag 缺失 | 替换字符串不匹配 | 检查原始代码，调整替换模式 |
| layer_idx 未定义 | 类无 layer_idx 属性 | 添加 `self.layer_idx = _extract_layer_index(prefix)` |
| **日志无输出** | **vLLM 未设置 TORCHDYNAMO_DISABLE=1** | **必须设置此环境变量！** |
| **MLP layer=-** | **MLP 未定义 layer_idx** | **MLP 也需要 `self.layer_idx = _extract_layer_index(prefix)`** |

### ⚠️ 特别注意: vLLM 必须设置 TORCHDYNAMO_DISABLE=1

**原因**: vLLM 默认启用 `torch.compile`，会优化掉 tensor dump 代码，导致日志不输出。

**症状**: 服务启动正常，推理正常，但日志中没有 `[TD_CMP]` 输出。

**解决方案**: 启动命令中必须包含 `export TORCHDYNAMO_DISABLE=1`

```bash
# 正确 ✓
export TORCHDYNAMO_DISABLE=1
export TENSOR_DUMP_ENABLE=1
vllm serve /path/to/model

# 错误 ✗ (日志无输出)
export TENSOR_DUMP_ENABLE=1
vllm serve /path/to/model
```

**详见**: `references/review-guide.md`

---

## 通用环境变量

| 变量 | 默认 | 说明 |
|-----|------|------|
| `TENSOR_DUMP_ENABLE` | 0 | 主开关，设为 1 启用 |
| `TENSOR_DUMP_DEVICE` | npu:0 | **目标设备** |
| `TENSOR_DUMP_LAYERS` | all | 过滤层，如 `0,1,2` |
| `TENSOR_DUMP_TAGS` | all | 过滤 tags |

---

## 设备过滤

通过 `TENSOR_DUMP_DEVICE` 环境变量控制：

| 设置 | 效果 |
|------|------|
| `npu:1` | 只打点 npu:1 |
| `cuda:0` | 只打点 cuda:0 |
| `npu:*` | 打点所有 NPU 设备 |
| `cuda:*` | 打点所有 GPU 设备 |
| `*` | 不进行设备过滤 |

---

## 日志格式

```
[TD_CMP] layer=0 attn.out device=npu:1 shape=[2048,5120] dtype=bfloat16 l1_norm=1.152184e+05
```

| 字段 | 说明 |
|------|------|
| `layer` | 层索引（- 表示模型级别） |
| `attn.out` | Tag 名称 |
| `device` | 设备 |
| `shape` | tensor 形状 |
| `dtype` | 数据类型 |
| `l1_norm` | L1 范数 |

---

## 快速开始

### ⚠️ vLLM 必须设置 TORCHDYNAMO_DISABLE=1

**原因**: vLLM 默认启用 `torch.compile`，会优化掉 tensor dump 代码，导致日志不输出。

### vLLM

```bash
# 1. 分析模型
grep -E "class.*\(|use_moe|gate\(" /path/to/model.py

# 2. 注入
python scripts/inject_tensor_dump.py --framework vllm --model-path /path/to/model.py --backup

# 3. 自检验证 (必须!)
#    - 语法检查
python -m py_compile /path/to/model.py
#    - 检查 Tag 数量
grep -c '_td_compare_log' /path/to/model.py
#    - 检查必须 Tags 存在
for tag in input_ids hs.in final_hs logits attn.in attn.out hs.out mlp.output; do
    grep "\"$tag\"" /path/to/model.py > /dev/null && echo "✓ $tag" || echo "✗ $tag 缺失"
done

# 4. 启动服务 (必须设置 TORCHDYNAMO_DISABLE=1!)
export TORCHDYNAMO_DISABLE=1  # 必须设置，否则日志无输出!
export TENSOR_DUMP_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
vllm serve /path/to/model --tensor-parallel-size 8

# 5. 验证日志输出
grep "\[TD_CMP\]" /tmp/vllm.log | head -20
```

### SGLang

```bash
# 1. 分析模型
grep -E "class.*\(|forward_prepare|forward_core|layer_communicator" /path/to/model.py

# 2. 注入
python scripts/inject_tensor_dump.py --framework sglang --model-path /path/to/model.py --backup

# 3. 自检验证 (必须!)
python -m py_compile /path/to/model.py
grep '_td_compare_log' /path/to/model.py | grep '"' | sort | uniq
grep "self.layer_id = layer_id" /path/to/model.py

# 4. 启动
export TENSOR_DUMP_ENABLE=1
export MIMO_COMPARE=1
python -m sglang.launch_server --model-path /path/to/model

# 5. 验证
grep "\[TD_CMP\]" logs/sglang.log | head -20
```

---

## 框架差异速查

| 特性 | vLLM | SGLang |
|------|------|--------|
| 层 ID 属性 | `self.layer_idx` (通过 `extract_layer_index(prefix)`) | `self.layer_id` |
| Attention | 单一 forward / **MLA Attention** | forward_prepare/forward_core (分离式) |
| 残差管理 | 内联 `residual = hidden_states + x` | LayerCommunicator |
| MoE 路径 | 单一 forward_normal | forward_normal + forward_deepep |
| AllReduce | `maybe_all_reduce_tensor_model_parallel` | `tensor_model_parallel_all_reduce` |

### Attention 类型说明

| 类型 | 模型示例 | 说明 |
|------|---------|------|
| **标准Attention** | Qwen3, LLaMA | 使用 `qkv_proj`, `o_proj` |
| **MLA Attention** | DeepSeek-V2/V3, GLM-4.5/4.6/4.7, GLM-5 | 使用 `MultiHeadLatentAttentionWrapper` |

### layer_idx 的重要性

| 场景 | 有 layer_idx | 无 layer_idx |
|------|-------------|-------------|
| 多层对比 | ✅ 可区分第几层 | ❌ 无法区分 |
| 定位问题层 | ✅ 精准定位到第N层 | ❌ 无法定位 |
| GPU/NPU 对比 | ✅ 可逐层对比 | ❌ 无法逐层对比 |

**注意**: MLP/Attention/DecoderLayer 等模块需要通过 `extract_layer_index()` 提取层索引，否则日志中 `layer=None`，无法定位问题。

详见: **`references/framework-integration.md`** - 框架集成指南

---

## 优先级建议

| 优先级 | Tags | 使用场景 |
|--------|------|---------|
| ⭐⭐⭐ 必须 | `hs.in`, `hs.out`, `attn.out`, `mlp.output`, `logits` | 快速定位精度问题 |
| ⭐⭐ 重要 | `input_ids`, `final_hs`, `moe.router_logits`, `moe.output` | 详细分析流程 |
| ⭐ 有助 | `qkv_proj.out`, `rotary_emb.out`, `prepare_mlp.out.residual` | 深度调试 |

> 完整 Tag 列表请参阅 [references/tensor-tags-guide.md](./references/tensor-tags-guide.md)

---

## Scripts

| 脚本 | 功能 |
|------|------|
| `inject_tensor_dump.py` | vLLM 自动化注入 |
| `inject_tensor_dump_sglang.py` | SGLang 专用注入 |
| `compare_tags.py` | 注入 vs 参考 Tag 对比 |
| `verify_tags.py` | Tag 完整性验证 |
| `analyze_logs.py` | 日志分析 |
| `rollback_tensor_dump.py` | 回滚 |

### 使用示例

```bash
# vLLM
python scripts/inject_tensor_dump.py --framework vllm --model-path /path/to/model.py --backup

# SGLang
python scripts/inject_tensor_dump_sglang.py --model-path /path/to/model.py --backup

# 验证
python scripts/verify_tags.py /path/to/model.py --model-type xxx

```

---

## 参考文件

| 参考文件 | 内容 | 使用场景 |
|---------|------|---------|
| **[tensor-tags-guide.md](./references/tensor-tags-guide.md)** | Tag 标准参考 | Tag 命名规范、优先级、完整列表 |
| **[framework-integration.md](./references/framework-integration.md)** | 注入手册 | vLLM/SGLang 详细代码模板 |
| **[verification-debugging.md](./references/verification-debugging.md)** | 验证调试 | 语法检查、错误排查 |
| **[checklist.md](./references/checklist.md)** | 检查清单 | 注入前准备检查 |
| **[review-guide.md](./references/review-guide.md)** | 检视指南 | 注入后自检步骤 |
