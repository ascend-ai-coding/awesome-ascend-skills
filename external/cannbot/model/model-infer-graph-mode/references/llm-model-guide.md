# LLM 模型图模式改造指南

> 本文档专门针对 **LLM 推理模型** 的图模式适配，提炼核心改造思路和最佳实践。
>
> **适用场景**：GPT、Llama、DeepSeek、Qwen 等 Transformer 架构的大语言模型推理。

---

## 核心原则

> **图模式适配的本质：将动态变化的东西提取为模型输入，模型内部尽量保证静态。**

图编译后会生成静态计算图，任何动态行为都可能导致图中断（graph break）或重编译。改造的关键是识别并隔离动态因素：

| 动态因素 | 问题表现 | 解决思路 |
|---------|---------|---------|
| **内存地址变化** | Guard 失败、重编译 | 预分配固定大小，原地更新 |
| **Shape 变化** | 图中断、多次编译 | 固定 shape 或通过参数控制 |
| **Python 控制流** | Graph Break | 使用 Tensor 操作或模式参数 |
| **`.item()` 调用** | 强制 Graph Break | 保持 Tensor 或外部传入 |

---

## LM 模型图模式适配工作流

> **重要**：对于 LLM 推理模型，必须区分 prefill 和 decode 阶段，通常只对 decode 做图模式优化。

```
LLM 图模式适配流程
    │
    ├─→ 1. 区分 Prefill/Decode
    │       ├── Prefill: 保持 Eager 模式（序列长度变化大）
    │       ├── Decode: 图模式优化（固定单 token 输入）
    │       └── 添加 is_prefill 参数
    │
    ├─→ 2. 改造模型代码
    │       ├── 预分配 KV Cache
    │       ├── 位置等变化参数提取出来作为输入传入
    │       └── 见下方各模块改造指南
    │
    ├─→ 3. 配置图模式
    │       └── 见推荐配置章节
    │
    └─→ 4. 验证性能
            ├── 正常 → 完成
            └── 性能劣化 → 定位重编译问题
```

---

## 重编译问题定位与解决

> **关键**：如果图模式性能劣化，**必须**定位是否发生了重编译！

### 定位重编译

```python
# 开启重编译日志
torch._logging.set_logs(recompiles=True)

# 运行模型
output = compiled_model(input)
# 如果发生重编译，会打印类似：
# [recompiles] Recompiling function <func_name> for reason: <reason>
```

### 重编译解决方案

```
dynamic=False: 检测到重编译
    │
    └─→ 分析重编译原因
            ├── 固定 shape 但仍重编译 → dynamic=False + skip_guard_eval_unsafe=True
            └── 输入 shape 变化 → dynamic=True
```

---

## 模块改造指南

### 1. KV Cache 模块改造

**改造目标**：消除 KV Cache 动态扩展导致的 shape 变化，实现固定大小 cache 的原地更新。

**核心思路**：

1. **预分配策略**：在模型初始化时分配固定大小的 cache
2. **原地更新原则**：使用原地更新算子写入新值，避免重新分配
3. **有效长度控制**：通过参数控制实际参与计算的长度
4. **返回优化**：图模式下不返回 KV cache（已原地更新）

**问题模式 vs 改造模式**：

```python
# 问题模式：动态扩展 KV cache
key = torch.cat([past_key, new_key], dim=1)  # shape 变化！

# 改造模式：固定大小预分配 + 原地更新
# 1. 初始化时预分配
def _init_kv_cache(self, batch_size, max_seq_len, device):
    cache_shape = (batch_size, 1, max_seq_len, head_dim)
    self.kv_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

# 2. forward 中原地更新
def forward(self, ..., kv_len, past_key_value):
    torch_npu.scatter_update_(past_key_cache, kv_len, new_key_states, dim=-2)
```

**常见问题**：

| 问题现象 | 根因 | 解决方案 |
|---------|------|---------|
| 每次 decode 触发重编译 | `torch.cat` 扩展 KV cache | 预分配固定大小，原地更新 |
| 内存占用过大 | 预分配浪费 | 结合 PagedAttention 按 block 管理 |
| 返回 KV cache 开销大 | 图模式下返回大量 tensor | 已原地更新，无需返回 |

---

### 2. Rotary Embedding 模块改造

**改造目标**：消除动态计算，实现静态图 cos/sin 查询。

> 如果已经使用了融合算子，没有触发静态图的限制，则无需改造，否则需要进行改造

**核心思路**：

1. **预计算 cos/sin**：初始化时计算所有位置值并缓存
2. **索引查询**：通过 `index_select` 或切片获取
3. **外层计算优化**：在模型外层统一计算，传入各层

**实现示意**：

```python
def forward(self, x, kv_len, is_prefill=True):
    if is_prefill:
        cos = self.cos_cached[:seq_len]  # prefill：切片
    else:
        cos = torch.index_select(self.cos_cached, dim=0, index=kv_len.view(-1))  # decode：索引
    return cos.to(x.dtype), sin.to(x.dtype)
```

---

### 3. Attention 模块改造

**改造目标**：使 Attention 计算图模式友好，支持 Flash Attention 等融合算子。

**核心思路**：

1. **使用 NPU 原生融合算子**：优先使用 NPU 提供的融合 attention 算子
2. **有效长度参数化**：通过参数控制，避免大规模 attention mask
3. **区分 prefill/decode**：使用模式参数选择不同计算路径

> **提示**：可使用 `model-infer-fusion` skill 获取融合算子指导。

---

### 4. Buffer/Parameter 模块改造

**改造目标**：避免 buffer/parameter 地址变化触发 guard 失败。

**核心思路**：

1. **预分配策略**：初始化时分配最大可能大小
2. **原地更新原则**：使用 `copy_()`、`fill_()` 等原地操作
3. **只读访问**：通过 `index_select`、切片等只读方式访问

---

### 5. 动态信息外部化设计

**改造目标**：将动态变化的信息从模型内部移到输入参数。

| 动态信息 | 内部计算 | 外部传入 |
|---------|--------------|--------------|
| 位置索引 | `position_ids = torch.arange(seq_len)` | 作为参数传入 |
| 序列长度 | `seq_len = hidden_states.size(1)` | `actual_seq_lengths` 参数 |
| 写入位置 | 内部计算 `kv_len` | `kv_len` 参数 |
| 模式切换 | 内部判断 | `is_prefill` 参数 |

**forward 签名设计参考**：

```python
def forward(
    self,
    input_ids: torch.LongTensor,
    # 位置相关（Tensor 形式支持图追踪）
    position_ids: Optional[torch.LongTensor] = None,
    kv_len: Optional[torch.IntTensor] = None,  # KV 写入位置

    # 序列长度（List[int] 传给 NPU 算子）
    actual_seq_lengths_kv: Optional[List[int]] = None,
    actual_seq_lengths_q: Optional[List[int]] = None,

    # 模式控制
    is_prefill: bool = False,

    # KV Cache
    past_key_values: Optional[Tuple[torch.Tensor]] = None,

    # 预计算的 cos/sin（避免重复计算）
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    ...
):
    pass
```

---

## 注意事项

**不要在模型 forward 中使用 `.item()`**——将 Tensor 转换为 Python 标量会触发 Graph Break。

```python
# 错误写法 - 会导致 Graph Break
max_pos_id = position_ids.max().item() + 1

# 正确写法 - 使用静态参数或预计算
max_pos_id = MAX_SEQ_LEN  # 作为常量传入
```

---

## 推荐配置

### npugraph_ex 后端（推荐用于 LLM Decode）

```python
import torch
import torch_npu

model = YourModel().npu()
opt_model = torch.compile(
    model,
    backend="npugraph_ex",
    fullgraph=True,
    dynamic=False,  # LLM decode 固定 shape
    options={
        # FX图优化
        "inplace_pass": True,
        "input_inplace_pass": True,
        "pattern_fusion_pass": True,
        # 内存优化
        "reuse_graph_pool_in_same_fx": True,
        "clone_input": True,
        "clone_output": False,
        # 性能优化
        "remove_noop_ops": True,
    }
)
```

### GE 图模式

```python
import torch
import torch_npu
import torchair
from torchair import patch_for_hcom

patch_for_hcom()  # 集合通信入图（有 TP/EP 并行时需调用）

config = torchair.CompilerConfig()
# 根据需要配置 inference_config, ge_config 等
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

---

## 区分 Prefill/Decode 实践指南

> **核心思想**：为模型添加独立的 `prefill()` 和 `decode()` 方法，通过 `is_prefill` 参数区分执行路径。

### 代码示例

```python
# === 模型层 ===
class MyModelForCausalLM(nn.Module):
    def forward(self, input_ids, position_ids, past_key_values, is_prefill=False, **kwargs):
        # is_prefill 控制不同执行路径
        if is_prefill:
            # Prefill 专属：SP all-gather、取最后 token logits
            pass
        else:
            # Decode 专属：多流并行、原地更新 KV cache
            pass
        return logits

    def prefill(self, **kwargs):
        return self.forward(is_prefill=True, **kwargs)

    def decode(self, **kwargs):
        return self.forward(is_prefill=False, **kwargs)

# === Runner 层 ===
class MyRunner:
    def model_inference(self, model_inputs, is_prefill=False):
        if is_prefill:
            return self.model.prefill(**model_inputs)
        else:
            return self.model.decode(**model_inputs)  # 适合图模式
```

### Prefill vs Decode 关键差异

| 组件 | Prefill | Decode |
|------|---------|--------|
| **输入** | 变长序列（提示词） | 固定数量 token |
| **图模式** | 通常不图化 | 适合 `dynamic=False` 图模式 |

---

## 相关文档

- **npugraph_ex 详细指南**：`npugraph_ex-guide.md`
- **GE 图模式详细指南**：`ge-graph-guide.md`