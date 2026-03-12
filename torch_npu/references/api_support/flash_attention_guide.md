# Flash Attention 适配指引

本文档详细说明如何在 torch_npu 中使用 Flash Attention 算子。

---

## 概述

Flash Attention 是一种高效的注意力机制实现，通过减少内存访问次数来加速训练和推理。torch_npu 提供了两个主要 API：

- **`torch_npu.npu_prompt_flash_attention`**：用于 prompt 阶段（预填充）
- **`torch_npu.npu_incre_flash_attention`**：用于增量推理（解码阶段）

---

## 设备兼容性

⚠️ **重要限制**：Flash Attention 算子**仅支持 Ascend910B 设备**。

**检查设备兼容性**：
```python
import torch
import torch_npu

if torch.npu.is_available():
    device_name = torch.npu.get_device_name(0)
    print(f"Device: {device_name}")
    # 确保设备名称包含 "910B"
    if "910B" not in device_name:
        print("Warning: Flash Attention only supported on Ascend910B")
```

---

## API 详细说明

### 1. npu_prompt_flash_attention

用于 **prompt 阶段**的 Flash Attention，适用于序列预填充场景。

#### API 签名

```python
torch_npu.npu_prompt_flash_attention(
    query,                  # [必填] 查询张量
    key,                    # [必填] 键张量
    value,                  # [必填] 值张量
    padding_mask=None,      # [选填] 填充掩码
    atten_mask=None,        # [选填] 注意力掩码
    pse_shift=None,         # [选填] 位置编码偏移
    actual_seq_lengths=None,# [选填] 实际序列长度
    num_heads=1,            # [必填] 注意力头数
    scale_value=1.0,        # [选填] 缩放因子
    input_layout="BSH",     # [必填] 输入布局
    sparse_mode=0,          # [选填] 稀疏模式
    pre_tokens=65535,       # [选填] 前向 token 数
    next_tokens=65535       # [选填] 后向 token 数
)
```

#### 输入布局

支持以下输入布局：
- **"BSH"**：`[batch, seq_len, num_heads * head_dim]`
- **"BNSD"**：`[batch, num_heads, seq_len, head_dim]`（推荐）

#### 完整示例

```python
import torch
import torch_npu
import math

# 1. 检查设备兼容性
assert torch.npu.is_available(), "NPU not available"
device_name = torch.npu.get_device_name(0)
assert "910B" in device_name, "Flash Attention requires Ascend910B"

# 2. 构造输入张量（BNSD 布局）
batch_size = 1
num_heads = 32
seq_len = 2048
head_dim = 128

query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16).npu()
key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16).npu()
value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16).npu()

# 3. 计算缩放因子
scale = 1.0 / math.sqrt(head_dim)

# 4. 调用 Flash Attention
output = torch_npu.npu_prompt_flash_attention(
    query=query,
    key=key,
    value=value,
    num_heads=num_heads,
    input_layout="BNSD",  # 推荐使用 BNSD 布局
    scale_value=scale,
    pre_tokens=65535,
    next_tokens=65535,
    sparse_mode=0
)

print(f"Output shape: {output.shape}")  # [1, 32, 2048, 128]
```

---

### 2. npu_incre_flash_attention

用于 **增量推理**的 Flash Attention，适用于自回归生成场景（如 GPT 解码）。

#### API 签名

```python
torch_npu.npu_incre_flash_attention(
    query,                      # [必填] 查询张量
    key,                        # [必填] 键张量
    value,                      # [必填] 值张量
    padding_mask=None,          # [选填] 填充掩码
    atten_mask=None,            # [选填] 注意力掩码
    pse_shift=None,             # [选填] 位置编码偏移
    actual_seq_lengths=None,    # [选填] 实际序列长度
    antiquant_scale=None,       # [选填] 反量化缩放
    antiquant_offset=None,      # [选填] 反量化偏移
    block_table=None,           # [选填] 块表
    dequant_scale1=None,        # [选填] 反量化缩放1
    quant_scale1=None,          # [选填] 量化缩放1
    dequant_scale2=None,        # [选填] 反量化缩放2
    quant_scale2=None,          # [选填] 量化缩放2
    quant_offset2=None,         # [选填] 量化偏移2
    kv_padding_size=None,       # [选填] KV 填充大小
    num_heads=1,                # [必填] 注意力头数
    scale_value=1.0,            # [选填] 缩放因子
    input_layout="BSH",         # [必填] 输入布局
    num_key_value_heads=0,      # [选填] KV 头数（GQA）
    block_size=0,               # [选填] 块大小
    inner_precise=1             # [选填] 内部精度
)
```

#### 布局转换

增量推理 API **要求 BSH 布局**，如果输入是 BNSD 需要先转换：

```python
def trans_BNSD2BSH(tensor: torch.Tensor):
    """
    将 BNSD 布局转换为 BSH 布局
    输入: [batch, num_heads, seq_len, head_dim]
    输出: [batch, seq_len, num_heads * head_dim]
    """
    tensor = torch.transpose(tensor, 1, 2)  # [batch, seq_len, num_heads, head_dim]
    tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
    return tensor
```

#### 完整示例

```python
import torch
import torch_npu
import math

# 1. 检查设备兼容性
assert torch.npu.is_available(), "NPU not available"
device_name = torch.npu.get_device_name(0)
assert "910B" in device_name, "Flash Attention requires Ascend910B"

# 2. 布局转换函数
def trans_BNSD2BSH(tensor: torch.Tensor):
    tensor = torch.transpose(tensor, 1, 2)
    tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
    return tensor

# 3. 构造输入（BNSD 布局）
batch_size = 1
num_heads = 32
query_len = 1  # 增量推理：每次只生成 1 个 token
kv_len = 2048  # KV cache 长度
head_dim = 128

q = torch.randn(batch_size, num_heads, query_len, head_dim, dtype=torch.float16).npu()
k = torch.randn(batch_size, num_heads, kv_len, head_dim, dtype=torch.float16).npu()
v = torch.randn(batch_size, num_heads, kv_len, head_dim, dtype=torch.float16).npu()

# 4. 转换为 BSH 布局
q_FA = trans_BNSD2BSH(q)  # [1, 1, 32*128] = [1, 1, 4096]
k_FA = trans_BNSD2BSH(k)  # [1, 2048, 4096]
v_FA = trans_BNSD2BSH(v)  # [1, 2048, 4096]

# 5. 计算缩放因子
scale = 1.0 / math.sqrt(head_dim)

# 6. 调用增量 Flash Attention
output = torch_npu.npu_incre_flash_attention(
    query=q_FA,
    key=k_FA,
    value=v_FA,
    num_heads=num_heads,
    input_layout="BSH",  # 必须使用 BSH 布局
    scale_value=scale
)

print(f"Output shape: {output.shape}")  # [1, 1, 4096]
```

---

## 使用场景对比

| 场景 | API | 输入布局 | 序列长度 | 典型用途 |
|------|-----|---------|---------|---------|
| **预填充** | `npu_prompt_flash_attention` | BNSD（推荐）| 固定 | Prompt 编码、BERT 前向 |
| **增量解码** | `npu_incre_flash_attention` | BSH（必须）| 动态 | GPT 生成、自回归解码 |

---

## 精度注意事项

### 1. 数据类型

- **推荐**：`torch.float16`（FP16）
- **支持**：`torch.bfloat16`（BF16，需硬件支持）

### 2. 精度损失

Flash Attention 使用 **FP16/BF16 累加**，可能与 FP32 参考实现有小误差：

```python
# 精度对比示例
def verify_fa_accuracy():
    # 构造输入
    q = torch.randn(1, 32, 128, 128, dtype=torch.float32)
    k = torch.randn(1, 32, 128, 128, dtype=torch.float32)
    v = torch.randn(1, 32, 128, 128, dtype=torch.float32)
    
    # CPU 参考（FP32）
    def standard_attention(q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(128)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
    
    ref_output = standard_attention(q, k, v)
    
    # NPU Flash Attention（FP16）
    q_npu = q.half().npu()
    k_npu = k.half().npu()
    v_npu = v.half().npu()
    
    fa_output = torch_npu.npu_prompt_flash_attention(
        q_npu, k_npu, v_npu,
        num_heads=32,
        input_layout="BNSD",
        scale_value=1.0 / math.sqrt(128)
    ).cpu().float()
    
    # 计算误差
    diff = torch.abs(ref_output - fa_output)
    rel_diff = diff / (torch.abs(ref_output) + 1e-8)
    
    print(f"Max abs diff: {diff.max().item():.6e}")  # 预期 < 1e-2
    print(f"Max rel diff: {rel_diff.max().item():.6e}")  # 预期 < 1e-2
    
    # Flash Attention 允许更大误差（< 1e-2）
    assert (rel_diff < 1e-2).all(), "Accuracy check failed"
```

### 3. 确定性计算

如果启用确定性计算模式，Flash Attention 会**退化为标准 attention 实现**：

```python
import torch_npu

# 启用确定性计算
torch_npu.npu.set_deterministic(True)

# 此时 Flash Attention 会使用标准实现，精度更高但速度更慢
output = torch_npu.npu_prompt_flash_attention(...)
```

---

## 性能优化建议

### 1. 选择合适的布局

- **Prompt 阶段**：使用 **BNSD** 布局（更直观，性能相当）
- **增量推理**：必须使用 **BSH** 布局（API 要求）

### 2. 序列长度对齐

建议将序列长度对齐到 **64 或 128 的倍数**，以获得最佳性能：

```python
def pad_to_multiple(tensor, multiple=64):
    """将序列长度对齐到 multiple 的倍数"""
    seq_len = tensor.shape[2]
    if seq_len % multiple != 0:
        pad_len = multiple - (seq_len % multiple)
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))
    return tensor
```

### 3. 批处理优化

- **小批量**：`batch_size=1` 时性能最优
- **大批量**：增加 `batch_size` 可提高吞吐量，但内存占用增加

### 4. 性能对比

| 实现方式 | 显存复杂度 | 时间复杂度 | 加速比 |
|---------|----------|----------|--------|
| 标准 Attention | O(N²) | O(N²) | 1x |
| Flash Attention | O(N) | O(N²) | 2-4x |

---

## 常见问题

### 问题 1：设备不支持

**错误信息**：
```
RuntimeError: Flash Attention only supported on Ascend910B
```

**解决**：
- 检查设备：`torch.npu.get_device_name(0)`
- 使用标准 attention 实现作为替代

### 问题 2：布局不匹配

**错误信息**：
```
RuntimeError: Invalid input layout
```

**解决**：
- `npu_prompt_flash_attention`：推荐使用 BNSD
- `npu_incre_flash_attention`：必须使用 BSH

### 问题 3：精度异常

**现象**：输出为 NaN 或 Inf

**解决**：
1. 检查输入数据范围（避免过大/过小值）
2. 使用 FP32 输入转换为 FP16（而非直接生成 FP16）
3. 启用确定性计算模式

### 问题 4：内存不足

**错误信息**：
```
RuntimeError: NPU out of memory
```

**解决**：
1. 减小 `batch_size`
2. 减小 `seq_len`
3. 使用梯度检查点（gradient checkpointing）

---

## 已弃用 API

❌ **不要使用以下 API**：

```python
# ❌ 非公开接口，不应直接调用
torch_npu._npu_flash_attention(...)

# ❌ 不存在
torch_npu.contrib.flash_attention(...)
```

✅ **使用公开接口**：

```python
# ✅ 正确
torch_npu.npu_prompt_flash_attention(...)
torch_npu.npu_incre_flash_attention(...)
```

---

## 完整示例：LLM 推理

```python
import torch
import torch_npu
import math

class FlashAttentionLLM:
    def __init__(self, num_heads=32, head_dim=128):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        
        # KV cache
        self.k_cache = None
        self.v_cache = None
    
    def prefill(self, query, key, value):
        """Prompt 阶段：使用 BNSD 布局"""
        output = torch_npu.npu_prompt_flash_attention(
            query=query,
            key=key,
            value=value,
            num_heads=self.num_heads,
            input_layout="BNSD",
            scale_value=self.scale
        )
        
        # 缓存 KV
        self.k_cache = key
        self.v_cache = value
        
        return output
    
    def decode(self, query):
        """增量推理：转换为 BSH 布局"""
        def trans_BNSD2BSH(tensor):
            tensor = torch.transpose(tensor, 1, 2)
            tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
            return tensor
        
        q_FA = trans_BNSD2BSH(query)
        k_FA = trans_BNSD2BSH(self.k_cache)
        v_FA = trans_BNSD2BSH(self.v_cache)
        
        output = torch_npu.npu_incre_flash_attention(
            query=q_FA,
            key=k_FA,
            value=v_FA,
            num_heads=self.num_heads,
            input_layout="BSH",
            scale_value=self.scale
        )
        
        # 更新 KV cache
        self.k_cache = torch.cat([self.k_cache, query], dim=2)
        self.v_cache = torch.cat([self.v_cache, query], dim=2)
        
        return output

# 使用示例
model = FlashAttentionLLM(num_heads=32, head_dim=128)

# Prefill
q = torch.randn(1, 32, 128, 128, dtype=torch.float16).npu()
k = torch.randn(1, 32, 128, 128, dtype=torch.float16).npu()
v = torch.randn(1, 32, 128, 128, dtype=torch.float16).npu()
output = model.prefill(q, k, v)

# Decode（自回归生成）
for _ in range(10):
    q_new = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
    output = model.decode(q_new)
```

---

## 参考文档

- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [torch_npu API 文档](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000451.html)
- [torch_npu GitHub 仓库](https://github.com/Ascend/pytorch)
- [源码参考 - test_prompt_flash_attention.py](https://github.com/Ascend/pytorch/blob/master/test/custom_ops/test_prompt_flash_attention.py)
- [源码参考 - test_incre_flash_attention.py](https://github.com/Ascend/pytorch/blob/master/test/custom_ops/test_incre_flash_attention.py)
