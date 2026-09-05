# RoPE 迁移实例：GPU → NPU

## 触发条件

当 Agent 迁移 RoPE（Rotary Position Embedding，旋转位置编码）算子到 NPU 时，参考本文档。典型场景：

- 将 GPU 版 RoPE kernel 迁移到 Ascend 910_95
- 迁移使用 `tl.make_block_ptr` + `libdevice.pow` 的多维 Block Pointer kernel
- 遇到 RoPE 相关的 Grid 维度变化、Block Pointer shape 变化、autotune 配置等问题

---

## 核心知识：迁移 Diff 分析

### 源文件对照

| 版本 | 文件路径 |
|------|---------|
| GPU | `rope.py` |
| NPU | `rope_npu.py` |

### Diff 总览

| 改动类别 | GPU 版本 | NPU 版本 | 影响范围 |
|---------|---------|---------|---------|
| 导入 | `from triton.language.extra import libdevice` | `from triton.language.extra.cann import libdevice` | 全局 |
| 设备检测 | 本地定义 `is_hopper()` 使用 `torch.cuda` | 注释掉 `is_hopper()` / `is_ampere()` | autotune 配置 |
| Autotune 配置 | 基于 GPU 架构返回 1 个固定 Config | 列表推导式搜索空间，含 `multibuffer` 参数 | kernel 装饰器 |
| Grid 维度 | 3D Grid `(cdiv(max_len, BLOCK_M), bs, head)` | 1D Grid `(bs,)`，循环内遍历 seq_len 和 head | kernel 启动 |
| Block Pointer shape | 2D `(len, DIM)` | 3D `(len, head, DIM)` | 5 个 block_ptr |
| Block Pointer strides | 2D `(head * DIM, 1)` | 3D `(head * DIM, DIM, 1)` | 5 个 block_ptr |
| Block Pointer order | `(1, 0)` | `(2, 1, 0)` | 5 个 block_ptr |
| 旋转计算 | `x1 = x0 * cos - y0 * sin`（2D 广播） | `x1 = x0 * cos[:, None, :] - y0 * sin[:, None, :]`（3D 广播） | 核心计算 |
| libdevice.pow 路径 | `triton.language.extra.libdevice` | `triton.language.extra.cann.libdevice` | inv_freq 计算 |
| kernel 参数 | `head` 为普通参数 | `head: tl.constexpr` | kernel 签名 |
| 新增参数 | 无 | `max_seq_len`（普通参数） | kernel 签名 |
| 循环结构 | 无循环（3D Grid 天然并行） | `for start_m in range(tasks)` 循环遍历 seq_len | kernel 主体 |
| 测试代码 | `device` 未指定（CPU） | `device="npu"` + `torch_npu.profiler` | `__main__` |

---

## 代码模式：GPU → NPU 代码对比

### 1. 导入与 libdevice 路径

**GPU 版本：**
```python
import os
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from .utils import is_hopper, is_ampere
```

**NPU 版本：**
```python
import os
import torch
import torch_npu
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice
```

**迁移要点：**
- 添加 `import torch_npu`
- `libdevice` 路径从 `triton.language.extra.libdevice` 改为 `triton.language.extra.cann.libdevice`
- NPU 版本注释掉了 `is_hopper()` / `is_ampere()` 检测函数（不再需要 GPU 架构分支）
- 移除了 `.utils` 的相对导入

### 2. Autotune 配置变化

**GPU 版本 — 基于 GPU 架构返回单个 Config：**
```python
def is_hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major == 9

def get_configs():
    if is_hopper():
        return [triton.Config({"BLOCK_M": 32}, num_stages=2, num_warps=4)]
    elif is_ampere():
        return [triton.Config({"BLOCK_M": 32}, num_stages=1, num_warps=4)]
    else:
        return [triton.Config({"BLOCK_M": 16}, num_stages=3, num_warps=16)]

if os.environ.get("TRITON_DEBUG") == "1":
    configs = [triton.Config({"BLOCK_M": 4}, num_stages=1, num_warps=1)]

@triton.autotune(get_configs(), key=["DIM", "REVERSE"])
```

**NPU 版本 — 列表推导式搜索空间 + multibuffer 参数：**
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": BM, "multibuffer": MB})
        for BM in [100, 75, 50, 25, 10, 5, 2]
        for MB in [True, False]
    ],
    key=["DIM", "REVERSE"]
)
```

**迁移要点：**

| 维度 | GPU | NPU |
|------|-----|-----|
| Config 数量 | 1（按架构选择） | 7 x 2 = 14 |
| `num_stages` | 有（GPU 有效） | 无（NPU 忽略） |
| `num_warps` | 有（GPU 有效） | 无（NPU 忽略） |
| `multibuffer` | 无 | 有（NPU 特有，控制 ping-pong 流水线） |
| BLOCK_M 候选值 | 4/16/32 | 2/5/10/25/50/75/100 |
| key | `["DIM", "REVERSE"]` | `["DIM", "REVERSE"]`（保持一致） |

**注意**：NPU 版本的 BLOCK_M 候选值包含非 2 的幂次值（5, 25, 75, 100），这在 NPU 上可能导致对齐问题。建议优先使用 2 的幂次值（2, 4, 8, 16, 32, 64, 128）。

### 3. Kernel 签名变化

**GPU 版本：**
```python
@triton.jit
def rope_kernel(
    in_ptr, pos_ptr, cu_seqlens, out_ptr,
    head, base,
    DIM: tl.constexpr, REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr
):
```

**NPU 版本：**
```python
@triton.jit
def rope_kernel(
    in_ptr, pos_ptr, cu_seqlens, out_ptr,
    head: tl.constexpr, base,
    DIM: tl.constexpr,
    max_seq_len,
    REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
```

**迁移要点：**
- `head` 从普通参数变为 `tl.constexpr`：NPU 版本需要 `head` 在编译期确定，因为 Block Pointer 的 `block_shape` 依赖 `head` 值
- 新增 `max_seq_len` 参数：用于循环内计算任务数 `tasks = tl.cdiv(max_seq_len, BLOCK_M)`
- 参数顺序调整：`head` 移到 `tl.constexpr` 参数组

### 4. Grid 维度与循环结构变化（核心差异）

**GPU 版本 — 3D Grid，无循环：**
```python
def rope_impl(input, position, offset, max_len, base=10000., reverse=False):
    len, head, dim = input.size()
    out = input.new_empty(len, head, dim)
    bs = offset.size(0) - 1
    grid = lambda META: (triton.cdiv(max_len, META["BLOCK_M"]), bs, head)
    rope_kernel[grid](input, position, offset, out, head, base, dim, reverse)
    return out
```

**NPU 版本 — 1D Grid，循环遍历：**
```python
def rope_impl(input, position, offset, max_len, base=10000., reverse=False):
    len, head, dim = input.size()
    out = input.new_empty(len, head, dim)
    bs = offset.size(0) - 1
    grid = (bs,)
    rope_kernel[grid](input, position, offset, out, head, base, dim, max_len, reverse)
    return out
```

**迁移要点：**

| 维度 | GPU | NPU |
|------|-----|-----|
| Grid 维度 | 3D `(seq_blocks, bs, head)` | 1D `(bs,)` |
| 并行策略 | 每个 (seq_block, batch, head) 组合一个 program | 每个 batch 一个 program，循环内遍历 seq_len 和 head |
| Grid 大小 | `cdiv(max_len, BLOCK_M) * bs * head` | `bs` |
| 核数利用 | 可能超过物理核数，触发分批调度 | 核数等于 batch size，通常小于物理核数 |

**NPU 版本 kernel 内循环结构：**
```python
start_b = tl.program_id(0)
begin = tl.load(cu_seqlens + start_b)
len = tl.load(cu_seqlens + start_b + 1) - begin
tasks = tl.cdiv(max_seq_len, BLOCK_M)
for start_m in range(tasks):
    if start_m * BLOCK_M < len:
        # ... Block Pointer 操作
```

- 每个 program 处理一个 batch 的所有 seq_len blocks 和所有 head
- `if start_m * BLOCK_M < len` 用于跳过超出实际序列长度的 block
- 这种设计减少了 Grid 大小，避免 coreDim 超限

### 5. Block Pointer 维度变化（核心差异）

**GPU 版本 — 2D Block Pointer：**
```python
x0_block_ptr = tl.make_block_ptr(
    base = in_ptr + begin * head * DIM + start_h * DIM,
    shape = (len, DIM),
    strides = (head * DIM, 1),
    offsets = (start_m * BLOCK_M, 0),
    block_shape = (BLOCK_M, DIM // 2),
    order = (1, 0)
)
```

**NPU 版本 — 3D Block Pointer：**
```python
x0_block_ptr = tl.make_block_ptr(
    base = in_ptr + begin * head * DIM,
    shape = (len, head, DIM),
    strides = (head * DIM, DIM, 1),
    offsets = (start_m * BLOCK_M, 0, 0),
    block_shape = (BLOCK_M, head, DIM // 2),
    order = (2, 1, 0)
)
```

**迁移要点：**

| 维度 | GPU | NPU |
|------|-----|-----|
| shape | `(len, DIM)` | `(len, head, DIM)` |
| strides | `(head * DIM, 1)` | `(head * DIM, DIM, 1)` |
| offsets | `(start_m * BLOCK_M, 0)` | `(start_m * BLOCK_M, 0, 0)` |
| block_shape | `(BLOCK_M, DIM // 2)` | `(BLOCK_M, head, DIM // 2)` |
| order | `(1, 0)` | `(2, 1, 0)` |
| base 偏移 | `begin * head * DIM + start_h * DIM` | `begin * head * DIM` |

**关键变化解释：**
- GPU 版本通过 3D Grid 的 `start_h` 维度在 base 中偏移 `start_h * DIM`，每个 program 只处理一个 head
- NPU 版本将 head 维度纳入 Block Pointer 的 shape/strides/block_shape，一个 program 处理所有 head
- `order=(2, 1, 0)` 表示最内维度（DIM）连续，符合行优先内存布局
- NPU 版本的 `block_shape=(BLOCK_M, head, DIM // 2)` 一次性加载所有 head 的数据

### 6. 旋转计算广播维度变化

**GPU 版本 — 2D 广播：**
```python
offset_n = tl.arange(0, DIM // 2)
inv_freq = libdevice.pow(base, -2.0 / DIM * offset_n)
freqs = pos[:, None] * inv_freq[None, :]
sin = tl.sin(freqs)
cos = tl.cos(freqs)
if REVERSE:
    sin = -sin
x1 = x0 * cos - y0 * sin
y1 = x0 * sin + y0 * cos
```

**NPU 版本 — 3D 广播：**
```python
offset_n = tl.arange(0, DIM // 2)
inv_freq = libdevice.pow(base, -2.0 / DIM * offset_n)
freqs = pos[:, None] * inv_freq[None, :]
sin = tl.sin(freqs)
cos = tl.cos(freqs)
if REVERSE:
    sin = -sin
x1 = x0 * cos[:, None, :] - y0 * sin[:, None, :]
y1 = x0 * sin[:, None, :] + y0 * cos[:, None, :]
```

**迁移要点：**
- GPU 版本：`x0` 形状为 `(BLOCK_M, DIM//2)`，`cos` 形状为 `(BLOCK_M, DIM//2)`，2D 广播自动对齐
- NPU 版本：`x0` 形状为 `(BLOCK_M, head, DIM//2)`，`cos` 形状为 `(BLOCK_M, DIM//2)`，需要 `cos[:, None, :]` 扩展到 3D
- `sin` 和 `cos` 不依赖 head 维度，通过 `[:, None, :]` 在 head 维度广播

### 7. 测试代码改动

**GPU 版本：**
```python
v = rope_impl(input, pos, offset, MAX_LEN, base=2.)
pad_input = pad(input, size, MAX_LEN)
rope = RotaryPositionalEmbeddings(DIM, base=2)
roped = rope(pad_input.transpose(0,1)).transpose(0,1)
unpad_rope = unpad(roped, size)
print(torch.allclose(unpad_rope, v))
```

**NPU 版本：**
```python
DEVICE = torch.device("npu")
v = rope_impl(input.to(DEVICE), pos.to(DEVICE), offset.to(DEVICE), MAX_LEN, base=2.)

# NPU Profiler
with torch_npu.profiler.profile(...) as prof:
    for i in range(30):
        v = rope_impl(input.to(DEVICE), pos.to(DEVICE), offset.to(DEVICE), MAX_LEN, base=2.)
        torch.npu.synchronize()
        prof.step()

pad_input = pad(input, size, MAX_LEN)
rope = RotaryPositionalEmbeddings(DIM, base=2)
roped = rope(pad_input.transpose(0,1).to(DEVICE)).transpose(0,1)
unpad_rope = unpad(roped, size)
print(torch.allclose(unpad_rope, v, rtol=1e-3, atol=1e-3))
```

**迁移要点：**
- 输入数据需要 `.to(DEVICE)` 移到 NPU
- 添加 `torch_npu.profiler` 性能剖析
- 使用 `torch.npu.synchronize()` 确保 kernel 执行完成
- 容差从默认值放宽到 `rtol=1e-3, atol=1e-3`（BF16 精度下合理）

---

## 910_95 特别注意

### 1. Grid 大小与物理核数

NPU 版本 Grid 大小为 `bs`（batch size），通常远小于物理核数。如果 batch size 很小（如 1-4），大量 AI Core 将空闲。可考虑：

- 将 Grid 扩展为 `(bs * head,)` 或 `(bs * head * seq_blocks,)`，增加并行度
- 但需相应调整 kernel 内的 program_id 计算逻辑

### 2. Block Pointer 3D 化的 UB 压力

NPU 版本的 `block_shape=(BLOCK_M, head, DIM // 2)` 一次性加载所有 head 的数据。当 `head` 较大时（如 32 或 64），UB 压力显著增加：

```
UB 占用估算（FP16）：
BLOCK_M=100, head=32, DIM//2=32: 100 * 32 * 32 * 2B = 204800B ≈ 200KB
```

910_95 UB 为 256KB，上述配置接近上限。如果 head 更大，需要：
- 减小 BLOCK_M
- 或将 head 维度也做循环切分（类似 GPU 版本每个 program 处理一个 head）

### 3. multibuffer 与 UB 空间

910_95 默认 `multibuffer=False`（详见 [07-compile-params.md](07-compile-params.md)）。当 autotune 搜索到 `multibuffer=True` 时，UB 可用空间减半（128KB），可能导致 UB 溢出。建议：

- 对大 head 场景，优先使用 `multibuffer=False`
- 或减小 BLOCK_M 以适应双缓冲

### 4. BLOCK_M 非对齐值

当前 NPU 版本的 BLOCK_M 候选值包含 5, 25, 75, 100 等非 2 的幂次值。在 910_95 上：
- 32B 对齐要求：FP16 元素数需为 16 的倍数
- BLOCK_M=5 不满足对齐要求，可能导致性能下降或编译错误
- 建议替换为 `[2, 4, 8, 16, 32, 64, 128]`

### 5. libdevice.pow 精度

`libdevice.pow(base, -2.0 / DIM * offset_n)` 在 NPU 上使用 CANN libdevice 实现。910_95 的 Cube 单元不直接支持 pow 运算，该操作在 Vector 单元上执行，精度与 GPU 可能存在微小差异。建议：

- 使用 `rtol=1e-3, atol=1e-3` 的容差进行验证
- 如果精度不满足要求，可考虑预计算 inv_freq 表并传入 kernel

### 6. head 作为 tl.constexpr 的影响

将 `head` 声明为 `tl.constexpr` 意味着：
- 每个不同的 `head` 值会触发一次新的编译
- 如果模型中 head 数量固定（通常如此），编译开销可接受
- 如果需要支持动态 head 数量，应考虑将 head 从 constexpr 中移除，但这需要重新设计 Block Pointer 的 block_shape

---

## 相关文档链接

- [01-migration-overview.md](../docs_for_triton_agent/01-migration-overview.md) — GPU → NPU 迁移概览
- [02-api-differences.md](../docs_for_triton_agent/02-api-differences.md) — API 差异对照
- [03-tiling-and-grid.md](../docs_for_triton_agent/03-tiling-and-grid.md) — NPU Tiling 与 Grid 配置
- [07-compile-params.md](../docs_for_triton_agent/07-compile-params.md) — NPU 编译参数速查
- [10-autotune-on-npu.md](../docs_for_triton_agent/10-autotune-on-npu.md) — NPU Autotune 配置指南
- `rope.py` — GPU 版源码
- `rope_npu.py` — NPU 版源码
