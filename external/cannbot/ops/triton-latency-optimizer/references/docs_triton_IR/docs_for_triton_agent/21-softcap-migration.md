# Softcap 迁移实例：GPU → NPU

## 触发条件

当 Agent 迁移 Softcap 算子（注意力分数软截断 `softcap * tanh(x / softcap)`）到 NPU 时，参考本文档。典型场景：

- 将 GPU 版 Softcap kernel 迁移到 Ascend 910_95
- 迁移包含前向（fwd）和反向（bwd）两个子 kernel 的逐元素算子
- 遇到 Softcap 相关的 Autotune 重构、BLOCK_NUM 多块循环、VF 融合等迁移问题

---

## 核心知识：迁移 Diff 分析

### 源文件对照

| 版本 | 文件路径 |
|------|---------|
| GPU | `softcap.py` |
| NPU | `softcap_npu.py` |

### Diff 总览

| 改动类别 | GPU 版本 | NPU 版本 | 影响范围 |
|---------|---------|---------|---------|
| 导入 | `from triton.language.extra import libdevice` + `from maybe_triton_jit import maybe_triton_jit` | `import torch_npu` + `from triton.language.extra.cann import libdevice` | 全局 |
| 设备检测 | `from .utils import is_hopper, is_ampere` | `from utils import is_hopper, is_ampere`（注释掉旧 Config，未实际使用） | autotune 配置 |
| Autotune 装饰器 | `@maybe_triton_jit(configs=..., key=[])` | `@triton.autotune(configs=[...], key=["n_elements"])` | fwd/bwd kernel |
| Autotune 配置 | 基于 GPU 架构返回 1 个固定 Config（含 `num_stages`/`num_warps`） | 列表推导式搜索空间，含 `multibuffer` 和 `BLOCK_NUM` 参数 | fwd/bwd kernel |
| Kernel 参数 | `BLOCK_SIZE: tl.constexpr` | `BLOCK_SIZE: tl.constexpr` + `BLOCK_NUM: tl.constexpr` | fwd/bwd kernel |
| 核内循环 | 无循环，每个 program 处理一个 `BLOCK_SIZE` 块 | `for i in range(BLOCK_NUM)` 多块循环 | fwd/bwd kernel |
| tanh 函数 | `libdevice.tanh`（`triton.language.extra`） | `libdevice.tanh`（`triton.language.extra.cann`） | fwd/bwd kernel |
| 编译参数 | 无 | `debug=True`、`enable_vf_fusion=True` | fwd/bwd kernel 启动 |
| Grid 计算 | `triton.cdiv(numel, META["BLOCK_SIZE"])` | `triton.cdiv(numel, META["BLOCK_SIZE"] * META["BLOCK_NUM"])` | fwd/bwd kernel 启动 |
| 测试代码 | `device="cuda"` + if/else 打印 | `device="npu"` + `torch_npu.profiler` + assert 断言 | `__main__` |

---

## 代码模式：GPU → NPU 代码对比

### 1. 导入差异

**GPU 版本：**
```python
import torch
import triton
import triton.language as tl
from maybe_triton_jit import maybe_triton_jit
from triton.language.extra import libdevice
from .utils import is_hopper, is_ampere
```

**NPU 版本：**
```python
import torch
import torch_npu
import triton
import triton.language as tl
# from maybe_triton_jit import maybe_triton_jit
from triton.language.extra.cann import libdevice
# import triton.language.extra.ascend.libdevice as libdevice
from utils import is_hopper, is_ampere
```

**迁移要点：**
- GPU 版本使用 `from triton.language.extra import libdevice`，NPU 版本改为 `from triton.language.extra.cann import libdevice`（CANN 后端路径）
- NPU 版本新增 `import torch_npu`，用于 NPU 设备和 profiler
- GPU 版本使用 `from maybe_triton_jit import maybe_triton_jit`（自定义装饰器），NPU 版本注释掉，改用标准 `@triton.autotune`
- GPU 版本使用相对导入 `from .utils import ...`，NPU 版本改为 `from utils import ...`（绝对导入）

### 2. Autotune 装饰器与配置变化

**GPU 版本 — `@maybe_triton_jit` + 固定 Config：**
```python
def get_fwd_config():
    if is_hopper():
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=2, num_warps=4)]
    elif is_ampere():
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=4, num_warps=8)]
    else:
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=2, num_warps=16)]

@maybe_triton_jit(
    configs=get_fwd_config(),
    key=[],
)
@triton.jit
def softcap_fwd_kernel(...):
```

**NPU 版本 — `@triton.autotune` + 搜索空间：**
```python
# 旧的 GPU 架构分支 Config 被注释掉
# @triton.autotune(
#     configs=get_fwd_config(),
#     key=[],
# )

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": BM, "multibuffer": MB, "BLOCK_NUM": BN})
        for BM in [18725, 16384, 8192, 4096, 2048, 1024, 512]
        for MB in [True, False]
        for BN in [64, 32, 16, 8, 4, 2, 1]
    ],
    key=["n_elements"]
)
@triton.jit
def softcap_fwd_kernel(...):
```

**迁移要点：**

| 维度 | GPU 版本 | NPU 版本 |
|------|---------|---------|
| 装饰器 | `@maybe_triton_jit`（自定义） | `@triton.autotune`（标准） |
| Config 来源 | 函数返回，基于 GPU 架构分支 | 列表推导式，搜索空间 |
| Config 数量 | 1（按架构选一个） | 7 x 2 x 7 = 98 |
| BLOCK_SIZE 候选值 | 固定 2048 | [512, 1024, 2048, 4096, 8192, 16384, 18725] |
| GPU 专属参数 | `num_stages`、`num_warps` | 无（NPU 不支持） |
| NPU 专属参数 | 无 | `multibuffer`、`BLOCK_NUM` |
| key | `[]`（空） | `["n_elements"]`（按数据量搜索） |

**关键变化解释：**
- `num_stages` 和 `num_warps` 是 GPU 专属参数，NPU 不支持，必须移除
- `multibuffer` 是 NPU 专属的编译器级别双缓冲控制参数
- `BLOCK_NUM` 是 NPU 多块循环参数，控制每个 program 处理的块数
- `key` 从空列表改为 `["n_elements"]`，使 autotune 能根据数据量选择不同配置

### 3. 前向 Kernel 核内结构变化

**GPU 版本 — 无循环，单块处理：**
```python
@triton.jit
def softcap_fwd_kernel(
    x_ptr, y_ptr, n_elements, softcap,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = softcap * (libdevice.tanh(x.to(tl.float32) / softcap)).to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)
```

**NPU 版本 — BLOCK_NUM 多块循环：**
```python
@triton.jit
def softcap_fwd_kernel(
    x_ptr, y_ptr, n_elements, softcap,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_NUM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    for i in range(BLOCK_NUM):
        block_start = pid * BLOCK_SIZE * BLOCK_NUM + i * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = softcap * (libdevice.tanh(x.to(tl.float32) / softcap)).to(x.dtype)
        tl.store(y_ptr + offsets, y, mask=mask)
```

**迁移要点：**

| 维度 | GPU 版本 | NPU 版本 |
|------|---------|---------|
| Kernel 参数 | 仅 `BLOCK_SIZE` | `BLOCK_SIZE` + `BLOCK_NUM` |
| 循环结构 | 无循环，每个 program 处理一个块 | `for i in range(BLOCK_NUM)` 多块循环 |
| offset 计算 | `pid * BLOCK_SIZE` | `pid * BLOCK_SIZE * BLOCK_NUM + i * BLOCK_SIZE` |
| 总数据量/program | `BLOCK_SIZE` | `BLOCK_SIZE * BLOCK_NUM` |
| 计算逻辑 | 完全一致 | 完全一致 |

**BLOCK_NUM 多块循环的意义：**

```
GPU 版本 (无循环):
  Program 0: [0 ~ BLOCK_SIZE-1]
  Program 1: [BLOCK_SIZE ~ 2*BLOCK_SIZE-1]
  每个 program 处理 1 个块

NPU 版本 (BLOCK_NUM 多块循环):
  Program 0: [block0, block1, ..., blockN-1]
  Program 1: [block0', block1', ..., blockN-1']
  每个 program 处理 BLOCK_NUM 个块
  每个 block 大小 = BLOCK_SIZE
```

- NPU 上减少 program 数量可降低调度开销
- 多块循环让单个 program 处理更多数据，提高 Cube/Vector 利用率
- 当 `BLOCK_NUM=1` 时，两者行为等价

### 4. 反向 Kernel 差异

**GPU 版本：**
```python
@triton.jit
def softcap_bwd_kernel(
    dy_ptr, x_ptr, dx_ptr, n_elements, softcap,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dy = tl.load(dy_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = libdevice.tanh(x.to(tl.float32) / softcap).to(x.dtype)
    dx = dy * (1 - y * y)
    tl.store(dx_ptr + offsets, dx, mask=mask)
```

**NPU 版本：**
```python
@triton.jit
def softcap_bwd_kernel(
    dy_ptr, x_ptr, dx_ptr, n_elements, softcap,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_NUM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    for i in range(BLOCK_NUM):
        block_start = pid * BLOCK_SIZE * BLOCK_NUM + i * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        dy = tl.load(dy_ptr + offsets, mask=mask)
        x = tl.load(x_ptr + offsets, mask=mask)
        y = libdevice.tanh(x.to(tl.float32) / softcap).to(x.dtype)
        dx = dy * (1 - y * y)
        tl.store(dx_ptr + offsets, dx, mask=mask)
```

**迁移要点：**
- 反向 kernel 的迁移模式与前向完全对称
- 同样是添加 `BLOCK_NUM` 参数 + 多块循环
- 计算逻辑 `dx = dy * (1 - y * y)` 保持不变

### 5. 编译参数添加（NPU 版本特有）

**GPU 版本 — 无编译参数：**
```python
softcap_fwd_kernel[grid](
    x, y, numel, softcap,
)
```

**NPU 版本 — 添加编译参数：**
```python
softcap_fwd_kernel[grid](
    x, y, numel, softcap,
    debug=True,
    enable_vf_fusion=True,
)
```

**编译参数说明：**

| 参数 | 值 | 含义 |
|------|-----|------|
| `debug` | True | 启用调试信息输出 |
| `enable_vf_fusion` | True | 启用 VF（Vector Function）融合，将多个 Vector 操作融合为一个，减少指令下发开销 |

**注意：** 反向 kernel 仅添加了 `enable_vf_fusion=True`，未添加 `debug=True`。

**VF 融合对 Softcap 的影响：**

Softcap 前向计算 `y = softcap * tanh(x / softcap)` 包含以下 Vector 操作序列：
1. `x.to(fp32)` — 类型转换
2. `x / softcap` — 标量除法
3. `tanh(...)` — 双曲正切
4. `softcap * ...` — 标量乘法
5. `.to(x.dtype)` — 类型转换回原精度

启用 `enable_vf_fusion=True` 后，编译器会尝试将这些操作融合为更少的指令，减少 GM↔UB 搬运次数。

### 6. Grid 计算变化

**GPU 版本：**
```python
grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)
```

**NPU 版本：**
```python
grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"] * META["BLOCK_NUM"]),)
```

**迁移要点：**
- GPU 版本每个 program 处理 `BLOCK_SIZE` 个元素
- NPU 版本每个 program 处理 `BLOCK_SIZE * BLOCK_NUM` 个元素
- Grid 大小 = `numel / (BLOCK_SIZE * BLOCK_NUM)`，program 数量更少
- 当 `BLOCK_NUM=1` 时，两者等价

### 7. 测试代码改动

**GPU 版本：**
```python
x = torch.randn((1024, 1024), dtype=dtype, device="cuda").requires_grad_()
y = Softcap.apply(x, softcap)
dy = torch.randn_like(y)
y.backward(dy)
triton_dx, x.grad = x.grad.clone(), None

ref = softcap * torch.tanh(x.to(torch.float32) / softcap).to(dtype)
ref.backward(dy)
torch_dx, x.grad = x.grad.clone(), None
atol = 1e-3
rtol = 1e-3
if torch.allclose(y, ref, atol=atol, rtol=rtol):
    print("✅ [Fwd]Triton and Torch match")
else:
    print("❌ [Fwd]Triton and Torch differ")
```

**NPU 版本：**
```python
x = torch.randn((1024, 1024), dtype=dtype, device="npu").requires_grad_()

# NPU Profiler
with torch_npu.profiler.profile(...) as prof:
    prof.start()
    for i in range(30):
        y = Softcap.apply(x, softcap)
        torch.npu.synchronize()
        prof.step()
        dy = torch.randn_like(y)
        y.backward(dy)
        triton_dx, x.grad = x.grad.clone(), None

ref = softcap * torch.tanh(x.to(torch.float32) / softcap).to(dtype)
ref.backward(dy)
torch_dx, x.grad = x.grad.clone(), None
...
# 改为自动化断言，失败直接抛异常
assert torch.allclose(y, ref, atol=atol, rtol=rtol), "[Fwd] Triton 和 Torch 前向计算结果不匹配"
assert torch.allclose(triton_dx, torch_dx, atol=atol, rtol=rtol), "[Bwd] Triton 和 Torch 反向梯度计算结果不匹配"
```

**迁移要点：**

| 维度 | GPU 版本 | NPU 版本 |
|------|---------|---------|
| 设备 | `device="cuda"` | `device="npu"` |
| Profiler | 无 | `torch_npu.profiler` 性能剖析 |
| 同步 | 无 | `torch.npu.synchronize()` 确保 kernel 执行完 |
| 验证方式 | `if/else` 打印 | `assert` 断言（更适合自动化测试） |
| 迭代次数 | 1 次 | 30 次（profiler 循环） |

---

## 910_95 特别注意

### 1. 搜索空间过大

当前 NPU 版本的搜索空间为 7 x 2 x 7 = 98 个配置，首次运行 autotune 耗时极长。建议：

- 使用 `TRITON_PRINT_AUTOTUNING=1` 观察最优配置
- 根据最优配置的分布缩小搜索范围
- 将配置数控制在 20 个以内
- 考虑移除 `BLOCK_NUM` 维度，固定为 1 或 2，减少组合数

### 2. BLOCK_SIZE 非对齐值

当前搜索空间包含 `BLOCK_SIZE=18725`，这不是 2 的幂次值。在 910_95 上：
- 32B 对齐要求：FP16 元素数需为 16 的倍数
- 18725 不满足此要求，可能导致性能下降
- 建议替换为 `[512, 1024, 2048, 4096, 8192, 16384]`

### 3. multibuffer 对 UB 的影响

910_95 默认 `multibuffer=False`（详见 [07-compile-params.md](07-compile-params.md)）。当 autotune 选择 `multibuffer=True` 时：
- UB 可用空间减半（从 256KB 降至 128KB）
- `BLOCK_SIZE=16384` 在 FP16 下需要 32KB（单 buffer），双缓冲需要 64KB
- 反向 kernel 需要同时缓冲 `dy` 和 `x`，双缓冲需要 128KB
- 仍在 128KB 限制内，但余量很小

### 4. enable_vf_fusion 的 UB 开销

`enable_vf_fusion=True` 会增加 UB 占用（融合操作需要更多中间缓冲）。在 910_95 上：
- 如果同时启用 `multibuffer=True`，UB 可能溢出
- 建议在 autotune 中将 `enable_vf_fusion` 也作为搜索维度
- 或者在 `multibuffer=True` 时关闭 `enable_vf_fusion`

### 5. num_stages 和 num_warps 必须移除

GPU 版本的 autotune Config 中包含 `num_stages` 和 `num_warps` 参数，这些是 GPU 专属参数：
- `num_stages`：控制 GPU shared memory 的流水线级数
- `num_warps`：控制每个 thread block 的 warp 数量
- NPU 不支持这两个参数，迁移时必须移除，否则会报编译错误

### 6. libdevice 路径变更

GPU 版本使用 `from triton.language.extra import libdevice`，NPU 版本需改为 `from triton.language.extra.cann import libdevice`：
- GPU 路径：`triton.language.extra.libdevice` → 调用 CUDA 实现的数学函数
- NPU 路径：`triton.language.extra.cann.libdevice` → 调用 CANN 实现的数学函数
- 两者 API 一致（如 `libdevice.tanh`），但底层实现不同

### 7. debug=True 的生产环境影响

NPU 版本在 fwd kernel 启动时传入了 `debug=True`，这会：
- 增加编译时间
- 产生额外的调试输出
- 可能影响性能

建议在生产环境中移除 `debug=True`，仅在调试阶段使用。

### 8. maybe_triton_jit 装饰器替换

GPU 版本使用自定义的 `@maybe_triton_jit` 装饰器，NPU 版本需替换为标准的 `@triton.autotune`：
- `@maybe_triton_jit` 可能是项目自定义的装饰器，用于条件性启用 JIT
- NPU 版本直接使用 `@triton.autotune`，这是 Triton 标准库提供的 autotune 机制
- 注意 `key` 参数需从 `[]` 改为 `["n_elements"]`，以根据数据量选择不同配置

---

## 相关文档链接

- [01-migration-overview.md](../docs_for_triton_agent/01-migration-overview.md) — GPU → NPU 迁移概览
- [07-compile-params.md](../docs_for_triton_agent/07-compile-params.md) — NPU 编译参数速查
- [10-autotune-on-npu.md](../docs_for_triton_agent/10-autotune-on-npu.md) — NPU Autotune 配置指南
- [12-store-merge.md](../docs_for_triton_agent/12-store-merge.md) — Store 合并优化
- `softcap.py` — GPU 基础版源码
- `softcap_npu.py` — NPU 优化版源码
