# 向量加法模式（Vector Addition Pattern）

## 概述

向量加法是 Triton 编程的入门示例，展示了最基本的 SPMD 编程模型：多个 program 并行处理数据的不同分片。在 NPU 上，向量加法是典型的 Vector 计算密集型操作，理解其模式是编写更复杂 kernel 的基础。

| 关键概念 | 说明 |
|---------|------|
| `tl.program_id` | 获取当前 program 的 ID，用于确定处理的数据范围 |
| `tl.arange` | 生成连续偏移量序列，构建指针块 |
| `BLOCK_SIZE: tl.constexpr` | 编译时常量，定义每个 program 处理的元素数 |
| `mask` | 边界保护，防止越界内存访问 |
| `tl.load / tl.store` | 从/向 Global Memory 读写数据 |
| `triton.cdiv` | 向上取整除法，计算 grid 大小 |
| `device='npu'` | NPU 设备指定，替代 CUDA 的 `'cuda'` |

## 完整代码示例

```python
import torch
import torch_npu

import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')
output_torch = x + y
output_triton = add(x, y)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

## 逐行解释

### Kernel 定义

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
```

- `@triton.jit`：将 Python 函数标记为 Triton JIT 编译的 kernel
- `x_ptr, y_ptr`：指向输入向量的指针，运行时自动从 `torch.Tensor` 转换
- `output_ptr`：指向输出向量的指针
- `n_elements`：向量元素总数，运行时参数
- `BLOCK_SIZE: tl.constexpr`：编译时常量，每个 program 处理的元素数量

### Program ID 与偏移量计算

```python
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

- `tl.program_id(axis=0)`：获取 1D grid 中当前 program 的 ID
- `block_start`：当前 program 起始元素的全局偏移
- `tl.arange(0, BLOCK_SIZE)`：生成 `[0, 1, ..., BLOCK_SIZE-1]` 的偏移序列
- `offsets`：当前 program 负责处理的所有元素的全局偏移

### 边界保护

```python
mask = offsets < n_elements
```

当 `n_elements` 不是 `BLOCK_SIZE` 的整数倍时，最后一个 program 处理的偏移可能超出范围。`mask` 为 `True` 的位置是有效数据，`False` 的位置将被忽略。

### 数据加载与计算

```python
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
output = x + y
```

- `tl.load`：从 Global Memory 加载数据到片上 SRAM
- `mask=mask`：仅加载有效位置的元素，越界位置填充 0
- `x + y`：Vector 核心执行逐元素加法

### 数据写回

```python
tl.store(output_ptr + offsets, output, mask=mask)
```

将计算结果写回 Global Memory，同样使用 `mask` 保护边界。

## Grid 配置策略

```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
```

| 配置项 | 说明 |
|-------|------|
| Grid 维度 | 1D，大小为 `ceil(n_elements / BLOCK_SIZE)` |
| `triton.cdiv` | 向上取整除法，确保覆盖所有元素 |
| `meta['BLOCK_SIZE']` | 从 kernel 的 constexpr 参数中获取 |

### BLOCK_SIZE 选择指南

| BLOCK_SIZE | 适用场景 | 说明 |
|-----------|---------|------|
| 256 | 小向量 / 调试 | 较少的并行度 |
| 1024 | 通用场景 | 默认推荐值 |
| 2048 | 大向量 | 更高的内存吞吐 |
| 4096 | 超大向量 | 最大化带宽利用 |

NPU 上 BLOCK_SIZE 应尽量选择 2 的幂次，且建议为 256 的倍数以获得更好的内存对齐。

## Autotune 配置

对于向量加法这种简单操作，手动选择 `BLOCK_SIZE` 通常足够。如需自动调优：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

- `key=['n_elements']`：当 `n_elements` 变化时重新调优
- Autotune 会自动选择最优的 `BLOCK_SIZE` 配置

## NPU 适配要点

### 1. 设备指定

```python
x = torch.rand(size, device='npu')
```

NPU 使用 `device='npu'` 而非 `device='cuda'`。需要 `import torch_npu` 来注册 NPU 后端。

### 2. 内存对齐

NPU 芯片亲和 512B 对齐场景。BLOCK_SIZE 选择应考虑数据类型的对齐要求：

| 数据类型 | 元素大小 | 推荐 BLOCK_SIZE（512B 对齐） |
|---------|---------|------------------------------|
| fp32 | 4B | 128 的倍数 |
| fp16 | 2B | 256 的倍数 |
| bf16 | 2B | 256 的倍数 |

### 3. UB 空间限制

NPU 上所有 tensor 总和不能超过 UB 限制：A2/A3 系列开启 double buffer 时为 96KB，关闭时为 192KB；910_95 系列开启 double buffer 时为 128KB，关闭时为 256KB。向量加法中同时存在的 tensor 为 `x`、`y`、`output`，需确保 `3 * BLOCK_SIZE * element_size` 不超过对应限制。

### 4. 无需 `is_cuda` 检查

NPU 版本不需要 `assert x.is_cuda` 的检查，可以移除或替换为 `assert x.is_npu`。

### 5. torch_npu 导入

NPU 版本需要额外导入 `torch_npu` 来注册 NPU 后端：

```python
import torch
import torch_npu
```

`torch_npu` 提供了 `device='npu'` 支持、NPU Stream 管理、NPU 设备属性查询等功能。

### 6. Benchmark 适配

NPU 上 benchmark 需要使用 `do_bench_npu` 替代 `do_bench`：

```bash
export TRITON_BENCH_METHOD=npu
```

或在代码中直接使用 NPU 设备进行测试：

```python
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')
```

## 完整 NPU 适配版代码

```python
import torch
import torch_npu

import triton
import triton.language as tl
from triton.runtime import driver


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='npu')
    y = torch.rand(size, device='npu')
    output_torch = x + y
    output_triton = add(x, y)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')
```

## 常见问题（Q&A）

**Q: 向量加法结果精度不对，出现 NaN？**

A: 检查输入数据是否包含 NaN/Inf，以及 mask 是否正确设置。未 mask 的越界 load 可能读到垃圾值。

**Q: BLOCK_SIZE 设为多大最合适？**

A: 默认 1024 是较好的起点。对于 NPU，建议 BLOCK_SIZE 为 256 的倍数。可通过 autotune 自动选择最优值。

**Q: 如何处理超大向量（元素数 > 2^31）？**

A: 使用 `al.int64()` 类型的偏移量，确保 `n_elements` 参数使用 64 位整数。

**Q: NPU 上向量加法比 PyTorch 慢？**

A: 向量加法是 memory-bound 操作，Triton kernel 的优势在于融合多个操作。单独的向量加法不会有明显加速，但可以与其他操作融合来减少内存访问。

## 相关文档

- [02-fused-softmax.md](./02-fused-softmax.md) - 融合 Softmax 模式
- [06-reduction-pattern.md](./06-reduction-pattern.md) - 归约操作模式
- 源码参考：[01-vector-add.py (Ascend)](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/tutorials/01-vector-add.py)
- 源码参考：[01-vector-add.py (upstream)](https://github.com/triton-lang/triton-ascend/tree/main/python/tutorials/01-vector-add.py)
