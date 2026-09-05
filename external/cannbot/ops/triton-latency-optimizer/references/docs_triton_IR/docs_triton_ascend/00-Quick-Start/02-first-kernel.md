# 02 - 第一个 Triton-Ascend Kernel

## 概述

本文档通过一个完整的向量加法（Vector Addition）示例，逐步介绍 Triton-Ascend kernel 的编写、启动和验证方法。你将学习 `@triton.jit` 装饰器、kernel 函数签名、`tl.program_id`、`tl.arange`、`tl.load`/`tl.store` 等核心 API 的用法，以及 NPU 平台上 kernel 启动的 grid 配置和与 GPU 版本的关键差异。

**关键词**：Triton kernel、@triton.jit、program_id、arange、load/store、grid配置、向量加法、NPU适配

---

## 关键概念

| 概念 | 说明 |
|------|------|
| `@triton.jit` | Just-In-Time 编译装饰器，将 Python 函数标记为 Triton kernel，由编译器在运行时编译为 NPU 可执行代码 |
| `tl.program_id(axis)` | 获取当前 program（计算核心任务）在指定轴上的 ID，用于确定当前核处理的数据范围 |
| `tl.arange(start, end)` | 生成从 start 到 end 的连续整数序列，用于计算数据偏移量 |
| `tl.load(ptr, mask)` | 从全局内存加载数据到片上内存（UB），mask 用于防止越界访问 |
| `tl.store(ptr, value, mask)` | 将计算结果从片上内存写回全局内存，mask 用于防止越界写入 |
| `tl.constexpr` | 编译时常量标记，标记的参数可在 shape 值等需要编译期确定的场景使用 |
| grid | kernel 启动时的并行配置，决定启动多少个 program（计算任务） |
| BLOCK_SIZE | 每个 program 处理的数据元素数量，是 tiling 策略的核心参数 |

---

## 完整的向量加法 Kernel 示例

以下是一个完整的、可在 Ascend NPU 上运行的向量加法示例：

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
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

---

## 逐步解释

### 1. `@triton.jit` 装饰器

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
```

`@triton.jit` 将 Python 函数标记为 Triton kernel。当 kernel 被调用时（如 `add_kernel[grid](...)`），Triton 编译器会将该函数编译为 NPU 可执行代码。编译过程为：

```
Python AST → TTIR → TTIR优化 → Ascend Passes → Linalg IR → BiSheng Compiler → NPU 二进制
```

**关键点**：
- kernel 函数的参数在编译时被解析为 Triton IR 的函数参数
- 指针类型参数（如 `x_ptr`）对应全局内存地址
- 标量参数（如 `n_elements`）在运行时传入
- `tl.constexpr` 标记的参数（如 `BLOCK_SIZE`）在编译时确定，可用作 shape 值

### 2. Kernel 函数签名

```python
def add_kernel(x_ptr,         # 第一个输入向量的指针
               y_ptr,         # 第二个输入向量的指针
               output_ptr,    # 输出向量的指针
               n_elements,    # 向量元素总数
               BLOCK_SIZE: tl.constexpr,  # 每个 program 处理的元素数
               ):
```

参数说明：

| 参数 | 类型 | 说明 |
|------|------|------|
| `x_ptr` | 指针 | 指向第一个输入向量在全局内存中的起始地址 |
| `y_ptr` | 指针 | 指向第二个输入向量在全局内存中的起始地址 |
| `output_ptr` | 指针 | 指向输出向量在全局内存中的起始地址 |
| `n_elements` | 标量 | 向量的元素总数，用于边界检查 |
| `BLOCK_SIZE` | constexpr | 每个 program 处理的元素数量，编译时常量 |

### 3. `tl.program_id` - 获取当前核的 ID

```python
pid = tl.program_id(axis=0)
```

`tl.program_id(axis=0)` 返回当前 program 在 grid 第 0 轴上的唯一 ID。在 1D grid 中，每个 program 获得一个从 0 开始的整数 ID，用于确定该核负责处理的数据范围。

**NPU 上的含义**：在 Ascend NPU 上，每个 program 对应一个 AI Core 上的计算任务。`pid` 就是该任务在并行任务队列中的编号。

### 4. `tl.arange` - 生成偏移量序列

```python
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

- `block_start = pid * BLOCK_SIZE`：计算当前 program 负责的数据起始位置
- `tl.arange(0, BLOCK_SIZE)`：生成 `[0, 1, 2, ..., BLOCK_SIZE-1]` 的连续整数序列
- `offsets`：最终的数据偏移量，表示当前 program 要访问的全局内存位置

例如，当 `BLOCK_SIZE=1024` 时：
- pid=0 → offsets = [0, 1, ..., 1023]
- pid=1 → offsets = [1024, 1025, ..., 2047]
- pid=2 → offsets = [2048, 2049, ..., 3071]

### 5. `tl.load` - 从全局内存加载数据

```python
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
```

- `mask = offsets < n_elements`：创建边界检查掩码，防止越界访问。当向量长度不是 BLOCK_SIZE 的整数倍时，最后一个 program 的部分 offset 会超出范围
- `tl.load(x_ptr + offsets, mask=mask)`：从全局内存（DRAM）加载数据到片上内存（UB），mask 为 False 的位置不会被加载

**NPU 上的含义**：`tl.load` 对应从 Global Memory 到 Unified Buffer (UB) 的数据搬运操作，由 Vector Core 执行。

### 6. 计算与 `tl.store` - 写回结果

```python
output = x + y
tl.store(output_ptr + offsets, output, mask=mask)
```

- `output = x + y`：在片上内存中执行逐元素加法
- `tl.store(output_ptr + offsets, output, mask=mask)`：将计算结果从片上内存写回全局内存

---

## Kernel 启动方式

### Grid 配置

```python
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

**grid 的含义**：

- `triton.cdiv(n_elements, meta['BLOCK_SIZE'])` 计算需要启动的 program 总数
- 例如 `n_elements=98432, BLOCK_SIZE=1024`，则 `cdiv(98432, 1024) = 97`，启动 97 个 program
- grid 使用 lambda 函数，可以在 autotune 改变 BLOCK_SIZE 时自动调整

**kernel 调用语法**：

```python
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
```

方括号 `[]` 中传入 grid 配置，圆括号 `()` 中传入 kernel 参数。

### num_warps 参数

在 GPU 版本中，`num_warps` 控制每个 program 中的 warp 数量。在 NPU 版本中，`num_warps` 的含义有所不同：

```python
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=4)
```

NPU 上的 `num_warps` 参数主要影响 SIMT 编译模式下的线程配置，在默认的 SIMD 编译模式下影响较小。NPUOptions 中默认 `num_warps=4`, `warp_size=32`。

---

## 与 GPU 版本的关键差异

### 1. 设备标识

```python
# GPU 版本
x = torch.rand(size, device='cuda')

# NPU 版本
x = torch.rand(size, device='npu')
```

### 2. 分核策略

GPU 拥有大量 SM（流多处理器），通常几十到上百个，可以直接按数据量分核。而 NPU 的 AI Core 数量通常在几十个量级，直接套用 GPU 的分核策略会导致大量核启动和初始化开销。

**GPU 风格（不推荐在 NPU 上使用）**：

```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
```

**NPU 推荐风格 - 固定核数 + 核内循环**：

```python
import triton.runtime.driver as driver

device = torch.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]

NUM_CORE = vectorcore_num
grid = (NUM_CORE,)

@triton.jit
def add_kernel_npu(x_ptr, y_ptr, output_ptr, n_elements,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
```

**核心思路**：
- 将分核数固定为物理核数（Vector Core 数量）
- 核内通过 `range(pid, NUM_BLOCKS, NUM_CORE)` 循环处理多个数据块
- 跨步分配确保任务均匀分布到各个核

### 3. 数据对齐要求

| 算子类型 | 尾轴对齐要求 |
|---------|------------|
| VV 类（纯 Vector 算子） | 尾轴大小能被 32 Bytes 整除 |
| CV 类（Cube+Vector 融合算子） | 尾轴大小能被 512 Bytes 整除 |

如果尾轴长度不足会自动补齐，可能导致性能下降。可通过转置操作将长轴放到低维来规避。

### 4. 片上内存（UB）大小限制

| NPU 系列 | UB 大小 |
|----------|--------|
| A2 系列 (Ascend910B) | 192 KB |
| A3 系列 (Ascend910_95 / Ascend950) | 256 KB |

BLOCK_SIZE 过大可能导致 UB 溢出，编译时会报错：

```
ub overflow, requires 3072256 bits while 1572864 bits available!
```

### 5. 存算并行（Multi-Buffer）

NPU 默认开启 multiBuffer（`multibuffer=True`），支持存算并行：在搬运第一批数据至片上内存的同时开始计算，形成"搬运+计算"重叠的流水线操作。这会将可用 UB 容量减半（double buffer）。

---

## 运行验证方法

### 方法一：直接运行 Tutorial 示例

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```

预期输出：

```
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

### 方法二：与 PyTorch 结果对比

```python
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')

output_torch = x + y
output_triton = add(x, y)

max_diff = torch.max(torch.abs(output_torch - output_triton))
print(f"最大误差: {max_diff}")
assert max_diff < 1e-6, "结果验证失败！"
```

### 方法三：使用 Autotune 自动调优

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel_autotune(x_ptr, y_ptr, output_ptr, n_elements,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

打印最优参数信息：

```bash
export TRITON_PRINT_AUTOTUNING=1
```

---

## NPU 适配要点

1. **固定核数优于动态分核**：将 grid 大小设为物理核数，核内循环处理数据，避免核启动开销。

2. **BLOCK_SIZE 选择**：需要兼顾 UB 容量限制和数据对齐要求。推荐使用 `triton.autotune` 自动搜索最优值。

3. **mask 必不可少**：当数据量不是 BLOCK_SIZE 的整数倍时，最后一个 program 的部分访问需要 mask 保护。

4. **device='npu'**：所有输入/输出张量必须创建在 NPU 设备上（`device='npu'`），否则 kernel 无法访问。

5. **纯 Vector 算子分核数**：对于不包含 `tl.dot` 的纯向量算子，分核数应等于 Vector Core 数量（通常为 AI Core 数量的 2 倍）。

6. **含 tl.dot 的算子分核数**：对于包含矩阵乘法的算子，分核数应等于 AI Core 数量。

---

## 常见问题

### Q1: 运行时报错 `device='npu'` 不可用

**A**: 确认已正确安装 torch_npu 并 source CANN 环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch_npu; print(torch.npu.is_available())"
```

### Q2: 编译时报错 `ub overflow`

**A**: BLOCK_SIZE 过大导致片上内存溢出。减小 BLOCK_SIZE 或使用子块划分（Tiling）策略：

```python
@triton.jit
def add_kernel_tiled(x_ptr, y_ptr, output_ptr, n_elements,
                     BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
```

### Q3: kernel 执行结果与 PyTorch 不一致

**A**: 检查以下几点：
- 输入张量是否在 `device='npu'` 上
- mask 是否正确设置（越界位置可能加载到未定义值）
- 数据类型是否匹配（bfloat16 精度低于 float32）

### Q4: 如何选择 BLOCK_SIZE？

**A**: 推荐使用 `triton.autotune` 自动搜索。手动选择时需考虑：
- UB 容量：`BLOCK_SIZE * element_size * 变量数` 不超过 UB 大小
- 对齐要求：VV 算子尾轴需 32 Bytes 对齐
- 2 的幂次：推荐 128, 256, 512, 1024 等

### Q5: GPU 上的 kernel 能直接在 NPU 上运行吗？

**A**: 大部分 Triton 语法兼容，但需注意：
- `device='cuda'` 改为 `device='npu'`
- 分核策略需要适配（固定核数 + 核内循环）
- UB 大小限制比 GPU shared memory 更严格
- 部分数据类型（如 uint）NPU 不支持

### Q6: coreDim 超限怎么办？

**A**: NPU 的 coreDim 不能超过 65535。当数据量极大时，应采用固定核数 + 核内循环策略，而非按数据量动态分核。

---

## 相关文档

- [01 - 环境搭建与验证](./01-environment-setup.md)：搭建 Triton-Ascend 开发环境
- [03 - 编译流程全景](./03-compilation-flow.md)：了解 kernel 从 Python 到 NPU 二进制的完整编译流程
- [源码参考 - 01-vector-add.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/tutorials/01-vector-add.py)
- [源码参考 - programming_guide.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/programming_guide.md)
- [源码参考 - compiler.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py)
