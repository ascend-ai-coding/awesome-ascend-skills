# Autotune 使用与 AutoTilingTuner

## 概述

Triton 的 `@triton.autotune` 装饰器是 Tiling 优化的核心配套工具，能够自动遍历预设的参数配置，通过实际运行对比性能，自动选择最优参数组合。Triton-Ascend 在此基础上扩展了 `AutoTilingTuner`，实现了自动分块调优功能，能够根据算子特征和硬件参数自动生成候选分块配置，大幅减少手动调参的工作量。

## 关键概念

| 概念 | 说明 | 使用场景 |
|------|------|----------|
| @triton.autotune | 自动调优装饰器，遍历预设配置选最优 | 手动指定候选配置 |
| Config | 单个参数配置，包含 kwargs、num_warps 等 | 定义候选参数组合 |
| AutoTilingTuner | 自动分块调优器，自动生成候选配置 | 无需手动指定配置 |
| key | 调优维度，参数取值依赖的输入维度 | 决定何时重新调优 |
| split_params | 分核轴参数，与 tl.program_id 关联 | 决定 grid 分核方式 |
| tiling_params | 分块轴参数，与 tl.arange 关联 | 决定核内分块大小 |
| reduction_axes | 归约轴参数，与 tl.sum/tl.max 关联 | 决定归约维度处理 |
| low_dim_axes | 低维轴参数，影响对齐和分块 | 尾轴等低维度的处理 |
| tile_generator | 分块配置生成器，基于硬件约束生成候选 | AutoTilingTuner 内部使用 |

## @triton.autotune 装饰器

### 基本用法

```python
import triton
import triton.language as tl

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
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

### Config 类

`triton.Config` 定义单个参数配置：

```python
triton.Config(
    kwargs,              # dict: 分块参数，如 {'BLOCK_SIZE': 1024, 'BLOCK_SIZE_SUB': 256}
    num_warps=4,         # int: warp 数量（NPU SIMD 模式下通常为 1）
    num_stages=2,        # int: 流水线阶段数
    num_ctas=1,          # int: CTA 数量
    num_buffers_warp_spec=0,
    num_consumer_groups=0,
    reg_dec_producer=0,
    reg_inc_consumer=0,
    pre_hook=None,       # function: 预处理钩子
)
```

### key 参数

`key` 参数指定调优维度——当这些参数的值发生变化时，会触发重新调优：

```python
# key 为列表：参数按顺序映射到轴名称 x, y, z, w, v, t
@triton.autotune(
    configs=[...],
    key=['M', 'N'],  # M -> x 轴, N -> y 轴
)

# key 为字典：显式指定轴名称与参数的映射
@triton.autotune(
    configs=[...],
    key={'x': 'M', 'y': 'N'},  # 显式映射
    hints={
        'split_params': {'x': 'BLOCK_M'},
        'tiling_params': {'y': 'BLOCK_N'},
        'low_dim_axes': ['y'],
        'reduction_axes': [],
    }
)
```

### 打印调优信息

设置环境变量可打印最优参数信息：

```bash
export TRITON_PRINT_AUTOTUNING=1
```

输出示例：
```
Ascend autotuning parse split axes: {'x': 'BLOCK_M'}, split axis pid dims: {'x': 0}, axis pid dims: {'x': 0}
Ascend autotuning parse tiling axes: {'y': 'BLOCK_N_SUB'}
Ascend autotuning parse low dimensional axes: ['y']
Generated configs number: 12
Triton autotuning for function matmul_kernel finished after 2.35s; best config selected: Config({'BLOCK_M': 128, 'BLOCK_N_SUB': 64}, num_warps=1, num_stages=1);
```

## AutoTilingTuner：自动分块调优

### 概述

`AutoTilingTuner` 是 Triton-Ascend 对标准 `Autotuner` 的扩展，能够根据算子的 AST 分析结果和硬件参数，自动生成候选分块配置，无需手动指定 `configs` 列表。

源码参考：[autotuner.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/autotuner.py#L45-L124)

### 触发条件

当 `@triton.autotune` 的 `configs` 参数为空列表或未提供时，`AutoTilingTuner` 会自动启用：

```python
# 自动模式：不提供 configs，AutoTilingTuner 自动生成
@triton.autotune(
    configs=[],  # 空列表触发自动生成
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K, ...):
    ...

# 手动+自动混合：提供部分 configs，自动生成额外的配置
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),  # 用户指定的配置
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(...):
    ...
```

### 轴名称约定

AutoTilingTuner 使用轴名称来标识不同维度的参数，轴名称分为普通轴和归约轴：

| 轴类型 | 名称列表 | 说明 |
|--------|----------|------|
| 普通轴 | x, y, z, w, v, t | 对应非归约维度 |
| 归约轴 | rx, ry, rz, rw, rv, rt | 前缀 'r' 表示归约维度 |

源码参考：[utils.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/utils.py#L98-L105)

```python
valid_axis_names = ["x", "y", "z", "w", "v", "t"]

# 判断是否为有效轴名：
# "x" -> True (普通轴)
# "rx" -> True (归约轴，去掉前缀 "r" 后为 "x")
# "a" -> False (无效轴名)
```

### key 的两种形式

#### 列表形式（自动映射）

```python
@triton.autotune(
    configs=[],
    key=['M', 'N'],  # M -> x 轴, N -> y 轴
)
```

当使用列表形式时，参数按顺序映射到轴名称 x, y, z, w, v, t。归约轴、低维轴等信息由 autoparser 自动推断。

#### 字典形式（显式映射）

```python
@triton.autotune(
    configs=[],
    key={'x': 'M', 'rx': 'K'},  # 显式指定轴名称
    hints={
        'split_params': {'x': 'BLOCK_M'},
        'tiling_params': {'x': 'BLOCK_M_SUB'},
        'low_dim_axes': ['y'],
        'reduction_axes': ['rx'],  # 注意：这里不带前缀 'r'
    }
)
```

当使用字典形式时，必须同时提供 `hints` 中的所有轴相关参数。

### hints 参数

`hints` 字典提供额外的调优提示：

| 参数 | 类型 | 说明 |
|------|------|------|
| split_params | Dict[str, str] | 分核轴参数映射，如 {'x': 'BLOCK_M'} |
| tiling_params | Dict[str, str] | 分块轴参数映射，如 {'y': 'BLOCK_N_SUB'} |
| low_dim_axes | List[str] | 低维轴列表，如 ['y'] |
| reduction_axes | List[str] | 归约轴列表（不带前缀 'r'），如 ['x'] |
| auto_gen_config | bool | 是否自动生成配置（默认 True） |

## Autoparser 自动解析

AutoTilingTuner 内部使用多个 Parser 自动从算子 AST 中提取轴信息：

源码参考：[autoparser.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/autoparser.py)

### SplitAxesParser

从 `tl.program_id` 语句中提取分核轴参数：

```python
# 识别模式：
pid = tl.program_id(0)           # -> program_id_vars
block_start = pid * BLOCK_SIZE   # -> split_axes: {'x': 'BLOCK_SIZE'}

# 也支持 for 循环中的 grid-stride 模式：
for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
    # -> 识别为 split axis
```

### TilingAxesParser

从 `tl.arange` 和 `for` 循环中的 `range` 语句提取分块轴参数：

```python
# 识别模式：
offsets = tl.arange(0, BLOCK_SIZE_SUB)  # -> tiling_axes

# for 循环中的 range：
for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
    xindex = xoffset_sub + tl.arange(0, XBLOCK_SUB)
    # -> XBLOCK_SUB 被识别为 tiling 参数
```

### ReductionAxesParser

从归约函数（`tl.sum`, `tl.max`, `tl.min`, `tl.argmax`, `tl.argmin`, `tl.xor_sum`）中提取归约轴：

```python
# 识别模式：
mean = tl.sum(x, axis=0) / N   # -> reduction_axes: ['x'] (axis=0 对应的轴)
m = tl.max(x, axis=-1)          # -> reduction_axes (负索引支持)
```

### LowDimsAxesParser

从 `tl.arange` 语句中提取低维轴，用于判断对齐和分块策略：

```python
# 识别模式：
cols = tl.arange(0, BLOCK_N)    # 如果 cols 参与切片操作且在最低维
x = tl.load(X + cols, mask=cols < N)
# -> low_dim_axes: ['y'] (如果 BLOCK_N 对应 y 轴)
```

### PtrNumsParser

统计算子中指针参数的数量，影响 UB 空间估算：

```python
# 识别模式：参数参与 tl.load 或 tl.store 的地址计算
# x_ptr, y_ptr, out_ptr -> ptr_nums = 3
```

## tile_generator 的分块配置生成逻辑

### 核心流程

源码参考：[tile_generator.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/tile_generator.py)

```
AutoTilingTuner._gen_tile_configs()
    |
    v
创建 KernelMeta（轴信息、硬件参数）
    |
    v
创建 TileGenerator(kernel_meta)
    |
    v
TileGenerator.descend_split_tiling()
    |
    +-- 1. descend_split_axis()     # 递减分核轴大小
    +-- 2. descend_tiling_not_low_dims()  # 递减非低维分块轴
    +-- 3. descend_all_low_dims()   # 递减所有低维轴
    |
    v
生成候选 Config 列表
    |
    v
_expand_simd_multibuffer_configs()  # 为每个配置生成 multiBuffer 变体
```

### 关键参数计算

```python
# TileGenerator 初始化时的关键计算
local_mem_size = ub_size_in_kbytes  # SIMD 模式使用 UB
max_numel_threshold = local_mem_size * 1024 // dtype_bytes // num_buffers
# 例：192 * 1024 / 2 / 3 = 32768 elements (float16, 3 buffers)

# 停止递减的阈值
stop_numel = min(1024 // dtype_bytes, max_total_numel // (num_vector_core * 2))
# 例：min(512, 32768 / 96) = min(512, 341) = 341
```

### 递减策略

TileGenerator 采用递减策略生成候选配置：

1. **分核轴递减**：从最大值开始，逐步减半，直到总 program 数不超过物理核数
2. **分块轴递减**：从最大值开始，按 2 的幂次递减
3. **低维轴递减**：所有低维轴同时递减，保持比例平衡
4. **对齐处理**：非 SIMT 模式下，分块大小按 32B 对齐向上取整

### 配置筛选

每个候选配置需通过以下筛选条件：

```python
# 1. UB 空间约束
tile_numel <= max_numel_threshold

# 2. 最小分块约束（避免过小的分块）
tile_numel >= stop_numel_threshold

# 3. 去重
not find_config(newcfg)
```

### multiBuffer 扩展

对于 SIMD 模式，每个基础配置会生成两个变体：

```python
# 基础配置
Config({'BLOCK_SIZE': 1024}, num_warps=1, num_stages=1)

# multiBuffer=True 变体（默认）
Config({'BLOCK_SIZE': 1024, 'multibuffer': True}, num_warps=1, num_stages=1)

# multiBuffer=False 变体
Config({'BLOCK_SIZE': 1024, 'multibuffer': False}, num_warps=1, num_stages=1)
```

## 代码示例

### 示例1：手动 autotune

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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 示例2：自动 autotune（列表形式 key）

```python
@triton.autotune(
    configs=[],  # 空列表，触发自动生成
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(B + (offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 示例3：自动 autotune（字典形式 key + hints）

```python
@triton.autotune(
    configs=[],
    key={'x': 'M', 'y': 'N', 'rx': 'K'},
    hints={
        'split_params': {'x': 'BLOCK_M', 'y': 'BLOCK_N'},
        'tiling_params': {'rx': 'BLOCK_K'},
        'low_dim_axes': ['y'],
        'reduction_axes': ['x'],  # 不带前缀 'r'
    }
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

### 示例4：使用 libentry + autotune

```python
from triton.runtime import libentry

@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("masked_fill"),
    key=['N'],
)
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N,
                       BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB

    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
        tl.store(out + offsets, input_vals, mask=mask)
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        overwrite_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, overwrite_vals, mask=mask)
```

## NPU 适配要点

1. **候选值须为 2 的幂次**：autotune 的参数候选值必须是 2 的幂次（128, 256, 512, 1024, ...）
2. **key 决定调优触发**：当 key 中指定的参数值变化时，会重新调优；相同值会复用缓存
3. **SIMD 模式下 num_warps=1**：NPU SIMD 模式下 num_warps 固定为 1，不需要调整
4. **multiBuffer 自动扩展**：AutoTilingTuner 会自动为每个配置生成 multiBuffer 开/关的变体
5. **并行编译**：默认启用并行编译加速调优，可通过 `TRITON_AUTOTUNE_PARALLEL_COMPILE=0` 关闭

## 常见问题 (Q&A)

**Q1: autotune 调优时间太长怎么办？**

A: (1) 减少候选配置数量；(2) 设置 `TRITON_AUTOTUNE_PARALLEL_COMPILE=1`（默认开启）启用并行编译；(3) 使用 `prune_configs_by` 参数提前剪枝不可行配置。

**Q2: 自动生成的配置数量为 0 怎么办？**

A: 检查 key 参数是否正确映射到算子参数，确保 split_params 和 tiling_params 能被 autoparser 正确识别。设置 `TRITON_PRINT_AUTOTUNING=1` 查看解析日志。

**Q3: 如何查看 autotune 选出的最优配置？**

A: 设置 `export TRITON_PRINT_AUTOTUNING=1`，运行后会打印最优配置信息，包括调优耗时和选中的配置参数。

**Q4: key 使用列表还是字典？**

A: 简单算子使用列表形式即可，autoparser 会自动推断轴信息。复杂算子（含归约轴、多维度分块）建议使用字典形式显式指定，避免自动推断错误。

**Q5: autotune 的缓存机制是怎样的？**

A: 调优结果按 key 值缓存，相同 key 值不会重复调优。如果需要强制重新调优，需要重启 Python 进程或清除 Triton 缓存目录。

## 相关文档

- [01-optimization-overview.md](./01-optimization-overview.md) - 优化策略总览
- [02-tiling-strategy.md](./02-tiling-strategy.md) - 分块策略详解
- [07-profiling-guide.md](./07-profiling-guide.md) - 性能分析与瓶颈定位

### 源码参考

- [autotuner.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/autotuner.py) - AutoTilingTuner 实现
- [tile_generator.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/tile_generator.py) - 分块配置生成器
- [autoparser.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/autoparser.py) - AST 自动解析器
- [utils.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/utils.py) - 硬件参数和工具函数
