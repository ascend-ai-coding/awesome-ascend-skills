# Autotune 自动调优

## 概述

Triton autotune 用于自动选择最优的 kernel 配置参数，主要包括影响分核（split）和切块（tiling）大小的参数，主要使用方式如下：

**三种种使用方式：**

| 方式 | 说明 | 适用场景 |
|------|------|---------|
| **自动 autotune** | 框架自动解析切分轴、tiling 轴，生成配置 | Vector 类算子，简化使用 |
| **半自动 autotune** | 用户手动传入 `hints` ，框架基于`hints`生成配置 | Vector 类算子，简化使用 |
| **自定义 autotune** | 用户手动传入 `triton.Config` 列表 | 需要精确控制搜索空间 |

上述三种方法的详细使用场景参考如下：
1. 如果这些参数能从 `tl.program_id`、`tl.arange`、`tl.range/range`、`mask/bounds` 表达式中被唯一识别出来，就尝试使用自动 autotune `configs=[]`
2. 如果 kernel 语义上适合自动 tiling，但 DSL 写法让 parser 解析不出来，就使用半自动 autotune显式传 `hints`
3. 如果某些 tiling 参数不可自由调整，例如某 kernel dsl 写法要求 grid 第一维必须固定为 `batch_size` 大小，或者根本没有暴露出可调的 tiling 参数，此时建议使用自定义 autotune。

**说明：** 当前 Triton-Ascend autotune 支持 block size、multibuffer（编译器优化），因硬件架构差异不支持 num_warps、num_stages 参数。

---

## 一、API 参考

### triton.autotune 装饰器

```python
@triton.autotune(
    configs=[...],           # Config 列表
    key=['x_size'],          # 触发重新评估的参数名
    hints={...}              # 显式指定轴与 tiling 参数的映射关系
    prune_configs_by=None,   # 配置剪枝函数
    reset_to_zero=None,      # 评估前重置为零的参数
    restore_value=None,      # 评估后恢复原值的参数
    pre_hook=None,           # 内核调用前的钩子
    post_hook=None,          # 内核调用后的钩子
    warmup=None,             # 预热时间（已弃用）
    rep=None,                # 重复时间（已弃用）
    do_bench=None,           # 自定义基准测试函数
    cache_results=False,     # 是否缓存结果到磁盘
)
@triton.jit
def kernel(...):
    ...
```

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `configs` | `list[triton.Config]` | Config 对象列表，每个代表一种 kernel 配置 |
| `hints`   | `dict` | **Ascend 扩展参数**，用于显式指定轴与 tiling 参数的映射关系                         |
| `key` | `list[str]` | 参数名列表，这些参数值变化时触发重新评估所有配置 |
| `prune_configs_by` | `dict` | 配置剪枝函数，用于减少评估的配置数量 |
| `reset_to_zero` | `list[str]` | 参数名列表，评估前重置为零（避免累积更新） |
| `restore_value` | `list[str]` | 参数名列表，评估后恢复原值 |
| `pre_hook` | `lambda` | 内核调用前的钩子函数 |
| `post_hook` | `lambda` | 内核调用后的钩子函数 |
| `do_bench` | `lambda` | 自定义基准测试函数 |
| `cache_results` | `bool` | 是否缓存调优结果到磁盘（默认 False） |

#### 重要提示

**避免累积更新：** 当所有配置都评估后，内核会运行多次。如果内核会更新某些值，这些值会被多次更新。使用 `reset_to_zero` 在评估前重置这些张量：

```python
@triton.autotune(
    configs=[...],
    key=['n'],
    reset_to_zero=['output_ptr'],  # 评估前重置 output 为零
)
@triton.jit
def kernel(output_ptr, ...):
    ...
```

**调试输出：** 设置环境变量 `TRITON_PRINT_AUTOTUNING=1`，Triton 会打印调优时间和最佳配置：

```bash
export TRITON_PRINT_AUTOTUNING=1
```

### triton.Config 类

```python
triton.Config(
    kwargs={'BLOCK_SIZE': 128},  # 传递给内核的元参数
    num_warps=4,                 # warp 数量（GPU）
    num_stages=3,                # 流水阶段数（GPU）
    num_ctas=1,                  # 块集群中的块数（SM90+）
    maxnreg=None,                # 最大寄存器数
    pre_hook=None,               # 调用前的钩子
    ir_override=None,            # 自定义 IR 文件名
)
```

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `kwargs` | `dict` | 传递给内核的元参数字典，如 `{'BLOCK_SIZE': 128}` |
| `num_warps` | `int` | GPU warp 数量，决定并行线程数（8 warps = 256 线程） |
| `num_stages` | `int` | 软件流水线阶段数，用于矩阵乘法优化 |
| `num_ctas` | `int` | 块集群中的块数量（仅 SM90+ GPU） |
| `maxnreg` | `int` | 单线程最大寄存器数量 |
| `pre_hook` | `lambda` | 内核调用前的钩子函数 |
| `ir_override` | `str` | 自定义 IR 文件名（.ttgir/.llir/.ptx/.amdgcn） |

#### Triton-Ascend 支持情况

| 参数 | 社区版 | Triton-Ascend | 说明 |
|------|--------|---------------|------|
| `kwargs` | ✅ | ✅ | 完全支持 |
| `num_warps` | ✅ | ❌ | NPU 架构差异，不支持 |
| `num_stages` | ✅ | ❌ | NPU 架构差异，不支持 |
| `multibuffer` | ❌ | ✅ | NPU 特有，多缓冲优化（通过 kwargs 传入） |
| `unit_flag` | ❌ | ✅ | NPU 特有，独立计算单元（通过 kwargs 传入） |

#### 使用示例

```python
# GPU 风格配置
configs = [
    triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
]

# Triton-Ascend 风格配置
configs = [
    triton.Config({'BLOCK_SIZE': 128, 'multibuffer': True}),
    triton.Config({'BLOCK_SIZE': 256, 'multibuffer': False}),
]
```

### prune_configs_by 配置剪枝

用于减少需要评估的配置数量，加速 autotune 过程：

```python
@triton.autotune(
    configs=[...],
    key=['n'],
    prune_configs_by={
        'perf_model': my_perf_model,    # 性能预测模型
        'top_k': 10,                     # 只评估 top_k 个配置
        'early_config_prune': my_prune_fn,  # 自定义剪枝函数
    }
)
@triton.jit
def kernel(...):
    ...
```

**剪枝函数签名：**

```python
def prune_fn(
    configs: List[triton.Config],
    named_args: Dict[str, Any],
    **kwargs
) -> List[triton.Config]:
    # 返回剪枝后的配置列表（至少返回一个）
    return pruned_configs
```

### pre_hook / post_hook 钩子

**pre_hook 签名：**

```python
def pre_hook(kwargs, reset_only):
    # kwargs: 传递给内核的所有参数
    # reset_only: 是否仅用于重置值
    pass
```

**post_hook 签名：**

```python
def post_hook(kwargs, exception):
    # kwargs: 传递给内核的所有参数
    # exception: 编译或运行时异常（无异常时为 None）
    pass
```

---

## 二、autotune使用工作流
使用Triton-Ascend autotune搜索最佳分核参数需要遵循以下工作流：

### step1.识别出 triton kernel 中哪些 `tl.constexpr` 参数是自由可调的 tiling 参数

系统首先识别 kernel 调用时**未传入**的参数作为候选项：
* Tensor 参数不可能是自动解析候选项；
* 普通运行时 shape 参数（如 n_rows、n_cols）通常属于 `key`；
* 真正的候选项通常是没有在 launch 处显式传值的 `tl.constexpr`；
* 如果某个 tl.constexpr 已经在 launch 时手动写死，它就不会再被当成自动解析候选项。

例如：
```python
@triton.jit
def kernel_func(
    output_ptr, input_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,    # 调用时传入，不可调
    XBLOCK: tl.constexpr,        # 未传入，可调
    XBLOCK_SUB: tl.constexpr,    # 未传入，可调
):
    ...

# 调用时只传入 BLOCK_SIZE
kernel_func[grid](y, x, n_rows, n_cols, BLOCK_SIZE=block_size)
# 可调参数候选项：XBLOCK, XBLOCK_SUB
```

#### step1.1.识别切分参数

切分（split）参数控制“一个 program 负责多大的一块数据”，它最常见的写法特征是：
1. 和 `tl.program_id(...)` 有直接关系；
2. 参与构造 block 起始位置；
3. 最后能通过 mask/bounds 表达式对应回某个 shape 轴。
例如：
```python
# 一维切分
pid = tl.program_id(0)
offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)# 可以知道BLOCK_M 是 split 参数
mask_m = offs_m < n_rows

# 二维切分（Ascend NPU 必须使用一维 Grid，通过一维 pid 映射到二维坐标）
pid = tl.program_id(0)
NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):  # 交错循环，每个核处理多个块
    pid_m = block_idx // NUM_BLOCKS_N  # 行块索引
    pid_n = block_idx % NUM_BLOCKS_N   # 列块索引
    # BLOCK_M 和 BLOCK_N 都是 split 参数
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask_m = offs_m < M
    mask_n = offs_n < N
```

#### step1.2.识别分块参数

分块（tiling）参数控制“在一个大的 split block 内，再按多大的子块去迭代”，它最常见的写法特征是：

1. 出现在 `tl.arange(0, PARAM)` 中；
2. 同时还出现在 for 循环的步长或循环次数推导中；
3. 最后能通过 mask/bounds 对应回某个轴长度参数。

例如：
```python
# 典型形态 1：步长是 tiling 参数
for k0 in tl.range(0, BLOCK_K, BLOCK_K_SUB):# 可以知道BLOCK_K_SUB 是 tiling 参数
    offs_k = k0 + tl.arange(0, BLOCK_K_SUB)# 可以知道BLOCK_K_SUB 是 tiling 参数
    mask_k = offs_k < k_size

# 典型形态 2：先计算循环次数
num_k_tiles = (k_size + BLOCK_K_SUB - 1) // BLOCK_K_SUB# 可以知道BLOCK_K_SUB 是 tiling 参数
for tile_id in range(num_k_tiles):
    offs_k = tile_id * BLOCK_K_SUB + tl.arange(0, BLOCK_K_SUB)# 可以知道BLOCK_K_SUB 是 tiling 参数
    mask_k = offs_k < k_size
```

#### step1.3.识别低维轴参数
**依据：** `tl.arange()` 切片操作

**识别规则：**
1. 必须通过 `tl.arange()` 计算
2. 必须进行切片操作
3. 在**非最低维**进行维度扩充才被识别为低维轴

```python
@triton.autotune(configs=[], key=["n_rows", "n_cols"])
@triton.jit
def kernel(...):
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        # row_offsets：切片在低维扩充，不是低维轴
        row_offsets = row_idx + tl.arange(0, XBLOCK_SUB)[:, None]
        # col_offsets：切片在高维扩充，是低维轴
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]

        xmask = row_offsets < n_rows
        ymask = col_offsets < n_cols

# 解析结果：low_dim_axes = ["y"]
```

#### step1.4.指针参数解析

**依据：** 参数是否参与 `tl.load()` 或 `tl.store()` 的第一个参数计算

```python
@triton.jit
def kernel(input_ptr, output_ptr, ...):
    # 直接参与
    input = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input, mask=mask)

    # 或间接参与
    inputs_ptr = input_ptr + offsets
    input = tl.load(inputs_ptr, mask=mask)

# 解析结果：指针参数 = input_ptr, output_ptr
```

#### step1.5.判断使用什么autotune
一般同时满足下面几条时，自动autotune的成功率比较高：

* split 参数能从 `tl.program_id` 路径判断出来；
* tiling 参数能从 `tl.arange + for(range/tl.range)` 路径判断出来；
* 每个轴都有比较清晰的 mask/bounds 表达式，例如：
    * `offs < n`
    * `offs < min(block_end, n)`
* `key` 能和运行时 shape 参数一一对应；

此时应当跳转到step2.尝试自动autotune执行。出现下面的情况，自动autotune可能会出现解析失败的情况：
* 没有和轴长度直接绑定的 mask/bounds
* 某个参数必须覆盖完整语义维度
    * 例如 `BLOCK_SIZE >= hidden_dim`
* grid 某一维被业务语义固定，不允许自由切块
* 一个参数同时影响两个轴，或者同时影响“核数 + tile 形状”
* kernel 没暴露出可调 `tl.constexpr`

如果step2.尝试自动autotune失败，并且满足以下条件：

* 能判断哪个参数属于 `split`，哪个属于 `tiling`；
* 知道每个参数对应轴的长度参数；

此时应当跳转到step3.尝试半自动autotune

否则，请直接尝试step4.尝试自定义autotune

### step2.尝试自动autotune
自动autotune完全由编译器决定尝试哪些参数组合，仅需要指定`key`。自动autotune模板如下：
```python
@triton.autotune(
    # configs 为空列表，表示不传入自定义配置
    # 此时 auto_gen_config 默认为 True，会自动生成 tiling 配置
    configs=[],
    key=["n_rows"],
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_rows
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(y_ptr + offs, x, mask=mask)
```

### step3.尝试半自动autotune
`hints` 是 Triton-Ascend 在 `autotune` 装饰器中新增的一个参数，类型为 `dict`，用于给 Triton-Ascend autotune 提供该 triton kernel 的一些关键信息，帮助 autotune 更好的生成 tiling 配置。

hints 参数说明:
* 当前 `hints` 参数中可以识别的字段有：
    * `split_params`: dict[str, str]，分核参数的映射关系，例如 `{"x": "BLOCK_M"}` 表示 `BLOCK_M` 是沿 `x` 轴切 program
    * `tiling_params`：dict[str, str]，切块参数的映射关系，例如 `{"y": "BLOCK_N"}` 表示 `BLOCK_N` 是沿 `y` 轴切 block
    * `low_dim_axes`：list[str]，低维轴的列表，例如 `["y"]` 表示 `y` 轴是低维轴
    * `reduction_axes`：list[str]，规约轴的列表，例如 `[]` 表示没有规约轴
    * `auto_gen_config`：bool，是否自动生成 tiling 配置，默认值为 `True`
* 注意：
    * 通过 `hints` 来显示指定轴关系时，autotune 中原本的参数 `key` 必须改为字典形式传入，因为后续 `split_params`、`tiling_params` 等参数都是按轴名来填写，需要和 `key` 里的轴名对应起来
    * 通过 `hints` 来显示指定轴关系时，`split_params`、`tiling_params`、`low_dim_axes`、`reduction_axes` 必须传入，即使某些参数为空
    * 合法的轴名称是 `x/y/z/w/v/t`，仅仅用做关系映射
    * `split_params` 和 `tiling_params` 为自动生成 tiling 算法必须的输入，`low_dim_axes` 和 `reduction_axes` 为 tiling 算法的可选输入，用于优化 tiling 效果，留空时 tiling 也能够自动生成，但可能会影响生成的候选 tiling 数量和质量
    * 当用户传入的 configs 不为空时，`auto_gen_config` 默认值为 `False`，如果希望此时也希望自动生成 tiling 配置并与用户传入的 configs 合并，需要显式在 `hints` 中传入 `"auto_gen_config": True`

使用示例：
```python
import triton
import triton.language as tl
import triton.backends.ascend.runtime

@triton.autotune(
    # configs 为空列表，表示不传入自定义配置
    # 此时 auto_gen_config 默认为 True，会自动生成 tiling 配置
    configs=[],
    
    # key 使用字典形式，轴名必须与 hints 中的轴名对应
    # "x" 对应 n_rows（行数），"y" 对应 n_cols（列数）
    # autotune 会根据这些维度值来缓存和选择最佳配置
    key={"x": "n_rows", "y": "n_cols"},
    
    # hints 参数：显式指定轴与 tiling 参数的映射关系
    hints={
        # split_params: 分核参数映射，指定沿哪个轴切分 program（任务）
        # "x": "BLOCK_M" 表示 BLOCK_M 沿 x 轴切分，即按行方向分核
        # 每个 program 处理 BLOCK_M 行数据
        "split_params": {"x": "BLOCK_M"},
        
        # tiling_params: 切块参数映射，指定沿哪个轴切分 block（数据块）
        # "y": "BLOCK_N" 表示 BLOCK_N 沿 y 轴切分，即按列方向切块
        # 每行数据在列方向上被切分为 BLOCK_N 大小的块，通过 for 循环处理
        "tiling_params": {"y": "BLOCK_N"},
        
        # low_dim_axes: 低维轴列表，用于优化 tiling 效果
        # ["y"] 表示 y 轴（列方向）是低维轴，访问连续性更好，适合作为内层循环
        "low_dim_axes": ["y"],
        
        # reduction_axes: 规约轴列表，本 kernel 无规约操作（如 sum/max 等）
        # 为空列表表示没有规约轴
        "reduction_axes": [],
        
        # auto_gen_config: 默认为 True，表示自动生成 tiling 配置
        # 由于 configs 为空，此处使用默认值 True 即可，无需显式传入
    },
)
@triton.jit
def kernel_with_hints(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]

    for n0 in range(0, n_cols, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)[None, :]
        mask_m = offs_m < n_rows
        mask_n = offs_n < n_cols
        mask = mask_m & mask_n

        x = tl.load(x_ptr + offs_m * n_cols + offs_n, mask=mask, other=0)
        tl.store(y_ptr + offs_m * n_cols + offs_n, x, mask=mask)
```

### step4.尝试自定义autotune
手写一组 `triton.Config` 传入参数 `configs` 中，手写 triton.Config 的总体原则
1. 对于影响 grid 发射核数的参数，一般我们尽量让其能够等于物理核数，如果数据量较小，也可能发射较少核数的时候能获得最优性能；对于影响 tile 块大小的参数，我们尽量在不产生 UB overflow 的情况下让其尽可能大，同时避免尾块的产生
2. 影响 grid 发射核数的参数：可以按照总长度从高到低设置为 X, X/2, X/4 等等的值，如果输入 shape 较大，可以设置为让 grid 发射核数正好等于物理核数的大小，例如 (X + num_cores - 1) // num_cores；这里是以一个切分轴为例，如果存在多个切分轴，那么就需要按照乘积来计算
3. 影响 tile 块大小的参数：起始值为切分轴参数（如果存在）或者轴长度，注意当该轴长度特别大的时候，我们可以直接从 16384 这样一个经验值开始取，然后按照 X / 2, X / 4 这样去取值
4. 上述按 2 的幂次方下降的值采样较为粗粒度，如果用户想要得到极致的最优性能，尤其是在输入大小不规则的情况下，需要在可能的最优区间内细粒度撒点，可以通过粗粒度采样后确认性能最优的大致区间后再进一步细分来实现。
5. 对于 vector 类算子，在设置了上述 tiling 大小的配置候选集后，可以加上 multibuffer 编译选项的调优。

示例：
```python
import triton
import triton.language as tl
import triton.backends.ascend.runtime


def get_configs():
    return [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN, "multibuffer": MB})
        for BM in [256, 128, 64, 32]
        for BN in [128, 64, 32, 16]
        for MB in [True, False]
    ]


@triton.autotune(
    configs=get_configs(),
    key=["n_rows", "n_cols"],
)
@triton.jit
def manual_config_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < n_rows) & (offs_n < n_cols)

    x = tl.load(x_ptr + offs_m * n_cols + offs_n, mask=mask, other=0)
    tl.store(y_ptr + offs_m * n_cols + offs_n, x, mask=mask)
```

#### BLOCK_SIZE调优
BLOCK_SIZE（分块大小，也包括BLOCK_M、BLOCK_N、BLOCK_K等名称的超参数）是影响 Triton 算子性能的关键参数之一。合理的 BLOCK_SIZE 配置可以充分利用硬件的并行计算能力和缓存局部性，从而显著提升性能。

##### 调优原则

1. 考虑内存带宽和缓存

- 较大的 BLOCK_SIZE 可以更好地利用数据局部性，减少重复加载
- 但过大的 BLOCK_SIZE 会导致缓存竞争和并行度下降
- 需要在带宽利用率和并行度之间取得平衡

2. 考虑数据类型

- 不同数据类型的向量宽度不同，BLOCK_SIZE 应与向量宽度匹配
- fp32、fp16、bfp16 等类型可能有不同的最优 BLOCK_SIZE

BLOCK_SIZE调优的经验规则：
- BLOCK_SIZE 通常选择 2 的幂次方


3. 关键点

- **避免过小 BLOCK_SIZE**：过小的 BLOCK_SIZE 会导致并行度不足，无法充分利用硬件

- **避免过大 BLOCK_SIZE**：过大的 BLOCK_SIZE 会导致缓存竞争和资源浪费

- **多维度协调**：BLOCK_M、BLOCK_N、BLOCK_K 需要联合调优，找到最优组合


##### 关键约束

1. 核心规则

```
BLOCK_SIZE 必须 <= 被分块的维度大小
```

2. 原因分析

当 BLOCK_SIZE > 维度大小时，padding 的 0.0 会被 `tl.sum()` 累加，污染统计结果。

```python
# ❌ 错误示例
W = 512
BLOCK_SIZE = 4096  # > W

for w_start in range(0, W, BLOCK_SIZE):  # 只循环一次
    w_offs = w_start + tl.arange(0, BLOCK_SIZE)  # [0, 1, ..., 4095]
    mask = w_offs < W  # 只有前 512 个是 True

    data = tl.load(..., mask=mask, other=0.0)
    # data = [有效数据×512, padding 0.0×3584]

    sum_val += tl.sum(data)  # 累加了 3584 个 0.0！
    # 结果：sum 被污染，后续计算全部错误
```

3. padding 比例

```
padding 比例 = (BLOCK_SIZE - 维度大小) / BLOCK_SIZE
```

| BLOCK_SIZE | 维度大小 | padding 比例 | 结果 |
|-----------|---------|-------------|------|
| 512 | 512 | 0% | ✅ 正确 |
| 1024 | 512 | 50% | ❌ 错误 |
| 4096 | 512 | 87.5% | ❌ 错误 |
| 8192 | 262144 | 0.03% | ✅ 正确 |

##### BLOCK_SIZE 的影响

1. 优点：增大 BLOCK_SIZE

```python
# BLOCK_SIZE = 512
for start in range(0, 262144, 512):  # 循环 512 次
    ...

# BLOCK_SIZE = 8192
for start in range(0, 262144, 8192):  # 循环 32 次
    ...
```

- ✅ 减少循环迭代次数
- ✅ 减少循环开销
- ✅ 更大的数据块，更好的内存带宽利用

2. 缺点：增大 BLOCK_SIZE

```python
# BLOCK_SIZE = 512
data = tl.load(..., other=0.0)  # 加载 512 个元素

# BLOCK_SIZE = 16384
data = tl.load(..., other=0.0)  # 加载 16384 个元素
```

- ❌ 更高的寄存器压力
- ❌ 可能降低并行度（SM 资源有限）
- ❌ 可能触发寄存器溢出

3. 最优值需要autotune搜索

| BLOCK_SIZE | 循环次数 | 寄存器压力 | 性能 |
|-----------|---------|-----------|------|
| 512 | 多 | 低 | 基准 |
| 2048 | 中 | 中 | 较好 |
| 4096 | 少 | 中 | 好 |
| **8192** | **更少** | **中高** | **最优** |
| 16384 | 最少 | 高 | 可能更慢 |

##### 案例：BatchNorm2d

1.  测试配置

- 数据尺寸：(64, 64, 512, 512)
- 分块维度：H×W = 262144
- 可用 BLOCK_SIZE 范围：[1, 262144]

2.  用于autotune的BLOCK_SIZE取值和结果

| BLOCK_SIZE取值 | 循环迭代次数 | 平均延迟 | 加速比 |
|-----------|-------------|---------|--------|
| 512 | 512 | ~20 ms | 基准 |
| 2048 | 128 | ~14 ms | 1.43x |
| 4096 | 64 | ~14 ms | 1.43x |
| **8192** | **32** | **11.67 ms** | **1.71x** |

##### 常见错误

1. 错误 1：BLOCK_SIZE 超过分块维度

```python
# ❌ 错误：分块 W 维度 (512)，BLOCK_SIZE > W
W = 512
BLOCK_SIZE = 4096  # > W

for w_start in range(0, W, BLOCK_SIZE):
    # padding 87.5%，结果错误
```

2. 错误 2：认为越大越好

```python
# ❌ 错误：盲目增大
BLOCK_SIZE = 16384  # 可能寄存器溢出

# ✅ 正确：实测确定
# 在 BatchNorm 中，8192 > 16384 性能更好
```

3. 错误 3：忽略硬件限制

```python
# ❌ 错误：超过硬件限制
BLOCK_SIZE = 65536  # 可能超过共享内存或寄存器限制

# ✅ 正确：考虑硬件限制
# 最大 BLOCK_SIZE 通常受限于：
# - 寄存器数量
# - 共享内存大小
# - warp 大小（通常 32 或 64）
```

##### 经验值

| 场景 | 推荐 BLOCK_SIZE |
|------|----------------|
| 小维度 (< 1024) | 维度大小或 512 |
| 中维度 (1K-64K) | 2048 或 4096 |
| 大维度 (> 64K) | 4096 或 8192 |
| 统计算子 (mean, sum) | 可偏大 (8192) |
| 复杂算子 (conv, matmul) | 偏小 (256-1024) |

##### 维度合并与 BLOCK_SIZE 的关系

维度合并扩展可用范围

```python
# 原始：分块 W 维度
W = 512
# 可用 BLOCK_SIZE: [1, 512]

# 维度合并：分块 H×W 维度
HW = H * W = 262144
# 可用 BLOCK_SIZE: [1, 262144]
```

**注意：能用更大的 BLOCK_SIZE 是维度合并的副作用，不是主要目的！**

维度合并的主要收益：
- 减少循环层数
- 减少重复计算
- 提高内存连续性

**核心原则：**
1. 确保 `BLOCK_SIZE <= 分块维度大小`
2. BLOCK_SIZE 不是越大越好
3. 通过 autotune 确定最优值
4. 考虑寄存器压力和并

## 三、性能采集方式

### 默认方式：benchmark

```python
# 默认使用 benchmark 方式获取片上计算时间
```

### Profiling 方式（小 shape 推荐）

```python
import os
os.environ["TRITON_BENCH_METHOD"] = "npu"  # 使用 profiler

# 对于小 shape 算子，能获取更准确的计算时间
# 但会显著增加整体 autotune 时间，请谨慎开启
```

---

## 四、问题定位

排查问题时，建议先启动环境变量debug：
```bash
export TRITON_PRINT_AUTOTUNING=1
```
日志中可以直接看到：

* 识别出的 split axes；
* 识别出的 tiling axes；
* 识别出的 low-dimensional axes；
* 识别出的 reduction axes；
* 生成的 config 数量。

### 自动生成 Profiling 结果

```python
@triton.autotune(
    auto_profile_dir="./profile_result",  # 输出目录
    configs=[...],
    key=[...],
)
@triton.jit
def kernel(...):
    ...
```

自动在指定目录生成最优 kernel 配置的 profiling 结果。

### 常见问题 1：configs=[] 解析失败

**原因：** 切分/tiling 参数无法从 DSL 唯一识别

**解决：**
1. 添加 `hints` 显式指定
2. 或改为手写 `configs`

### 常见问题 2：自动生成配置质量差

**原因：** 参数耦合方式不适合自动算法

**解决：** 手写 Config，参考分核优化原则

### 常见问题 3：Kernel 没有可调参数

**原因：** 所有 `tl.constexpr` 都被显式传入

**解决：** 移除部分显式传入，让 autotune 接管

### 常见问题 4：性能抖动大

**原因：** 小 shape 测试不稳定

**解决：**
```bash
export TRITON_BENCH_METHOD=npu
export TRITON_PRINT_AUTOTUNING=1
```

---

### 问题快速排查表

| 现象                              | 可能的原因                       | 建议动作                      |
| ------------------------------- | ---------------------------- | ------------------------- |
| configs=[] 直接解析失败              | split/tiling 轴没有从 DSL 唯一识别出来 | 先补 hints，再试             |
| parser 能识别一部分，但总差一个参数           | 某个参数没有和轴长度 mask 建立联系         | 改 DSL 写法或改手写 config      |
| kernel 完全没有合适的 tl.constexpr 可调项 | DSL 没暴露调参接口                  | 先改 kernel dsl，再谈 autotune |
| 自动生成能跑，但候选质量明显差                 | 当前算法不适合该 kernel 的参数耦合方式      | 手动构造 config 传入      |

---

## 总结

| 方式 | 适用场景 | 复杂度 |
|------|---------|--------|
| 自定义 autotune | 需要精确控制搜索空间 | 中 |
| 自动 autotune (configs=[]) | DSL 规范，参数清晰 | 低 |
| 半自动 autotune (hints) | 自动解析失败，但能人工判断 | 中 |

**限制：** 进阶用法仅支持 Vector 类算子，不支持 Cube 类算子。

**优先级：** 自动 autotune > 半自动 autotune (hints) > 自定义 autotune
