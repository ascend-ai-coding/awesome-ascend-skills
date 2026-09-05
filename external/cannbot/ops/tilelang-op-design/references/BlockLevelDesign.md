# Block Level Design

## 硬件架构

理解 block-level design 时，可以先有一个最基本的 Ascend 硬件图景：算子运行在 AI Core 上。AI Core 主要由三部分组成：`Cube` 计算单元负责矩阵类计算，`Vector` 计算单元负责逐元素、归约、`gather`、`scatter` 等向量类计算，`Scalar` 主要负责指令发射，也会参与循环控制和地址计算，其计算能力较弱，实现时应尽量减少 Scalar 侧计算。此外，`Cube` 和 `Vector` 计算单元可以并行运行，这也是后文进行 `C/V` 分工和流水设计的基础。

存储结构上，也可以先用一个和 GPU 对照的粗略理解：`global memory` 对应 `global memory`，`shared memory` 可近似对应 `Cube` 侧的 `L1 buffer` 和 `Vector` 侧的 `Unified Buffer`，`register memory` 可近似对应 `L0A/L0B/L0C buffer`。输入和输出 tensor 在开始时位于 `global memory` 上，`Cube Core` 和 `Vector Core` 之间需要通过 `global memory` 传递数据。block-level 设计里用到的 `workspace tensor` 同样位于 `global memory` 上，用于存储中间结果。

## Block Level Design 总览

Block-level design 用来确定 kernel 的块级组织方式，不展开具体计算细节，主要回答四件事：

1. 每个 block 负责输出张量中的哪一段区域。
2. 为了得到这段输出，block 内需要遍历哪些数据。
3. `C` 和 `V` 如何分工，以及如何交错执行。
4. 当不同 shape 需要不同块级组织方式时，是否需要设计多个 `T.prim_func` 模版并在 trace-time 选择。

`block` 指一个逻辑任务块，即输出张量上的一段工作区域，以及为了得到这段输出所需要遍历的一组输入数据。`block-level design` 讨论的就是这些逻辑任务块如何划分、如何遍历、以及它们在 kernel 内如何组织执行。

`T.Kernel(...)` 启动时给出的通常是实际参与执行的 AI Core 数量。`cid` 表示当前 AI Core 的编号。逻辑 block 到 AI Core 的映射关系通常写在 TileLang kernel 内部，例如使用 persistent-kernel 风格的循环、任务队列或显式索引计算来完成调度，常见写法如 `bx = coreIdx * tasksPerCore + localIdx`。因此，本文中的 block-level 设计描述的是**逻辑任务块如何在 kernel 内被划分、遍历，并分配给已经启动的 AI Core 执行**。

## 文件分工

通常需要同时修改两个文件：

- `design/block_level/*.py`
  作用：定义任务划分、流水骨架、workspace 和同步关系，细节计算保留给 tile-level。
  常见结构：
  - `pass_configs = {...}`：配置 TileLang 编译与调度选项。
  - `@tilelang.jit(...)`：定义 kernel generator。
    - `out_idx`：`main(...)` 参数列表中哪些位置是输出 Tensor。
    - `workspace_idx`：哪些位置是 workspace Tensor，可为单个下标或下标列表。
  - `def op(...):`：接收问题规模参数，计算 block 大小、循环次数和 workspace 形状。
  - `@T.prim_func` / `def main(...):`：定义设备侧 kernel 本体。
  - `with T.Kernel(block_num, is_npu=True) as (cid, vid):`：启动 kernel；`cid` 标识当前 block，`vid` 标识当前 block 内的向量侧分工，从 `0` 开始计数。
- `model_new_tilelang.py`
  作用：将 PyTorch 输入整理成 block-level 设计所要求的布局，必要时做轴合并、pack / unpack，并按 shape build 后调用 kernel。

## 任务划分

任务划分先确定 block 之间如何分工，再确定 block 内如何遍历。主要考虑以下几点：

1. 优先选择 block 间无写冲突、无额外同步的切分。
2. 若这种切分明显损失并行度或数据访问效率，可引入可控的跨 block 归并。
3. 原始 layout 不利于分工时，可做轴合并、拆分或重排。

### 例子一：Matmul

- 输出：`C[M, N]`
- 常见切分：沿 `M/N` 分给不同 block，`K` 在 block 内循环。

```python
C[bx * block_M:(bx + 1) * block_M,
  by * block_N:(by + 1) * block_N]
```

这种切分的特点是：

- 每个 block 独占一段输出区域，没有写冲突。
- `K` 留在 block 内时，可以直接复用局部累加结果。

当 `M` 和 `N` 较小、仅沿 `M/N` 切分无法提供足够并行度时，也可以进一步沿 `K` 方向切分到多个 block。此时多个 block 会分别计算同一段输出的部分累加结果，最终需要额外归并。是否采用这种方案，取决于新增并行度带来的收益能否覆盖归并成本。

### 例子二：Flash Attention

- 输出：`O[B, H, S, D]`
- 常见切分：每个 block 固定一个 `(batch, head, q_block)` 对应的输出区域，在 block 内顺序遍历所有 `KV` 分块。

```python
O[bz, by, bx * block_M:(bx + 1) * block_M, :]
```

### 例子三：Flash Attention GQA

GQA 通常需要先做布局整理，再定义 block 的分工。

典型变换：

- `Q: [B, H, S_q, D] -> [B * H_kv, group_size * S_q, D]`
- `K/V: [B, H_kv, S_kv, D] -> [B * H_kv, S_kv, D]`

完成轴合并后，再按 `(batch, kv_head, packed_q_block)` 分配 block。这样可以把原本分散的依赖关系整理成 block 内更容易连续处理的布局。

### 例子四：Pooling 与 Scan 类算子

这类算子常常需要先做重排，再设计 block 分工；否则后续实现很容易退化为标量循环。

- 对 pooling（如 `avg_pool2d`、`avg_pool3d`），通常应先把 channel 维移到最后：
  - `NCHW -> NHWC`
  - `NCDHW -> NDHWC`
  这样同一空间位置上的 channel 向量连续，便于一次向量加载并在窗口遍历时对多个 channel 做 SIMD 累加 / 比较。
- 对 `cumsum`、`cumprod` 这类 scan 算子，通常应先把用户指定的扫描轴移到 axis 0。
  这样可以把问题整理成“axis 0 串行、其余维并行”的形式，使每一步前缀更新能够同时处理多条独立序列。

如果后续 kernel 明显依赖某种 layout 才能进行向量加载或避免标量循环，那么这种重排应在 block-level 阶段就明确写出，而不是等到 tile-level 再临时决定。

## `model_new_tilelang.py`

`model_new_tilelang.py` 负责两件事：

1. 按 block-level 设计整理输入布局。
2. 按输入 shape build 并调用 kernel。

普通情况下，如果输入布局与 kernel 一致，可以直接 build：

```python
batch, heads, seq_len, dim = q.shape
kernel = tl_kernel(batch, seq_len, heads, dim)
```

如果 block-level 设计要求先合并轴，则先做 pack / reshape，再 build kernel。例如 GQA 中常见的 `Q` 变换为：

```python
q = q.reshape(batch, kv_heads, group_size, q_seq_len, dim).reshape(
    batch * kv_heads,
    group_size * q_seq_len,
    dim,
)
```

## 模版设计：按 Shape 选择不同 `prim_func`

很多算子并不存在一套 block-level 方案可以同时覆盖所有 shape。

因此，block-level design 可以先定义多个候选 `T.prim_func`，再在 Python trace-time 根据 shape 选择返回哪一个。这样做有几个好处：

- 设计层面可以把不同 shape 的分工方式表达清楚，而不是把所有分支揉进一个巨大的 kernel。
- tile-level 和 AscendC 转译时，结构会更清晰：通常一个 `prim_func` 对应一套独立 kernel 实现。
- host 侧可以直接根据 shape 选择最合适的 kernel，而不是在设备侧引入复杂运行时分支。

推荐模版如下：

```python
@tilelang.jit(out_idx=[...], pass_configs=pass_configs)
def op(shape0, shape1, ..., dtype="float32"):
    # 先根据 shape 计算 block 组织参数
    ...

    @T.prim_func
    def fast_path(...):
        with T.Kernel(block_num_fast, is_npu=True) as (cid, vid):
            ...

    @T.prim_func
    def fallback_path(...):
        with T.Kernel(block_num_fallback, is_npu=True) as (cid, vid):
            ...

    # 在 trace-time 决定返回哪一个 prim_func
    if some_shape_condition:
        return fast_path
    return fallback_path
```

### 例子：RMSNorm 按 `N` 选择不同 `prim_func`

`rms_norm` 是一个典型例子。其输出是 `Y[M, N]`，每一行都需要做：

- 计算 `mean(x^2)`
- 加 `eps`
- `rsqrt`
- 再乘回 `x` 和 `gamma`

当 `N` 较小时，可以一次处理多行，把 `Gamma[N]` 广播到 `(row_factor, N)`，从而提高吞吐；但当 `N` 较大时，这种做法会显著增加 UB 压力，因此更适合退化为单行路径。

一个推荐的 block-level 模版如下：

```python
@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def rms_norm(M, N, eps=1e-5, dtype="float32"):
    block_M = 64
    num_physical_cores = 20
    m_num = M // block_M
    usedCoreNum = min(num_physical_cores, m_num)
    tasksPerCore = (m_num + usedCoreNum - 1) // usedCoreNum
    vec_num = 2
    sub_block_M = block_M // vec_num

    row_factor = 8
    row_loops = sub_block_M // row_factor

    @T.prim_func
    def merge_n(
        X: T.Tensor((M, N), dtype),
        Gamma: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            coreIdx = cid
            for localIdx in T.serial(tasksPerCore):
                bx = coreIdx * tasksPerCore + localIdx
                with T.Scope("V"):
                    if bx < m_num:
                        for r in T.serial(row_loops):
                            row_base = bx * block_M + vid * sub_block_M + r * row_factor
                            # TODO(tile-level):
                            # - load X[row_base:row_base + row_factor, :]
                            # - broadcast Gamma
                            # - reduce over N for each row
                            # - write back row_factor rows

    @T.prim_func
    def single_row(
        X: T.Tensor((M, N), dtype),
        Gamma: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            coreIdx = cid
            for localIdx in T.serial(tasksPerCore):
                bx = coreIdx * tasksPerCore + localIdx
                with T.Scope("V"):
                    if bx < m_num:
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            # TODO(tile-level):
                            # - load X[row_idx, :]
                            # - load Gamma
                            # - reduce over N for this row
                            # - write back one row

    if N <= 1024:
        return merge_n
    return single_row
```

## 流水设计

流水设计讨论的是 block 内部的执行排布。目标是根据依赖关系安排 block 内各阶段的先后与重叠，使 Cube 和 Vector 计算单元在 steady-state 中尽量持续工作。

- `C` 指 Cube 计算单元，负责矩阵乘等 Cube 类计算。
- `V` 指 Vector 计算单元，负责逐元素、归约、softmax、激活和 merge 等 Vector 类计算（包括 gather 等基于索引的数据访问与重排操作）。

在 block-level 设计中，通常需要先把任务拆成若干子阶段，例如：

- `C1`、`C2`：不同的 Cube 阶段
- `V1`、`V2`：不同的 Vector 阶段

这些子阶段的划分取决于数据依赖和中间结果的交接方式。划分完成后，再按依赖关系灵活安排执行顺序，以尽量减少等待。

### 例子一：Matmul

- `C`：完成 matmul 主计算。
- `V`：完成 epilogue，例如激活、cast、写回。

### 例子二：Flash Attention

这个例子中，任务分解是固定的：

- `C1`：计算 `Q @ K_t^T`
- `V1`：完成 softmax update
- `C2`：计算 `P_t @ V_t`
- `V2`：将局部结果 merge 到输出

区别不在于如何分解任务，而在于如何调度这些阶段。

第一种是直接按单个 `KV` 分块的计算顺序组织：

如果按这种顺序处理每个 `t`，时间线是：

```text
t=0: C1(0) -> V1(0) -> C2(0) -> V2(0)
t=1: C1(1) -> V1(1) -> C2(1) -> V2(1)
t=2: C1(2) -> V1(2) -> C2(2) -> V2(2)
```

这种写法直接对应算法步骤，但问题也很直接：

- `C1` 结束后，Cube 单元必须等待 `V1`
- `V1` 结束后，Cube 单元才能进入 `C2`
- `C2` 结束后，Vector 单元才能进入 `V2`
- 当前 `t` 完成之前，下一个 `t` 的后续阶段无法推进

因此 Cube 和 Vector 会频繁互相等待。

第二种可参考 `flash_attention/design/block_level/flash_attention.py` 中采用的交错流水，通过 workspace、cross flag 和 ring buffer 将不同 `t` 的阶段交错执行。

其核心不是按单个 `t` 串行走完 `C1 -> V1 -> C2 -> V2`，而是在每个 `t` 上同时推进当前分块的前序阶段和更早分块的后续阶段。与 `flash_attention/design/block_level/flash_attention.py` 一致的抽象时间线可写为：

```text
t=0:
  C: C1(0)
  V: V1(0)

t=1:
  C: C1(1)
  V: V1(1)

t=2:
  C: C1(2) + C2(0)
  V: V1(2) + V2(0)

t=3:
  C: C1(3) + C2(1)
  V: V1(3) + V2(1)

...

tail:
  C: C2(...)
  V: V2(...)
```

这里体现的是 preload / prelaunch：先推进若干个前序阶段，再进入前序阶段与后续阶段交错执行的稳态；ring buffer 和 cross flag 用于管理中间结果与阶段交接。

这样做的结果是：

- `C1/C2` 与 `V1/V2` 不再按单个分块完全串行执行
- 不同分块的不同阶段可以在稳态中重叠
- steady-state 中 Cube 和 Vector 的等待会明显减少
