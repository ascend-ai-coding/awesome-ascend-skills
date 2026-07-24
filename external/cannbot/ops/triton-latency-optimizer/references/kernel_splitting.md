# Kernel 分裂优化

## 概述

多 Case 场景下，单个泛用 Kernel 往往无法在所有 Shape/Dtype 特征下都取得最优性能。
Kernel 分裂优化通过按 Case 特征生成专用 Kernel，并构建统一调度器进行自动路由，使每个
Case 都能使用最匹配的实现，同时保持对外接口和精度一致性。

**核心原则**：
1. **精度优先**：专用 Kernel 必须 100% 通过精度验证，否则回退到泛用 Kernel。
2. **性能底线**：专用 Kernel 性能必须不低于泛用 Kernel，否则不采用。
3. **调度透明**：对外接口不变，内部自动路由到最优 Kernel。
4. **渐进分裂**：按 Case 特征分组，不盲目一对一，优先合并相似特征。

## 适用条件

仅在同时满足以下条件时执行，否则跳过：
1. `total_cases > 1`（多 case 场景）。
2. Phase 4 最终 `speedup_vs_torch < 0.8`（泛用 Kernel 性能未达标）。

## 优化方法

### Step 1：Case 特征分析与分组

#### 1.1 加载参考经验

根据算子类型选择对应参考附录：

| 算子类型 | 识别特征 | 参考附录 |
|---------|---------|---------|
| **Reduce** | `sum/mean/max/min/softmax` 等单 kernel 归约操作 | 附录 A：Reduce 类算子 Kernel 分裂经验 |
| **NormStats+Apply** | stats 归约 + apply 逐元素双 kernel（BatchNorm/LayerNorm/GroupNorm/InstanceNorm/RMSNorm） | 附录 A（分组维度用归约规模 `group_elements` 或 `inner_size`，非单个空间轴长度） |
| **广播逐元素** | `add/sub/mul/div` + 存在 shape 不等 | 附录 B：广播逐元素算子 Kernel 分裂经验 |

#### 1.2 经验命中判定

**Reduce 类命中条件**：
- 任务包含归约操作，且存在多个 Case。
- 按 `inner_size`（reduce 轴位置）分组：`==1` → reduce-last，`>1` → reduce-non-last。
- 若命中，必须使用附录 A 中的分组建议和 BLOCK 配置。

**广播逐元素命中条件**：
- 任务为逐元素操作，且 Case 间存在 shape 不等（需要广播）。
- 按 `(out_ndim, broadcast_dims)` 分组：无广播 / 2D dim0/dim1 / 3D/4D。
- 若命中，必须使用附录 B 中的分组建议和 BLOCK 配置。

#### 1.3 未命中时的性能瓶颈分析与分组

若未命中任何参考附录，执行以下步骤：

1. **读取 Phase 4 最终性能文件**：
   - 优先读取：`{工作目录}/output/optimized_perf_result.json`
   - 备选读取：`{工作目录}/output/perf_result.json`

2. **筛选性能瓶颈用例**：从 `per_shape_results` 中提取 `speedup_vs_torch < 0.3` 的用例。

3. **分类归因**：

   | 归因维度 | 判定条件 | 典型原因 |
   |---------|---------|---------|
   | **Shape 过小** | 元素数 < 1024 | Kernel 启动开销占比过高 |
   | **Shape 过大** | 元素数 > 10M | 寄存器溢出、缓存未命中 |
   | **非对齐访问** | shape 非 2 的幂次 | mask 分支导致性能下降 |
   | **跨步访存** | stride > 1 且非连续 | 内存带宽利用率低 |
   | **特殊 dtype** | bf16/int8 等低精度 | 向量化策略不匹配 |

4. **生成分组**：为每个归因类别生成专属分组，输出分组清单：
   ```json
   [
     {
       "group_id": "grp_small_shape",
       "case_indices": [1, 3, 5],
       "features": {"element_count": "<1024", "dtype": "float16"},
       "bottleneck_reason": "kernel_launch_overhead",
       "baseline_perf": {"speedup_vs_torch": 0.15}
     }
   ]
   ```

### Step 2：专用 Kernel 生成

对每个分组，基于泛用 Kernel 进行特化。

#### 特化策略

| 策略 | 适用场景 | 优化方向 |
|------|---------|---------|
| **固定 constexpr** | Shape 固定或范围极小 | 将 BLOCK_SIZE、grid 等硬编码为 constexpr |
| **展开循环** | 小 Shape 场景 | 消除循环开销，完全展开 |
| **调整 Tiling** | 特定 Shape 比例 | 优化 tile 尺寸匹配 UB 容量 |
| **简化边界** | 尺寸对齐的 Case | 移除 mask 检查，使用无分支 load/store |
| **专用规约** | 特定规约轴 | 选择最优的 reduce 策略（原子/二分/树形） |

#### 生成要求

- 每个专用 Kernel 命名格式：`{op_name}_kernel_{group_id}`。
- 保持输入输出签名与泛用 Kernel 一致。
- 代码必须完整可编译，禁止占位符。

#### 专用 Kernel 的进一步优化

生成专用 Kernel 后，**必须继续应用本 skill 优化点 1-13 及 Block Size Scaling**，对每个专用 Kernel 进行进一步优化：

1. 按顺序检查 13 个优化点（constexpr / tiling / 分核 / ...）。
2. 命中则应用对应优化策略。
3. 执行 `references/checklist.md` 检查确保代码规范。
4. 输出优化后的专用 Kernel。

### Step 3：精度验证

对每个专用 Kernel 独立调用 `triton-op-verifier` 的 `verify.py`：

```bash
python3 <triton-op-verifier-path>/scripts/verify.py \
    --op_name <op_name> \
    --verify_dir <split_verify_dir>/<group_id> \
    --triton_impl_name <group_kernel_name> \
    --timeout 300
```

**判定**：
- `passed_cases == total_cases` → 通过，进入 Step 4。
- 任何失败 → 该组标记为 `fallback_to_base`，跳过后续步骤。

### Step 4：性能测试

对每个专用 Kernel 独立调用 `triton-op-verifier` 的 `benchmark.py`：

```bash
python3 <triton-op-verifier-path>/scripts/benchmark.py \
    --op_name <op_name> \
    --verify_dir <split_verify_dir>/<group_id> \
    --triton_impl_name <group_kernel_name> \
    --warmup 5 --repeats 50 \
    --output <split_perf_dir>/<group_id>_perf.json
```

**判定**：
- `speedup_vs_torch >= baseline_speedup` → 采纳。
- 否则 → 标记为 `fallback_to_base`。

### Step 5：调度器构建

构建统一的 `ModelNew` 类，必须将路由逻辑封装在独立的 `_route` 方法中，`forward()` 仅负责调用该方法：

```python
class ModelNew(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 初始化所有专用 Kernel 和泛用 Kernel
        self.base_kernel = BaseKernel(...)
        self.specialized_kernels = {
            group_id: SpecializedKernel(group_id, ...)
            for group_id, adopted in adopted_groups.items()
        }

    def forward(self, *args):
        # forward 保持极简，仅调用一次路由函数
        return self._route(*args)

    def _route(self, *args):
        # 路由逻辑全部在此
        # 1. 提取输入 shape/dtype 特征
        # 2. 匹配分组规则
        # 3. 返回对应 kernel 启动结果，若无匹配则返回 base_kernel 结果
        if condition_1:
            return kernel_1[grid](...)
        elif condition_2:
            return kernel_2[grid](...)
        else:
            return self.base_kernel[grid](...)
```

**关键约束**：
- **禁止**在 `forward()` 中直接编写 `if-elif-else` 路由分支。
- **必须**使用 `_route` 方法封装路由逻辑。
- 路由开销必须 < 0.1ms（使用简单的 shape 比较，禁止复杂计算）。

### Step 6：集成验证

对分裂后的完整代码执行全量验证：

1. 使用 `verify.py` 验证所有 Case 精度通过。
2. 使用 `benchmark.py` 测试整体性能。
3. 生成分裂汇总，包含：
   - 每组采用的 Kernel 类型（specialized / base）。
   - 每组性能对比（vs baseline）。
   - 整体几何平均加速比。

## 输出格式

最终输出 `split_kernel.py`，结构如下：

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

# === 泛用 Kernel（保留原样） ===
@triton.jit
def {op_name}_base_kernel(...): ...

# === 专用 Kernel 1 ===
@triton.jit
def {op_name}_kernel_grp1(...): ...

# === 专用 Kernel 2 ===
@triton.jit
def {op_name}_kernel_grp2(...): ...

# === 调度器 ===
class ModelNew(nn.Module):
    def __init__(self, ...): ...
    def _route(self, ...): ...
    def forward(self, ...): ...
```

## 关键点

1. **触发条件严格**：仅 `total_cases > 1` 且 `speedup_vs_torch < 0.8` 时执行，否则跳过。
2. **精度零妥协**：任何专用 Kernel 精度不通过，立即回退到泛用 Kernel。
3. **性能底线**：专用 Kernel 必须 ≥ 泛用 Kernel 性能，否则不采用。
4. **路由封装**：路由逻辑必须封装在 `_route` 方法中，`forward` 仅调用该方法。
5. **代码自包含**：所有 Kernel 和调度逻辑必须在同一文件内。
6. **禁止过度分裂**：相似 Case 必须合并分组，禁止 1-to-1 无意义分裂。
7. **回退安全**：路由逻辑必须包含兜底机制，确保 100% 覆盖所有 Case。

---

# 附录 A：Reduce 类算子 Kernel 分裂经验

## A.1 分组维度

| 维度 | 判定条件 | 分组 |
|------|---------|------|
| **Reduce 轴位置** | `inner_size == 1` vs `inner_size > 1` | reduce-last / reduce-non-last |
| **Reduce 规模** | `reduce_size < 256` / `256~4096` / `>4096` | 小/中/大 |
| **数据类型** | fp16/bf16/fp32 | 不同精度需不同向量化策略 |

---

## A.2 Reduce-Last 特化（`inner_size == 1`）

### A.2.1 适用场景
Reduce 轴在最后维度，数据在内存中连续，可直接线性访存。

### A.2.2 Kernel 实现

```python
@triton.jit
def sum_kernel_reduce_last(
    input_ptr, output_ptr,
    outer_size, reduce_size,
    BLOCK_RED: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # Grid 动态分配：按 VEC_CORE_NUM 限制
    tiles_per_pid = (outer_size + num_pids - 1) // num_pids
    tile_start = pid * tiles_per_pid
    tile_end = tl.minimum(tile_start + tiles_per_pid, outer_size)

    for outer_idx in range(tile_start, tile_end):
        acc = tl.full((), 0.0, tl.float32)
        base_offset = outer_idx * reduce_size

        # 沿 reduce 轴分块累加
        for r_start in range(0, reduce_size, BLOCK_RED):
            r_end = tl.minimum(r_start + BLOCK_RED, reduce_size)
            r_offsets = r_start + tl.arange(0, BLOCK_RED)
            r_mask = r_offsets < r_end

            vals = tl.load(input_ptr + base_offset + r_offsets, mask=r_mask, other=0.0)
            vals = vals.to(tl.float32)
            acc += tl.sum(vals, axis=0)

        tl.store(output_ptr + outer_idx, acc.to(output_ptr.dtype.element_ty))
```

### A.2.3 关键优化点

| 优化点 | 做法 | 收益 |
|-------|------|------|
| **连续访存** | `base_offset + r_offsets` 线性偏移，无 stride | 最大化内存带宽利用率 |
| **固定 constexpr** | `BLOCK_RED` 声明为 constexpr | 编译期优化，消除运行时分支 |
| **Grid 限制** | `grid = (min(outer_size, VEC_CORE_NUM),)` | 避免过度并行导致调度开销 |
| **标量累加器** | `acc = tl.full((), 0.0, tl.float32)` | 单值累加，减少寄存器压力 |

### A.2.4 Grid 配置

```python
grid = (min(outer_size, self.VEC_CORE_NUM),)
sum_kernel_reduce_last[grid](
    x, output,
    outer_size, reduce_size,
    BLOCK_RED=1024,
)
```

---

## A.3 Reduce-Non-Last 特化（`inner_size > 1`）

### A.3.1 适用场景
Reduce 轴不在最后维度，需要 2D tile 策略同时处理 reduce 轴和 inner 轴。

### A.3.2 Kernel 实现

```python
@triton.jit
def sum_kernel_reduce_non_last(
    input_ptr, output_ptr,
    outer_size, reduce_size, inner_size,
    BLOCK_RED: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # 2D tile 映射：outer × inner 展平为 1D grid
    inner_tiles = (inner_size + BLOCK_INNER - 1) // BLOCK_INNER
    total_tiles = outer_size * inner_tiles

    tiles_per_pid = (total_tiles + num_pids - 1) // num_pids
    tile_start = pid * tiles_per_pid
    tile_end = tl.minimum(tile_start + tiles_per_pid, total_tiles)

    for tile_idx in range(tile_start, tile_end):
        outer_idx = tile_idx // inner_tiles
        inner_tile = tile_idx % inner_tiles
        inner_start = inner_tile * BLOCK_INNER
        inner_end = tl.minimum(inner_start + BLOCK_INNER, inner_size)
        inner_len = inner_end - inner_start

        # 向量累加器：shape = (BLOCK_INNER,)
        acc = tl.zeros((BLOCK_INNER,), dtype=tl.float32)
        base_offset = outer_idx * reduce_size * inner_size + inner_start

        for r_start in range(0, reduce_size, BLOCK_RED):
            r_end = tl.minimum(r_start + BLOCK_RED, reduce_size)

            # 2D offset 计算
            r_offs = r_start + tl.arange(0, BLOCK_RED)[:, None]
            i_offs = tl.arange(0, BLOCK_INNER)[None, :]
            in_offsets = base_offset + r_offs * inner_size + i_offs

            mask_r = r_offs < r_end
            mask_i = i_offs < inner_len
            mask = mask_r & mask_i

            vals = tl.load(input_ptr + in_offsets, mask=mask, other=0.0)
            vals = vals.to(tl.float32)
            acc += tl.sum(vals, axis=0)

        # 2D store
        out_base = outer_idx * inner_size + inner_start
        out_offsets = out_base + tl.arange(0, BLOCK_INNER)
        out_mask = tl.arange(0, BLOCK_INNER) < inner_len
        tl.store(output_ptr + out_offsets, acc.to(output_ptr.dtype.element_ty), mask=out_mask)
```

### A.3.3 关键优化点

| 优化点 | 做法 | 收益 |
|-------|------|------|
| **2D tiling** | 同时 tile reduce 轴和 inner 轴 | 充分利用并行度 |
| **向量累加器** | `acc = tl.zeros((BLOCK_INNER,), dtype=tl.float32)` | 批量累加，减少循环次数 |
| **Offset 计算** | `r_offs * inner_size + i_offs` 二维广播 | 精确映射 2D 内存布局 |
| **Grid 计算** | `total_tiles = outer_size * inner_tiles` | 按 `VEC_CORE_NUM` 限制 |

### A.3.4 Grid 配置

```python
inner_tiles = (inner_size + 63) // 64
total_tiles = outer_size * inner_tiles
grid = (min(total_tiles, self.VEC_CORE_NUM),)
sum_kernel_reduce_non_last[grid](
    x, output,
    outer_size, reduce_size, inner_size,
    BLOCK_RED=64,
    BLOCK_INNER=64,
)
```

---

## A.4 分组建议

| Case 特征 | 推荐 Kernel | BLOCK 配置 |
|----------|------------|-----------|
| `inner_size == 1`, reduce_size 任意 | reduce-last | `BLOCK_RED=1024` |
| `inner_size > 1`, reduce_size ≤ 256 | reduce-non-last | `BLOCK_RED=64, BLOCK_INNER=64` |
| `inner_size > 1`, reduce_size > 256 | reduce-non-last | `BLOCK_RED=128, BLOCK_INNER=32` |

---

## A.5 调度器实现

```python
import torch_npu
import triton.runtime.driver as driver

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        properties = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
        self.VEC_CORE_NUM = properties["num_vectorcore"]
        self.AI_CORE_NUM = properties["num_aicore"]

    def forward(self, x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
        return self._route(x, dim, keepdim)

    def _route(self, x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
        original_dim = dim
        if dim is None:
            x = x.reshape(1, -1)
            dim = 1
        if dim < 0:
            dim = x.ndim + dim

        shape = x.shape
        outer_size = math.prod(shape[:dim]) if dim > 0 else 1
        reduce_size = shape[dim]
        inner_size = math.prod(shape[dim + 1:]) if dim + 1 < x.ndim else 1

        if keepdim:
            out_shape = list(shape)
            out_shape[dim] = 1
        else:
            out_shape = list(shape[:dim]) + list(shape[dim + 1:])

        output = torch.empty(out_shape, dtype=x.dtype, device=x.device)

        # === 路由决策 ===
        if inner_size == 1:
            # reduce-last: 直接沿 reduce 轴连续访存
            grid = (min(outer_size, self.VEC_CORE_NUM),)
            sum_kernel_reduce_last[grid](
                x, output,
                outer_size, reduce_size,
                BLOCK_RED=1024,
            )
        else:
            # reduce-non-last: 2D tile 策略
            inner_tiles = (inner_size + 63) // 64
            total_tiles = outer_size * inner_tiles
            grid = (min(total_tiles, self.VEC_CORE_NUM),)
            sum_kernel_reduce_non_last[grid](
                x, output,
                outer_size, reduce_size, inner_size,
                BLOCK_RED=64,
                BLOCK_INNER=64,
            )

        if original_dim is None and not keepdim:
            output = output.squeeze()

        return output
```

**关键设计**：
- **无字典映射**：直接用 `if-else` 分支，零开销
- **内联 grid 计算**：每个分支独立计算 grid
- **constexpr 固定**：BLOCK 值硬编码

---

# 附录 B：广播逐元素算子 Kernel 分裂经验

## B.1 分组维度

| 维度 | 判定条件 | 分组 |
|------|---------|------|
| **是否广播** | `x.shape == y.shape` | 无广播 / 有广播 |
| **广播维度位置** | `broadcast_dims` 位置 | dim0 / dim1 / 多维 |
| **输出维度数** | `out_ndim` | 2D / 3D / 4D |

---

## B.2 无广播特化

### B.2.1 适用场景
`x.shape == y.shape`，纯逐元素操作，无广播开销。

### B.2.2 Kernel 实现

```python
@triton.jit
def add_no_broadcast_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    blocks_per_core = tl.cdiv(num_blocks, tl.num_programs(0))
    start_block = pid * blocks_per_core
    end_block = min(start_block + blocks_per_core, num_blocks)

    for block_idx in range(start_block, end_block):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        out = x + alpha * y

        tl.store(out_ptr + offsets, out, mask=mask)
```

### B.2.3 关键优化点

| 优化点 | 做法 | 收益 |
|-------|------|------|
| **大 BLOCK** | 推荐范围 `4096~16384`，通过 autotune 选择 | 充分利用带宽 |
| **Grid 动态** | `grid = (min(tl.cdiv(n_elements, BLOCK_SIZE), VEC_CORE_NUM),)` | 避免过度并行 |
| **循环分块** | `blocks_per_core` 分配 | 每个 core 处理多个 block |

---

## B.3 2D 广播特化

### B.3.1 适用场景
2D 张量，`y` 在某一维度上广播（如 `y=[M,1]` 或 `y=[1,N]`）。

### B.3.2 Broadcast Dim1（`y=[M,1]`）

```python
@triton.jit
def add_broadcast_2d_dim1_kernel(
    x_ptr, y_ptr, out_ptr, M, N, alpha,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_blocks_m * num_blocks_n
    blocks_per_core = tl.cdiv(num_blocks, tl.num_programs(0))
    start_block = pid * blocks_per_core
    end_block = min(start_block + blocks_per_core, num_blocks)

    for block_idx in range(start_block, end_block):
        bm = block_idx // num_blocks_n
        bn = block_idx % num_blocks_n
        row_offs = bm * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offs = bn * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = row_offs < M
        col_mask = col_offs < N
        mask_2d = row_mask[:, None] & col_mask[None, :]

        x = tl.load(x_ptr + row_offs[:, None] * N + col_offs[None, :], mask=mask_2d, other=0.0)
        y = tl.load(y_ptr + row_offs[:, None], mask=row_mask[:, None], other=0.0)
        out = x + alpha * y

        tl.store(out_ptr + row_offs[:, None] * N + col_offs[None, :], out, mask=mask_2d)
```

### B.3.3 Broadcast Dim0（`y=[1,N]`）

```python
# y load 改为 1D 列维度
y = tl.load(y_ptr + col_offs, mask=col_mask, other=0.0)
```

### B.3.4 关键优化点

| 优化点 | 做法 | 收益 |
|-------|------|------|
| **2D tiling** | `BLOCK_M` 推荐 `4~16`，`BLOCK_N` 推荐 `512~2048`，通过 autotune 选择 | 匹配广播模式 |
| **广播 load** | `y` 仅加载一次，利用 broadcast 语义 | 减少内存访问 |
| **2D mask** | `row_mask[:, None] & col_mask[None, :]` | 精确边界处理 |

---

## B.4 通用高维广播特化

### B.4.1 适用场景
3D/4D 张量，广播模式复杂，不适合在 kernel 内处理。

### B.4.2 特化策略
Host 端 `expand + contiguous + view(-1)`，退化为 1D kernel。

### B.4.3 Kernel 实现

```python
@triton.jit
def add_broadcast_1d_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    blocks_per_core = tl.cdiv(num_blocks, tl.num_programs(0))
    start_block = pid * blocks_per_core
    end_block = min(start_block + blocks_per_core, num_blocks)

    for block_idx in range(start_block, end_block):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        out = x + alpha * y

        tl.store(out_ptr + offsets, out, mask=mask)
```

### B.4.4 Host 端预处理

```python
out_shape = torch.broadcast_shapes(x.shape, y.shape)
x_expanded = x.expand(out_shape).contiguous().view(-1)
y_expanded = y.expand(out_shape).contiguous().view(-1)
output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
out_flat = output.view(-1)

n_elements = out_flat.numel()
# BLOCK_SIZE 和 grid 通过 autotune 或动态计算
BLOCK_SIZE = 8192  # 推荐初始值，可 autotune
num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
grid_size = min(num_blocks, self.VEC_CORE_NUM)
add_broadcast_1d_kernel[grid_size](
    x_expanded, y_expanded, out_flat, n_elements, alpha, BLOCK_SIZE
)
```

### B.4.5 关键优化点

| 优化点 | 做法 | 收益 |
|-------|------|------|
| **Host 端展平** | 利用 PyTorch 的 `expand` 处理广播 | kernel 简化为 1D |
| **内存连续化** | `contiguous()` 确保线性访存 | 避免跨步访存 |
| **复用 1D kernel** | 与无广播路径共享 kernel 代码 | 减少代码量 |

---

## B.5 广播信息提取

```python
def _get_broadcast_info(self, x_shape, y_shape):
    if x_shape == y_shape:
        return len(x_shape), [], False

    max_ndim = max(len(x_shape), len(y_shape))
    x_padded = [1] * (max_ndim - len(x_shape)) + list(x_shape)
    y_padded = [1] * (max_ndim - len(y_shape)) + list(y_shape)

    out_shape = []
    broadcast_dims = []
    for i in range(max_ndim):
        if x_padded[i] == y_padded[i]:
            out_shape.append(x_padded[i])
        elif x_padded[i] == 1:
            out_shape.append(y_padded[i])
        elif y_padded[i] == 1:
            out_shape.append(x_padded[i])
            broadcast_dims.append(i)
        else:
            raise ValueError(f"Incompatible shapes: {x_shape}, {y_shape}")

    return len(out_shape), tuple(broadcast_dims), True
```

---

## B.6 分组建议与参数范围

| Case 特征 | 推荐 Kernel | BLOCK 范围 | 推荐 autotune 配置 |
|----------|------------|-----------|-------------------|
| `x.shape == y.shape` | no-broadcast | `BLOCK_SIZE: [4096, 8192, 16384]` | `@triton.autotune(configs=[...], key=['n_elements'])` |
| 2D, `y` 在 dim1 广播 | broadcast-2d-dim1 | `BLOCK_M: [4, 8, 16], BLOCK_N: [512, 1024, 2048]` | `@triton.autotune(configs=[...], key=['M', 'N'])` |
| 2D, `y` 在 dim0 广播 | broadcast-2d-dim0 | `BLOCK_M: [4, 8, 16], BLOCK_N: [512, 1024, 2048]` | `@triton.autotune(configs=[...], key=['M', 'N'])` |
| 3D/4D 任意广播 | generic-1d | `BLOCK_SIZE: [4096, 8192, 16384]` | `@triton.autotune(configs=[...], key=['n_elements'])` |

---

## B.7 调度器实现

```python
import torch_npu
import triton.runtime.driver as driver

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        properties = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
        self.VEC_CORE_NUM = properties["num_vectorcore"]
        self.AI_CORE_NUM = properties["num_aicore"]

    def _get_broadcast_info(self, x_shape, y_shape):
        # ... 同上 ...

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return self._route(x, y, alpha)

    def _route(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        out_ndim, broadcast_dims, is_broadcast = self._get_broadcast_info(x.shape, y.shape)

        if not is_broadcast:
            if not x.is_contiguous(): x = x.contiguous()
            if not y.is_contiguous(): y = y.contiguous()
            output = torch.empty_like(x)
            n_elements = x.numel()

            # BLOCK_SIZE 可通过 autotune 选择，此处为推荐初始值
            BLOCK_SIZE = 8192
            grid = (min(triton.cdiv(n_elements, BLOCK_SIZE), self.VEC_CORE_NUM),)
            add_no_broadcast_kernel[grid](x, y, output, n_elements, alpha, BLOCK_SIZE)
            return output

        # === 广播路由 ===
        key = (out_ndim, broadcast_dims)

        if key == (2, (1,)):
            output = torch.empty_like(x)
            M, N = x.shape
            BLOCK_M, BLOCK_N = 8, 1024  # 推荐初始值，可 autotune
            num_blocks_m = triton.cdiv(M, BLOCK_M)
            num_blocks_n = triton.cdiv(N, BLOCK_N)
            grid = (min(num_blocks_m * num_blocks_n, self.VEC_CORE_NUM),)
            add_broadcast_2d_dim1_kernel[grid](x, y, output, M, N, alpha, BLOCK_M, BLOCK_N)
            return output

        elif key == (2, (0,)):
            output = torch.empty_like(x)
            M, N = x.shape
            BLOCK_M, BLOCK_N = 8, 1024
            num_blocks_m = triton.cdiv(M, BLOCK_M)
            num_blocks_n = triton.cdiv(N, BLOCK_N)
            grid = (min(num_blocks_m * num_blocks_n, self.VEC_CORE_NUM),)
            add_broadcast_2d_dim0_kernel[grid](x, y, output, M, N, alpha, BLOCK_M, BLOCK_N)
            return output

        else:
            out_shape = torch.broadcast_shapes(x.shape, y.shape)
            x_expanded = x.expand(out_shape).contiguous().view(-1)
            y_expanded = y.expand(out_shape).contiguous().view(-1)
            output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            out_flat = output.view(-1)

            n_elements = out_flat.numel()
            BLOCK_SIZE = 8192
            grid = (min(triton.cdiv(n_elements, BLOCK_SIZE), self.VEC_CORE_NUM),)
            add_broadcast_1d_kernel[grid](x_expanded, y_expanded, out_flat, n_elements, alpha, BLOCK_SIZE)
            return output
```

**关键设计**：
- **Key 匹配**：`(out_ndim, broadcast_dims)` 元组作为路由 key
- **内联分支**：每个分支直接写 kernel 调用
- **Host 预处理**：3D/4D 走 `expand + contiguous + view(-1)` 路径
- **参数可调**：BLOCK 值为推荐初始值，支持 `@triton.autotune` 自动调优



---

## 来自 SKILL.md 的原始描述（优化点 18：Kernel 分裂优化）

**适用条件**：多 Case 场景下泛用 Kernel 性能未达标（`total_cases > 1` 且 `speedup_vs_torch < 0.8`）

**典型代码特征**：
```python
# 特征 1：单个泛用 Kernel 需要覆盖差异巨大的多组 Shape
class Model(nn.Module):
    def forward(self, x):
        # x 可能是 [1024]、[128, 64] 或 [32, 128, 64] 等差异显著的 shape
        return self.kernel[grid](x, ...)

# 特征 2：不同 Case 的瓶颈归因明显不同
# 例如小 shape 受 kernel 启动开销主导，大 shape 受内存带宽主导

# 特征 3：存在 reduce/广播等可特化的模式
# reduce-last / reduce-non-last 或广播 dim0 / dim1 / 多维
```

**判断逻辑**：
1. 确认当前处于多 Case 场景（`total_cases > 1`）。
2. 读取 Phase 4 最终性能数据，检查整体几何平均 `speedup_vs_torch < 0.8`。
3. 按算子类型匹配经验文档：
   - Reduce 类算子（`sum/mean/max/min/softmax/layernorm`）→ 命中
   - 广播逐元素算子（`add/sub/mul/div` 且存在 shape 不等）→ 命中
4. 若未命中经验文档，则分析 `per_shape_results` 中 `speedup_vs_torch < 0.3` 的瓶颈 Case，检查是否存在可归因类别（Shape 过小/过大、非对齐访问、跨步访存、特殊 dtype）。
5. 若存在可归因的瓶颈分组 → 命中，进入参考文档。

**命中条件**：多 Case 场景、泛用 Kernel 性能未达标，且存在可特化的分组特征

**参考文档**：`references/kernel_splitting.md`

---
