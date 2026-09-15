# TileLang Ascend 性能关注项清单

本文件用于生成、改写或评审 TileLang Ascend 算子时快速排查常见潜在性能劣化模式。遇到性能关注项时，优先按“替代写法”调整；如果必须临时保留，需要记录 shape、dtype、原因和后续优化计划。

## 核心原理：搬运单元容量最大化

NPU 上每一次 DMA 调用、每一次 tile 迭代、每一条向量指令都携带**固定开销**（DMA 建链、地址计算、同步信号、流水线起停）。性能取决于固定开销与有效数据量的比值：

```
有效性能 = 有用数据量 / (有用数据量 + 固定开销)
```

本清单中的所有反模式，本质都是**让固定开销在有效数据中占比过高**。按根因归类：

| 根因类别 | 对应反模式 | 固定开销倍数 |
|---------|-----------|-------------|
| 搬运 0 次有效数据（纯冗余） | Wrapper 侧 permute+contiguous | ∞（100% 开销，0 有效增量） |
| 搬运次数爆炸 | 逐行 DataCopyPad 循环 | N 倍（N=行数） |
| 单次搬运量过小 | tile size 过小 | TILE_SIZE/actual_tile 倍 |
| 流水线空转 | 单缓冲 BUFFER_NUM=1 | 每次迭代一次空泡 |
| 布局限制导致被迫冗余搬运 | 强制 2D reshape | 100% 全量拷贝 |

**三维分解（outerSize, dimSize, innerSize）** 是消除上述根因的共同使能条件：它提供 `rowSize`、`inputRowSize`、`stride` 参数，使 kernel 能直接在原始内存布局上操作任意维度，无需 wrapper 做布局变换。

## 目录

- [使用方式](#使用方式)
- [硬件 buffer size 信息表（A2/A3，用于评估内存）](#硬件-buffer-size-信息表a2a3用于评估内存)
- [launch core 数需要重点关注](#launch-core-数需要重点关注)
- [Vector Core 内逐元素/逐行 for loop 计算](#vector-core-内逐元素逐行-for-loop-计算)
- [冗余全局同步](#冗余全局同步)
- [基础指令拼接未融合](#基础指令拼接未融合)
- [tile size 过小导致片上内存浪费](#tile-size-过小导致片上内存浪费)
- [AIC/AIV 混合算子未开启 CV overlap](#aicaiv-混合算子未开启-cv-overlap)
- [纯 AIV memory bound 算子未做流水/双 buffer](#纯-aiv-memory-bound-算子未做流水双-buffer)
- [正交轴串行化（Scalar Scan on Parallelizable Axis）](#正交轴串行化scalar-scan-on-parallelizable-axis)
- [Wrapper 侧数据搬运（permute/contiguous/reshape）](#wrapper-侧数据搬运permutecontiguousreshape)
- [逐行 DataCopyPad 循环（替代：2D strided）](#逐行-datacopypad-循环替代2d-strided)
- [单缓冲 BUFFER_NUM=1](#单缓冲-buffer_num1)
- [强制 2D reshape（替代：三维分解）](#强制-2d-reshape替代三维分解)
- [评审记录模板](#评审记录模板)

---

## 使用方式

- 写新算子前先扫一遍本清单，尽量避免为了功能正确引入明显性能风险。
- 性能优化前用本清单做第一轮静态检查，再结合 msprof 数据定位瓶颈。
- 修改后重新检查精度和性能；没有收益或引入内存超限时回退本轮修改。

## 硬件 buffer size 信息表（A2/A3，用于评估内存）

下表容量信息适用于 Ascend A2/A3 硬件；其他硬件型号需要按对应规格重新确认。

| 存储层级 | 容量（字节） | 典型用途 |
|----------|--------------|----------|
| L0A | 65536 | Cube A 操作数 |
| L0B | 65536 | Cube B 操作数 |
| L0C | 131072 | Cube 累加结果 |
| L1 | 524032 | Cube 侧数据缓存、GM 到 L0 的中间层 |
| UB | 196352 | Vector 侧计算 buffer |
| L2 | 201326592 | GM 访问的片上缓存层，AIC/AIV 通过 GM workspace 交互时可受益 |

容量评估时按“所有同层 buffer + pipeline stage/buffer num 倍数 + 临时 buffer”计算。`T.tile.broadcast`、双 buffer、`T.Pipelined(num_stages>1)` 都会增加片上内存占用，必须留出余量。

注意：可以开启 memory planning pass，编译器会在必要时做片上 buffer 复用。因此理论内存值只作为评估参考，最终需要通过实际编译和运行验证是否存在内存不足。

开启方式：在 JIT 的 `pass_configs` 中设置 `TL_ASCEND_MEMORY_PLANNING=True`。

```python
pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(pass_configs=pass_configs)
def kernel(...):
	...
```

如果代码中使用 `import tilelang as tl`，则写作 `tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING`。

---

## launch core 数需要重点关注

### 关注项 A：任务数高于物理 AI Core 数，按任务数 launch

**识别特征**：逻辑任务数 `block_num` 明显大于 A2/A3 的 24 个 AI Core，但代码使用：

```python
with T.Kernel(block_num, is_npu=True) as (cid, vid):
	# 每个 cid 只处理一个 block
	...
```

**性能原因**：逻辑任务数远高于物理核数时，按任务数 launch 会放大 kernel 初始化、workspace 分配、地址计算和同步开销。很多临时 buffer 还会随 `block_num` 线性膨胀，而同一时刻实际只会有物理 core 数量的任务并行执行。

**替代写法**：使用 Fixed Core，按物理核数 launch，再在每个 core 内手动分配多个逻辑任务。

```python
core_num = 24

with T.Kernel(core_num, is_npu=True) as (cid, vid):
	single_core_load = T.ceildiv(block_num, core_num)
	for block_idx in T.serial(cid * single_core_load, (cid + 1) * single_core_load):
		if block_idx < block_num:
			# 处理逻辑任务 block_idx
			...
```

**检查点**：
- workspace 优先按 `core_num` 维度分配，再通过 `cid` 复用。
- 尾块必须处理 `block_idx < block_num`，避免越界。
- 如果每个逻辑任务耗时差异很大，静态连续分配可能负载不均，需要结合任务形状重新设计映射。

### 关注项 B：任务数低于物理 AI Core 数，仍按 24 核 launch

**识别特征**：`block_num < 24`，但代码固定使用：

```python
with T.Kernel(24, is_npu=True) as (cid, vid):
	...
```

**性能原因**：空闲 core 不做有效计算，但仍会进入 kernel、执行部分初始化和分支判断；如果存在全局同步或 workspace 初始化，空 core 还可能放大等待和资源占用。

**替代写法**：按实际任务数 launch。

```python
launch_core_num = T.min(block_num, 24)

with T.Kernel(launch_core_num, is_npu=True) as (cid, vid):
	# cid 天然落在有效任务范围内
	...
```

如果 `block_num` 是 Python 侧静态整数，也可以直接在 host 侧计算：

```python
core_num = min(block_num, 24)

with T.Kernel(core_num, is_npu=True) as (cid, vid):
	...
```

---

## Vector Core 内逐元素/逐行 for loop 计算

**识别特征**：在 Vector 侧用 Python `range` 或 `T.serial` 循环对 UB 中的小切片反复执行同一类 scalar/tile 操作，常见于 softmax、归一化、mask、逐行缩放。

```python
for row in range(block_M):
	T.tile.sub(acc_ub[row, :], acc_ub[row, :], max_ub[row])
```

**性能原因**：每一行/每个元素单独发起一次指令，会降低 Vector 指令利用率，并引入大量 scalar 地址计算、循环控制和指令下发开销。Vector Core 更适合一次处理连续 tile。

**替代写法**：先把低维标量/向量 broadcast 到与目标 tile 相同形状，再一次性调用向量化 tile 指令。

```python
max_2d = T.alloc_ub([block_M, block_N], dtype)

T.tile.broadcast(max_2d, max_ub, axis=1)
T.tile.sub(acc_ub, acc_ub, max_2d)
```

**收益和代价**：
- 收益：减少循环控制、scalar 指令和多次 tile 指令下发，提升 Vector 计算利用率。
- 代价：broadcast 后的变量需要额外 UB 空间，broadcast 本身也需要执行并可能产生临时 buffer。
- 适用：UB 有余量、原始循环次数较多、每轮计算量偏小的场景。
- 不适用：存在真实迭代依赖（例如前一轮结果影响后一轮）且无法数学等价重排的场景。

**评审建议**：看到 `for row in range(...)`、`for col in range(...)` 里反复调用 `T.tile.add/sub/mul/div/max/min` 时，优先确认能否改成 broadcast + 整 tile 计算。

### 子模式：`T.if_then_else` 逐元素条件分支导致串行循环 ⭐

**识别特征**：kernel 中用 `T.if_then_else` 做逐元素条件分支（如 `x < threshold` 时用近似公式），被 `ascend_lower_parallel` pass 降级为串行循环。

```python
# 反模式：T.if_then_else 逐元素条件分支
for i, j in T.Parallel(block_M, block_N):
    x = x_ub[i, j]
    y_ub[i, j] = T.if_then_else(x < threshold, approx_fn(x), full_fn(x))
```

**性能原因**：`T.if_then_else` 无法被 `ascend_lower_parallel` pass 向量化，整个 `T.Parallel` 循环降级为**串行标量迭代**（每 tile 数千次）。串行循环内每元素都执行条件判断 + 分支选择 + 标量读写，性能灾难性下降。

**与"基础指令拼接未融合"的区别**：后者是多个 `T.tile.xxx` 可融合为一条复合指令（如 `mul+add`→`mul_add_dst`）；本子模式是条件分支本身导致无法向量化，即使分支内的操作都是向量化的。

**替代写法**：用数学等价变换消除条件分支（完整工具箱见 [tilelang-op-design ascend-constraints.md §7](../../tilelang-op-design/references/ascend-constraints.md)）：

```python
# 正模式：用数学等价公式消除分支，全部用 T.tile.xxx 向量化
# 核心思路：将条件分支改写为无条件分支的等价数学表达式
# 例：log-sum-exp trick 将 ln(1+exp(x)) 改写为 max(x,0) + ln(1+exp(-|x|))
#     使 exp 参数恒非正，不溢出，且全向量化无需条件判断
T.tile.max(t1_ub, a_ub, 0.0)       # max(x, 0)
T.tile.abs(t0_ub, a_ub)            # |x|
T.tile.mul(t0_ub, t0_ub, -1.0)     # -|x|（exp 参数 ≤ 0）
T.tile.exp(t0_ub, t0_ub)           # exp(-|x|)，结果有界
# ... 后续 add/ln 步骤完成等价变换
```

**评审建议**：看到 `T.if_then_else` 在 `T.Parallel` 循环内做逐元素条件分支时，**强制要求**改用数学等价变换。只有当所有等价变换都不可行时才允许条件分支，并在 design 文档显式标注性能代价。

---

## 冗余全局同步

**识别特征**：循环体内或每个小步骤后频繁出现：

```python
T.barrier_all()
...
T.barrier_all()
...
T.sync_all()
```

**性能原因**：`barrier_all` / `sync_all` 会扩大等待范围。即使只有局部数据依赖，也会让无关 pipeline 或 core 一起等待，造成 MTE、Vector、Cube 的气泡。同步放在内层循环时，开销会被循环次数放大。

**替代方向**：
- 删除没有生产者/消费者依赖的同步。
- 用 `T.set_flag(src, dst, event_id)` / `T.wait_flag(src, dst, event_id)` 约束具体 pipeline 之间的依赖。
- AIC/AIV 跨核交互用 `T.set_cross_flag` / `T.wait_cross_flag` 或交给 `T.Pipelined` 自动管理。
- 对多次任务后才需要一致性的场景，考虑增加同步间隔，参考 `T.Pipelined(..., cross_interval=N)`。

**性能劣化模式示例**：每轮 gather 后都全局同步。

```python
for i in T.serial(num_blocks):
	T.copy(src[i, :], ub_tmp)
	T.barrier_all()
	T.tile.add(out_ub, out_ub, ub_tmp)
	T.barrier_all()
```

**改写示意**：只在真正需要消费搬运结果的位置等待，或改成流水/双 buffer。

```python
for i in T.serial(num_blocks):
	side = i % 2
	T.copy(src[i, :], ub_tmp[side, :])
	# 若开启自动同步且生成代码正确，可不手写 barrier。
	T.tile.add(out_ub, out_ub, ub_tmp[side, :])
```

**检查点**：先验证同步是否必需，再删改；如果关闭自动同步或进入 Expert 模式，必须通过生成的 Ascend C 代码和精度测试确认同步语义没有被破坏。

---

## 基础指令拼接未融合

**识别特征**：连续出现多个基础 element-wise 指令，且整体等价于硬件/TileLang 已有复合指令或激活函数。

### `mul + add` / 累加 pattern

**性能劣化模式示例**：

```python
T.tile.mul(tmp_ub, x_ub, w_ub)
T.tile.add(acc_ub, acc_ub, tmp_ub)
```

**替代写法**：

```python
T.tile.mul_add_dst(acc_ub, x_ub, w_ub)  # acc_ub = x_ub * w_ub + acc_ub
```

如果是 `dst = scalar * src + dst`，使用 `T.tile.axpy`：

```python
T.tile.axpy(dst_ub, src_ub, scale)  # dst_ub = scale * src_ub + dst_ub
```

注意：PTO 后端场景若没有 `axpy`，优先确认能否使用 `mul_add_dst` 等价表达；所有接口以 `tilelang/language/ascend_tile.py` 为准。

### `max(x, 0)` pattern

**性能劣化模式示例**：

```python
T.tile.max(out_ub, x_ub, 0.0)
```

**替代写法**：

```python
T.tile.relu(out_ub, x_ub)
```

其他激活函数也按“数学等价优先”原则尝试替换，例如 `leaky_relu` 等。替换前确认 dtype、边界值和 NaN 行为是否满足算子精度要求。

### `sqrt + div` pattern

**性能劣化模式示例**：

```python
T.tile.sqrt(tmp_ub, x_ub)
T.tile.div(out_ub, one_ub, tmp_ub)
```

**替代写法**：

```python
T.tile.rsqrt(out_ub, x_ub)  # out_ub = 1 / sqrt(x_ub)
```

如果后续计算是 `y / sqrt(x)`，优先尝试 `rsqrt` 后再乘：

```python
T.tile.rsqrt(inv_sqrt_ub, x_ub)
T.tile.mul(out_ub, y_ub, inv_sqrt_ub)
```

**检查点**：融合会改变中间舍入路径，尤其是 fp16/bf16 场景，必须重新跑精度。

---

## tile size 过小导致片上内存浪费

**识别特征**：L0A/L0B/L0C/L1/UB 实际使用量远小于容量。

**性能原因**：tile 太小会增加逻辑任务数、循环次数、同步次数和 GM 往返次数；片上缓存没有充分复用，单次搬运/计算粒度也可能不足以打满带宽或算力。

**替代方向**：在不超过片上容量的前提下，优先成倍扩大 tile size。

```python
# 性能劣化模式示例：UB 占用很小，任务数很多
block_M = 16
block_N = 64

# 尝试：按 2 倍递增，并重新计算 UB/L1/L0 占用
block_M = 32
block_N = 128
```

**容量估算示意**：

```text
UB 使用量 = sum(buffer_elements * dtype_bytes * buffer_num)
L1 使用量 = sum(l1_tile_elements * dtype_bytes * stages)
L0C 使用量 = block_M * block_N * accum_dtype_bytes * stages
```

**AIC/AIV 交互说明**：AIC 和 AIV 之间通常通过 GM workspace 交互。当 workspace 访存量小于 L2 容量（A2/A3 约 201 MB）时，数据更可能命中 L2 cache，从而获得高于普通 GM 往返的带宽收益。设计 workspace 时应尽量让交互数据连续、按 core 复用，并避免无意义地扩大到超过 L2 cache 容量。

**检查点**：
- 扩大 tile size 后确认 `T.Pipelined` stage、双 buffer、broadcast 临时 buffer 的总占用仍不超限。
- 理论内存估算只作参考；开启 memory planning pass 后可能复用部分 buffer，最终以内存规划结果和实际运行是否报内存不足为准。
- tile 变大可能降低并行任务数；当任务数低于物理 core 数时，需要同步调整 launch core 数。
- 尾块处理和 mask 逻辑要随 tile size 一起复查。

---

## AIC/AIV 混合算子未开启 CV overlap

**识别特征**：通过 `get_kernel_source()` 看到生成代码同时包含 `IS_ASCEND_AIC` 和 `IS_ASCEND_AIV`，但主循环仍是 Cube 写 workspace、Vector 读 workspace 的串行结构，未使用 `T.Pipelined`。

```python
for k in T.serial(loop_k):
	# AIC: compute and write workspace
	...
	# AIV: read workspace and vector compute
	...
```

**性能原因**：CV 融合算子的 AIC 和 AIV 通过 workspace 串接。如果没有核间流水，Vector 往往要等 Cube 产出，Cube 也可能等 Vector 消费，形成明显核间气泡。

**替代写法**：用 `T.Pipelined` 表达核间流水，并开启自动 CV combine/sync。

```python
pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
	tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

for k in T.Pipelined(loop_k, num_stages=2):
	# AIC: write workspace for iteration k
	...
	# AIV: read previous/ready workspace and compute
	...
```

**调参建议**：
- 从 `num_stages=2` 开始，逐步增大；`num_stages` 不能超过循环次数。
- 循环次数较多、Cube/Vector 耗时差距大时，较大的 `num_stages` 可能收益更好。
- 如果同步开销占比高，尝试 `cross_interval=2`，但需要验证并行度损失是否抵消收益。
- 不要嵌套多个 `T.Pipelined`；核间用 `T.Pipelined` 时，核内 double buffer 推荐手写 flat pattern。

---

## 纯 AIV memory bound 算子未做流水/双 buffer

**识别特征**：`get_kernel_source()` 只包含 `IS_ASCEND_AIV`；每轮循环都按 `GM -> UB -> Vector -> GM` 串行执行。

```python
for i in T.serial(loop_n):
	T.copy(x[i, :], x_ub)
	T.tile.exp(y_ub, x_ub)
	T.copy(y_ub, y[i, :])
```

**性能原因**：纯 AIV 算子常常 memory bound。如果搬入、计算、搬出完全串行，Vector 计算无法掩盖 GM/UB 搬运延迟，MTE 和 Vector pipeline 都容易出现空泡。

**替代写法 A：使用 `T.Pipelined`**

```python
for i in T.Pipelined(loop_n, num_stages=2):
	T.copy(x[i, :], x_ub)
	T.tile.exp(y_ub, x_ub)
	T.copy(y_ub, y[i, :])
```

**替代写法 B：手动双 buffer**

手动双 buffer 只分配双份 buffer 不够，还需要手动控制每个 stage 的同步关系，确保搬入、计算、搬出不会读写同一份未就绪 buffer。下面代码只展示 buffer 轮转位置；实际实现中需要按 pipeline 使用 `T.set_flag` / `T.wait_flag` 明确控制 stage 依赖。

```python
x_ub = T.alloc_ub([2, block_N], dtype)
y_ub = T.alloc_ub([2, block_N], dtype)

for i in T.serial(loop_n):
	side = i % 2
	# 手动双 buffer 时，需要在每个 stage 之间配套 set_flag / wait_flag。
	T.copy(x[i, :], x_ub[side, :])
	T.tile.exp(y_ub[side, :], x_ub[side, :])
	T.copy(y_ub[side, :], y[i, :])
```

**检查点**：
- 双 buffer 会让 UB 占用乘以 2；再叠加 broadcast 或临时 buffer 时尤其要重新估算。
- 如果 `loop_n` 很小，pipeline 启停开销可能抵消收益，需要实测。

---

## 多行 tile 循环内逐块归约

**识别特征**：多行 tile（`[rows, block_S]`，`rows > 1`）的循环内，每块都在执行 `reduce_sum`：

```python
# ❌ 多行 tile 循环内逐块归约
for si in T.serial(s_num):
    T.copy(src[...], buf_cal)               # [rows, block_S]
    T.reduce_sum(buf_cal, col_buf, dim=-1)  # [rows, 1]
    T.reduce_sum(col_buf, scalar, dim=0)    # dim=0 行为不确定！
    T.tile.add(total, total, scalar)
```

**性能/正确性原因**：（1）`reduce_sum(dim=0)` 在 `[rows, 1]` 形状上行为不确定，可能产生 NaN；（2）每轮额外发射 2 条归约指令，抵消了多行 tile 的收益。

**替代写法**：循环内只用 `tile.add` 累积到完整 `[rows, block_S]` buffer，循环外用 `reduce_sum(dim=-1)` 两步归约：

```python
# ✅ 循环内累积 → 循环外归约
accum = T.alloc_ub([rows, block_S], cal_dtype)
T.tile.fill(accum, 0.0)
for si in T.serial(s_num):
    T.copy(src[...], buf_cal)
    T.tile.add(accum, accum, buf_cal)   # 只累积

# 循环外
row_buf = T.alloc_ub([rows], cal_dtype)
result = T.alloc_ub([1], cal_dtype)
T.reduce_sum(accum, row_buf, dim=-1)   # [rows, block_S] → [rows]
T.reduce_sum(row_buf, result, dim=-1)  # [rows] → scalar
```

**检查点**：循环内是否有 `reduce_sum` 调用？循环外是否只用 `dim=-1`？详见 optimization-guide.md §2.13 铁律 P1。

---

## 正交轴串行化（Scalar Scan on Parallelizable Axis）

**识别特征**：两层嵌套循环中，外层遍历正交轴（如行），内层沿扫描轴逐元素处理，且外层各迭代独立无依赖。内层操作是标量（`if`/`GetValue`/`SetValue`），而非 `T.tile` 向量操作。

```python
# ❌ 正交轴被串行化，内层是标量操作
for r in range(sub_block_R):
    for i in range(1, L):
        if a[r, i] <= v[r, i - 1]:    # 标量 if-else
            v[r, i] = a[r, i]
            idx[r, i] = i
        else:
            v[r, i] = v[r, i - 1]
            idx[r, i] = idx[r, i - 1]
```

**性能原因**：扫描轴有真依赖无法消除，但正交轴各元素独立——串行处理正交轴浪费了向量化的并行能力。标量操作导致指令数和 icache miss 随并行轴规模线性增长。

**与 §Vector Core 内逐元素/逐行 for loop 计算的区别**：
- 那里的循环内已经是 `T.tile` 操作（broadcast 即可消除循环）
- 这里的循环内是标量 `if`/`GetValue`/`SetValue`，且扫描轴有依赖无法直接消除

**替代写法**：识别正交轴 → transpose 使并行轴内存连续 → 折叠为 `T.tile.compare/select/min` 向量操作。详见 optimization-guide.md §2.15。

```python
# ✅ 正交轴向量化：transpose [Rows, L] -> [L, Rows]，向量扫描
for j in range(1, L):
    T.copy(A[j, col_base:col_base + N], curr_ub)      # N 行连续
    T.tile.compare(mask, curr_cal, run_min_cal, "LE")  # 向量比较
    T.tile.select(run_idx, mask, idx_curr, run_idx, ...) # 向量选择
    T.tile.min(run_min_cal, run_min_cal, curr_cal)     # 向量 min (NaN 正确传播)
```

**检查点**：
- 两层嵌套循环中，外层各迭代是否独立（交换顺序结果不变）？
- 内层是否是标量 `if`/`GetValue`/`SetValue` 而非 `T.tile` 操作？
- 扫描轴是否有真依赖（无法直接消除内层循环）？
- 若全部满足，参考 §2.15 正交轴向量化

---

## 纯 Vector 算子的 AUTO_CV_COMBINE 误分核风险

**识别特征**：纯 Vector 算子（`get_kernel_source()` 只有 `IS_ASCEND_AIV`），pass_configs 中开了 `TL_ASCEND_AUTO_CV_COMBINE: True`，且 kernel 内使用了 `T.alloc_var`。

**风险**：某些 lowering 形态可能把 `alloc_var` 的定义和使用错误分到不同核，但这不是
“纯 Vector + AUTO_CV_COMBINE + alloc_var”必然失败的全局规则；仓内也有该组合的正向
测试。必须先检查生成代码或最小复现，确认变量确实跨核且没有正确传递。

确认发生误分核后，纯 Vector 算子可关闭不需要的 `AUTO_CV_COMBINE`：

```python
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    # 已通过生成代码确认误分核时，关闭不需要的 AUTO_CV_COMBINE
```

## Wrapper 侧数据搬运（permute/contiguous/reshape）

**识别特征**：`model_new_ascendc.py` 的 `forward()` 中存在 `x.permute(perm).contiguous()`、`x.reshape(M, N)` 等 PyTorch 张量变换操作，且这些操作导致全量数据拷贝。

**性能原因**：`permute` + `contiguous()` 等于对整个输入张量做一次全量拷贝（GM→GM），开销与张量大小成正比。对于大张量（如 `[8192, 2048]`），这相当于 16~32MB 的额外数据搬移。参考实现（PyTorch）通过 `torch.chunk` 创建 view（零拷贝）后直接操作非连续视图，效率远高于 wrapper 的全量拷贝。

**影响量级**：dim≠-1 的 case 加速比可从 1~2x 降至 0.13~0.18x（10 倍劣化）。

**替代方向**：kernel 原生支持任意 dim 参数，通过三维分解 `[outerSize, dimSize, innerSize]` 在 GM 中直接通过 stride/offset 定位数据，无需 wrapper 做任何布局变换。

```python
# ❌ wrapper 强制 permute+contiguous
def forward(self, x, dim=-1):
    if dim != x.ndim - 1:
        perm = list(range(x.ndim))
        perm.pop(dim)
        perm.append(dim)
        x = x.permute(perm).contiguous()  # 全量拷贝！
    return torch.ops.npu.swiglu(x.reshape(M, N), 1)

# ✅ kernel 原生处理任意 dim
def forward(self, x, dim=-1):
    return torch.ops.npu.swiglu(x, dim)  # kernel 内部三维分解
```

**检查点**：
- forward() 中是否有 `permute` / `contiguous` / `reshape` 组合？
- 是否有 `x.to(dtype)` 全量精度转换（应移入 kernel Cast）？
- wrapper_copy_bytes / kernel_compute_bytes 是否为 0？

---

## 逐行 DataCopyPad 循环（替代：2D strided）

**识别特征**：`CopyIn()` 或 `CopyOut()` 中存在 for 循环，每次迭代调用一次 1D `DataCopyPad` 处理单行数据。

```cpp
// ❌ 逐行循环，32 次 DMA 建链
for (int32_t r = 0; r < rows; ++r) {
    DataCopyPad(aLocal[ubOff], inputGm_[inRowOff], rowCp, pp);
}
```

**性能原因**：每次 `DataCopyPad` 调用有固定的 DMA 建链开销（地址计算、通道分配、同步信号）。当行数多（如 65536 行）但每行元素少（如 64 个）时，有效数据量很小但调用次数爆炸，固定开销 dominate。

**替代方向**：使用 2D `DataCopyPad` 的 `blockCount > 1` + `srcStride` 模式，一次 DMA 传输多行跨行数据。

```cpp
// ✅ 2D strided，一次 DMA 加载 32 行
DataCopyExtParams copyParams{
    static_cast<uint16_t>(numRows),  // blockCount = 行数
    rowSize * sizeof(T),              // blockLen = 每行字节数
    rowSize * sizeof(T),             // srcStride = 行间隔（跳过 b 块）
    0, 0                              // dstStride = 0 (UB 连续)
};
DataCopyPad(aLocal, xGm[aBaseOffset], copyParams, padParams);
```

**影响量级**：同一 shape/dtype/totalOutput，逐行循环 vs 2D strided 延迟差 20~30 倍。

**检查点**：
- CopyIn/CopyOut 中是否有 for 循环调用 DataCopyPad？
- 能否用 blockCount > 1 替代循环？

---

## 单缓冲 BUFFER_NUM=1

**识别特征**：`TQue` 声明使用 `BUFFER_NUM = 1`，CopyIn/Compute/CopyOut 完全串行执行。

```cpp
// ❌ 单缓冲，无流水重叠
TQue<TPosition::VECIN, 1> aQueue;
TQue<TPosition::VECOUT, 1> yQueue;
```

**性能原因**：`BUFFER_NUM=1` 时，tile N 的 CopyIn 必须等 tile N-1 的 CopyOut 完成后才能开始。三级流水（CopyIn→Compute→CopyOut）完全串行，无法利用 MTE2/MTE3 与 Vector 的并行能力。

**替代方向**：使用 `BUFFER_NUM=2`（双缓冲），tile N 的 Compute 与 tile N+1 的 CopyIn 重叠。

```cpp
// ✅ 双缓冲，流水重叠
TQue<TPosition::VECIN, 2> aQueue;
TQue<TPosition::VECOUT, 2> yQueue;
```

**影响量级**：计算密集型 case（大 float32）提升约 20~30%；搬运密集型 case 提升较小。

**检查点**：
- 所有 VECIN/VECOUT 队列的 BUFFER_NUM 是否 ≥ 2？
- UB 空间不足降为 1 时是否在 PERF_DESIGN.md 记录原因？
- 循环中同时持有的 queue tensor 数量 + 1 是否 ≤ BUFFER_NUM？

**注意**：`BUFFER_NUM` 过大不会线性提升性能（pipeline depth 受三级流水限制），通常 2 即可。

---

## 强制 2D reshape（替代：三维分解）

**识别特征**：Host 侧（op_host）假设输入为 2D `[M, N]`，wrapper 必须将任意维张量 reshape 为 2D 后才能调用 kernel。

```cpp
// ❌ 仅支持 2D
at::Tensor swiglu(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    int32_t M = x.size(0);
    int32_t N = x.size(1);
    ...
}
```

**性能原因**：强制 2D 意味着 wrapper 必须将 dim 参数移到末尾位置（permute+contiguous），然后 reshape 为 2D。这对 dim≠-1 的输入造成全量拷贝。即使 dim=-1，reshape 为 2D 也丢失了原始维度的语义信息，无法利用 stride 跨行访问。

**替代方向**：Host 侧将任意 shape 分解为 `[outerSize, dimSize, innerSize]` 三维逻辑结构，kernel 通过 `rowSize = halfDim * innerSize` 和 `inputRowSize = dimSize * innerSize` 直接在 GM 中定位 a/b 数据。

```cpp
// ✅ 三维分解，支持任意 dim
at::Tensor swiglu(const at::Tensor &x, int64_t dim) {
    int32_t normDim = dim < 0 ? dim + x.dim() : dim;
    int64_t outerSize = 1, innerSize = 1;
    for (int32_t i = 0; i < normDim; ++i) outerSize *= x.size(i);
    for (int32_t i = normDim + 1; i < x.dim(); ++i) innerSize *= x.size(i);
    int64_t dimSize = x.size(normDim);
    ...
}
```

**检查点**：
- `get_kernel_source()` 是否只有 `IS_ASCEND_AIV`（纯 Vector）？
- pass_configs 是否开了 `AUTO_CV_COMBINE`？
- kernel 内是否使用了 `T.alloc_var`？
- 三者同时满足 → 检查生成代码中变量定义/使用的核归属；仅在确认误分核后关闭
  `AUTO_CV_COMBINE`
- op_host 是否假设固定维度数（如 `x.dim() == 2`）？
- 是否有 `TORCH_CHECK(x.dim() == ...)` 限制？
- 能否用 `[outerSize, dimSize, innerSize]` 三维分解替代？


## 评审记录模板

发现性能关注项但暂不修改时，在优化记录中写清楚：

```text
- 关注项：Vector for loop 逐行计算
- 位置：examples/<op>/<file>.py::<kernel>
- shape/dtype：...
- 暂不修改原因：UB 余量不足，broadcast 后可能超限
- 后续方案：减小其他临时 buffer 或改用分块 broadcast，再验证精度和性能
```