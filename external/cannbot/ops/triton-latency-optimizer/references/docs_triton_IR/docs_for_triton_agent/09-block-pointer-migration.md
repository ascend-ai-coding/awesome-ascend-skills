# Block Pointer 迁移文档

## 触发条件

当 Agent 迁移使用 `tl.make_block_ptr` 的 GPU kernel 时，需要按照本文档进行适配。典型场景包括：
- GPU kernel 中调用了 `tl.make_block_ptr` 创建 block pointer
- 使用了 `tl.load(block_ptr, boundary_check=..., padding_option=...)` 加载数据
- 使用了 `tl.store(block_ptr, value, boundary_check=...)` 存储数据
- 使用了 `tl.advance(block_ptr, offsets)` 推进指针偏移
- 使用了 `load_if` / `store_if` 辅助函数封装 boundary_check 逻辑

---

## 核心知识：Block Pointer 在 NPU 上的行为差异

### 1. tl.make_block_ptr 支持情况

`tl.make_block_ptr` 在 NPU 上已支持，但存在以下关键差异：

| 特性 | GPU | NPU | 说明 |
|------|-----|-----|------|
| 数据类型 | 支持所有类型 | 不支持 uint8/uint16/uint32/uint64/fp64 | 硬件限制，需替换为对应 int 类型或 fp32 |
| stride 参数 | 可通过交换顺序实现转置 | 只能反映真实内存布局 | NPU 不支持 stride 交换转置 |
| order 参数 | 控制遍历顺序 | 控制遍历顺序 + 表达转置语义 | NPU 上 order 兼具转置功能 |
| 维度支持 | 1-5 维 | 1-5 维 | 无差异 |
| 连续访存偏好 | 灵活 | 强偏好连续访存 | 离散访存性能极差 |

**stride/order 规则**：
- NPU 上 `stride` 必须反映真实的物理内存布局，不能通过交换 stride 实现转置
- 转置语义只能通过调整 `order` 参数表达
- `order` 中靠前的维度是内层循环（变化最快），靠后的是外层循环

```python
# GPU 写法：通过 stride 交换实现转置（NPU 不支持）
block_ptr = tl.make_block_ptr(
    base=x_ptr,
    shape=(M, N),
    strides=(1, M),      # stride 交换
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(0, 1),
)

# NPU 写法：通过 order 参数表达转置
block_ptr = tl.make_block_ptr(
    base=x_ptr,
    shape=(M, N),
    strides=(N, 1),      # stride 反映真实内存布局
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),        # 通过 order 表达转置语义
)
```

### 2. boundary_check 和 padding_option 的行为差异

| 参数 | GPU | NPU | 说明 |
|------|-----|-----|------|
| `boundary_check` | 正常工作 | 正常工作，但需注意与复杂控制流搭配 | 指定需要边界检查的维度 |
| `padding_option="zero"` | 正常工作 | 部分场景可能不完全支持 | 建议使用 `mask` + `other` 替代 |
| `padding_option` 其他值 | 支持 | 支持有限 | 优先使用 `padding_option="zero"` |

**NPU 适配建议**：
- `load_if` / `store_if` 辅助函数在 NPU 上可以继续使用，其内部逻辑（根据 EVEN_M/EVEN_N 选择 boundary_check 维度）在 NPU 上行为一致
- 当 `padding_option` 出现问题时，可改用 `mask` + `other=0.0` 的方式替代
- NPU 特有的 `care_padding` 参数：当 `mask` 不为 None 而 `other` 为 None 时，`care_padding=True`（默认）会自动填充零值；设为 `care_padding=False` 可提升性能，但 padding 区域值为未定义

### 3. tl.advance 的行为差异

**关键差异**：在 NPU 上，`tl.advance` 与复杂循环/分支搭配时可能出现编译失败。

| 行为 | GPU | NPU |
|------|-----|-----|
| 基本用法 | 无限制 | 正常工作 |
| 与 for 循环搭配 | 正常工作 | 可能编译失败 |
| 与 if 分支 + for 循环搭配 | 正常工作 | 可能编译失败 |
| advance 后是否需要重新 load | 不需要 | advance 后仍需重新 load |

**NPU 上 advance 后仍需重新加载**：

```python
# GPU 写法：advance 后 block_ptr 自动更新，下次 load 使用新位置
for start_n in range(begin, end, BLOCK_N):
    k = load_if(k_block_ptr, True, False)
    k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))

# NPU 写法：advance 后同样需要重新 load，但需注意编译问题
# 如果 advance 与复杂控制流搭配编译失败，改用重新创建 block_ptr
```

**替代方案**：当 `tl.advance` 不可用时，在循环内重新调用 `tl.make_block_ptr`：

```python
# 替代方案：重新创建 block_ptr 替代 advance
for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
    task_hz_idx = block_idx // NUM_BLOCKS_M
    task_m_idx = block_idx % NUM_BLOCKS_M
    q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(task_m_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
```

### 4. NPU 上不能在 for 循环内使用 return

**核心限制**：Triton JIT 编译器不支持在 `for` 或 `while` 循环内使用 `return` 语句，也不支持 `continue` 语句。

```python
# GPU 写法：在循环内使用 return 跳过无效任务（NPU 不支持）
for block_idx in range(start_block, end_block, step):
    if start_m * BLOCK_M >= q_len:
        return  # 编译错误！

# NPU 写法：改用 if 条件跳过
for block_idx in range(start_block, end_block, step):
    if start_m * BLOCK_M < q_len:
        # 将原来的逻辑放在 if 块内
        ...
```

**迁移规则**：
- `return` → 改为 `if` 条件包裹，将 return 后的逻辑放在 if 块内
- `continue` → 改为 `if` 条件包裹，将 continue 后的逻辑放在 if 块内
- 多层 return/continue → 嵌套 if 条件

### 5. 离散访存 vs 连续访存

NPU 对连续访存有强偏好。GPU 上常见的将多维数据展平为一维、使用 stride 表示跨步的写法，在 NPU 上性能极差。

```python
# GPU 风格：离散访存（NPU 性能差）
block_ptr = tl.make_block_ptr(
    base=input_ptr,
    shape=(1024,),
    strides=(32,),       # 每次跳 32 个元素，离散
    offsets=(i_t * 16,),
    block_shape=(BT,),
    order=(0,),
)

# NPU 优化：连续访存
block_ptr = tl.make_block_ptr(
    base=input_ptr,
    shape=(1024, 32),    # 二维张量
    strides=(32, 1),     # 最低维度连续
    offsets=(i_t * BT, 0),
    block_shape=(BT, 32),
    order=(1, 0),        # 先行后列
)
```

---

## 代码模式：迁移修改点

### 修改点 1：Grid 从多维改为 1D + 核内循环

GPU 版本使用多维 Grid（如 3D Grid `(NUM_BLOCKS_M, q_head, batch_size)`），NPU 版本改为 1D Grid + 核内循环处理多个任务块。

```python
# GPU 写法：3D Grid
start_m = tl.program_id(0)
start_qh = tl.program_id(1)
start_b = tl.program_id(2)

# NPU 写法：1D Grid + 核内循环
pid = tl.program_id(0)
NUM_BLOCKS_M = tl.cdiv(MAX_Q_LEN, BLOCK_M)
NUM_BLOCKS = NUM_BLOCKS_M * BATCH_SIZE * q_head
start_block, end_block, step = pid, NUM_BLOCKS, AICORE_NUM

for block_idx in range(start_block, end_block, step):
    task_hz_idx = block_idx // NUM_BLOCKS_M
    start_m = block_idx % NUM_BLOCKS_M
    start_b = task_hz_idx // q_head
    start_qh = task_hz_idx % q_head
```

### 修改点 2：return 改为 if 条件跳过

```python
# GPU 写法
if start_m * BLOCK_M >= q_len:
    return

# NPU 写法
if start_m * BLOCK_M < q_len:
    # 所有后续逻辑放在 if 块内
    ...
```

### 修改点 3：make_block_ptr 移入循环内

GPU 版本在循环外创建 block_ptr，NPU 版本因核内循环处理多个任务块，需要在循环内重新创建 block_ptr。

```python
# GPU 写法：循环外创建 block_ptr
q_block_ptr = tl.make_block_ptr(
    base=q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
    shape=(q_len, QK_DIM),
    strides=(q_head * QK_DIM, 1),
    offsets=(start_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, QK_DIM),
    order=(1, 0)
)
for start_n in range(begin, end, BLOCK_N):
    ...
    k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))

# NPU 写法：循环内创建 block_ptr
for block_idx in range(start_block, end_block, step):
    ...
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
        shape=(q_len, QK_DIM),
        strides=(q_head * QK_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, QK_DIM),
        order=(1, 0)
    )
```

### 修改点 4：Softmax 计算方式适配

GPU 版本使用 `tl.math.exp2` 配合 `log2e` 常数进行指数运算，NPU 版本在 fwd_kernel 中改用 `tl.math.exp` 直接计算。

```python
# GPU 写法：使用 exp2 + log2e
log2e: tl.constexpr = 1.4426950408889634
qk_scale = scale * log2e
alpha = tl.math.exp2((m - m_new) * qk_scale)
p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
l = m * scale + tl.log(l)

# NPU 写法（fwd_kernel）：使用 exp 直接计算
qk_scale = scale  # 不乘 log2e
s = s * qk_scale + tl.where(mask, 0.0, -2.0 ** 30)
m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
p = tl.math.exp(s - m_new[:, None])
alpha = tl.math.exp(m - m_new)
m = m + tl.log(l)  # 注意：不乘 scale
```

**注意**：NPU 上 `tl.max` 需要添加 `propagate_nan=True` 参数，避免 NaN 值在 maximum 操作中被忽略导致数值不稳定。

### 修改点 5：Mask 融合到 softmax 计算

GPU 版本先计算 `s = tl.dot(q, k)`，再用 `tl.where` 应用 mask；NPU 版本将 mask 融合到 softmax 的加法中，减少一次 UB 操作。

```python
# GPU 写法：先 dot 再 mask
s = tl.dot(q, k)
boundary_mask = (offset_n < k_len)[None, :]
s = tl.where(mask & boundary_mask, s, -2**30)
m_new = tl.maximum(m, tl.max(s, 1))

# NPU 写法：mask 融合到 softmax 加法
s = tl.dot(q, k)
s = s * qk_scale + tl.where(mask, 0.0, -2.0 ** 30)  # mask 融合
m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
```

### 修改点 6：compile_hint 优化

NPU 版本使用 `extension.compile_hint` 打断 Vector Function 融合，防止 UB 溢出。详见 [14-compile-hint-and-extension.md](14-compile-hint-and-extension.md)。

```python
import triton.language.extra.cann.extension as extension

ds = tl.where(mask, ds, 0.0)
extension.compile_hint(ds, "break_vf")
dq += tl.dot(ds, k)
```

### 修改点 7：num_warps / num_stages 参数移除

NPU 上 `num_warps` 和 `num_stages` 参数无效，需要从 Config 中移除。

```python
# GPU 写法
triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=3, num_warps=4)

# NPU 写法
triton.Config({"BLOCK_M": 128, "BLOCK_N": 32})
```

---

## load_if / store_if 辅助函数的 NPU 适配

`load_if` 和 `store_if` 在 NPU 上可以继续使用，无需修改核心逻辑。但需注意以下差异：

### load_if

```python
@triton.jit
def load_if(block_ptr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        return tl.load(block_ptr)
    elif EVEN_M:
        return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
    elif EVEN_N:
        return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
    else:
        return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")
```

**NPU 注意事项**：
- `padding_option="zero"` 在 NPU 上基本可用，但部分场景可能不完全支持
- 如果遇到 `padding_option` 相关问题，可改用 `mask` + `other=0.0` 方式
- 对于 block pointer 的 load，`care_padding` 参数不直接适用（它是普通 `tl.load` 的扩展参数）

### store_if

```python
@triton.jit
def store_if(block_ptr, value, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        tl.store(block_ptr, value)
    elif EVEN_N:
        tl.store(block_ptr, value, boundary_check=(0,))
    elif EVEN_M:
        tl.store(block_ptr, value, boundary_check=(1,))
    else:
        tl.store(block_ptr, value, boundary_check=(0, 1))
```

**NPU 注意事项**：
- `store_if` 在 NPU 上行为与 GPU 一致
- 离散 mask 的 store 在 NPU 上会被拆解为 atomic {load, select, store} 组合，可能存在泛化性问题
- 优先使用连续 mask 或 block pointer 方式

---

## FlashAttention 中 Block Pointer 的实际迁移 Diff

以下基于 `flash_attention.py`（GPU 版本）和 `flash_attention_npu_v8.py`（NPU 版本）的实际对比：

### Diff 1：Grid 调度模式

| 维度 | GPU | NPU |
|------|-----|-----|
| Grid 形状 | 3D `(cdiv(max_seqlen_q, BLOCK_M), q_head, batch_size)` | 1D `(AICORE_NUM,)` |
| program_id | 3 个维度 `tl.program_id(0/1/2)` | 1 个维度 `tl.program_id(0)` |
| 任务分配 | 每个 program 处理一个 (m_block, head, batch) | 每个 program 循环处理多个任务块 |

### Diff 2：fwd_kernel 结构变化

```python
# GPU: 3D program_id + return 跳过
start_m = tl.program_id(0)
start_qh = tl.program_id(1)
start_b = tl.program_id(2)
if start_m * BLOCK_M >= q_len:
    return

# NPU: 1D program_id + for 循环 + if 条件跳过
pid = tl.program_id(0)
for block_idx in range(pid, NUM_BLOCKS, AICORE_NUM):
    start_m = block_idx % NUM_BLOCKS_M
    start_qh = task_hz_idx % q_head
    start_b = task_hz_idx // q_head
    if start_m * BLOCK_M < q_len:
        ...
```

### Diff 3：make_block_ptr 创建位置

```python
# GPU: 循环外创建，循环内 advance
q_block_ptr = tl.make_block_ptr(...)  # 循环外
k_block_ptr = tl.make_block_ptr(...)
for start_n in range(begin, end, BLOCK_N):
    k = load_if(k_block_ptr, ...)
    k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))

# NPU: 外层循环内创建，内层循环内 advance
for block_idx in range(...):
    q_block_ptr = tl.make_block_ptr(...)  # 外层循环内
    k_block_ptr = tl.make_block_ptr(...)
    for start_n in range(begin, end, BLOCK_N):
        k = load_if(k_block_ptr, ...)
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
```

### Diff 4：Softmax 计算差异

```python
# GPU: exp2 + log2e
log2e: tl.constexpr = 1.4426950408889634
qk_scale = scale * log2e
s = tl.dot(q, k)
s = tl.where(mask & boundary_mask, s, -2**30)
m_new = tl.maximum(m, tl.max(s, 1))
alpha = tl.math.exp2((m - m_new) * qk_scale)
p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
acc *= alpha[:, None]
acc += tl.dot(p.to(dtype), v)
l = m * scale + tl.log(l)

# NPU: exp 直接计算 + mask 融合 + propagate_nan
qk_scale = scale  # 不乘 log2e
s = tl.dot(q, k)
s = s * qk_scale + tl.where(mask, 0.0, -2.0 ** 30)  # mask 融合
m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
p = tl.math.exp(s - m_new[:, None])
alpha = tl.math.exp(m - m_new)
pv = tl.dot(p.to(dtype), v)
acc = acc * alpha[:, None] + pv  # 合并乘加
m = m + tl.log(l)  # 不乘 scale
```

### Diff 5：Mask 从函数计算改为预计算 tensor 加载

```python
# GPU: 在 kernel 内通过 mask_fn 函数计算 mask
q_attn_arg = load_if(q_attn_arg_block_ptr, False, True)
k_attn_arg = load_if(k_attn_arg_block_ptr, False, True)
mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN)

# NPU: 预计算 mask tensor，通过 block_ptr 加载
mask_block_ptr = tl.make_block_ptr(
    base=mask_tensor_ptr + start_b * MAX_Q_LEN * MAX_K_LEN,
    shape=(q_len, k_len),
    strides=(MAX_K_LEN, 1),
    offsets=(start_m * BLOCK_M, begin),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0)
)
mask = load_if(mask_block_ptr, False, False)
```

### Diff 6：新增 kernel 参数

NPU 版本新增了以下 constexpr 参数：

| 参数 | 说明 |
|------|------|
| `AICORE_NUM` | AI Core 数量，用于核内循环步长 |
| `MAX_Q_LEN` | 最大 Q 序列长度，用于 mask tensor 寻址 |
| `MAX_K_LEN` | 最大 K 序列长度，用于 mask tensor 寻址 |
| `BATCH_SIZE` | batch 大小，用于任务块总数计算 |

### Diff 7：kernel launch 参数

```python
# GPU: 多维 grid + num_stages/num_warps
grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), q_head, batch_size)
fwd_kernel[grid](..., BLOCK_M=128, BLOCK_N=32)

# NPU: 1D grid + NPU 特有编译选项
NUM_CORES = AICORE_NUM
grid = (NUM_CORES,)
fwd_kernel[grid](
    ...,
    AICORE_NUM=NUM_CORES,
    MAX_Q_LEN=max_seqlen_q,
    MAX_K_LEN=max_seqlen_k,
    BATCH_SIZE=batch_size,
    multibuffer=True,
    enable_mixed_cv=True,
    enable_auto_bind_sub_block=True,
    sync_solver=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    enable_flatten=False,
    set_workspace_multibuffer=2,
)
```

---

## 910_95 特别注意

### 1. UB 容量差异

| 硬件型号 | UB 容量 | 开启 double buffer 后 |
|---------|---------|---------------------|
| A2/A3 系列 | 192 KB | 96 KB |
| Ascend910_95/950 | 256 KB | 128 KB |

910_95 系列 UB 容量更大，但仍需注意 BLOCK_SIZE 选择，避免 UB 溢出。

### 2. MultiBuffer 默认行为

| 硬件型号 | MultiBuffer 默认值 | 说明 |
|---------|-------------------|------|
| A2/A3 系列 | `True` | 默认开启存算并行 |
| 910_95 系列 | `False` | 默认关闭，需手动开启 |

910_95 系列默认关闭 MultiBuffer，如需开启存算并行，需在 Config 或 kernel launch 时显式设置 `multibuffer=True`。开启后 UB 可用容量减半。

### 3. L0C -> UB 直通路径

910_95 系列支持 L0C -> UB 直通路径（通过 FixPipe），CV 融合算子无需经过 GM 中转，性能更优。非 910_95 系列需要 L0C -> GM -> UB，多一次 GM 往返。

```
# 910_95 数据通路
GM -> L1 -> L0A/L0B -> Cube -> L0C -> UB -> Vector -> UB -> GM

# 非 910_95 数据通路
GM -> L1 -> L0A/L0B -> Cube -> L0C -> GM -> UB -> Vector -> UB -> GM
```

### 4. L0C 容量差异

| 硬件型号 | L0C 容量 |
|---------|---------|
| A2/A3 系列 | 128 KB |
| 910_95 系列 | 256 KB |

910_95 系列的 L0C 容量翻倍，支持更大的矩阵乘法分块。

### 5. FFTS 不支持

910_95 系列不支持 FFTS（Fast Fourier Transform Scheduling），编译器会自动禁用。可通过 `TRITON_DISABLE_FFTS=true` 环境变量在其他架构上手动禁用。

### 6. FP8 数据类型

910_95 系列额外支持 FP8 数据类型，A2/A3 系列不支持。

### 7. Reg-based 架构

910_95 采用 Reg-based 架构，支持 SIMT 模式和 DCache + RF 128KB，与 A2/A3 的 Mem-based 架构有显著差异。

---

## 迁移检查清单

迁移使用 `tl.make_block_ptr` 的 GPU kernel 时，按以下清单逐项检查：

- [ ] **数据类型**：确认 block pointer 操作的数据类型不是 uint8/uint16/uint32/uint64/fp64
- [ ] **stride/order**：确认 stride 反映真实内存布局，转置语义通过 order 参数表达
- [ ] **Grid 模式**：将多维 Grid 改为 1D Grid + 核内循环
- [ ] **return/continue**：将循环内的 return/continue 改为 if 条件跳过
- [ ] **make_block_ptr 位置**：将 block_ptr 创建移入外层循环内
- [ ] **advance 兼容性**：检查 advance 是否与复杂控制流搭配，必要时改用重新创建 block_ptr
- [ ] **连续访存**：检查是否存在离散访存模式，改为连续访存
- [ ] **Softmax 适配**：检查 exp2/log2e 计算，考虑改用 exp 直接计算
- [ ] **propagate_nan**：在 tl.maximum/tl.max 中添加 propagate_nan 参数
- [ ] **compile_hint**：在关键位置添加 `extension.compile_hint` 优化
- [ ] **num_warps/num_stages**：从 Config 中移除这些无效参数
- [ ] **UB 容量**：根据目标硬件（A2/A3 vs 910_95）调整 BLOCK_SIZE
- [ ] **MultiBuffer**：910_95 需手动开启 multibuffer=True
- [ ] **kernel launch 参数**：添加 NPU 特有的编译选项（multibuffer, enable_mixed_cv, sync_solver 等）

---

## 相关文档链接

- [Block Pointer 迁移注意事项（源文档）](../docs_triton_ascend/06-Migration-from-GPU/04-block-pointer-migration.md)
- GPU 版 FlashAttention：`flash_attention.py`
- NPU 版 FlashAttention：`flash_attention_npu_v8.py`
- [内存操作 API](../docs_triton_ascend/02-Core-API/01-memory-ops.md)
- [NPU 内存层次](../docs_triton_ascend/01-Programming-Model/03-memory-model.md)
- [代码迁移模式](../docs_triton_ascend/06-Migration-from-GPU/02-code-migration-patterns.md)
- [编译错误排查](../docs_triton_ascend/07-Debugging/03-compile-errors.md)
- [Tiling 策略](../docs_triton_ascend/05-Performance-Optimization/02-tiling-strategy.md)
- [FAQ](../docs_triton_ascend/09-Reference/05-faq.md)
