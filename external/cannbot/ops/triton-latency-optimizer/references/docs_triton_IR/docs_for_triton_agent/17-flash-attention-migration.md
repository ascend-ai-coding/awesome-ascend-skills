# Flash Attention 迁移实例文档

## 触发条件

Agent 迁移 Flash Attention 类算子时（包含 `tl.dot` + softmax + mask 的注意力计算模式，含前向和反向传播）。

---

## 核心知识：迁移 Diff 分析

本文档基于以下两个文件的逐行对比：

- **GPU 版本**：`flash_attention.py`
- **NPU 版本**：`flash_attention_npu_v8.py`

共提取 **12 个关键 Diff 点**，按迁移修改顺序排列。

---

## Diff 1：Autotune 配置变化（移除 num_stages/num_warps，NPU 特有 Config）

### 变化说明

GPU 版本的 `triton.Config` 使用 `num_stages` 和 `num_warps` 控制软件流水线深度和 warp 并行度。NPU 版本移除这两个参数，因为 Ascend 架构没有 GPU 意义上的 warp 和软件流水线 stage 概念。NPU 使用 `multibuffer`、`enable_mixed_cv` 等编译参数替代。

此外，NPU 版本新增了 `get_bwd_qkv_configs()` 函数，用于融合反向 QKV kernel 的 autotune 配置。

### GPU 代码

```python
def get_fwd_configs():
    if is_hopper():
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4)
        ]
    elif is_ampere():
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=3, num_warps=4)
        ]
    else:
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=2, num_warps=4)
        ]
```

### NPU 代码

```python
def get_fwd_configs():
    if is_hopper():
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4)
        ]
    elif is_ampere():
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=3, num_warps=4)
        ]
    else:
        configs = [
            triton.Config(
                {
                    "BLOCK_M": BM,
                    "BLOCK_N": BN,
                },
            )
            for BM in [32, 64, 128]
            for BN in [32, 64, 128]
        ]
        return configs

def get_bwd_qkv_configs():
    return [
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}),
    ]
```

### 迁移要点

1. NPU 的 `else` 分支（非 Hopper/Ampere）移除 `num_stages` 和 `num_warps`
2. NPU 新增 `get_bwd_qkv_configs()` 用于融合反向 QKV kernel
3. `@maybe_triton_jit` 装饰器替换为 `@triton.autotune`（NPU 版本使用标准 autotune）

---

## Diff 2：Grid 调度变化（3D Grid -> 1D Grid + 核内循环，Grid 收缩到 AICORE_NUM）

### 变化说明

GPU 版本使用 3D Grid `(cdiv(max_seqlen_q, BLOCK_M), q_head, batch_size)`，每个 program 处理一个 (M块, head, batch) 组合。NPU 版本将 Grid 收缩为 1D `(AICORE_NUM,)`，每个 AICore 通过核内循环遍历所有任务块，采用 interleaved 分配策略（`pid, NUM_BLOCKS, AICORE_NUM`）实现负载均衡。

### GPU 代码

```python
# 前向 kernel 内部
start_m = tl.program_id(0)
start_qh = tl.program_id(1)
start_b = tl.program_id(2)

# 调用处
grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), q_head, batch_size)
fwd_kernel[grid](...)
```

### NPU 代码

```python
# 前向 kernel 内部
pid = tl.program_id(0)
NUM_BLOCKS_M = tl.cdiv(MAX_Q_LEN, BLOCK_M)
NUM_BLOCKS = NUM_BLOCKS_M * BATCH_SIZE * q_head

start_block, end_block, step = pid, NUM_BLOCKS, AICORE_NUM

for block_idx in range(start_block, end_block, step):
    task_hz_idx = block_idx // NUM_BLOCKS_M
    start_m = block_idx % NUM_BLOCKS_M
    start_b = task_hz_idx // q_head
    start_qh = task_hz_idx % q_head
    start_kvh = start_qh * kv_head // q_head
    # ... 原来的 kernel 逻辑放在 if 条件内

# 调用处
NUM_CORES = AICORE_NUM
grid = (NUM_CORES,)
fwd_kernel[grid](
    ...,
    AICORE_NUM=NUM_CORES,
    MAX_Q_LEN=max_seqlen_q,
    MAX_K_LEN=max_seqlen_k,
    BATCH_SIZE=batch_size,
)
```

### 迁移要点

1. Grid 从 3D `(M_blocks, heads, batch)` 变为 1D `(AICORE_NUM,)`
2. Kernel 入参新增 `AICORE_NUM`、`MAX_Q_LEN`、`MAX_K_LEN`、`BATCH_SIZE` 等 constexpr 参数
3. 使用 `range(pid, NUM_BLOCKS, AICORE_NUM)` 的 interleaved 模式分配任务
4. 从 `block_idx` 反推 `start_m`、`start_qh`、`start_b`：先除以 `NUM_BLOCKS_M` 得到 hz 索引，再分别除以/取模 `q_head` 得到 batch 和 head
5. `start_kvh` 的计算从 `start_qh // (q_head // kv_head)` 改为 `start_qh * kv_head // q_head`（等价但避免整数除法精度问题）

### 设备属性获取

```python
import triton.runtime.driver as driver

device = torch.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
AICORE_NUM = properties["num_aicore"]
VECTOR_NUM = properties["num_vectorcore"]
```

---

## Diff 3：for 循环内 return -> if 条件跳过

### 变化说明

GPU 版本在 kernel 入口处使用 `return` 提前退出不合法的 program（如 `start_m * BLOCK_M >= q_len`）。但 Triton 不允许在 `for`/`while` 循环内使用 `return` 语句。由于 NPU 版本将整个 kernel 逻辑包裹在核内 `for` 循环中，必须将 `return` 替换为 `if` 条件守卫。

### GPU 代码

```python
start_m = tl.program_id(0)
start_qh = tl.program_id(1)
start_b = tl.program_id(2)

q_start = tl.load(cu_seqlens_q + start_b)
q_end = tl.load(cu_seqlens_q + start_b + 1)
q_len = q_end - q_start
if start_m * BLOCK_M >= q_len:
    return
```

### NPU 代码

```python
for block_idx in range(start_block, end_block, step):
    task_hz_idx = block_idx // NUM_BLOCKS_M
    start_m = block_idx % NUM_BLOCKS_M
    start_b = task_hz_idx // q_head
    start_qh = task_hz_idx % q_head

    q_start1 = tl.load(cu_seqlens_q + start_b)
    q_end = tl.load(cu_seqlens_q + start_b + 1)
    q_len = q_end - q_start1
    if start_m * BLOCK_M < q_len:
        # 原来的 kernel 逻辑放在 if 内
        ...
```

### 迁移要点

1. 所有 `return` 替换为 `if <valid_condition>:` 的正向守卫
2. 条件取反：`if start_m * BLOCK_M >= q_len: return` -> `if start_m * BLOCK_M < q_len:`
3. 嵌套的 `return` 也需要同样处理（如 `if begin >= k_len: return` -> `if begin < k_len:`）
4. 同理，`continue` 也不被支持，需要用 `if` 条件守卫替代

---

## Diff 4：Block Pointer 使用差异

### 变化说明

Block Pointer 的创建和使用方式基本一致，但 NPU 版本有两处关键差异：

1. **mask 相关的 block pointer 被替换**：`q_attn_arg_block_ptr` 和 `k_attn_arg_block_ptr` 被注释掉，替换为 `mask_block_ptr`（指向预计算的 mask_tensor）
2. **`tl.multiple_of` 提示被移除**：NPU 版本在部分 kernel 中注释掉了 `start_n = tl.multiple_of(start_n, BLOCK_N)` 等对齐提示

### GPU 代码

```python
q_attn_arg_block_ptr = tl.make_block_ptr(
    base = q_attn_arg_ptr + q_start,
    shape = (q_len,),
    strides = (1,),
    offsets = (start_m * BLOCK_M,),
    block_shape = (BLOCK_M,),
    order = (0,)
)
k_attn_arg_block_ptr = tl.make_block_ptr(
    base = k_attn_arg_ptr + k_start,
    shape = (k_len,),
    strides = (1,),
    offsets = (begin,),
    block_shape = (BLOCK_N,),
    order = (0,)
)

# 循环内
for start_n in range(begin, end, BLOCK_N):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    k_attn_arg = load_if(k_attn_arg_block_ptr, False, True)
    ...
    k_attn_arg_block_ptr = tl.advance(k_attn_arg_block_ptr, (BLOCK_N,))
```

### NPU 代码

```python
mask_block_ptr = tl.make_block_ptr(
    base=mask_tensor_ptr + start_b * MAX_Q_LEN * MAX_K_LEN,
    shape=(q_len, k_len),
    strides=(MAX_K_LEN, 1),
    offsets=(start_m * BLOCK_M, begin),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0)
)

# 循环内
for start_n in range(begin, end, BLOCK_N):
    # start_n = tl.multiple_of(start_n, BLOCK_N)  # NPU 版本注释掉
    mask = load_if(mask_block_ptr, False, False)
    ...
    mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_N))
```

### 迁移要点

1. 新增 `mask_tensor_ptr` 参数和 `mask_block_ptr` block pointer
2. `mask_block_ptr` 的 base 地址为 `mask_tensor_ptr + start_b * MAX_Q_LEN * MAX_K_LEN`，按 batch 偏移
3. `mask_block_ptr` 的 shape 为 `(q_len, k_len)`，strides 为 `(MAX_K_LEN, 1)`
4. 移除 `q_attn_arg_block_ptr` 和 `k_attn_arg_block_ptr`（不再需要动态计算 mask）
5. `tl.multiple_of` 在部分 kernel 中可保留或移除，对 NPU 性能影响不大

---

## Diff 5：mask 计算方式变化（从动态计算 mask_fn -> 预计算 mask_tensor）

### 变化说明

GPU 版本在 kernel 内部通过 `mask_fn` 函数动态计算每个 block 的 mask，输入为 `q_attn_arg`、`k_attn_arg`、`offset_m`、`offset_n`。NPU 版本将 mask 预计算为一个 `[batch, max_q_len, max_k_len]` 的 bool tensor，在 kernel 外部通过 `generate_mask_fn_vectorized` 函数生成，kernel 内部直接从 `mask_tensor` 加载。

### GPU 代码

```python
@triton.jit
def mask_fn(q_attn_arg, k_attn_arg, q_offset, k_offset, TYPE: tl.constexpr):
    tril_causal = q_offset[:, None] >= k_offset[None, :]
    triu_causal = q_offset[:, None] <= k_offset[None, :]
    if TYPE == 1:
        return (
            (triu_causal &
                ((q_attn_arg[:, None] == k_attn_arg[None, :]) |
                (k_attn_arg[None, :] == 0))) |
            (q_offset[:, None] == k_offset[None, :]))
    if TYPE == 2:
        return (
            (tril_causal &
                ((q_attn_arg[:, None] == k_attn_arg[None, :]) |
                (k_attn_arg[None, :] == 0))) |
            (q_offset[:, None] == k_offset[None, :]))

# kernel 内使用
mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN)
```

### NPU 代码

```python
# kernel 内使用（直接从预计算 tensor 加载）
mask = load_if(mask_block_ptr, False, False)

# kernel 外预计算
def generate_mask_fn_vectorized(q_seq_list, k_seq_list, bs, max_q_len, max_k_len,
                                 q_attn_arg, k_attn_arg):
    device = "cpu"
    mask_fn = torch.zeros((bs, max_q_len, max_k_len), dtype=torch.bool, device=device)
    for b_i in range(bs):
        cur_q_len = q_seq_list[b_i]
        cur_k_len = k_seq_list[b_i]
        q_positions = torch.arange(cur_q_len, device=device).view(-1, 1)
        k_positions = torch.arange(cur_k_len, device=device).view(1, -1)
        causal_mask = (q_positions <= k_positions)
        q_attn_slice = torch.tensor(q_attn_arg[:cur_q_len], device=device, dtype=torch.int32).view(-1, 1)
        k_attn_slice = torch.tensor(k_attn_arg[:cur_k_len], device=device, dtype=torch.int32).view(1, -1)
        attn_args_mask = (q_attn_slice == k_attn_slice) | (k_attn_slice == 0)
        q_offset_mask = (q_positions == k_positions)
        result_mask = ((causal_mask.bool() & attn_args_mask.bool()) | q_offset_mask.bool()).to(torch.bool)
        mask_fn[b_i, :cur_q_len, :cur_k_len] = result_mask
    return mask_fn
```

### 迁移要点

1. 将 `mask_fn` 的计算逻辑从 kernel 内移到 kernel 外，在 CPU 上预计算
2. mask_tensor 形状为 `[batch, max_q_len, max_k_len]`，dtype 为 `torch.bool`
3. mask_tensor 需要作为参数传入 `FlashAttentionFunc.forward()`，并保存到 `ctx` 中供反向使用
4. kernel 内部通过 `mask_block_ptr` 加载对应 block 的 mask
5. 预计算方式减少了 kernel 内的计算量，但增加了显存占用（`batch * max_q_len * max_k_len` bytes）
6. `mask_fn` 函数在 NPU 版本中仍保留（用于 TYPE==1 分支的 `.to(tl.int32)` 显式类型转换），但不再在主循环中调用

---

## Diff 6：Softmax 计算变化（tl.math.exp2 -> tl.math.exp，scale 位置变化）

### 变化说明

GPU 版本使用 `tl.math.exp2`（以 2 为底的指数函数）配合 `log2e` 常数实现快速 softmax，scale 在 exp2 之前乘入（`qk_scale = scale * log2e`）。NPU 版本改用 `tl.math.exp`（自然指数），scale 直接乘到 QK 点积结果上，不再使用 `log2e` 转换。

### GPU 代码

```python
log2e: tl.constexpr = 1.4426950408889634
qk_scale = scale * log2e

# 前向
s = tl.dot(q, k)
boundary_mask = (offset_n < k_len)[None, :]
s = tl.where(mask & boundary_mask, s, -2**30)
m_new = tl.maximum(m, tl.max(s, 1))
alpha = tl.math.exp2((m - m_new) * qk_scale)
p = tl.math.exp2((s - m_new[:, None]) * qk_scale)

# 反向
p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
```

### NPU 代码

```python
qk_scale = scale  # 不再乘 log2e

# 前向
s = tl.dot(q, k)
s = s * qk_scale + tl.where(mask, 0.0, -2.0 ** 30)
m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
p = tl.math.exp(s - m_new[:, None])

# 反向（bwd_kv_kernel / bwd_qkv_kernel 仍使用 exp2）
p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
```

### 迁移要点

1. **前向 kernel**：`exp2` -> `exp`，`qk_scale = scale * log2e` -> `qk_scale = scale`
2. **反向 kernel**：仍保留 `exp2` + `log2e` 的方式（`bwd_kv_kernel`、`bwd_qkv_kernel`、`bwd_q_kernel`）
3. **scale 位置变化**：GPU 先做 `tl.where(mask, s, -2**30)` 再隐式 scale；NPU 将 scale 和 mask 合并为一步 `s * qk_scale + tl.where(mask, 0.0, -2.0**30)`
4. **l 值计算变化**：GPU 版本 `l = m * scale + tl.log(l)`；NPU 版本 `m = m + tl.log(l)`（因为 scale 已在前面的 s 中乘入）

---

## Diff 7：CV 流水线优化（tl.dot(p,v) 提前，p_sum/alpha 延后）

### 变化说明

NPU 版本对前向 kernel 的 softmax 循环体进行了 CV 流水线优化。将 `tl.dot(p.to(dtype), v)`（Cube 操作）提前到 `p_sum` 和 `alpha` 计算（Vector 操作）之前，使得 Cube 和 Vector 可以并行执行：当 Cube 执行 `pv = tl.dot(p, v)` 时，Vector 可以同时计算 `p_sum` 和 `alpha`。

### GPU 代码

```python
m_new = tl.maximum(m, tl.max(s, 1))
alpha = tl.math.exp2((m - m_new) * qk_scale)
p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
p_sum = tl.sum(p, 1)
acc *= alpha[:, None]
v = load_if(v_block_ptr, False, True)
acc += tl.dot(p.to(dtype), v)
l = l * alpha + p_sum
m = m_new
```

### NPU 代码

```python
m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
p = tl.math.exp(s - m_new[:, None])
v = load_if(v_block_ptr, False, True)
pv = tl.dot(p.to(dtype), v)          # Cube 操作提前
p_sum = tl.sum(p, 1)                  # Vector 操作延后
alpha = tl.math.exp(m - m_new)        # Vector 操作延后
acc = acc * alpha[:, None] + pv       # 合并 acc 更新
l = l * alpha + p_sum
m = m_new
```

### 迁移要点

1. **v 的加载提前**：从 `acc *= alpha` 之后加载，提前到 `p` 计算之后立即加载
2. **`tl.dot(p, v)` 提前**：计算 `pv` 后暂存，不立即累加到 `acc`
3. **`p_sum` 和 `alpha` 延后**：在 `pv` 计算之后再计算
4. **`acc` 更新合并**：`acc *= alpha; acc += pv` 合并为 `acc = acc * alpha[:, None] + pv`
5. 这种重排使得 Cube（dot）和 Vector（sum/exp）可以流水线并行，提升硬件利用率

---

## Diff 8：propagate_nan 参数添加

### 变化说明

NPU 版本在 `tl.maximum` 和 `tl.max` 中添加了 `propagate_nan` 参数，确保当输入包含 NaN 时结果也传播 NaN。这是因为在 Flash Attention 的 online softmax 中，如果 `m` 或 `s` 包含 NaN（如全 mask 列导致 `-inf` 运算），默认行为可能吞没 NaN 导致后续计算错误。

### GPU 代码

```python
m_new = tl.maximum(m, tl.max(s, 1))
```

### NPU 代码

```python
m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
```

### 迁移要点

1. `tl.max(s, 1, propagate_nan=True)`：归约操作启用 NaN 传播
2. `tl.maximum(m, ..., propagate_nan=tl.PropagateNan.ALL)`：逐元素操作启用 NaN 传播
3. `tl.PropagateNan` 枚举值：
   - `tl.PropagateNan.NONE`（默认）：不传播 NaN
   - `tl.PropagateNan.ALL`：传播所有 NaN
4. 这是 NPU 特有的参数，GPU 版 Triton 不支持

---

## Diff 9：编译参数配置（enable_mixed_cv, sync_solver 等）

### 变化说明

NPU 版本在 kernel 调用时传入一组 Ascend 特有的编译参数，用于控制 Cube-Vector 协同、多缓冲流水线、同步策略等。这些参数是 NPU 性能优化的核心，不同的 kernel 类型使用不同的参数组合。

### GPU 代码

```python
fwd_kernel[grid](
    q, k, v, o, l,
    q_attn_arg, k_attn_arg,
    cu_seqlens_q, cu_seqlens_k,
    q_head, kv_head, scale,
    QK_DIM=qk_dim, V_DIM=v_dim,
    MASK_FN=mask_fn, SPARSE_OPT=sparse_opt,
    DTYPE=(19 if q.dtype == torch.float16 else 14),
)
```

### NPU 代码

```python
# 前向 kernel（CV 融合）
fwd_kernel[grid](
    q, k, v, o, l,
    q_attn_arg, k_attn_arg, mask_tensor,
    cu_seqlens_q, cu_seqlens_k,
    q_head, kv_head, scale,
    QK_DIM=qk_dim, V_DIM=v_dim,
    MASK_FN=mask_fn, SPARSE_OPT=sparse_opt,
    DTYPE=(19 if q.dtype == torch.float16 else 14),
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

# 反向预处理 kernel（纯 Vector）
bwd_preprocess_ifmn[(NUM_CORES,)](
    o, do, d, ...,
    multibuffer=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
)

# 反向 QKV 融合 kernel（CV 融合）
bwd_qkv_kernel[(NUM_CORES,)](
    q, k, v, dq, dk, dv, ...,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    enable_flatten=False,
    sync_solver=True,
    enable_mixed_cv=True,
)
```

### 编译参数速查

| 参数 | CV 融合 kernel | 纯 Vector kernel | 说明 |
|------|---------------|-----------------|------|
| `multibuffer` | `True` | `True` | 启用 ping-pong 多缓冲流水线 |
| `enable_mixed_cv` | `True` | 不需要 | 启用 Cube/Vector 混合调度 |
| `enable_auto_bind_sub_block` | `True` | 不需要 | 自动绑定子块，优化 CV 分配 |
| `sync_solver` | `True` | 不需要 | 自动求解 CV 间同步点 |
| `limit_auto_multi_buffer_of_local_buffer` | `"no-limit"` | `"no-limit"` | 不限制 UB 多缓冲分配 |
| `enable_flatten` | `False` | 不需要 | CV 融合不能展平 |
| `set_workspace_multibuffer` | `2` | 不需要 | workspace 双缓冲 |

> 完整编译参数说明见 [07-compile-params.md](07-compile-params.md)。

### 迁移要点

1. **判断 kernel 类型**：包含 `tl.dot` 的为 CV 融合 kernel，纯向量运算的为 Vector kernel
2. **CV 融合 kernel** 需要完整参数集（7 个参数）
3. **纯 Vector kernel** 只需 `multibuffer` 和 `limit_auto_multi_buffer_of_local_buffer`
4. 910_95 平台 `multibuffer` 默认为 `False`，需显式设置为 `True`

---

## Diff 10：bwd_preprocess 的 NPU 特化版本（bwd_preprocess_ifmn）

### 变化说明

GPU 版本的 `bwd_preprocess` 按 `(M块, head, batch)` 的 3D Grid 调度，每个 program 处理一个 (M块, head, batch) 组合。NPU 版本提供了两个实现：

1. **`bwd_preprocess`**：1D Grid 版本，与 fwd_kernel 类似的 interleaved 调度（代码中已注释掉）
2. **`bwd_preprocess_ifmn`**：更高效的特化版本，将数据按 `(seq, head, dim)` 的 3D block 加载，利用 `i-f-m-n` 内存布局的连续性

`bwd_preprocess_ifmn` 的核心思路是：将 `d = sum(o * do, dim=1)` 的计算从按 (M块, head) 分配改为按 seq 段分配，每个 task 处理一段连续的序列，在该段内同时遍历所有 head 和 dim。

### GPU 代码

```python
@triton.jit
def bwd_preprocess(
    o_ptr, do_ptr, d_ptr,
    cu_seqlens_q,
    q_head,
    V_DIM: tl.constexpr, DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    start_h = tl.program_id(1)
    start_b = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q + start_b)
    q_end = tl.load(cu_seqlens_q + start_b + 1)
    q_len = q_end - q_start
    if start_m * BLOCK_M >= q_len:
        return

    q_start = q_start.to(tl.int64)
    o_block_ptr = tl.make_block_ptr(
        base = o_ptr + q_start * q_head * V_DIM + start_h * V_DIM,
        shape = (q_len, V_DIM),
        strides = (q_head * V_DIM, 1),
        offsets = (start_m * BLOCK_M, 0),
        block_shape = (BLOCK_M, V_DIM),
        order = (1, 0),
    )
    # ... 类似加载 do，计算 d = sum(o * do, 1)
```

### NPU 代码

```python
@triton.jit
def bwd_preprocess_ifmn(
    o_ptr,
    do_ptr,
    d_ptr,
    Q_HEAD_NUM: tl.constexpr,
    V_DIM: tl.constexpr,
    DTYPE: tl.constexpr,
    TASK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    task_id = tl.program_id(0)
    task_start = task_id * TASK_SIZE

    for block_id in tl.range(0, NUM_BLOCKS):
        o_block_ptr = tl.make_block_ptr(
            base = o_ptr + task_start * Q_HEAD_NUM * V_DIM,
            shape = (TASK_SIZE, Q_HEAD_NUM, V_DIM),
            strides = (Q_HEAD_NUM * V_DIM, V_DIM, 1),
            offsets = (block_id * BLOCK_SIZE, 0, 0),
            block_shape = (BLOCK_SIZE, Q_HEAD_NUM, V_DIM),
            order = (2, 1, 0),
        )
        o = load_if(o_block_ptr, False, True).to(tl.float32)
        # ... 类似加载 do，计算 d = tl.sum(o * do, 2)
```

### 调用方式

```python
if ctx.v_dim > 64:
    BLOCK_SIZE = 16
else:
    BLOCK_SIZE = 32
NUM_CORES = VECTOR_NUM  # 使用 Vector 核心而非 AI Core
TASK_SIZE = triton.cdiv(ctx.max_seqlen_q * ctx.batch_size, VECTOR_NUM)
if TASK_SIZE < BLOCK_SIZE:
    NUM_CORES = triton.cdiv(ctx.max_seqlen_q * ctx.batch_size, BLOCK_SIZE)
    TASK_SIZE = BLOCK_SIZE
NUM_BLOCKS = triton.cdiv(TASK_SIZE, BLOCK_SIZE)
bwd_preprocess_ifmn[(NUM_CORES,)](
    o_ptr=o, do_ptr=do, d_ptr=d,
    Q_HEAD_NUM=ctx.q_head, V_DIM=ctx.v_dim, DTYPE=ctx.dtype,
    TASK_SIZE=TASK_SIZE, NUM_BLOCKS=NUM_BLOCKS, BLOCK_SIZE=BLOCK_SIZE,
    multibuffer=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
)
```

### 迁移要点

1. `bwd_preprocess_ifmn` 使用 **3D block pointer** `(TASK_SIZE, Q_HEAD_NUM, V_DIM)`，一次性加载所有 head
2. 归约维度从 `axis=1` 变为 `axis=2`（在 V_DIM 维度上求和）
3. 使用 `VECTOR_NUM` 而非 `AICORE_NUM` 作为核心数（纯 Vector 操作不需要 Cube）
4. `TASK_SIZE` 动态计算：`cdiv(max_seqlen_q * batch_size, VECTOR_NUM)`
5. `BLOCK_SIZE` 根据 `V_DIM` 选择：`V_DIM > 64` 时用 16，否则用 32
6. 不需要 `cu_seqlens_q` 参数，按连续段分配任务

---

## Diff 11：bwd_qkv 融合 kernel

### 变化说明

GPU 版本将反向传播分为三个独立 kernel：`bwd_preprocess`、`bwd_kv_kernel`、`bwd_q_kernel`。NPU 版本新增了 `bwd_qkv_kernel`，将 `bwd_kv_kernel` 和 `bwd_q_kernel` 融合为一个 kernel，通过 `tl.atomic_add` 累加 `dq`，避免多次遍历 Q/K/V 数据。

### GPU 代码

```python
# 分两步执行
# 步骤1：bwd_kv_kernel 计算 dk, dv
bwd_kv_kernel[grid](q, k, v, dk, dv, do, l, d, ...)

# 步骤2：bwd_q_kernel 计算 dq
bwd_q_kernel[grid](q, k, v, dq, do, l, d, ...)
```

### NPU 代码

```python
@triton.autotune(
    list(filter(keep, get_bwd_qkv_configs())),
    key = ["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
    reset_to_zero = ["dq_ptr"]
)
@triton.jit
def bwd_qkv_kernel(
        q_ptr, k_ptr, v_ptr,
        dq_ptr, dk_ptr, dv_ptr,
        do_ptr, l_ptr, d_ptr,
        q_attn_arg_ptr, k_attn_arg_ptr,
        mask_tensor_ptr,
        cu_seqlens_q, cu_seqlens_k,
        batch_size,
        max_q_len, max_k_len,
        q_head, kv_head,
        scale,
        QK_DIM: tl.constexpr, V_DIM: tl.constexpr,
        MASK_FN: tl.constexpr, SPARSE_OPT: tl.constexpr, DTYPE: tl.constexpr,
        NUM_CORES: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # ... 与 bwd_kv_kernel 类似的外层循环结构
    for start_m in range(begin, end, BLOCK_M):
        mask = load_if(mask_block_ptr, False, False)
        if not SPARSE_OPT or tl.sum(mask.cast(tl.int32)) != 0:
            q = load_if(q_block_ptr, False, True)
            s = tl.dot(q, k)
            l = load_if(l_block_ptr, False, True)
            p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
            p = tl.where(mask, p, 0.0)
            do = load_if(do_block_ptr, False, True)
            p = p.to(dtype)
            # dv
            dv += tl.dot(tl.trans(p), do)
            d = load_if(d_block_ptr, False, True)
            dp = tl.dot(do, v)
            ds = p * (dp - d[:, None])
            ds = tl.where(mask, ds, 0.0)
            ds = ds.to(dtype)
            # dq（使用 atomic_add 累加）
            dq_offs_base = (q_start + start_m) * q_head * QK_DIM + start_qh * QK_DIM
            row_offs = tl.arange(0, BLOCK_M)
            row_mask = start_m + row_offs < end
            col_offs = tl.arange(0, QK_DIM)
            extension.compile_hint(ds, "break_vf")
            dq = tl.dot(ds, tl.trans(k))
            dq *= scale
            tl.atomic_add(
                dq_ptr + (dq_offs_base + row_offs[:, None] * q_head * QK_DIM + col_offs[None, :]),
                dq,
                mask=row_mask[:, None]
            )
            # dk
            dk += tl.dot(tl.trans(ds), q)
```

### 调用方式

```python
if use_fused_bwd_qkv:
    dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)  # 需要 zeros 初始化
    NUM_CORES = AICORE_NUM
    bwd_qkv_kernel[(NUM_CORES,)](
        q, k, v, dq, dk, dv, do, l, d, ...,
        NUM_CORES=NUM_CORES,
        limit_auto_multi_buffer_of_local_buffer="no-limit",
        enable_flatten=False,
        sync_solver=True,
        enable_mixed_cv=True,
    )
    dq = dq.to(q.dtype)  # float32 -> 原始 dtype
else:
    # 回退到分离的 bwd_kv_kernel + bwd_q_kernel
    ...
```

### 迁移要点

1. **融合优势**：减少 Q/K/V 数据的重复加载，一次遍历同时计算 dq/dk/dv
2. **`dq` 使用 `tl.atomic_add`**：因为多个 K 块可能对同一个 Q 块贡献梯度，需要原子累加
3. **`dq` 初始化为 `torch.float32` 的 zeros**：`atomic_add` 需要初始值为 0，使用 float32 避免精度问题
4. **`reset_to_zero = ["dq_ptr"]`**：autotune 的 `reset_to_zero` 确保每次 autotune 运行前清零 dq
5. **`dq` 需要类型转换**：计算完成后 `dq = dq.to(q.dtype)`
6. **`extension.compile_hint(ds, "break_vf")`**：在 `ds` 和 `tl.dot(ds, tl.trans(k))` 之间打断 Vector Fusion
7. **`use_fused_bwd_qkv` 参数**：`FlashAttentionFunc.forward()` 新增此参数控制是否使用融合版本

---

## Diff 12：extension.compile_hint(ds, "break_vf") 的使用

在 `bwd_q_kernel` 和 `bwd_qkv_kernel` 中，NPU 版本在 `ds` 计算完成后、`tl.dot(ds, k)` 之前插入 `extension.compile_hint(ds, "break_vf")`，打断 Vector Function（VF）融合，防止过长的 VF 融合链导致 UB 溢出。

### GPU 代码

```python
ds = p * (dp - d[:, None])
boundary_mask = (offset_n < k_len)[None, :]
ds = tl.where(mask & boundary_mask, ds, 0.0)
dq += tl.dot(ds.to(dtype), k)
```

### NPU 代码

```python
ds = p * (dp - d[:, None])
ds = ds.to(dtype)
ds = tl.where(mask, ds, 0.0)
extension.compile_hint(ds, "break_vf")
dq += tl.dot(ds, k)
```

### 迁移要点

1. **导入方式**：`import triton.language.extra.cann.extension as extension`
2. **插入位置**：在 `ds` 的所有计算完成后、`tl.dot` 之前
3. **作用**：打断 Vector Function 融合，防止 UB 溢出
4. **适用场景**：当出现 `UB overflow` 错误时，可在关键位置添加 `break_vf`

> 完整 compile_hint API 说明见 [14-compile-hint-and-extension.md](14-compile-hint-and-extension.md)。

---

## 910_95 特别注意

### 1. multibuffer 默认值差异

910_95 平台 `multibuffer` 默认为 `False`，其他平台默认为 `True`。在 910_95 上必须显式设置 `multibuffer=True`。

### 2. enable_mixed_cv 平台限制

`enable_mixed_cv` 仅在 910_95（Ascend950）平台上生效。参见 [Config.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Tools/bishengir-compile/Config.h#L351-L363)。

### 3. UB 溢出回退策略

出现 UB 溢出时：
1. 减小 `BLOCK_M` / `BLOCK_N`
2. 设置 `set_workspace_multibuffer=1`
3. 设置 `limit_auto_multi_buffer_of_local_buffer="no-limit"`
4. 在关键位置添加 `extension.compile_hint(ds, "break_vf")`

### 4. propagate_nan 的重要性

910_95 上 NaN 传播行为可能与 GPU 不同，在 online softmax 的 `tl.maximum` 和 `tl.max` 中必须添加 `propagate_nan` 参数。

> 完整 910_95 硬件规格见 [00-hardware-quick-ref.md](00-hardware-quick-ref.md)。

---

## 迁移检查清单

| 序号 | 检查项 | 状态 |
|------|--------|------|
| 1 | Config 移除 `num_stages`/`num_warps` | [ ] |
| 2 | Grid 从 3D 改为 1D + 核内循环 | [ ] |
| 3 | `return` 替换为 `if` 条件守卫 | [ ] |
| 4 | `mask_fn` 动态计算替换为预计算 `mask_tensor` | [ ] |
| 5 | 前向 softmax 从 `exp2` 改为 `exp` | [ ] |
| 6 | CV 流水线优化：`tl.dot(p,v)` 提前 | [ ] |
| 7 | `propagate_nan` 参数添加 | [ ] |
| 8 | 编译参数配置（CV 融合 vs 纯 Vector） | [ ] |
| 9 | `bwd_preprocess_ifmn` 替换 `bwd_preprocess` | [ ] |
| 10 | `bwd_qkv_kernel` 融合 + `atomic_add` | [ ] |
| 11 | `extension.compile_hint(ds, "break_vf")` 插入 | [ ] |
| 12 | 910_95 平台 `multibuffer=True` 显式设置 | [ ] |

---

## 相关文档链接

- [07-compile-params.md](../docs_for_triton_agent/07-compile-params.md) - NPU 编译参数速查
- [05-cv-pipeline-optimization.md](../docs_for_triton_agent/05-cv-pipeline-optimization.md) - CV 流水线优化
- [14-compile-hint-and-extension.md](../docs_for_triton_agent/14-compile-hint-and-extension.md) - compile_hint 与扩展 API 速查
- [09-block-pointer-migration.md](../docs_for_triton_agent/09-block-pointer-migration.md) - Block Pointer 迁移指南
- [02-api-differences.md](../docs_for_triton_agent/02-api-differences.md) - GPU/NPU API 差异

### 源码参考

- `flash_attention.py` - GPU 版本 Flash Attention
- `flash_attention_npu_v8.py` - NPU 版本 Flash Attention
- [core.py: PropagateNan](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L22) - PropagateNan 枚举定义
- [core.py: maximum](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1957) - maximum 函数（含 propagate_nan 参数）
- [standard.py: max](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/standard.py#L174) - max 归约函数（含 propagate_nan 参数）
- [aux_ops.py: compile_hint](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/aux_ops.py#L135-L151) - compile_hint 函数定义
- [compiler.py: NPUOptions](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L846) - NPU 编译参数数据类
