# Ascend NPU 上 Triton API 约束（静态检视用）

本文档仅包含可通过阅读代码识别的 API 约束。

## 1. Masking — 必须遵守

Ascend 对越界访问**零容错**。静态检视时，检查所有 `tl.load`/`tl.store` 是否有 `mask=` 参数：

```python
# ❌ 缺少 mask
x = tl.load(x_ptr + offsets)

# ✅ 有 mask
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# ✅ make_block_ptr（自动处理边界）
block_ptr = tl.make_block_ptr(
    base=ptr, shape=(M, N), strides=(stride_m, stride_n),
    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
data = tl.load(block_ptr)
block_ptr = tl.advance(block_ptr, (0, BLOCK_N))
```

## 2. BLOCK_SIZE 约束（静态可查）

| 检查项 | 代码中如何识别 |
|--------|---------------|
| BLOCK_SIZE 非 constexpr | 函数参数无 `: tl.constexpr` 声明 |
| 矩阵 BLOCK 非 16 倍数 | `BLOCK_M=100`、`BLOCK_N=50` 等字面量 |
| BLOCK_K 未对齐 | 未按 `kalign = 32 // dtype_bytes` 计算 |

```python
# BLOCK_K 对齐（来自官方测试用例）
dtype_bytes = torch.tensor(0, dtype=eval('torch.' + dtype)).element_size()
kalign = 32 // dtype_bytes
BLOCK_K = min(max(K, kalign), 32)
```

## 3. 精度约束（静态可查）

| 代码模式 | 问题 |
|----------|------|
| `tl.sum(x_fp16, ...)` 无前置 `.to(tl.float32)` | 归约未升精度 |
| `tl.dot(a, b)` 无显式 `out_dtype` | 浮点默认 fp32、int8 仅 int32 可选，显式指定非必要 |
| `tl.exp(x)` 而非 `tl.exp(x - max_x)` | Softmax 数值不稳定 |

**矩阵乘法精度模式**（来自官方测试用例）：

```python
if dtype == "int8":
    accumulator_type = tl.int32
else:
    accumulator_type = tl.float32

accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
c = accumulator.to(c_ptr.dtype.element_ty)
```

## 4. Grid 配置约束（静态可查）

| 代码模式 | 问题 |
|----------|------|
| `grid = (20,)` 等字面量 | 硬编码核数 |
| 矩阵 kernel 用 `num_vectorcore` | 含 `tl.dot` 应用 AI Core |
| 矩阵乘法无 `tl.dot`（用逐元素乘加实现 matmul/GEMV） | Cube Core 吞吐远高于 Vector Core，即使 N=1 也必须 pad 后用 `tl.dot` |
| 逐元素 kernel 用 `num_aicore` | 无 `tl.dot` 应用 Vector Core |

```python
import triton.runtime.driver as driver
device = torch.npu.current_device()
# 含 tl.dot
num_aicore = driver.active.utils.get_device_properties(device)["num_aicore"]
# 不含 tl.dot
num_vectorcore = driver.active.utils.get_device_properties(device)["num_vectorcore"]
```

## 5. Atomic 操作约束（静态可查）

| 代码模式 | 问题 |
|----------|------|
| `for ...: tl.atomic_cas/or/xor/and/xchg(...)` | 不支持在 loop 中使用 |
| 多核 kernel 中 `ret = tl.atomic_add(...)` 并使用 `ret` | 不支持多核 add + 保存中间结果 |

## 6. 特定 Op 约束（静态可查）

以下约束基于官方测试用例的实际使能情况：

| Op | 约束 | 测试用例状态 |
|----|------|-------------|
| `tl.dot` | 输入仅支持 int8/fp16/fp32/bf16 | generalization_cases 使能 |
| `dot_scaled` | ⚠ 有条件支持（lhs/rhs 仅 bf16/fp16，scale 仅 int8 ue8m0，输出仅 fp32） | 有条件使能 |
| `tl.sort` | 支持 1D~5D | generalization_cases 和 pytest_ut 均使能 |
| `tl.gather` | 支持多轴（axis 0~4） | generalization_cases 使能；pytest_ut 标记 skip |
| `permute`/`trans` (2,1,0) | 3D 不相邻轴转置 | generalization_cases 注释掉；pytest_ut test_permute_full 使能 |
| `permute`/`trans` | 不支持 int64 | generalization_cases 使能但排除 int64 |
| `tensor_descriptor` | make/load/store 需配套使用 | generalization_cases 使能 |

## 7. 代码模式约束（静态可查）

| 代码模式 | 问题 |
|----------|------|
| `for i in range(N):` 在 kernel 中 | loop 次数少且固定时可考虑 `tl.static_range`；loop 数较大时收益不明显甚至可能劣化，不应盲目替换 |
| `import numpy`/`import xxx` 在 kernel 中 | kernel 内不可调用第三方库 |
| BLOCK_SIZE 参数无 `: tl.constexpr` | 必须为编译时常量 |
| `tensor.item()` 在 Host 循环中 | CPU-NPU 同步瓶颈 |

## 8. 控制流约束（静态可查）

Triton kernel 编译为 MLIR 结构化控制流（`scf.for`/`scf.while`），不支持 early exit。

| 代码模式 | 问题 |
|----------|------|
| `for/while` 循环内 `return` | 编译错误："Cannot have return statements inside while or for"（含子函数中的 return，transitively checked） |
| `for` 循环内 `break` | 编译错误："unsupported AST node type: Break" |

```python
# ❌ 循环内 return
for i in range(N):
    if cond:
        return val  # 编译错误

# ❌ 循环内 break
for i in range(N):
    if cond:
        break  # 编译错误

# ✅ 替代方案 1：while + 布尔标志
i = 0
active = True
while i < N and active:
    result = tl.where(cond, desired, result)
    active = tl.where(cond, False, True)  # 手动终止
    i += 1

# ✅ 替代方案 2：tl.where mask（推荐）
for i in range(N):
    result = tl.where(active_mask, compute(i), result)
```

## 9. Tensor 索引约束（静态可查）

Triton tensor 不支持 Python 风格 `[]` 下标操作（读取和赋值均不支持）。

| 代码模式 | 问题 |
|----------|------|
| `tensor[i] = val` | AssertionError（`visit_Subscript` 中 ctx 非 Load） |
| `val = tensor[i]` | AssertionError |
| `tensor[i:j]` 切片 | 编译错误，Python 切片不支持 |

```python
# ❌ 索引赋值
local_vector = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
local_vector[i] = 123.0  # AssertionError

# ✅ tl.where 替代
idx = tl.arange(0, BLOCK_SIZE)
local_vector = tl.where(i == idx, 123.0, local_vector)

# ✅ tl.full（更直接）
local_vector = tl.full((BLOCK_SIZE,), 123.0, dtype=tl.float32)

# ❌ 从 tensor 按索引取值
val = tensor[i]  # AssertionError

# ✅ tl.gather 替代
val = tl.gather(tensor, index, axis=0)

# ❌ 切片
subx = x[1:3]  # 编译错误

# ✅ tl.extract_slice 替代
subx = tl.extract_slice(x, offsets=(1,), sizes=(2,), strides=(1,))
```

## Ascend 扩展 API

```python
import triton.language.extra.cann.extension as extension

# extract_slice / insert_slice：分片处理大 tensor
acc_i = extension.extract_slice(acc, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
acc = extension.insert_slice(acc, acc_i, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))

# extension.sort：2D/3D 多维排序
x = extension.sort(x, descending=False, dim=1)
```

## 10. Ascend 高性能 API

### tl.parallel(bind_sub_block=True)

将 post-dot 逐元素操作（激活、cast、store）分配到 2 个 Vector Core 并行执行。

```python
# ✅ 正确用法：post-dot 操作并行化
for s in tl.parallel(0, 2, bind_sub_block=True):
    sub = tl.extra.ascend.extract_slice(acc, (s * HALF, 0), (HALF, N), (1, 1))
    sub = tl.sigmoid(sub)  # activation
    sub = sub.to(out_dtype)
    tl.store(out_ptr + offsets_for(s), sub, mask=mask)
```

约束：需 `extract_slice`/`insert_slice` 配合分片；仅适用于 dot 后的逐元素操作。

### tl.load(care_padding=False)

masked 位置返回随机值而非 `other`，减少额外计算开销，约 5-10% 性能提升。

```python
# ✅ 正确用法：下游不依赖 mask 位置的值
x = tl.load(ptr + offsets, mask=mask, care_padding=False)
# mask 为 False 的位置值不确定，但后续计算中被 mask 屏蔽
result = tl.where(mask, x * scale, 0.0)
```

约束：仅在 masked 位置不被下游使用时安全（如 result 被 `tl.where(mask, ...)` 再次屏蔽）。

### tl.cast(overflow_mode=...)

控制类型转换溢出行为：

| 参数 | 行为 | 性能 |
|------|------|------|
| `"trunc"`（默认） | 截断高位比特 | 快 |
| `"saturate"` | 钳位到目标类型 min/max | A2/A3 上走 FP32，较慢 |

```python
# ✅ 正确用法：需要饱和截断时显式指定
y = tl.cast(x, tl.int8, overflow_mode="saturate")  # clamp to [-128, 127]

# ⚠ 默认 trunc 可能静默丢失数据
y = tl.cast(x, tl.int8)  # 大值会被截断而非钳位
```

约束：`saturate` 在 A2/A3 上会走 FP32 路径，注意性能影响。

### sync_block_set / sync_block_wait

Cube-Vector 同步原语，用于 dot 后 Vector 操作与 Cube 计算的同步。

```python
# ✅ 正确用法：Cube 计算后通知 Vector
tl.dot(a, b, acc)
tl.sync_block_set(sender="cube", receiver="vector", event_id=0)

# Vector 侧等待 Cube 完成
tl.sync_block_wait(sender="cube", receiver="vector", event_id=0)
result = acc.to(tl.float16)
```

约束：
- `event_id` 范围 0-15
- **P0**：多个并行块使用 sync 时，event_id 不可冲突

### index_select_simd()

`triton.language.extra.ascend.libdevice` 中的并行索引选择，比 gather 更高效。

```python
from triton.language.extra.ascend.libdevice import index_select_simd

# ✅ 正确用法：dim 不能是最后一维，read_shape[dim] 必须为 -1
# src: (M, K), index: (M,), output: (M, N) where N = len(index)
result = index_select_simd(src, index, dim=0)  # dim=0 < ndim-1=1 ✅
```

约束：
- `dim < ndim - 1`（不能是最后一维）
- `read_shape[dim]` 必须为 `-1`
- P1：检查 gather 模式是否可用 `index_select_simd` 替代

## 参考资源

- [Triton-Ascend 官方仓库](https://gitcode.com/Ascend/triton-ascend)
- [API 数据类型支持矩阵](ascend-api-dtype-matrix.md)
