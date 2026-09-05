# CV 流水线优化

## 触发条件

当 Agent 处理**同时包含 `tl.dot`（矩阵乘法）和向量运算**的融合算子时，应考虑应用 CV 流水线优化。典型场景包括：

- FlashAttention 及其变体（QK 矩阵乘 + Softmax + PV 矩阵乘）
- 矩阵乘法 + Bias + 激活函数（GELU / ReLU / Sigmoid）
- 矩阵乘法 + LayerNorm / Quantize
- 任何 Cube/Vector 计算交叉进行的算子

---

## Cube/Vector 分离架构原理

昇腾 NPU 采用 **Cube/Vector 分离架构**，每个 AI Core 由 1 个 Cube Core 和 2 个 Vector Core 组成：

| 硬件单元 | 执行运算 | 数据通路 | 每个AI Core数量 |
|----------|----------|----------|:---:|
| Cube Core | 矩阵乘法（`tl.dot`） | L1 -> L0C | 1 |
| Vector Core | 逐元素运算、归约、激活函数 | UB | 2 |

### 关键内存层级与数据通路

```
GM (Global Memory)
  |  PIPE_MTE1        |  PIPE_MTE2
  v                   v
L1 (Cube缓存)       UB (Unified Buffer, Vector缓存)
  |                   ^
  v                   |
L0C (Cube输出)  --fixpipe(PIPE_FIX)--> UB    (仅 910_95)
              \--L0C->GM->UB-->            (A2/A3)
```

- **L0C**：Cube 的输出缓冲区，数据以 NZ（Narrow Z）格式存储
- **UB**：Unified Buffer，Vector 的输入/输出缓冲区
- **fixpipe**：连接 L0C 和 UB 的硬件流水线，支持格式转换和融合后处理（**仅 910_95/950 支持**）

### PIPE 枚举速查

| 枚举值 | 硬件单元 | 数据通路 |
|--------|----------|----------|
| `PIPE_M` | Cube 矩阵单元 | Cube 计算流水线 |
| `PIPE_V` | Vector 向量单元 | Vector 计算流水线 |
| `PIPE_FIX` | Fixpipe 流水线 | L0C -> GM/L1/UB |
| `PIPE_MTE1` | 搬运引擎 | GM -> L1 |
| `PIPE_MTE2` | 搬运引擎 | GM -> UB |
| `PIPE_MTE3` | 搬运引擎 | UB -> GM |

> 源码参考：[pipe-and-core.md](../docs_triton_ascend/03-Ascend-Extensions/02-pipe-and-core.md)

---

## CV 流水线优化核心思想

### 问题：串行执行导致算力浪费

在未优化的代码中，Cube 和 Vector 计算串行执行。当 Vector 执行逐元素运算时，Cube 空闲等待；当 Cube 执行矩阵乘法时，Vector 空闲等待。这导致流水线气泡，硬件利用率低。

### 核心思想：让 Cube 提前执行，Vector 与 Cube 并行

通过**指令重排序**，将没有数据依赖的 Vector 计算延后，让 Cube 计算提前启动。当 Cube 执行矩阵乘法时，Vector 可以同时执行独立的向量运算，实现 Cube/Vector 流水并行。

```
优化前（串行）：
  Vector: [p_sum] [alpha] [acc*=alpha] ----等待----> [acc+=pv]
  Cube:   ----等待 Vector 完成----> [tl.dot(p,v)]

优化后（并行）：
  Vector: ----等待----> [p_sum] [alpha] [acc=acc*alpha+pv]
  Cube:   [tl.dot(p,v)]                (与 p_sum/alpha 并行)
```

---

## 指令重排序优化（代码模式）

### 经典模式：FlashAttention 中的 tl.dot(p,v) 提前

#### 优化前

```python
k = load_if(k_block_ptr, True, False)
s = tl.dot(q, k)
s = tl.where(mask & boundary_mask, s, -2**30)
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

**问题**：`alpha`、`p_sum`、`acc *= alpha` 等 Vector 计算在 `tl.dot(p, v)` 之前执行，Cube 需要等待 Vector 完成后才能启动。

#### 优化后

```python
k = load_if(k_block_ptr, True, False)
s = tl.dot(q, k)
s = tl.where(mask & boundary_mask, s, -2**30)
m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
v = load_if(v_block_ptr, False, True)
pv = tl.dot(p.to(dtype), v)
p_sum = tl.sum(p, 1)
alpha = tl.math.exp2((m - m_new) * qk_scale)
acc = acc * alpha[:, None] + pv
l = l * alpha + p_sum
m = m_new
```

**优化要点**：

1. **提前 Cube 计算**：将 `tl.dot(p.to(dtype), v)` 提前到 `p_sum` 和 `alpha` 计算之前
2. **并行执行**：`p_sum` 和 `alpha` 的 Vector 计算与 `tl.dot` 的 Cube 计算并行执行
3. **延迟依赖计算**：`acc = acc * alpha[:, None] + pv` 需要等待 Cube 计算结果，放在最后执行
4. **propagate_nan**：`tl.max` 和 `tl.maximum` 中添加 `propagate_nan=True` 保持数值稳定性

> 实际应用参考：`flash_attention_npu_v8.py:337-346`

---

## propagate_nan 参数用法

在 CV 流水线优化中，指令重排序可能改变浮点运算的执行顺序，引入 NaN 传播风险。`propagate_nan` 参数确保在 `tl.max` / `tl.maximum` 等聚合操作中正确传播 NaN，保持数值稳定性。

### 用法

```python
m_new = tl.maximum(
    m,
    tl.max(s, 1, propagate_nan=True),
    propagate_nan=tl.PropagateNan.ALL
)
```

| 参数 | 位置 | 值 | 说明 |
|------|------|-----|------|
| `propagate_nan` | `tl.max` 参数 | `True` | 当输入包含 NaN 时，结果传播 NaN 而非忽略 |
| `propagate_nan` | `tl.maximum` 参数 | `tl.PropagateNan.ALL` | 两个输入中任一为 NaN 时传播 NaN |

### 为什么需要 propagate_nan

在 FlashAttention 的 Online Softmax 计算中，`m_new = tl.maximum(m, tl.max(s, 1))` 用于追踪当前最大值。如果 `s` 中某些行全被 mask 为负无穷（如 causal mask 的上三角），`tl.max` 可能产生 NaN。不加 `propagate_nan` 时，NaN 可能被 `tl.maximum` 静默忽略，导致后续 `exp` 计算结果错误。

> 实际应用参考：`flash_attention_npu_v8.py:337`

---

## 编译参数优化

CV 融合算子和纯 Vector 算子需要不同的编译参数组合，详见 [07-compile-params.md](07-compile-params.md)。

### 纯 Vector 算子

```python
xx_vector_kernel[grid](
    ...,
    enable_flatten=True,
    multibuffer=True,
)
```

### Cube/Vector 融合算子

```python
xx_cube_vector_kernel[grid](
    ...,
    enable_auto_bind_sub_block=True,
    enable_flatten=False,
    set_workspace_multibuffer=2,
    sync_solver=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    multibuffer=True,
    enable_mixed_cv=True,
)
```

> 如果精度出现问题，按 [07-compile-params.md](07-compile-params.md) 中的回退策略逐步回退参数。

### 分核策略

CV 融合算子的分核数应等于 **Cube 核数量**（`aicore_num`），而非 Vector 核数量：

```python
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
aicore_num = properties["num_aicore"]

grid = (aicore_num,)
```

纯 Vector 算子使用 `vectorcore_num` 分核。

> 实际应用参考：`flash_attention_npu_v8.py:1025-1028`

---

## 910_95 特别注意

### fixpipe 数据通路

910_95/950 支持 **fixpipe 直接从 L0C 到 UB** 的数据通路，这是 CV 融合高性能的关键：

```
910_95:  Cube -> L0C --fixpipe(PIPE_FIX)--> UB -> Vector   (零 GM 访问)
A2/A3:   Cube -> L0C --PIPE_FIX--> GM --PIPE_MTE2--> UB -> Vector  (2 次 GM 访问)
```

fixpipe API 用法和对齐要求详见 [11-fixpipe-and-bias-fusion.md](11-fixpipe-and-bias-fusion.md)。

### 910_95 限制清单

| 限制项 | 说明 |
|--------|------|
| fixpipe 仅 910_95 支持 | A2/A3 需通过 GM 中转，性能损失较大 |
| 512B 对齐要求 | CV 融合算子要求 Tensor 尾轴大小能被 512Bytes 整除 |
| sync_block event_id 范围 0-15 | 共 16 个独立事件，需避免冲突 |
| multibuffer 默认 False | 需显式设置 `multibuffer=True` |

> 更多 910_95 硬件规格详见 [00-hardware-quick-ref.md](00-hardware-quick-ref.md)。

---

## 精度回退策略

当应用 CV 流水线优化后出现精度问题时，按以下顺序逐步回退参数：

### 回退步骤

```
步骤 1: 移除 enable_mixed_cv=True
         ↓ 仍有精度问题
步骤 2: sync_solver=True → sync_solver=False
         ↓ 仍有精度问题
步骤 3: limit_auto_multi_buffer_of_local_buffer="no-limit" → 移除该参数
         ↓ 仍有精度问题
步骤 4: set_workspace_multibuffer=2 → 移除该参数
         ↓ 仍有精度问题
步骤 5: enable_auto_bind_sub_block=True → False
         ↓ 仍有精度问题
步骤 6: multibuffer=True → False
         ↓ 仍有精度问题
步骤 7: 移除 propagate_nan 参数
         ↓ 仍有精度问题
步骤 8: 撤销指令重排序，恢复原始代码顺序
```

### 回退原则

- **每次只回退一个参数**，定位具体导致精度问题的参数
- **优先回退影响较小的参数**（如 `enable_mixed_cv`），保留性能收益较大的参数
- **纯 Vector 算子**：回退 `enable_flatten` 和 `multibuffer` 即可
- **CV 融合算子**：从 `enable_mixed_cv` 开始回退，逐步扩大回退范围
- 每次回退后**必须进行精度验证**

---

## 性能收益预估

| 优化手段 | 预估收益 | 说明 |
|----------|----------|------|
| 指令重排序（CV 流水并行） | 10%-30% | FlashAttention 等算子的端到端加速 |
| 编译参数优化（multibuffer 等） | 5%-15% | 减少流水线气泡，提高指令发射效率 |
| fixpipe 数据通路（仅 910_95） | 10%-20% | 消除 GM 写回和重新加载开销 |
| 综合优化 | 20%-40% | 多种优化手段叠加 |

### 收益条件

- Cube/Vector 计算比例接近 1:1 时收益最大
- 矩阵乘法后处理越复杂，CV 融合收益越高
- 纯 Vector 或纯 Cube 算子无法从此优化中获益

---

## compile_hint 辅助优化

在反向传播等复杂算子中，可使用 `compile_hint("break_vf")` 打断 Vector Function 融合，防止过长的 VF 融合链导致 UB 溢出：

```python
import triton.language.extra.cann.extension as extension

ds = p * (dp - d[:, None])
ds = tl.where(mask, ds, 0.0)
ds = ds.to(dtype)
extension.compile_hint(ds, "break_vf")
dq += tl.dot(ds, k)
```

`"break_vf"` 提示编译器在此处打断 Vector Function（VF）融合，防止过长的 VF 融合链导致 UB 溢出，确保后续 `tl.dot` 能正确调度到 Cube 核心执行。详见 [14-compile-hint-and-extension.md](14-compile-hint-and-extension.md)。

> 实际应用参考：`flash_attention_npu_v8.py:804`

---

## 风险提示

| 风险 | 说明 | 缓解措施 |
|------|------|----------|
| 精度问题 | 编译参数调整或指令重排序可能导致精度变化 | 每次调整后进行精度验证，按回退策略逐步回退 |
| 数据依赖错误 | 错误的重排序可能破坏数据依赖关系 | 仔细分析依赖关系，验证计算结果 |
| 性能回退 | 某些场景下优化可能无效或负优化 | 进行性能对比测试，必要时回退 |
| 910_95 专属功能 | fixpipe 仅 910_95 支持，A2/A3 需 GM 中转 | 根据目标平台选择合适的优化策略 |

---

## 相关文档链接

| 文档 | 路径 | 说明 |
|------|------|------|
| CV 融合算子详解 | [05-cv-fusion.md](../docs_triton_ascend/05-Performance-Optimization/05-cv-fusion.md) | fixpipe/sync_block/sub_vec_id |
| PIPE/CORE 枚举 | [02-pipe-and-core.md](../docs_triton_ascend/03-Ascend-Extensions/02-pipe-and-core.md) | 硬件流水线与核心类型 |
| fixpipe 操作 | [03-fixpipe.md](../docs_triton_ascend/03-Ascend-Extensions/03-fixpipe.md) | L0C->UB 数据通路详解 |
| FlashAttention 实际应用 | `flash_attention_npu_v8.py` | CV 优化的完整代码示例 |
