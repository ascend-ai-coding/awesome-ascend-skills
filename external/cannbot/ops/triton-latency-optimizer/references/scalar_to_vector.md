# Scalar 转 Vector 优化模式

## 概述

在昇腾 NPU Triton kernel 开发中，scalar 操作会占用 Scalar 计算单元，无法充分利用 NPU 的 Vector 计算单元（支持 256 字节 SIMD 并行）。通过将 scalar 操作转换为 vector 操作，可以显著提升算子性能，尤其在数据并行计算场景中效果更为明显。

## 优化场景

当在代码中遇到以下模式时，可应用此优化：

1. **标量广播操作**：使用 Python 标量或标量指针与向量数据进行计算
2. **标量规约操作**：使用标量变量进行累加、累乘等规约计算
3. **标量控制流**：使用 if-else 分支处理向量数据
4. **标量索引计算**：使用标量循环生成内存访问索引
5. **标量数学函数**：使用 Python math 模块进行逐元素数学计算
6. **整数比较**：避免int类型的比较
7. **整数除法和取余**：避免int类型的除法和取余
8. **Scatter-Add 并行轴选择**：scatter-add 类算子（embedding backward、scatter_add 等）选错并行轴，导致 atomic 调用次数爆炸

## 优化方法

### 1. 标量广播 → 向量广播

**原始代码（scalar 计算）**

```python
scalar_val = 0.5  # Python 标量
x = tl.load(x_ptr + offsets, mask=mask)
result = x * scalar_val  # scalar 广播，无法启用 vector 加速
```

**优化后代码（vector 计算）**

```python
scalar_val = 0.5
x = tl.load(x_ptr + offsets, mask=mask)
scalar_vec = tl.full([BLOCK_SIZE], scalar_val, dtype=x.dtype)  # 转为 vector
result = x * scalar_vec  # 启用 vector 广播计算
```

### 2. 标量规约 → 向量分块规约

**原始代码（scalar 计算）**

```python
sum_val = 0.0  # 标量累加器
for n in range(N):
    val = tl.load(x_ptr + row_offset + n)
    sum_val += val  # 标量加法，循环依赖
result = sum_val
```

**优化后代码（vector 计算）**

```python
acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # vector 累加器
for n_start in range(0, N, BLOCK_SIZE):
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
    n_mask = n_offsets < N
    x_vec = tl.load(x_ptr + row_offset + n_offsets, mask=n_mask)
    acc += x_vec  # vector 加法，无循环依赖
result = tl.sum(acc, axis=0)  # vector 规约
```

### 2.1 2D 向量化规约

**原始代码（每个 program 只处理 1 个 expert，1D 标量比较）**

```python
@triton.jit
def count_naive(topk_ids_ptr, expert_num_tokens_ptr, num_experts, topk_numel,
                expert_map, HAS_EXPERT_MAP: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    curr_expert = tl.program_id(0)  # 标量索引
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)  # 1D 累加器
    for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
        expert_ids = tl.load(topk_ids_ptr + x * BLOCK_SIZE + offsets, ...)
        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)  # 标量比较
        acc = acc + has_curr_expert
    tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))
```

**优化后代码（每个 program 同时处理 EXPERT_BLOCK 个 expert，2D 向量化）**

```python
@triton.jit
def count_2d(topk_ids_ptr, expert_num_tokens_ptr, num_experts: tl.constexpr,
             topk_numel: tl.constexpr, expert_map, HAS_EXPERT_MAP: tl.constexpr,
             BLOCK_SIZE: tl.constexpr, EXPERT_BLOCK: tl.constexpr):
    # 向量索引：同时表示 EXPERT_BLOCK 个 expert
    curr_expert = tl.program_id(0) * EXPERT_BLOCK + tl.arange(0, EXPERT_BLOCK)
    offsets = tl.arange(0, BLOCK_SIZE)
    # 2D 累加器：(EXPERT_BLOCK, BLOCK_SIZE)
    acc = tl.zeros((EXPERT_BLOCK, BLOCK_SIZE), dtype=tl.float32)
    cntx = (topk_numel - 1) // BLOCK_SIZE + 1
    for x in range(cntx):
        mask = offsets < (topk_numel - x * BLOCK_SIZE)
        expert_ids = tl.load(topk_ids_ptr + x * BLOCK_SIZE + offsets, mask=mask, other=-1)
        # 广播：(EXPERT_BLOCK, 1) == (1, BLOCK_SIZE) -> (EXPERT_BLOCK, BLOCK_SIZE)
        has_curr_expert = (expert_ids[None, :] == curr_expert[:, None]).to(tl.float32)
        has_curr_expert = tl.where(mask, has_curr_expert, 0.0)
        acc = acc + has_curr_expert
    # 沿 BLOCK_SIZE 轴规约，每行一个结果
    tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc, axis=1))
```

**关键变换**：
- `curr_expert` 从标量 → 向量（`tl.arange(0, EXPERT_BLOCK)`）
- `acc` 从 `(BLOCK_SIZE,)` 1D → `(EXPERT_BLOCK, BLOCK_SIZE)` 2D
- 比较操作通过广播实现 2D 向量化：`expert_ids[None, :] == curr_expert[:, None]`
- Grid = `(num_experts // EXPERT_BLOCK,)`，更接近物理核数


### 2.2 Tiled Reduction 中的标量累加器（反模式）

Tiled reduction kernel 的常见写法：每 tile 加载向量数据，用 `tl.sum` 归约，
结果累加到标量。**`tl.sum` 掩盖了标量依赖——检查优化点 5 时必须识别此模式。**

**识别特征**（3 条件同时满足即命中）：
1. `tl.full((), ...)` 或 `0.0` 初始化的标量变量
2. 在 `for tile in range(0, N, BLOCK)` 循环内
3. 该标量参与 `+= tl.sum(vector, axis=0)` 累加

**反模式示例（Tiled Reduction）**：

```python
acc = tl.full((), 0.0, tl.float32)          # 标量累加器
for offset in range(0, N, BLOCK):
    val = tl.load(ptr + offset + tl.arange(0, BLOCK), mask=mask)
    acc = acc + tl.sum(val, axis=0)          # tl.sum 归约 + 标量依赖
```

**优化（向量累加器 + 延后归约）**：

```python
acc = tl.zeros([BLOCK], dtype=tl.float32)   # 向量累加器
for offset in range(0, N, BLOCK):
    val = tl.load(ptr + offset + tl.arange(0, BLOCK), mask=mask)
    acc += val                               # 向量加法，无循环依赖
result = tl.sum(acc, axis=0) / N             # 循环结束后一次性归约
```

**适用范围**：所有 Tiled Reduction 算子——
Normalization stats kernel、Sum/Mean over axis、Softmax numerator、Variance 计算等。


### 3. 标量控制流 → 向量掩码

**场景一：两个分支的计算逻辑的定义域一样，均不会出现nan或inf等无效输出**
**原始代码（scalar 控制流）**

```python
# 莫格
x = tl.load(x_ptr + offsets, mask=mask)
if x > 0:  # 标量条件，导致 warp divergence
    result = tl.exp(x)
else:
    result = tl.cos(x)
```

**优化后代码（vector 掩码）**

```python
x = tl.load(x_ptr + offsets, mask=mask)
cond_mask = x > 0  # vector 比较，返回布尔向量
exp_result = tl.exp(x)
log_result = tl.cos(x)
result = cond_mask * exp_result + ~cond_mask * log_result  # vector 加法，无分支
```

**场景二：两个分支的计算逻辑定义域不一样，至少有一个会出现nan或inf等无效输出**
**原始代码（scalar 控制流）**

```python
# 莫格
x = tl.load(x_ptr + offsets, mask=mask)
if x > 0:  # 标量条件，导致 warp divergence
    result = tl.exp(x)
else:
    result = tl.log(x)
```

**优化后代码（vector 掩码）**

```python
x = tl.load(x_ptr + offsets, mask=mask)
cond_mask = x > 0  # vector 比较，返回布尔向量
exp_result = tl.exp(x)
log_result = tl.log(x)
result = tl.where(cond_mask, exp_result, log_result)  # vector 选择，无分支
```

### 4. 标量索引 → 向量索引

**原始代码（scalar 索引）**


```python
for i in range(BLOCK_SIZE):
    idx = tl.load(indices_ptr + start + i)  # 标量加载
    val = tl.load(input_ptr + idx)  # 标量地址计算
    tl.store(output_ptr + start + i, val)
```

**优化后代码（vector 索引）**

```python
# 方案一：先将indices和input都搬到ub，再使用tl.gather收集并输出，适用于input较小的场景，加速比较大
indices_offsets = indices_start + tl.arange(0, BLOCK_SIZE)
indices_mask = indices_offsets < indices_len
indices = tl.load(indices_ptr + indices_offsets, mask=indices_mask)
input_mask = indices_offsets < input_len
input_data = tl.load(input_ptr + indices_offsets, mask=input_mask)
gathered_data = tl.gather(input_data, indices, 0)
tl.store(output_ptr + indices_offsets, gathered_data, mask=indices_mask)

# 方案二：将indices搬到ub，再使用al.gather_out_to_ub扩展接口收集GM上的input数据，再进行输出，对UB空间占用较小，但是当前这个接口无法编译通过
# import triton.language.extra.cann.extension as al
# pid = tl.program_id(0)
# indices_start = pid * BLOCK_SIZE
# indices_offsets = indices_start + tl.arange(0, BLOCK_SIZE)
# indices_mask = indices_offsets < indices_len
# indices = tl.load(indices_ptr + indices_offsets, mask=indices_mask)
# gathered_data = al.gather_out_to_ub(
#     src=input_ptr,
#     index=indices,
#     index_boundary=input_len,
#     dim=0,
#     src_stride=(1, ),
#     end_offset=(indices_len, ),
#     start_offset=(0, )
# )
# tl.store(output_ptr + indices_offsets, gathered_data, mask=indices_mask)

```

### 5. 标量数学函数 → 向量数学函数

**原始代码（scalar 数学）**

```python
import math  # 错误：使用 Python math 模块

x = tl.load(x_ptr + offsets, mask=mask)
result = math.exp(x)  # 这将导致标量循环
```

**优化后代码（vector 数学）**
```python
x = tl.load(x_ptr + offsets, mask=mask)
result = tl.exp(x)  # vector 指数函数
# 其他 vector 函数：tl.log, tl.sqrt, tl.sin, tl.pow 等
```

### 6. int类型比较 → 向量比较

**优化前（整数比较退化为标量）**

```python
is_invalid_tok = tok < 0          # i64/i32类型
valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)   # 标量比较
```

**优化后（转换为浮点数启用向量）**

```python
is_invalid_tok = tok.to(tl.float32) < 0               # 转换float32类型
valid_block = (block_id.to(tl.float32) < max_num_blocks_per_req) & (block_id.to(tl.float32) >= 0)   # vector比较
```

### 7. int类型除法/取余 → 向量除法/取余

**优化前（整数除法/取余退化为标量）**

```python
c = a // b          # i32标量除法
d = a % b           # i32标量取余
```

**优化后（转换为浮点数启用向量）**

```python
c = a.to(tl.float32) // b.to(tl.float32)            # 转换float32类型
d = a - (a // b) * b  # 公式转换
```

### 8. atomic_* 标量操作 → atomic_* 向量操作

**原始代码（scalar 操作）**

```python
for idx in range(0, BLOCK_SIZE):
    tl.atomic_add(output_ptr + idx, block_sum)      # 标量的原子加
```

**优化后代码（vector 操作）**

```python
h_offs = tl.arange(0, BLOCK_SIZE)
block_vals = tl.full((BLOCK_SIZE,), block_sum, dtype=output_dtype)
tl.atomic_add(output_ptr + h_offs, block_vals)      # 向量化的原子加
# 其他可向量化的 atomic_* 操作：tl.atomic_max, tl.atomic_min 等
```

### 9. Scatter-Add 并行轴选择

**识别条件**：算子语义为 `output[indices[i]] += values[i]` 模式，即按稀疏索引做累加。典型算子包括 `embedding_dense_backward`、`scatter_add`、`index_add` 等。

**⚠️ 关键判断：并行轴应该选哪个维度？**

| 并行维度 | 问题 |
|----------|------|
| 沿 **indices** 并行 | 多个 core 可能写同一 output 行，必须用 atomic 竞争；每个 atomic 只能处理小向量宽度 |
| 沿 **E（数据宽度）** 并行 | 每个 core 处理不同 E 区间，**天然无冲突**；每次 atomic 可用大 BLOCK_E 向量化 |

**反模式（错误：沿 indices 并行）**

```python
# ❌ grid = indices 维度，每个 core 处理一批 indices
# 问题 1：不同 core 的 indices 可能映射到同一 output 行 → 必须 atomic
# 问题 2：每个 atomic 只能处理 1 个 E 元素（tl.atomic_add 不支持 2D）
# 问题 3：E 维度循环（for e in range(E)）导致 E 次 atomic_add 调用
@triton.jit
def scatter_add_bad(grad_ptr, indices_ptr, out_ptr, N, E, ...):
    pid = tl.program_id(0)
    # 处理 pid 对应的 indices 块
    for e in range(E):  # ❌ E 次 atomic，每次只处理 1 个元素
        val = tl.load(grad_ptr + idx_offs * E + e, ...)
        tl.atomic_add(out_ptr + t_idx * E + e, val, ...)
```

**正确模式（沿 E 维度并行）**

```python
# ✅ grid = E 维度切片，每个 core 处理一段 E 区间
# 核心洞察：不同 core 处理不同 E 区间 → 无冲突，无需 atomic 竞争
# 每个 core 扫描全部 indices，但只处理自己的 E 区间
# 每次 atomic_add 用大 BLOCK_E（128/256），大幅减少调用次数
@triton.jit
def scatter_add_good(grad_ptr, indices_ptr, out_ptr, N, E,
                     num_cores: tl.constexpr, BLOCK_E: tl.constexpr):
    pid = tl.program_id(0)
    # 每个 core 处理 E 维度的一段
    e_per_core = tl.cdiv(E, num_cores)
    e_start = pid * e_per_core
    e_end = tl.minimum(e_start + e_per_core, E)

    for e_block_start in range(e_start, e_end, BLOCK_E):
        e_offs = tl.arange(0, BLOCK_E)
        e_mask = e_offs < tl.minimum(BLOCK_E, e_end - e_block_start)

        # 扫描全部 indices，但只加载当前 E 区间的数据
        for idx_pos in range(N):
            t_idx = tl.load(indices_ptr + idx_pos)
            val = tl.load(grad_ptr + idx_pos * E + e_block_start + e_offs,
                         mask=e_mask, other=0.0)
            tl.atomic_add(out_ptr + t_idx * E + e_block_start + e_offs,
                         val, mask=e_mask)  # 256 个元素的向量化 atomic！
```

**决策流程**

```
拿到 scatter-add 类算子：
1. 识别两个维度：indices 维度（稀疏）和 E 维度（稠密连续）
2. 自问：哪个维度无冲突？
   - indices：有冲突（多 core 可能写同行）→ 需要 atomic
   - E：无冲突（不同 core 写不同列）→ 最优并行轴
3. 沿 E 分配 core，每个 core 扫描全部 indices
4. BLOCK_E 尽可能大（128/256），减少 atomic 调用次数
```

**实测效果**

| 算子 | 错误方案（沿 indices 并行） | 正确方案（沿 E 并行） |
|------|---------------------------|---------------------|
| embedding_dense_backward | 0.03x | **1.88x** |
| 提升倍数 | — | **62x** |

**关键原则**：对于 scatter-add 类算子，**并行轴应该选数据稠密维度，而不是稀疏索引维度**。数据宽度维度天然无冲突，可以用大 BLOCK 向量化 atomic。


### 10. 标量扫描循环 → 矩阵乘（Cube 单元）

**场景**：沿对角/三角区域逐元素累加的标量扫描循环（`for j in range(BC)` 内部用 `tl.where(m_i, ..., 0.)` 掩码 + 逐元素乘法累加），且累加结构等价于一个带掩码的矩阵乘 `[BC,BC] @ [BC,BK]`。这是标量循环的进阶向量化——直接升级到 Cube 矩阵单元。

**原始代码（scalar 扫描循环）**

```python
# 逐元素扫描对角块，BC 次循环
for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
    b_dAqk = tl.load(dAqk_local + o_dA + j, mask=m_dA, other=0)   # [BC] 标量列
    b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)        # [BK] 标量行
    m_i = o_i[:, None] >= j                                         # 三角掩码
    b_gqk = exp2(b_g - b_gkj[None, :])
    b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)  # 逐元素三乘累加
    p_kj += H * K
```

**优化后代码（整块矩阵乘 + 掩码）**

```python
# 一次性加载整块，用 tl.dot 矩阵乘
p_gn = g_local + (i_ti + min(BC // 2, T_local - i_ti - 1)) * H * K + o_k
b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]   # [1, BK] 参考点

p_dAqk = tl.make_block_ptr(dAqk_local, (T_local, BT), (H*BT, 1),
                            (i_ti, i_i * BC), (BC, BC), (1, 0))
b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)    # [BC, BC] 整块

b_g_diag = b_g - b_gn
b_k_exp = b_k * exp2(-b_g_diag)                                    # [BC, BK]
# 矩阵乘：[BC, BC] @ [BC, BK] → [BC, BK]，Cube 单元并行
b_dq2 += tl.dot(b_dAqk, b_k_exp) * exp2(b_g_diag)
```

**关键变换**：
- `for j in range(BC)` 逐列循环 → 一次 `tl.load` 整个 `[BC, BC]` 块
- `b_dA[:, None] * b_kj[None, :]` 逐元素三乘 → `tl.dot(b_dAqk, b_k_exp)` 矩阵乘（Cube 单元）
- `tl.where(m_i, ..., 0.)` 每步掩码 → 掩码编码进输入块的三角结构（加载前已清零）

**数值稳定参考点技巧**：`exp2(b_g)` 当 b_g 较大时溢出。减去块内中点 `b_gn`，`exp2(b_g - b_gn)` 安全。等价性来自 `exp2(b_g - b_gn) = exp2(b_g) * exp2(-b_gn)`，可拆分到 dot 两侧。

**适用判定**：
- ✅ `for j in range(BC)` 编译期常量扫描循环
- ✅ 循环内是 `scalar_col[:, None] * row[None, :]` 逐元素乘 + 三角掩码累加
- ✅ 累加结构等价于矩阵乘 `A[i,j] * B[j,k]`
- ✅ 掩码为三角/对角形状，可编码进输入块
- ❌ 循环体含递推依赖（如前向代入 `b_Ai = tl.where(o_i==i, b_a, b_Ai)` 用本步结果更新累加器）→ **不可矩阵乘化**，跳过

**性能收益**：BC 次串行标量循环 → 1 次矩阵乘，典型加速 3-8x（BC=16 时）。

## 关键点

1. **类型一致性**：确保 vector 操作中的数据类型一致，避免隐式类型转换回退到 scalar
2. **广播语义**：使用 `tl.full` 或 `tl.broadcast_to` 显式创建 vector 常量，避免 Python 标量自动广播
3. **规约融合**：使用 `tl.sum`, `tl.max`, `tl.min` 等内置 vector 规约函数，而非手动循环累加
4. **分支消除**：使用 `mask` 和加法结合 替代 if-else，避免标量控制流导致的 SIMD divergence
5. **内存对齐**：确保 BLOCK_SIZE 满足 128 字节对齐（如 FP16 类型下 BLOCK_SIZE 为 64 的倍数），以启用 vector 加载/存储
6. **索引比较类型转换**：对于 tl.where 中的整数索引比较，先 .to(tl.float32) 再比较，以启用向量比较指令。
7. **标量比较类型转换**：对于 `int32` 和 `int64` 整数标量比较，先 .to(tl.float32) 再比较，以启用向量比较指令。
8. **标量除法类型转换**：对于 `int32` 和 `int64` 中的整数除法操作，先 .to(tl.float32) 再计算，以启用向量计算指令。
9. **标量取余类型转换**：对于 `int32` 和 `int64` 中的整数取余操作， 使用`a - (a // b) * b`的形式计算，以启用向量计算指令。
10. **原子操作向量化**：对 `atomic_add` 这一类的 `atomic_*` 标量操作进行向量化，可消除循环的标量操作开销
11. **Scatter-Add 并行轴**：scatter-add 类算子优先沿数据宽度(E)维度并行，而非稀疏 indices 维度，避免 atomic 冲突并用大 BLOCK_E 向量化

## 性能收益

将 scalar 操作转换为 vector 操作，可在昇腾 NPU 上获得显著性能提升：

- **标量广播优化**：10-20% 加速，通过消除标量指令开销
- **标量规约优化**：5-128 倍加速（取决于 vector 并行度，FP16 理论加速比 128 倍）
- **控制流优化**：消除 warp divergence，提升 SIMD 利用率至 90%+
- **整体 kernel 优化**：在 LayerNorm、Softmax 等带宽受限算子中，端到端性能提升 2-3 倍

**实测数据参考**：在 LayerNorm 算子中，将 mean/variance 计算从标量累加改为 vector 分块规约，UB 利用率从 35% 提升至 78%，kernel 执行时间减少 62%。

---

## 来自 SKILL.md 的原始描述（优化点 5：Scalar 转 Vector 优化）

**适用条件**：代码中存在标量操作，可转换为向量操作以充分利用 NPU Vector 计算单元

**典型代码特征**：
```python
# 特征 1：标量广播操作
scalar_val = 0.5  # Python 标量
result = x * scalar_val  # scalar 广播，无法启用 vector 加速

# 特征 2：标量规约操作
sum_val = 0.0  # 标量累加器
for n in range(N):
    val = tl.load(x_ptr + n)
    sum_val += val  # 标量加法

# 特征 3：标量控制流
if x > 0:  # 标量条件，导致 SIMD 分支分化
    result = tl.exp(x)
else:
    result = tl.cos(x)

# 特征 4：int 类型比较/除法/取余
is_invalid = tok < 0  # int 类型比较，退化为标量
c = a // b  # int 类型除法，退化为标量
d = a % b   # int 类型取余，退化为标量

# 特征 5：atomic_* 标量操作
for idx in range(0, BLOCK_SIZE):
    tl.atomic_add(output_ptr + idx, block_sum)  # 标量的原子加
```

**判断逻辑**：
- 检查是否存在 Python 标量与向量数据的计算（标量广播）
- 检查是否存在标量累加器（如 `sum_val = 0.0`）
- 检查是否存在 `if-else` 控制流处理向量数据
- 检查是否存在 `int32/int64` 类型的比较、除法、取余操作
- 检查是否存在 `atomic_add` 这一类的 `atomic_*` 标量操作
- 如果存在以上任一情况 → 涉及
- 如果所有操作都已使用向量形式 → 不涉及，跳过

**命中条件**：代码中存在标量操作，可转换为向量操作

**参考文档**：`references/scalar_to_vector.md`

---
