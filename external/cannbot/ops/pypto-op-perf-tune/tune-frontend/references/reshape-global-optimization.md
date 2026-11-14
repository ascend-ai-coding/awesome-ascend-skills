#### Reshape 全局优化

**核心目标**：尽量将 reshape 提前到 loop 外面，使用 `reshape inplace`，减少循环体中的 reshape，从而减少数据搬运。

##### 1. 逐 Reshape 系统分析

对算子中**每一个** `pypto.reshape` 调用，逐个分析其是否必须出现在最内层循环体中。

**分析方法**：
1. 逐行阅读算子代码，记录每一个 `pypto.reshape` 调用的位置（loop 外 / loop 内 / 最内层循环体）
2. 分析每个 reshape 的输入 tensor 来源（原始输入 / 中间计算结果）和目标 shape
3. 判断该 reshape 是否**一定需要**在当前位置执行，还是可以提前到更外层

**分析表格模板**：

| # | 代码位置 | 输入 Tensor | 源 Shape | 目标 Shape | 是否在 loop 内 | 是否可外提 | 外提方式 |
|---|---------|------------|---------|-----------|--------------|-----------|---------|
| 1 | kernel 入口处 | query（原始输入） | `[B,N,S,D]` | `[B*N*S,D]` | 否 | — | 已在外层 |
| 2 | loop 内，第 L32 行 | query（原始输入） | `[B,N,S,D]` | `[B*N*S,D]` | 是 | ✅ 可外提 | 方式 1：挪到 loop 前，inplace |
| 3 | 最内层 loop 内 | matmul 输出 | `[M,K]` | `[M,N,H]` | 是 | ❌ 不可外提 | 依赖循环变量，保留 |

##### 2. Reshape 优化方式

逐一确认每个可外提的 reshape 的优化方式：

**方式 1：原始输入 reshape 外提**

对原始输入（函数参数）的 reshape，挪到算子入口（所有 loop 之前），并使用 `inplace=True`，直接完成 shape 变换，避免冗余数据拷贝。

```python
# ✅ 正确：reshape 挪到算子入口，inplace=True
q_grouped = pypto.reshape(query, [num_kv_heads, num_heads_per_group, head_dim], inplace=True)
k_cache = pypto.reshape(key_cache, [kv_len, num_kv_heads, head_dim], inplace=True)

for i in pypto.loop(num_blocks, ...):
    # loop 内直接使用已 reshape 的 tensor，无额外搬运
    scores = pypto.matmul(q_grouped, k_cache_block, ...)

# ❌ 错误：reshape 放在 loop 内部，每次循环重复执行数据拷贝
for i in pypto.loop(num_blocks, ...):
    q_grouped = pypto.reshape(query, [num_kv_heads, num_heads_per_group, head_dim])  # 冗余搬运
```

**方式 2：高维计算提前合轴**

如果循环体内的计算超过两维（如 3D/4D），NPU 指令对多维 tensor 处理不友好，性能较差。应在进入循环前对原始输入 `reshape inplace` 合轴为 2D，避免循环体内出现 reshape。

```python
# ✅ 正确：循环前合轴为 2D，循环内无 reshape
query_2d = pypto.reshape(query, [batch * heads * seq_q, dim], inplace=True)
key_2d = pypto.reshape(key, [batch * heads * seq_kv, dim], inplace=True)
value_2d = pypto.reshape(value, [batch * heads * seq_kv, dim], inplace=True)

for b_idx in pypto.loop(batch, ...):
    for n_idx in range(heads):
        q_offset = b_idx * heads * seq_q + n_idx * seq_q + q_start
        q_block = pypto.view(query_2d, [BLOCK, dim], [q_offset, 0], ...)
        # ... 计算，循环体中无 reshape
```

**方式 3：冗余 reshape 删除（source==target）**

检查每个 reshape 的源 shape 是否等于目标 shape（常见于代码迭代过程中残留的无效 reshape），直接删除无效调用：

```python
# ❌ 冗余：源 shape 等于目标 shape
k_embed = pypto.reshape(k_embed, [8, 128])   # [8,128] → [8,128]

# ✅ 删除后直接使用
# k_embed 已经是 [8, 128]，无需 reshape
```

**检查方法**：逐行扫描所有 `pypto.reshape` 调用，比对源 shape 与目标 shape 是否相同。此检查应在阶段 A3（Reshape 全局分析表）中完成，在表格的"冗余?"列中标记。

**🔥 案例**：[Decode Attention Vector 合轴优化](../cases/vector-axis-merge-softmax.md)（-6.0%，任务数 -18.5%，含 4 轮迭代失败分析）

**方式 4：squeeze/unsqueeze 替换为 reshape inplace 外提**

⚠️ `pypto.squeeze` 和 `pypto.unsqueeze` **不支持** `inplace=True` 参数，无法直接原地修改 tensor shape。

对原始输入（函数参数）的 squeeze/unsqueeze 操作，如果出现在循环体内部，应替换为等价的 `pypto.reshape(..., inplace=True)` 并挪到算子入口（所有 loop 之前），消除循环内的重复数据搬运。

```python
# ✅ 正确：squeeze 替换为 reshape inplace 外提
# 原代码：pypto.squeeze(query, dim=1) → [B,1,S,D] → [B,S,D]
query_3d = pypto.reshape(query, [batch, seq_len, head_dim], inplace=True)  # 算子入口，只执行一次

for i in pypto.loop(num_blocks, ...):
    scores = pypto.matmul(query_3d, key_block, ...)  # loop 内直接使用

# ❌ 错误：squeeze 放在 loop 内部，不支持 inplace，每次循环重复搬运
for i in pypto.loop(num_blocks, ...):
    q_sq = pypto.squeeze(query, dim=1)  # 冗余搬运
```

**等价映射规则**：

| 原始操作 | 等价 reshape |
|---------|-------------|
| `pypto.squeeze(x, dim=d)` | `pypto.reshape(x, [不含 size-1 维的 shape], inplace=True)` |
| `pypto.unsqueeze(x, dim=d)` | `pypto.reshape(x, [插入 size-1 维后的 shape], inplace=True)` |

**约束**：
- 仅对**原始输入**（函数参数）可替换为 `reshape(inplace=True)`，中间结果和输出 tensor 不能 inplace reshape
- 替换后需检查循环体内所有引用该 tensor 的位置是否使用了正确的 shape
- 替换后需在 reshape 后、使用前重新设置 `set_vec_tile_shapes` 以匹配新 shape
