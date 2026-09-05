# Sort/Select 算子优化

> 适用于需要迭代选择元素的算子：NMS、TopK、ArgSort 等

## 核心约束

Triton Ascend 不支持 `break`/`continue`/`return` 和 Python `if` 分支，必须用 `tl.where` + mask 实现条件逻辑。

| 禁止语法 | 替代方案 | 说明 |
|---------|---------|------|
| `if cond:` | `tl.where(cond, a, b)` | 所有条件必须用 SIMD 友好的方式表达 |
| `break`/`continue` | 用循环变量 + mask 控制 | 循环次数固定，用 mask 跳过无效迭代 |
| `return` | 无法提前返回 | 所有路径必须执行到函数末尾 |
| 标量条件赋值 `x = y if cond` | `x = tl.where(cond, y, x)` | 标量变量更新必须用 `tl.where` |

### 1.2 迭代选择的标准模式

对于需要"每次从剩余元素中选一个最优"的算法（如NMS），标准模式是：

```python
# 模式：selection-sort 风格的迭代选择
for step in range(max_select):
    # 1. 线性扫描找最优候选
    best_idx = -1
    best_score = threshold
    for i in range(n_elements):
        score = tl.load(scores_ptr + i)
        higher = (score > best_score) & active
        best_idx = tl.where(higher, i, best_idx)
        best_score = tl.where(higher, score, best_score)

    # 2. 检查是否找到有效候选
    found = (best_idx != -1) & active

    # 3. 记录结果（仅当 found 时）
    tl.store(output_ptr + count, best_idx.to(tl.int32), mask=found)
    count = tl.where(found, count + 1, count)

    # 4. 标记已选（通过修改内存状态）
    tl.store(scores_ptr + best_idx, sentinel_value, mask=found)

    # 5. 根据选中元素更新其他元素状态（算子特定逻辑）
    # ... 例如 NMS 中计算 IoU 并抑制重叠 box
```

**关键原则**：
- 用**内存值**（如将 score 设为哨兵值）表示"已选/已抑制"状态，而非标量 flag
- 用 `tl.where` 做所有条件选择，不用 Python `if`
- 用 `mask=` 参数控制 `tl.load`/`tl.store` 的执行

---

## 2. 算子特定实现

### 2.1 NMS (Non-Maximum Suppression)

#### 算法语义

验证框架对比的是 PyTorch 参考实现（如 `30_NMS.py`），其语义通常包含：

1. **先过滤**：只保留满足门槛条件的元素（如 `score > scores_threshold`）
2. **再降序排序**：参考实现通常用 `torch.argsort(..., descending=True, stable=True)` 确定顺序
3. **迭代选择**：按排序后的顺序遍历，若当前元素未被抑制则选中
4. **依赖抑制**：选中后，根据算子特定规则抑制其他元素（如 NMS 的 IoU 阈值）
5. **数量限制**：最多输出 `max_output_size` 个，达到即停止
6. **输出格式**：输出张量前 `num_selected` 个有效，其余为 0 或哨兵值

**关键：降序关系来自参考实现的排序步骤**。Triton kernel 中没有显式排序，而是通过迭代选择最高分来隐式复现降序语义。

#### 参考实现

```python
@triton.jit
def select_kernel(
    values_ptr,           # 用于比较的值
    selected_indices_ptr, # 输出：选中的原始索引
    num_selected_ptr,     # 输出：实际选中数量
    n_elements,
    max_output_size: tl.constexpr,
    threshold: tl.constexpr,
):
    pid = tl.program_id(0)
    active = (pid == 0)
    selected_count = 0

    for step in range(max_output_size):
        # 1. 线性扫描找最优候选
        best_idx = -1
        best_val = threshold
        for i in range(n_elements):
            val = tl.load(values_ptr + i)
            better = (val > best_val) & active
            best_idx = tl.where(better, i, best_idx)
            best_val = tl.where(better, val, best_val)

        # 2. 检查是否找到有效候选
        found = (best_idx != -1) & active

        # 3. 记录结果
        tl.store(selected_indices_ptr + selected_count,
                 best_idx.to(tl.int32), mask=found)
        selected_count = tl.where(found, selected_count + 1, selected_count)

        # 4. 标记已选，防止重复
        tl.store(values_ptr + best_idx, sentinel_value, mask=found)

        # 5. 算子特定逻辑：根据选中元素更新其他元素状态
        #    - NMS：读取选中元素的数据，计算与其他元素的关系（如 IoU），
        #            将满足条件的其他元素标记为已选/已抑制
        #    - TopK：无需此步骤
        #    - 其他算子：根据业务规则更新其他元素的值或标记

    tl.store(num_selected_ptr, selected_count, mask=active)
```

**关键点**:
- `grid=(1,)` 单核执行，顺序依赖算法天然难以并行
- `best_idx = -1` 初始值，配合 `found = (best_idx != -1)` 判断是否找到有效元素
- `mask=found` 保护所有依赖 `best_idx` 的 load/store，避免 -1 越界
- 写入顺序自然为降序，与参考实现 `argsort(descending=True)` 语义一致

## 算子特定扩展

### NMS

在通用模式阶段5加入：读取选中 box 坐标，计算与其他 box 的 IoU，将 IoU >= threshold 的 box 的 score 设为哨兵值（抑制）。

**关键点**:
- `scores_f32 = scores.float().contiguous()` 保证连续内存访问
- 输出前 `num_selected` 个为原始索引（按 score 降序），其余为 0

### TopK

无抑制逻辑，阶段5为空。将哨兵值设为 `-float('inf')`。

---

## TopK 实现规范

### 推荐架构：tile-wise partial sort + merge

```python
TILE = min(4096, next_pow2(N))
COMBINED = next_pow2(2 * K)

for c in range(0, N, TILE):
    offsets = c + tl.arange(0, TILE)
    mask = offsets < N
    tile = tl.load(input_ptr + offsets, mask=mask, other=pad_val)
    sorted_tile = tl.sort(tile, descending=largest)
    chunk_topk = tl.gather(sorted_tile, tl.arange(0, K), axis=0)

    if c == 0:
        best = chunk_topk
    else:
        merged = tl.full((COMBINED,), pad_val, dtype=...)
        merged = tl.insert_slice(merged, best, 0)
        merged = tl.insert_slice(merged, chunk_topk, K)
        sorted_merged = tl.sort(merged, descending=largest)
        best = tl.gather(sorted_merged, tl.arange(0, K), axis=0)
```

### 禁止模式

- `for rank in range(k): for i in range(slice_size):` 的选择排序
- `tl.sum(tl.where(tl.arange(0, BLOCK_SIZE) == rank, sorted_vals, 0.0))` 标量提取
- 原地修改输入（如用哨兵值覆盖输入张量）
- 动态大缓冲 `num_chunks * k` + 选择排序兜底
- 固定 `BLOCK_ROWS=1/2/4` 等常数，未按 UB 预算动态计算
- 多 tile 合并用 in-register gather + `idx % K` + `tl.where` 拼接

## TopK 高性能代码骨架

以下模板为**伪代码骨架**，用于说明 8 条 TopK/Argsort 约束的实现形态，而非可直接运行的代码。约束 1~4 来自基本规范，约束 5~8 为性能关键，必须在 designer/coding 阶段直接满足。

| 编号 | 约束 | 在骨架中的位置 |
|------|------|----------------|
| 1 | 用 `tl.sort`/`ext.sort` 做 tile-wise 排序 | kernel 内 `sort(tile)` |
| 2 | top-k 提取用 `tl.gather` | kernel 内 `gather(sorted_tile, topk_idx)` |
| 3 | 合并缓冲区固定为 `next_pow2(2 * K)` | Host 侧 `COMBINED` + kernel 内 temp workspace 宽度 |
| 4 | 目标维 permute 到末维；tile 不足填充 `±inf` | kernel 入口前完成 permute；`load(..., other=pad_val)` |
| 5 | UB 预算动态 `ROWS_PER_BLOCK`，禁止固定常数 | Host 侧 `ROWS_PER_BLOCK` 计算 |
| 6 | 多 tile 合并必须使用 temp buffer | kernel 内 store/load workspace 后 `sort(merged)` |
| 7 | kernel 启动传入 `multibuffer=False, unit_flag=False` | Host 侧 `launch(...)` |
| 8 | 循环遍历实际 `n_cols`，禁止遍历 `next_pow2(n_cols)` | kernel 内 `for c in arange(0, n_cols, TILE)` |

```python
# 约束 5：Host 侧按 UB 预算动态计算 ROWS_PER_BLOCK
TILE        = min(4096, next_pow2(n_cols))
COMBINED    = next_pow2(2 * K)
ROWS_PER_BLOCK = max(1, min(ceil(n_rows / num_cores),
                            UB_BUDGET // (TILE * element_size * BUFFER_COEFF)))
grid        = (min(ceil(n_rows / ROWS_PER_BLOCK), num_cores),)

# 约束 3/6：分配 temp workspace：[n_rows, COMBINED]，用于连续 2K merge
temp = alloc_workspace([n_rows, COMBINED], dtype=x.dtype)

@triton.jit
def topk_kernel(x_ptr, out_ptr, temp_ptr, n_rows, n_cols,
                TILE, K, COMBINED, ROWS_PER_BLOCK, pad_val, largest):
    pid       = program_id(0)
    row_offs  = pid * ROWS_PER_BLOCK + arange(0, ROWS_PER_BLOCK)
    row_mask  = row_offs < n_rows

    col_offs  = arange(0, TILE)
    topk_idx  = arange(0, K)
    best      = full([ROWS_PER_BLOCK, K], pad_val)

    # 约束 8：遍历实际 n_cols，禁止遍历 next_pow2(n_cols)
    for c in arange(0, n_cols, TILE):
        load_mask = row_mask & (c + col_offs < n_cols)

        # 约束 1/4：tile-wise sort，不足处用 pad_val 填充
        tile        = load(x_ptr + row_offs[:,None] * n_cols + (c + col_offs)[None,:],
                           mask=load_mask, other=pad_val)
        sorted_tile = sort(tile, dim=1, descending=largest)

        # 约束 2：用 gather 提取 top-k
        chunk_topk  = gather(sorted_tile, topk_idx, axis=1)

        # 约束 6：temp buffer merge，禁止 in-register gather + modulo + where
        store(temp_ptr + row_offs[:,None] * COMBINED + arange(0, COMBINED)[None,:],
              full([ROWS_PER_BLOCK, COMBINED], pad_val), mask=row_mask)
        store(temp_ptr + row_offs[:,None] * COMBINED + arange(0, K)[None,:],
              best, mask=row_mask)
        store(temp_ptr + row_offs[:,None] * COMBINED + (K + arange(0, K))[None,:],
              chunk_topk, mask=row_mask)

        merged = load(temp_ptr + row_offs[:,None] * COMBINED + arange(0, COMBINED)[None,:],
                      mask=row_mask, other=pad_val)
        sorted_merged = sort(merged, dim=1, descending=largest)
        best = gather(sorted_merged, topk_idx, axis=1)

    store(out_ptr + row_offs[:,None] * K + arange(0, K)[None,:],
          best, mask=row_mask)

# 约束 7：启动时传入 compiler hint
launch(topk_kernel, grid, ...,
       multibuffer=False, unit_flag=False)
```

## 常见错误

```python
# 错误：Python if 分支
if score > best_score:
    best_idx = i

# 正确：tl.where
best_idx = tl.where(score > best_score, i, best_idx)
```

```python
# 错误：标量 flag 累积
keep = True
for j in range(n):
    if iou >= threshold:
        keep = False

# 正确：通过内存状态传递
tl.store(scores_ptr + j, -1.0, mask=suppress)
```

```python
# 错误：先收集所有保留元素再截断（破坏降序）
# 正确：每次迭代只选一个，天然满足降序和数量限制
```
