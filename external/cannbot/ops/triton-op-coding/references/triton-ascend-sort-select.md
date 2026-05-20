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
