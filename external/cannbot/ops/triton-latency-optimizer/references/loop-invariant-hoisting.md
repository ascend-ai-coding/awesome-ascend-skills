# 循环不变量外提优化

## 快速识别（看到这些模式立即加载本文档）

| 抽象模式 | 检查方法 |
|---------|---------|
| 循环内 `tl.load` 的索引不含当前循环变量 | 检查 load 的索引表达式，看是否缺少 `for` 循环的迭代变量 |
| 索引通过 `//` 整除映射到更粗粒度 | 检查是否有 `index = something // const` 形式，且该 index 用于 load |
| 循环内有参数指针的 load | 检查 load 的指针是否为参数指针（非主数据输入指针） |

## 核心模式

**模式**：在循环内重复加载相同的值，且该值只与外层索引相关。

**检测条件**（同时满足）：

1. 存在嵌套循环结构
2. 内层循环中有 `tl.load(param_ptr + index_expr)`
3. `index_expr` 只依赖外层循环变量，不依赖内层循环变量

**收益**：当内层循环次数 >> 外层循环次数时，减少重复加载次数。

---

## 模式匹配

### 模式 1：参数索引只依赖外层变量

```python
# 问题代码
for outer_idx in range(outer_size):
    for inner_idx in range(inner_size):
        param_idx = outer_idx  # 只依赖外层变量
        val = tl.load(param_ptr + param_idx)  # 重复加载相同值
        ...

# 优化后
for outer_idx in range(outer_size):
    param_idx = outer_idx
    val = tl.load(param_ptr + param_idx)  # 提到外层
    for inner_idx in range(inner_size):
        ...
```

### 模式 2：参数索引通过整除映射到外层

```python
# 问题代码
for block in range(num_blocks):
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    param_idx = offsets // SPATIAL_SIZE  # 映射到更粗粒度的索引
    val = tl.load(param_ptr + base + param_idx)  # 同一 param_idx 的多个元素重复加载
    ...

# 优化后
for coarse_idx in range(num_coarse):
    val = tl.load(param_ptr + base + coarse_idx)  # 每个粗粒度索引只加载一次
    for fine_idx in range(num_fine):
        offsets = coarse_idx * fine_size + fine_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        ...
```

---

## 检测方法

```
Step 1: 找到循环内的 tl.load 语句
        ↓
Step 2: 分析 load 的索引表达式
        ↓
Step 3: 判断索引是否依赖内层循环变量
        ↓ 不依赖内层变量，只依赖外层变量或常量
        ↓
Step 4: 计算内外层循环次数
        ↓ 内层次数 >> 外层次数
        ↓
Step 5: 可外提到外层循环
```

---

## 收益分析

| 场景 | 内层次数 | 外层次数 | 加载次数变化 | 收益 |
|------|---------|---------|-------------|------|
| 归一化算子 per-channel 参数 | N/C | C | N → C | N/C 倍 |
| 通用情况 | inner | outer | inner×outer → outer | inner 倍 |

实际收益取决于：
- 内外层循环次数比例
- load 操作在总计算中的占比
- 内存访问延迟

---

## 示例

### 归一化算子场景

```python
# 问题：每个元素都加载 weight/bias
for block in range(num_blocks):
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    channel = offsets // spatial_size
    w = tl.load(weight_ptr + channel)  # 相同 channel 重复加载
    ...

# 优化：按 channel 分组，每个 channel 只加载一次
for c in range(num_channels):
    w = tl.load(weight_ptr + c)  # 提到 channel 循环外
    for block in range(blocks_per_channel):
        offsets = c * spatial_size + block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        ...
```

### 通用场景：查找表访问

```python
# 问题：循环内重复访问查找表同一位置
for batch in range(N):
    for elem in range(E):
        table_idx = batch % TABLE_SIZE  # 只依赖外层 batch
        val = tl.load(table_ptr + table_idx)
        ...

# 优化
for batch in range(N):
    table_idx = batch % TABLE_SIZE
    val = tl.load(table_ptr + table_idx)  # 提到外层
    for elem in range(E):
        ...
```

### 场景 3：小查找表在循环内被随机索引访问（结合离散访存优化）

**问题代码**：

```python
# expert_map 是一个小查找表（如 128 个元素），在循环内通过随机索引访问
for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
    expert_ids = tl.load(topk_ids_ptrs, ...)  # 运行时随机值
    # 虽然 expert_map 指针是常量，但索引 expert_ids 是随机的
    # 传统外提无法解决，因为每个循环迭代的 expert_ids 都不同
    expert_map_ptrs = expert_map + expert_ids
    mapped_ids = tl.load(expert_map_ptrs, ...)  # 循环内离散全局内存访问
```

**分析误区**：
- 表面上看 `expert_map` 是参数指针，索引表达式 `expert_ids` 在循环内变化，似乎不符合传统"循环不变量外提"条件
- 但实际上 `expert_map` 的内容（查找表本身）在循环内完全不变，只是**访问方式**是随机的

**正确优化**：将查找表整体预加载到 UB，循环内用 `tl.gather` 局部查找

```python
# 循环外：将整个 expert_map 加载到 UB（真正的不变量）
expert_map_data = tl.load(expert_map + tl.arange(0, num_experts)).to(tl.float32)

for x in range(cntx):
    e_ids = tl.load(topk_ids_ptr + x * BLOCK_SIZE + offsets, mask=mask, other=0)
    # 循环内：从 UB 局部 buffer gather，而非全局离散加载
    expert_ids = tl.gather(expert_map_data, e_ids, 0)
    ...
```

**关键洞察**：当循环内出现 `tl.load(small_table + random_index)` 模式时，应将"外提"理解为**将整个查找表预加载到局部 buffer**，而非简单地将 `tl.load` 语句移到循环外。

---

## 注意事项

1. **不是所有 load 都能外提**：只有索引不依赖内层变量时才能外提

2. **权衡代码复杂度**：外提可能增加代码复杂度，需权衡收益

3. **结合其他优化**：常与循环重排、分块优化结合使用

---

## 常见错误

```python
# ❌ 错误：索引依赖内层变量，不能外提
for outer in range(N):
    for inner in range(M):
        val = tl.load(ptr + outer + inner)  # inner 在索引中
        ...

# ✅ 正确：这种情况下不能外提
```

```python
# ❌ 错误：外提后逻辑变了
for c in range(C):
    w = tl.load(weight_ptr + c)
    for s in range(S):
        x = tl.load(input_ptr + c * S + s)
        # 如果原来有 mask 条件，外提后可能访问越界

# ✅ 正确：确保外提后不会越界
for c in range(C):
    if c < valid_channels:  # 添加条件判断
        w = tl.load(weight_ptr + c)
    for s in range(S):
        ...
```
