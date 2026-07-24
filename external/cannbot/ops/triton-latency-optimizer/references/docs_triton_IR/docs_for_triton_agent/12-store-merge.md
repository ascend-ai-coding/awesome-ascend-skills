# Store 合并优化

## 触发条件

当 Agent 发现 Triton kernel 中存在**多次 `tl.store` 对相同基地址、不同偏移的存储操作**，且满足以下条件时，应考虑应用 Store 合并优化：

- 多条 `tl.store` 的目标地址共享同一个 `base_ptr`
- 各条 store 的 offset 可以合并为一块**连续的**地址偏移
- 各条 store 的 data 可以整合为一个 tensor

典型代码模式：

```python
tl.store(base_ptr + offset_1, data_1)
tl.store(base_ptr + offset_2, data_2)
...
tl.store(base_ptr + offset_n, data_n)
```

---

## 核心知识

### 为什么需要合并

在 NPU 上，每次 `tl.store` 都会触发一次独立的内存写操作。多次 store 意味着多次独立的 MTE3（UB -> GM）搬运指令，每次搬运都有启动开销。如果这些 store 的目标地址是连续的，可以：

1. **在 UB 中预先整合数据**：将多个 data tensor 拼接为一个连续的大 tensor
2. **用一次 `tl.store` 完成写入**：一次连续内存搬运，减少指令开销和启动延迟

### 合并的前提条件

| 条件 | 说明 |
|------|------|
| 基地址相同 | 所有 store 必须指向同一个 `base_ptr` |
| offset 合并后连续 | `offset_1, offset_2, ..., offset_n` 合并后必须构成一整块连续地址，不能有间隔 |
| data 可整合 | 各 data tensor 可以通过 `tl.where` 等方式放入一个统一的大 tensor 中 |

### 合并方法

**offset 部分**：确认 offset 合并后连续，使用 `tl.arange` 直接生成一整块偏移。

**data 部分**：
- 使用 `tl.zeros` 创建一个与合并后偏移等大的 data tensor
- 使用 `tl.where` 将各 data 放到对应位置
- 如果某一部分 data 都是固定值，使用 `tl.full` 初始化，减少 `tl.where` 次数

**mask 部分**：如果原始代码中有 mask，需要按逻辑合并处理。

---

## 代码模式

### 优化前：3 次独立 store

```python
ps1_idx_0 = pid * 3
ps1_idx_1 = ps1_idx_0 + 1
ps1_idx_2 = ps1_idx_0 + 2

ps1_mask_0 = ps1_idx_0 < num_experts * 3
ps1_mask_1 = ps1_idx_1 < num_experts * 3
ps1_mask_2 = ps1_idx_2 < num_experts * 3

tl.store(problem_sizes1_ptr + ps1_idx_0, data0, mask=ps1_mask_0)
tl.store(problem_sizes1_ptr + ps1_idx_1, data1, mask=ps1_mask_1)
tl.store(problem_sizes1_ptr + ps1_idx_2, data2, mask=ps1_mask_2)
```

**问题**：3 次独立 `tl.store`，3 次 MTE3 搬运指令，每次只写 1 个元素，启动开销大。

### 优化后：1 次合并 store

```python
offs_n = tl.arange(0, 3)[None, :]

data = tl.zeros((BLOCK_SIZE, 3), dtype=tl.int32)
data = tl.where(offs_n == 0, data0, data)
data = tl.where(offs_n == 1, data1, data)
data = tl.where(offs_n == 2, data2, data)

base_offsets = pid[:, None] * 3 + offs_n

tl.store(problem_sizes1_ptr + base_offsets, data, mask=mask[:, None])
```

**改进**：1 次 `tl.store`，1 次 MTE3 搬运指令，写入连续的 3 个元素。

### 含固定值的场景

如果某个 data 是固定值（如 0），可以用 `tl.full` 初始化，省掉一次 `tl.where`：

```python
offs_n = tl.arange(0, 3)[None, :]

data = tl.full((BLOCK_SIZE, 3), FILL_VALUE, dtype=tl.int32)
data = tl.where(offs_n == 0, data0, data)
data = tl.where(offs_n == 1, data1, data)

base_offsets = pid[:, None] * 3 + offs_n

tl.store(problem_sizes1_ptr + base_offsets, data, mask=mask[:, None])
```

---

## 910_95 特别注意

### UB 容量约束

合并后的大 data tensor 会占用更多 UB 空间。910_95 的 UB 为 248KB（910B 为 192KB），空间更充裕，但仍需注意：

| 硬件 | UB 可用容量 | double buffer 后 |
|------|-----------|----------------|
| 910B | 192 KB | 96 KB |
| 910_95 | 248 KB | 124 KB |

合并后 data tensor 的大小 = `BLOCK_SIZE * N * sizeof(dtype)`，其中 N 为合并的 store 数量。需确保加上其他 UB 占用后不超出容量限制。

### MultiBuffer 交互

910_95 默认 `multibuffer=False`，UB 可用 248KB。如果手动开启 `multibuffer=True`，UB 可用容量减半为 124KB，合并后的 data tensor 可能导致 UB 溢出，需要重新评估 BLOCK_SIZE。

### 连续写入对齐

910_95 上 UB -> GM 的 MTE3 搬运要求 32B 对齐。合并后的连续 store 天然更容易满足对齐要求，这是此优化的额外收益。如果原始的多次 store 各自写入少量数据（如 1 个 int32 = 4B），单次搬运无法对齐，合并后写入 3 个 int32 = 12B 仍不足 32B，但至少减少了搬运次数。

---

## 相关文档

- [内存访问模式优化指南](../docs_for_triton_agent/04-memory-access-patterns.md) - 连续访存 vs 离散访存的性能差异
- [标量降级规避指南](../docs_for_triton_agent/06-scalar-degradation-avoidance.md) - `tl.where` 在特定条件下可能退化为标量操作
- [CV 流水线优化](../docs_for_triton_agent/05-cv-pipeline-optimization.md) - MTE3 搬运在 CV 流水线中的位置
- [Tiling 与 Grid 策略](../docs_for_triton_agent/03-tiling-and-grid.md) - UB 容量预算与 BLOCK_SIZE 选择
- [编译参数配置](../docs_for_triton_agent/07-compile-params.md) - multibuffer 配置与 910_95 差异
