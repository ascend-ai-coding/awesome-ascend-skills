# Tiling 优化

## 概述

在 NPU 架构中，内存带宽通常是性能瓶颈。

合并访存（Coalesced Access）：当一个向量指令访问连续的内存地址时，硬件可以一次性高效读取数据。

跨步访存（Strided Access）：如果向量化轴在非连续维度，硬件必须发起多次内存请求或进行复杂的地址重组，导致带宽利用率大幅下降。

计算效率：在连续轴上向量化可以利用 SIMD 单元进行纯向量加法，避免了在循环内频繁执行昂贵的跨 Lane 还原（Reduction）指令。

## 适用条件

处理多维张量（3D 及以上）的规约类（Reduction）或归一化类（Normalization）算子，且还原轴（Reduction Axis）并非内存布局中的最连续轴（通常为最后一维 N）。

## 优化方法

### 优化前（非连续轴向量化）

```python
# 假设 M 为还原轴，N 为连续轴（stride_n=1）
# 错误：在 M 上分块，导致访存不连续
for m_start in range(0, dim1, BLOCK_SIZE_M):
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    # 访存：ptr + (m_offsets * stride_m) + n_idx -> 跨步读
    vals = tl.load(input_ptr + m_offsets * stride_m + n_idx)
    acc += vals
result = tl.sum(acc) # 循环内或末尾需要还原向量
```

### 优化后（连续轴向量化）

```python
# 正确：在 N 上分块，利用连续性
offsets_n = n_start + tl.arange(0, BLOCK_SIZE_N)
acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

for m_idx in range(0, dim1):
    # 访存：ptr + (m_idx * stride_m) + offsets_n -> 连续合并读取
    vals = tl.load(input_ptr + m_idx * stride_m + offsets_n)
    acc += vals # 纯向量加法，极快
# 直接处理 acc 向量后写回
```

## 优化策略

1. **重置向量化轴**：将 BLOCK_SIZE 从还原轴转移到物理存储最连续的轴（通常是 dim_last）

2. **向量累加器**：在连续轴上维护一个累加器向量

3. **循环设计**：外层循环遍历还原轴（标量迭代或大步长迭代），内层直接进行向量加法

4. **粗粒度调度**：调整 Grid 配置，使每个 Program 处理更连续、更大块的数据（如整个 Batch），提升数据局部性

## 关键点

1. **合并访存**：向量化轴必须在内存最连续的维度上
2. **避免跨步访存**：确保 `tl.load` 的偏移量计算中，向量化部分作用于 `stride = 1` 的轴
3. **向量累加**：在连续轴上累加，避免在循环内执行昂贵的还原指令

## 判断逻辑

- 检查 `tl.load` 的偏移量计算：如果 `tl.arange` 产生的向量偏移量作用于 `stride > 1` 的轴，而存在 `stride = 1` 的轴仅被当作标量索引处理 → 涉及。
- 检查循环累加器：如果累加器在还原轴上分块，但访存模式导致了非连续内存读取 → 涉及。
- 如果 `tl.arange` 已经作用于内存最连续的轴（通常是张量的最后一维），且实现了合并访存 → 不涉及，跳过。

---

## 方法二：Host 侧张量布局变换（消除大步长 stride）

### 适用情形

当大步长并非来自分块策略（方法一可解决），而是来自**张量本身的头维度交织布局**时，需要在 Host 侧消除根源。典型场景：张量形状为 `[B, T, H, K]`（或 `[B, T, H]`），T 轴 stride = `H*K`，kernel 内 block_ptr stride 与手动指针偏移处处含乘 `H` 因子，导致同一 head 的连续时间步在内存中不相邻，访存合并性差。

**与方法一的区别**：方法一在 kernel 内重排向量化轴（不改数据布局，零额外开销）；方法二在 Host 侧用 `permute().contiguous()` 改变物理布局，消除大步长 stride（有额外拷贝开销，但根治交织布局问题）。

### 触发特征

1. **交织头维度布局**：张量形状为 `[B, T, H, K/V]` 或 `[B, T, H]`，T 轴 stride 含 `H` 因子
2. **kernel 内多处 `* H` 偏移**：`bos * H + i_h`、`H*K`、`H*V`、`H*BT` 等反复出现
3. **block_ptr stride 含 H 因子**：如 `tl.make_block_ptr(q, (T, K), (H*K, 1), ...)`

若张量已为头优先布局（`[H, B, T, K]`），或 H=1（无头维度交织），则不涉及。

### Host 侧变换

```python
# 1. 确定 stride_hz
stride_hz = B * T  # 非变长；变长时 stride_hz = total_len

# 2. 对需要变换的张量做 permute + contiguous，把 H 提前
q_t = torch.permute(q, (2, 0, 1, 3)).contiguous()       # [B,T,H,K] → [H,B,T,K]
beta_t = torch.permute(beta, (2, 0, 1)).contiguous()      # [B,T,H] → [H,B,T]
A_t = torch.permute(A, (2, 0, 1, 3)).contiguous()         # [B,T,H,BT] → [H,B,T,BT]

# 3. 输出张量也按新布局分配
dA_t = torch.zeros([H, B, T, BT], ...)
db_t = torch.zeros([H, B, T], ...)
```

### Kernel 侧变化

**偏移计算**：`bos * H + i_h` → `i_h * stride_hz + bos`

```python
# 优化前（交织布局，T 轴 stride = H*K）
q += (bos * H + i_h) * K
beta += bos * H + i_h

# 优化后（头优先布局，T 轴 stride = K）
q += (i_h * stride_hz + bos) * K
beta += i_h * stride_hz + bos
```

**block_ptr stride 简化**：

```python
# 优化前
p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), ...)   # 大步长
# 优化后
p_q = tl.make_block_ptr(q, (T, K), (K, 1), ...)     # 连续步长
```

**声明 stride_hz 为 do_not_specialize**：

```python
@triton.jit(do_not_specialize=['T', 'stride_hz'])
def kernel(..., stride_hz, ...):
```

### Host 侧结果还原

kernel 输出是 `[H, B, T, ...]` 布局，需 permute 回原布局：

```python
db = torch.permute(db_t, (1, 2, 0)).contiguous()
dA = torch.permute(dA_t, (1, 2, 0, 3)).contiguous()
```

### 性能收益与风险

**收益**：T 轴 stride 从 `H*K` 降为 `K`，同一 head 内连续时间步在内存中相邻，访存合并性提升、地址计算简化、硬件预取更有效。

**风险与权衡**：

| 风险 | 说明 | 缓解措施 |
|------|------|----------|
| permute + contiguous 开销 | Host 侧需额外内存拷贝 | 仅对 kernel 热路径中的张量变换；若 kernel 计算量远大于拷贝开销则值得 |
| stride_hz 语义变化 | 变长序列时 stride_hz = total_len，非 B*T | 确保 host 与 kernel 侧 stride_hz 定义一致 |
| 输出张量布局 | kernel 输出变为头优先布局 | host 侧需 permute 回原布局再返回 |

⚠️ 与方法一不同，方法二**跨越 kernel 边界**（需改 Host 调用层：输入 permute + 输出还原），任一处遗漏即布局错乱、精度全错。应用后须额外验证布局一致性。

### 两种方法选择

| 对比项 | 方法一（kernel 内重排轴） | 方法二（Host 侧布局变换） |
|-------|------------------------|------------------------|
| 大步长来源 | 分块策略不当 | 头维度交织布局 |
| 作用层面 | kernel 内（零额外开销） | Host 调用层（有拷贝开销） |
| 根治程度 | 改善当前 kernel 访存 | 消除 stride 源头 |
| 风险 | 低（纯改写） | 中（需成对 permute + 还原） |

**选择依据**：
- 大步长来自分块策略 → 方法一
- 大步长来自头维度交织布局（stride 含 H 因子）→ 方法二

---

## 来自 SKILL.md 的原始描述（优化点 2：Tiling 优化（连续轴向量化））

**适用条件**：处理多维张量（3D 及以上）的规约类或归一化算子，且规约轴并非内存布局中的最连续轴

**典型代码特征**：
```python
@triton.jit
def kernel(input_ptr, output_ptr, dim1, dim2, ...):
    # 特征 1：向量化偏移 tl.arange 作用在非连续轴（如 dim1/M 轴）
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    # 特征 2：访存偏移计算中，向量化部分乘上了较大的 stride
    input_offset = m_offsets * stride_m + n_idx * stride_n
    # 特征 3：循环内部频繁进行还原操作（如 tl.sum）将向量压缩为标量
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    ...
    total_sum = tl.sum(acc, axis=0)
```

**判断逻辑**：
- 检查 `tl.load` 的偏移量计算：如果 `tl.arange` 产生的向量偏移量作用于 `stride > 1` 的轴，而存在 `stride = 1` 的轴仅被当作标量索引处理 → 涉及
- 检查循环累加器：如果累加器在还原轴上分块，但访存模式导致了非连续内存读取 → 涉及
- 如果 `tl.arange` 已经作用于内存最连续的轴（通常是最后一张量的最后一维），且实现了合并访存 → 不涉及，跳过

**命中条件**：代码逻辑旨在对某维度进行还原，但其分块策略导致硬件执行了跨步访存

**参考文档**：`references/tiling_optimization.md`

---
