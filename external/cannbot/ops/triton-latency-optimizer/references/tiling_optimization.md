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
