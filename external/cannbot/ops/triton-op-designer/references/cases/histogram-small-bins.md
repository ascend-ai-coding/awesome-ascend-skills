---
name: triton-ascend-case-histogram-small-bins
description: "小 bins 直方图 / 小输出表规约类算子的草图设计：per-core 局部表 + 二次归约，避免全局原子竞争，输入展平、fp32 索引、动态核数"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  hardware: "Atlas A2, Atlas A3"
---

# 小 bins 直方图 / 小输出表规约草图设计

## 任务特征

- **操作类型**：每个输入元素通过标量映射落入一个小输出表（`bins <= 256` 或更小），例如 `torch.histc`、`torch.bincount`、`scatter_add` 的小输出场景。
- **写冲突**：多个元素可能映射到同一个输出位置，直接使用全局 `atomic_add` 会产生严重竞争。
- **输出表小**：局部表可以完整放进一个 program 的 UB / 寄存器空间。

## 默认架构：dual-kernel local + reduce

```text
Kernel 1 (local): 每个 VectorCore 统计自己数据段的局部表 -> partial[num_cores, bins]
Kernel 2 (reduce): 每个 bin 一个 program，累加所有 core   -> hist[bins]
```

**必须在草图中标注**：`@llm_hint: dual_kernel_candidate`

## Layer 1 约束（硬性规则）

1. **首版禁止全局原子 scatter-reduce**
   - 小输出表规约的首版实现不得使用单 kernel + 全局原子累加。
   - 全局原子仅在经过验证的特定场景（如极低竞争、超大输出表）中作为 fallback，且必须在草图中明确说明原因。

2. **必须双核结构**
   - Kernel 1 负责生成 `partial[num_cores, bins]`。
   - Kernel 2 负责按 `bins` 维度并行归约。

3. **必须输入展平**
   - 进入 kernel 前将输入 `reshape(-1)`，直方图语义对全量元素统计。

4. **bin 索引统一用 fp32 计算**
   - `scaled = (x_f32 - min) / bin_width`
   - `bin_idx = min(cast(scaled, int32), bins - 1)`
   - 避免 f16 边界元素因精度不足错桶。

5. **局部表必须零初始化**
   - 通过 zero-fill 方式分配并初始化局部表（如 `alloc([bins], init_zero)`）。

6. **动态核数匹配物理核数**
   - `NUM_CORES = min(PHYSICAL_CORES, max(1, ceil(N / BLOCK_SIZE)))`
   - 小 shape 用 1 核避免调度开销；大 shape 扩展到目标架构物理核数（910B2 为 48）。

7. **bins 必须作为编译期常量参数传入 kernel**
   - 便于编译器针对具体 bins 做展开和 UB 分配。

## 通用草图模板

```text
sketch histogram_small_bins {
  symbols: N, BINS;
  tensors: X[N]: f16|f32; HIST[BINS]: f32; PARTIAL[NUM_CORES, BINS]: f32;

  constexpr:
    BLOCK_SIZE = 32            # 初始值，coding 阶段可按 bins 调整
    PHYSICAL_CORES = 48        # 910B2; 其他架构按 npu-arch 读取
    NUM_CORES = min(PHYSICAL_CORES, max(1, (N + BLOCK_SIZE - 1) // BLOCK_SIZE))
    bin_width = (max - min) / BINS

  # @llm_hint("dual_kernel_candidate")
  # Kernel 1: per-core local histogram
  for core_idx in range(NUM_CORES):
    local_hist = alloc([BINS], init_zero)

    for block_start in range(core_idx * BLOCK_SIZE, N, NUM_CORES * BLOCK_SIZE):
      elems = min(BLOCK_SIZE, N - block_start)
      x_tile = load(X[block_start:block_start+elems])

      for j in range(elems):
        x_f32 = cast(x_tile[j], f32)
        valid = (x_f32 >= min) & (x_f32 <= max) & (~isnan(x_f32))
        scaled = (x_f32 - min) / bin_width
        bin_idx = min(cast(scaled, int32), BINS - 1)
        if valid:
          local_hist[bin_idx] += 1.0f

    store(local_hist -> PARTIAL[core_idx, 0:BINS])

  # Kernel 2: per-bin reduction
  for bin_idx in range(BINS):
    total = 0.0f
    for c in range(NUM_CORES):
      total += PARTIAL[c, bin_idx]
    store(total -> HIST[bin_idx])
}
```

## 与大规模索引直方图的区别

| 场景 | 输出维度 | 推荐策略 | 参考文档 |
|---|---|---|---|
| 小 bins 直方图（如 `histc`，bins<=256） | 小 | **per-core local table + reduce** | 本文档 |
| 大规模索引直方图（如 MoE expert 计数，365 类） | 大 | 预排序 + 二分查找 | `index-histogram.md` |

## 后续代码生成提示

- 代码生成阶段可参考两种向量化实现思路：
  1. **match-matrix**：构造 `[BLOCK_SIZE, BINS]` 布尔匹配矩阵，沿输入元素维度求和得到各 bin 计数。
  2. **conditional increment**：对 block 内每个有效元素直接执行 `local_hist[bin_idx] += 1`。
- 优先尝试 match-matrix；若编译器将其标量降级或 UB 溢出，再切换到 conditional increment。
- 具体 API 调用、指针计算、mask 处理由代码生成阶段根据当前 DSL 规范自行推导，不得直接复制历史算子的实现。
