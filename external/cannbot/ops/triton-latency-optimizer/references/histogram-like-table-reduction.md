# Histogram-like / Small-Output-Table Reduction 优化

## 1. 算子类别识别

当算子满足以下全部特征时，归为本类别：

1. 输入被展平后，每个元素通过标量索引落入一个小输出表。
2. 输出表维度较小（通常 `bins <= 256`），可完整放入单个 program 的 UB / 寄存器。
3. 存在跨线程/跨核写冲突（多个输入可能映射到同一输出位置）。

**典型算子**：`torch.histc`、`torch.bincount`、`scatter_add` / `scatter_reduce` 的小输出场景、自定义 bucket count。

## 2. 高频优化点

| 优先级 | 优化点 | 对应通用编号 | 触发条件 |
|---|---|---|---|
| 1 | **物理核数扩展** | 3（分核优化） | 当前 `MAX_CORES` 小于目标架构物理核数，且 local kernel 占主导 |
| 2 | **禁止全局 atomic scatter-reduce** | 5（Scalar 转 Vector）/ 负面规则 | 代码中出现全局原子累加用于 histogram/bincount 计数 |
| 3 | **BLOCK_SIZE vs bins 折中** | 1（入参静态化）+ 经验调参 | `bins` 较小且 UB 有剩余，尝试更大 BLOCK_SIZE |
| 4 | **IR 诊断 match-matrix scalarization** | 30（IR 分析） | local kernel 异常慢，需 dump `last_pass.mlir` 确认向量化匹配矩阵是否被降级 |

## 3. 物理核数扩展（最高优先级）

### 判定

- 代码中 `MAX_CORES` 使用固定小值（如 8）。
- `profile_plan.json` 或算子分项耗时显示 local kernel 是绝对瓶颈（reduce kernel 耗时 < 10%）。
- 目标架构物理核数未被用满（910B2 为 48）。

### 优化方法

将 `MAX_CORES` 提升到目标架构物理核数，并保持动态 grid：

```text
PHYSICAL_CORES = 目标架构物理核数
if N <= BLOCK_SIZE:
    num_cores = 1
else:
    tiles     = ceil(N / BLOCK_SIZE)
    num_cores = min(tiles, PHYSICAL_CORES)
```

### 典型收益

以 `6_Histc` 为例：

| 版本 | geomean speedup |
|---|---|
| `MAX_CORES=8` | 0.2379 |
| `MAX_CORES=48` | 0.6343 |

## 4. 禁止全局 atomic scatter-reduce

### 判定

- histogram / bincount 语义下出现单 kernel 全局原子累加（如 `atomic_add(output_ptr + bin_idx, ...)` 形式）。

### 为何禁用

全局计数器竞争在 NPU 上会被严重串行化。`6_Histc` 中尝试 atomic_add 后 geomean 从 0.63 跌至 0.02。

### 正确替代

走 dual-kernel `local partial table + per-bin reduce`，详见 `triton-op-designer/references/cases/histogram-small-bins.md` 和 `triton-op-coding/references/triton-ascend-reduce.md`。

## 5. BLOCK_SIZE vs bins 折中

### 原则

- `BLOCK_SIZE` 越大，每个 program 内部循环次数越少。
- 但 match-matrix 面积为 `BLOCK_SIZE * bins`，会占用 UB。
- 当 `bins` 较大时，不能盲目增大 `BLOCK_SIZE`。

### 推荐初始策略

| bins | 推荐 BLOCK_SIZE | 原因 |
|---|---|---|
| <= 64 | 64/128 | 矩阵面积小 |
| 65~128 | 32/64 | 平衡面积与循环次数 |
| > 128 | 32 | 防止 UB 溢出 |

### 验证方法

每次调整后必须跑全量 benchmark，因为该折中对编译器 UB 分配敏感。

## 6. IR 诊断：match-matrix scalarization

### 判定

- local kernel 耗时远高于预期。
- `last_pass.mlir` 中出现大量标量化的 `arith.cmpi` / `arith.andi` / `vector.reduction`。
- `avoid_scalar_lowering.md` 中列出的向量化降级条件被触发。

### 处理

- 尝试切换到 **conditional increment** 模式，减少显式矩阵构造。
- 减小 `BLOCK_SIZE` 或 `bins` tile，降低 UB 压力。
- 检查 `MAX_BINS` 是否确实作为编译期常量参数传入。

## 7. 瓶颈诊断 fallback

当 profiler 无法给出有效瓶颈标签（`bottleneck == unknown`）时，按 kernel 名拆分时间：

- 若 `*_local_kernel` 占绝对主导 → 优先尝试核数扩展、BLOCK_SIZE 折中、IR 诊断。
- 若 `*_reduce_kernel` 占主导 → 检查 reduce kernel 的 grid 是否过小、是否存在 bank 冲突或 load 不连续。

## 8. 与其他算子类别的关系

- **Tiled Reduction**：本类别不是简单的沿维规约，而是带动态输出索引的表规约。
- **Scatter/Gather**：当输出表变大时，可能向 scatter/gather 类别过渡，需要分块或 sorted-index 策略。
- **Index**：大规模索引直方图（输出维度大）更适合预排序 + 二分查找，见 `triton-op-designer/references/cases/index-histogram.md`。

## 9. 6_Histc 实验数据

| 迭代 | 改动 | geomean speedup |
|---|---|---|
| Phase 3 baseline | `MAX_CORES=8`, `BLOCK_SIZE=32` | 0.2379 |
| opt_iter_0 | `MAX_CORES=48` | 0.6343 |
| opt_iter_1 | 全局 `atomic_add` | 0.0231 |
| opt_iter_2 | 混合 `BLOCK_SIZE` | 0.6286 |
| opt_iter_2 复测 | — | 0.6349 |

结论：核数扩展是本类别最高收益点；atomic_add 必须避免；BLOCK_SIZE 折中收益有限。
