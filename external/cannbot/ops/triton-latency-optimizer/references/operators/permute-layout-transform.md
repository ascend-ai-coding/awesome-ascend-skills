# Permute / Layout-transform 算子优化

适用于 `permute`、`transpose`、`reshape`（触发 contiguous copy）等仅改变数据布局、不改变元素值的算子。

## 核心优化原则

1. **模式特化 > 通用 gather**
   - 避免把所有 permutation 都落入 1D element-wise gather/scatter。
   - 为常见模式实现专用 kernel：2D transpose、batch transpose、swap adjacent dims、reverse dims、move size-1 dims 等。
   - 每个专用 kernel 内部使用规则、连续的 `tl.load`/`tl.store`，让 NPU 可以向量化和合并访存。
   - **禁止以“硬编码 dims + 逐元素 `div`/`mod` 或 `tl.where` 链”冒充 pattern specialization；这仍属于 gather，不算真正的模式特化。**

2. **连续维度合并**
   - 若置换后某组相邻维度在输入或输出侧仍连续，应通过 `view` 将其合并为一个逻辑维度。
   - 示例：`[A, B, C, D] -> [B, A, C, D]` 中 `C*D` 连续，可降为对 `A x B` 矩阵做 `C*D` 次批量转置。

3. **View 短路**
   - identity permutation 直接返回 `x.view(out_shape)`。
   - 仅移动 size-1 维度时，内存顺序不变，同样返回 view，不启动 kernel。

4. **退化 shape 路由**
   - batch 维度为 1 时，batch transpose 可路由到更简单的 2D transpose。
   - 其他因 size-1 导致的特殊 layout 应单独处理，避免通用 kernel 的标量索引开销。

5. **Autotune tile 大小**
   - 对 2D/batch transpose 类 kernel，使用 `@triton.autotune` 覆盖多组 `(BLOCK_M, BLOCK_N)`。
   - 典型候选：`BLOCK_M` ∈ {16, 32, 64, 128}，`BLOCK_N` ∈ {32, 64, 128, 256}，配合 `num_warps=1`。

6. **num_cores-aware grid**
   - grid 大小上限设为 `num_cores`，kernel 内通过 `for block_idx in range(pid, total_blocks, grid_size)` 循环处理多 block。
   - 避免 grid 远大于物理核数带来的调度开销。

## 命中检查清单

| 检查项 | 未命中表现 | 优化动作 |
|---|---|---|
| 是否存在 view 短路 | identity/size-1-only 仍启动 kernel | 在 `forward()` 中提前返回 view |
| 是否使用通用 gather | 所有 permutation 走 1D `generic_permute_kernel` | 为常见模式增加专用 kernel 路径 |
| 专用 kernel 是否仍是 element-wise gather | 常见模式 kernel 内部仍用逐元素 `div`/`mod` 或 `tl.where` 链 | 重写为 tile-based 连续 `tl.load`/`tl.store`（对应优化点 2/8） |
| 是否合并连续维度 | kernel 中保留 3+ 层独立索引 | 通过 `view` 合并连续维度后再调用 kernel |
| 是否特化退化 shape | B==1 仍走 batch transpose | 增加 B==1 路由到 2D transpose |
| 是否 autotune tile | transpose kernel 使用固定 BLOCK | 增加 `@triton.autotune` |
| grid 是否按 num_cores 限制 | grid 远大于物理核数 | 限制 grid 并循环处理多 block |

## 功能与精度一致性

- 所有优化仅改变数据布局，不改变元素值和 dtype。
- view 短路必须验证输入输出形状一致且内存顺序等价。
- 专用 kernel 的 mask 必须覆盖非对齐边界。

## 与现有优化点的关系

- **优化点 1（入参静态化）**：`BLOCK_M/BLOCK_N/BLOCK_SIZE` 等应声明为 `tl.constexpr`。
- **优化点 2（Tiling 优化）**：常见模式专用 kernel 必须沿连续轴做 tile 向量化，避免跨步 gather。
- **优化点 8（维度合并）**：将连续维度合并为一个逻辑维度。
- **优化点 12（Grid 形状与多路径特化）**：不同 permutation 模式对应不同 kernel 路径。
- **优化点 13（Autotune）**：transpose tile 大小自动搜索。
- **优化点 14（混合策略自动选择）**：按 shape/dtype/pattern 选择专用或 fallback 路径。
- **优化点 16（连续拷贝聚合）**：identity permutation 使用连续拷贝或 view 短路。
