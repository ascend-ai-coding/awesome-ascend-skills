# Layout-transform / Permutation 算子设计要点

当算子属于 **permute / transpose / contiguous / reshape-as-copy** 等改变数据布局但不改变元素值的算子时，草图应优先采用以下模式。

## 1. 模式特化而非单一通用 gather

- 识别常见模式：identity、batch transpose、swap adjacent dims、reverse dims、move size-1 dims 等。
- 为每种常见模式设计独立 kernel 或独立路径；避免把所有情况都落入 1D element-wise gather，因为标量索引计算在 NPU 上代价高。
- 在草图中用 `@llm_hint: layout_pattern=<pattern>` 标注所识别的模式。

### 禁止事项
- 草图不得仅包含一个 `@llm_hint: layout_pattern=generic_permute` 的 flat gather kernel。
- identity permutation、仅移动 size-1 维度、或存在连续子序列保持原序的情况，必须标注 `@llm_hint: view_shortcut`，不启动 kernel。
- 2D / batch transpose 模式必须给出 tile 大小候选范围，供代码生成阶段使用 `@triton.autotune`。
- **禁止以“硬编码 dims + 逐元素 div/mod 或 `tl.where` 链”冒充 pattern specialization；每个专用路径必须设计为 tile-based 连续访存。**

## 2. 连续维度合并

- 若置换后某些相邻维度在输入/输出中仍保持连续，应将其合并为一个逻辑维度，以提升 `tl.load`/`tl.store` 的向量长度。
- 例如 `[A, B, C, D] -> [B, A, C, D]` 中 `C*D` 连续，可将问题降为对 `A x B` 矩阵的批量转置。

## 3. View 短路

- identity permutation 或仅移动 size-1 维度时，数据在内存中顺序不变，草图应标注 `@llm_hint: view_shortcut`，由 Host 直接返回 `x.view(out_shape)`，不启动 kernel。

## 4. 退化 shape 路由

- 当某个维度为 1 时，batch transpose 可退化为 2D transpose；草图中应显式标注此类特殊路径。

## 5. Grid 与向量化

- grid 大小按 `num_cores` 限制，kernel 内通过循环处理多个 block；tile 大小在 sketch 中给出候选范围，供代码生成阶段用 `@triton.autotune` 搜索。
