# 批量短归约与逐记录写回

适用于单次归约较短、外层记录较多，并在归约后继续逐记录归一化、缩放或其他逐元素 epilogue 的结构。每行先得到归约标量再写回整行时，同时读取 [行归约与逐元素写回](rowwise_reduce_epilogue.md)。

**证据等级：已实现并实测。** 下列结构已在真实 Ascend kernel 中落地，并有编译运行、精度和同口径性能对比证据；该等级只表示候选可信度，不替代当前瓶颈排序和目标 case 复验。

## 优化结构

- 记录按块处理，但 DMA 必须服从实际连续轴：record 连续的输入/输出可整块搬运，split-major 输入通常逐 record 搬连续的 `[split, feature]` 矩形。
- `SimdVF` 的 UB 特征轴按向量 lane 数 padding；GM 只搬有效 record/feature，短轴归约和写回用有效 lane mask，标量结果用单 lane 写回。
- 向量 partial 沿短特征轴累加，标量 partial 用 broadcast load 汇总；点积可用 masked `vcadd`/`vdupv`，并核对归约顺序。
- 显式 stage 维的 UB 用每个 core 的 wave 选择槽位并配 `annotate_manual_multi_buffer`；无显式 stage 维的流水 buffer 用 `annotate_buffer_versions`，同一 buffer 不重复混用两种寻址方式。
- `Persistent(..., num_stages=N)`、buffer 版本和需要的 `enable_offset` 保持一致，并从生成地址与 event 核对实际重叠；常驻量保持单版本。
- batching、SIMD footprint、DMA 形态和 buffer 生命周期共享同一布局时，先作为一个数据流假设落地，再对可独立拆分的部分做 A/B；尾块只访问有效 GM，允许无副作用地计算完整 UB footprint。
- `rows_per_tile` 与 `num_stages` 是独立搜索轴：批量粒度先按连续布局、UB 预算、DMA 粒度和任务 wave 选择几何候选，再对可行 tile 实测流水级数；不能用单记录上的 stage 搜索代替多记录候选。
- producer 内合并 partial 与 `n_splits==1` 连续快路是独立候选，只在数据依赖和目标 case 匹配时评估。

复用前须核对批量粒度、归约顺序、partial 数、搬运形态、buffer 版本和 launcher 参数；上下文不同的负结果只否决当前组合。

## 已验证结构组合

| 路径 | 已落地的物理结构 |
|---|---|
| 归约后归一化 | 多记录批处理和 lane-padded UB；split-major 输入按 record 搬连续矩形，显式 stage 输入配合自动版本输出；broadcast 标量与向量 partial 累加后批量写回 |
| 归约后逐元素 epilogue | 连续记录使用矩形 UB 和批量 DMA；每条记录独立生成 fp32 归约标量，随后在 tile 内逐记录写回，尾 tile 只访问有效 GM |
| 带参数梯度归约 | 输入输出批量 DMA，流式 buffer 多版本；masked lane reduction/broadcast 完成点积与回填，标量结果只由有效 lane 写回 |
