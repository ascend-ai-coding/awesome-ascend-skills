# Ascend CV 融合算子生成方法论 — Tiling 篇

> 基于 Sparse Flash Attention (SFA) 等 CV 融合算子优化实践总结
> 适用范围：所有需要选择 block/tile 大小的 Triton-Ascend 算子

---

## 1. 核心思想

Tiling 是在 **On-Chip 容量、DMA 吞吐、循环次数** 之间找最优切分。对 CV 融合算子，tiling 同时决定：

- Cube 每次计算的矩阵块大小
- Vector 每次后处理/Copy/Accum 的数据量
- 每个 CV 通道单次搬运的 shape
- 外层循环迭代次数

## 2. 参数命名

| 参数 | 通用含义 | CV 融合常见映射 |
|------|---------|----------------|
| `BLOCK_M` | 输出矩阵行方向块 | Query / Output 行数 |
| `BLOCK_N` | 输出矩阵列方向块 | KV / topk / reduction 维度 |
| `BLOCK_K` | 内积 reduction 维度 | head 维拆分 / feature 维拆分 |
| `BLOCK_H` | 参与一次 ROW_SPLIT 的分组维度 | head 分组 / batch 分组 |

## 3. 硬约束

1. **ROW_SPLIT 对齐**：被切分维度的一半需被 16 整除，通常等价于该维度 %32 == 0。
2. **NZ 布局对齐**：参与 Cube `16x16` tile 的维度必须 %16 == 0。
3. **On-Chip 容量**：单次 step 内所有 UB + L1 buffer 之和必须小于硬件上限。
4. **编译器兼容性**：过大或不规则 tile 可能触发断言，候选需先验证可编译。

## 4. On-Chip 内存估算模板

```
ub_buf  = dtype_size * (BLOCK_M / 2) * BLOCK_N      # ROW_SPLIT 后行数减半
l1_a    = dtype_size * BLOCK_K * BLOCK_N
l1_b    = dtype_size * BLOCK_M * BLOCK_K
l1_c    = dtype_size * BLOCK_M * BLOCK_N
```

具体算子按实际 buffer 数量和映射求和。引入双缓冲 / Batch 流水线后，上述结果需再乘以 ~2。

## 5. DMA / MTE 吞吐原则

- 搬运长度尽量是 16 或 32 的倍数
- 避免过“瘦”的 shape（如 `1xN`、`Mx1`）
- tile 越大循环越少，但内存压力和编译风险越大
- 优先选择能整除问题规模的 tile，减少尾块浪费

## 6. 候选验证流程

```
1. 固定其他优化（scope、CV 通道、同步信号等）
2. 按约束和容量生成候选（建议从 {32, 64, 96, 128} 开始，128 需单独验证编译）
3. 全量精度测试
4. msprof op 采集每个候选的 Task Duration
5. 选择延时最低的 tile
6. 后续优化改变内存或数据流时，回到步骤 2 重新评估
```

## 7. 重新评估时机

- 新增双缓冲 / Ping-Pong buffer
- 改变数据类型
- 合并更多计算到同一 scope
- 改变 CV 通道数量或方向
- 改变 `BLOCK_H` 或 ROW_SPLIT 策略
- 升级编译器 / CANN 版本

## 8. 借助 Autotune 自动化搜索

可用 `@triton.autotune` 替代手写候选表：

- **自定义 `configs=[...]` 最稳妥**；自动/半自动 autotune 对 Cube-Vector 混合核支持有限
- 用 `prune_configs_by` 剪掉不满足对齐、容量、尾块约束的配置
- 测速时设置 `TRITON_BENCH_METHOD=npu` 提升精度
- autotune 选出的配置仍需独立走精度和 `msprof op` 验证

## 9. Tiling 与 Batch 流水线的关系

引入 Batch 流水线后，必须重新评估 tiling：

- **内存占用接近翻倍**：按第 4 节估算后乘以 ~2，确认仍低于 L1/UB 上限。
- **可尝试更大的 `BLOCK_N`**：循环次数减少后，即使单块数据量变大，也可能整体更快。例如 SFA 中 topk=640 时，`BLOCK_N=128` 理论上循环更少，但当前平台 UB 溢出无法使用。
- **head padding 会增加内存和循环**：如果实际维度不是 `BLOCK_H` 倍数，padding 会让每个 program 多处理一些无效数据，可能抵消收益。必要时让 `BLOCK_H` 对齐实际维度。
- **Batch 流水线不改变 tiling 结论**：Batch 流水线与无流水线版本的 buffer 占用相同，因此原本因内存溢出无法使用的 tile 仍不能使用。
- **编译器选项**：某些平台需要配合 `vf_merge_level=1` 才能正确编译/调度 CV 融合核，但该选项可能带来轻微性能波动，需实测。

## 10. 示例：SFA 前向映射

- `BLOCK_M` → Query / Output 行数
- `BLOCK_N` → topk / 稀疏 KV 维度
- `BLOCK_K` → HEAD_DIM
- `BLOCK_H` → head 分组维度

在该算子上 `BLOCK_N=64` 综合最优，但其他算子需独立验证。调度层面，Batch 流水线（PIPE_STAGES=2）在 `BLOCK_N=64` 下可再获得 1.11x ~ 1.17x 收益，见 `references/operators/cv-fusion-pingpong.md`。

## 11. 常见 Tiling 陷阱

| 问题 | 原因 | 修复 |
|------|------|------|
| 尾块浪费大 | `BLOCK_N` 与 problem size 不整除 | 选择能整除或近似整除的 tile；若必须非整除，接受少量冗余 |
| UB / L1 overflow | tile 过大 + Batch 流水线 buffer doubling | 减小 tile，或减少 buffer 数量，或共享 buffer |
| 编译器断言失败 | tile 超出平台支持范围 | 退回更小 tile，验证编译 |
| 性能随 tile 增大反而下降 | DMA shape 变瘦 / 内存压力过大 | 使用 msprof 逐 candidate 实测，不盲目放大 |

---

*参考实现：SFA 前向 CV 融合参考代码*
*性能报告：SFA 前向 tiling 优化性能报告*
