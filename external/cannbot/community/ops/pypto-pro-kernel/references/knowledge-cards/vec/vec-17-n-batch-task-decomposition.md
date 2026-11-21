---
type: "PyPTO Performance Optimization Card"
title: "小归约维算子的 N 维 task 批处理"
description: "当归约/窗口维不超过 tile 行容量且 batch 维很大时，把每 task 处理一个 work item 改为每 task 批处理 NB 个完整 work item（相邻行块 GM 连续、窗口天然内部化），摊薄每 task 固定开销。"
status: "stable"
tags: ["pypto-pro", "vec", "scheduling", "task-decomposition"]
item_id: "vec-17"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-neighborhood.md#blocking-and-work-distribution"
      scope: "合并独立 work item 的连续行块以摊薄 transfer、Tile 与任务算术开销，同时保持每个邻域的数学边界。"
      boundary: "本卡无额外 halo 的结论依赖窗口裁剪在各 item 内；NB、容量与尾 item 数必须同源，不推广到跨 item 邻域。"
bound_hint: "MTE2、MTE3、SCALAR"
applicability: "归约/窗口维 ≤ tile 行容量且 batch 维大，单 task 有效工作量极小而每 task 固定开销（MTE descriptor、Tile 轮转、validshape、同步、任务算术）主导，work item 间独立且在合并视图中相邻行块 GM 连续"
target_api_gate: "仅限 Ascend 950PR 或 950DT；TilingKeyField 编译期模式分派、TilingData 运行时标量字段、pl.load 连续行块 2D 装载与嵌套 pl.range 均已在 950PR 交付算子核验；批大小 NB 与尾块钳位须 host/kernel 同源"
---
# 技术卡片 vec-17：小归约维算子的 N 维 task 批处理

- **适用 bound**：MTE2 / MTE3 / SCALAR（每 task 固定开销）
- **一句话**：把 N 个微 task 合成 ceil(N/NB) 个批量 task，每 task 的 MTE descriptor、Tile 轮转与任务算术等固定开销按 NB 倍摊薄；批处理合法性来自相邻 work item 行块连续、裁剪窗口落在自身 work item 内部。

## 何时用（诊断特征）

- 任务网格展开后 task 数远超 aivNum（每核需串行处理数十个 task），而单 task 的有效工作量极小——仅数行 × 数列，远不满一个 tile。
- 单 task 的 VF 指令量远小于配套固定开销（MTE2/MTE3 descriptor 下发、tile group 轮转、set_validshape、同步与任务 unravel 算术）：可由"每 task 指令数 × task 数 vs 固定开销 × task 数"的指令数模型初判，并用 trace/生成物归因确认。
- 归约/窗口维 C ≤ tile 行容量（如 c_chunk），batch 维 N 大；相邻 work item 在 [N·C, HW] 合并视图中是**连续行块**。
- per-item 分解在 C ≤ 容量时 halo 裁剪后窗口即整个 work item 自身——halo 本就没有携带额外行，批处理不需要装载任何额外数据。

## 何时不适用

- 归约/窗口维 > tile 行容量：一个 task 装不下完整 work item，必须保留带 halo 行的分段分解。
- work item 间存在递推、前缀、跨 item 归约或可见顺序依赖。
- 批后行块超出 tile/UB 容量；或 N 很小使任务数低于核数——此时 NB 必须退化为 1 维持并行度（见负载均衡公式），收益随之消失。
- 单 task 工作量已饱满、固定开销占比低时收益趋零。

## 原理

- 每 task 固定开销不随有效工作量缩小。task 数除以 NB 后，固定开销总量（descriptor、tile 操作、任务算术）同步除以 NB；单 task 的 tile 操作次数（load/cast/store 等）也按 NB 倍减少。
- [N·C, HW] 视图中相邻 work item 是相邻行块；窗口维 ≤ 行容量时，任意裁剪窗口 [max(0,c-r), min(C-1,c+r)] ⊆ 自身 work item 的 [0,C) 行——task 装载 [NB·C, hw_eff] 一个连续块即可覆盖全部窗口，**无需 halo 行**。注意这是批处理合法性的来源，不是字节收益：本形态下 per-item 分解的 halo 本就裁剪到 work item 边界，装载字节不变。
- 等价性：窗口行集合与升序累加顺序和 per-item 分解完全一致（C ≤ 容量时 per-item 的 off=r、u_c=i 与批内 ibase+c 寻址指向同一 GM 行、同一顺序）。

## 怎么改（before / after）

以下为已在 Ascend 950PR 交付算子核验的**嵌入片段**（n=批维、c=归约/窗口维）；kernel 上下文（datatype 特化、TileGroup/地址表、tiling 双侧合同）由具体算子提供。

before（kernel body，每 task 一个 work item + halo 逻辑）：

```python
# task 网格: N × ceil(C/c_chunk) × ceil(HW/hw_tile)
for task_id in pl.range(core_id, total_tasks, num_cores):
    n_idx = task_id // (c_blks * hw_blks)
    ...
    g_lo = pl.max(0, c0 - r)                     # halo 窗口（C<=c_chunk 时恒 [0,C)）
    g_hi = pl.min(c_dim, c0 + c_chunk + r)
    x_slot = x_group.next()
    pl.set_validshape(x_slot, [g_rows, hw_eff])
    pl.load(x_slot, x2d, [n_idx * c_dim + g_lo, hw_off])   # 每 task 一套 MTE/tile 开销
    ...
```

after（kernel body，每 task NB 个完整 work item，无 halo）：

```python
# task 网格: ceil(N/nb) × ceil(HW/hw_tile)；nb 经 TilingData 运行时标量下发
nb = tiling.nb
n_blks = (n_dim + nb - 1) // nb
for task_id in pl.range(core_id, n_blks * hw_blks, num_cores):
    n_blk = task_id // hw_blks
    hw_blk = task_id % hw_blks
    n0 = n_blk * nb
    m = pl.min(nb, n_dim - n0)                   # 尾块 task 的真实 item 数
    rows = m * c_dim
    x_slot = x_group.next()
    pl.set_validshape(x_slot, [rows, hw_eff])
    pl.load(x_slot, x2d, [n0 * c_dim, hw_off])   # 连续行块一次装载
    # VF 内 item→c→k 三层循环；窗口寻址 (ibase+max(0,c-r)) .. (ibase+min(C-1,c+r))
```

host 侧（模式选择 + NB 负载均衡；tile 几何类常量走 TilingKey 编译期分派，NB 是任务网格算术走 TilingData 运行时标量）：

```cpp
const bool wholeWindow = (depthRadius >= cSize - 1);
int64_t taskMode = (cSize > 0 && cSize <= cChunk)
                   ? (wholeWindow ? MODE_BATCH_WHOLE : MODE_BATCH)
                   : MODE_CHUNK;
int64_t itemsPerTask = 1;
if (taskMode != MODE_CHUNK) {
    const int64_t hwBlksForNb = Ops::Base::CeilDiv(hwSize, hwTile);
    itemsPerTask = std::min(cChunk / cSize,                        // 行容量上限: nb*C <= cChunk
                            std::max<int64_t>(1, (nSize * hwBlksForNb) / aivNum));
}                                                                  // 小 N 时退化为 1 保并行度
tilingData->nb = itemsPerTask;
```

## 性能与验证指标

比较 `Task Duration(us)`、task 总数与每核任务数、每 task 的 MME/tile 操作数、固定开销占比（trace 或生成物归因）。**待实测**：固定开销占比越高、批大小越大，摊薄收益预期越大；单 task 工作量饱满的形态收益趋零。建议以隔离对照臂（仅开关本项、其余不变）归因收益，不与其它优化合并报告。

验证方法：分解模式覆盖回归（C≤容量 / C>容量 / 尾块 task / 单 item / NB 退化为 1 / 空 tensor）+ 每个逻辑 work item 恰好处理一次的覆盖证明 + golden 容差对比；新旧输出逐位 diff（同窗口项同序时应逐位一致）可作为等价性补证。

## 技术限制与风险

- nb·C ≤ tile 行容量必须 host/kernel 双侧一致（用同一 TilingData 字段单源下发）；批大小上限 = 行容量 // C。
- NB 公式在任务已充足时取容量上限、任务不足时向 1 退化；采用前扫描少量有依据的 NB 值确定甜点。
- TilingKey 模式分派使编译实例数按 模式×档位×dtype 乘积增长，构建时间相应增长。
- 尾块 task（m < NB）的 validshape、行数与 store 范围必须按真实 item 数钳位。
- 装载从 [C, hw] 变为 [NB·C, hw]，单次 MTE 搬运变大；极小行距（HW 很小）下搬运形态需回归确认。
- 批内 VF 循环层数比 per-item 深（item→c→k 三层）；更深任务侧控制流的编译行为须单独核验。

## 参考资料

- 通用方法：一个 task 批处理多个独立 work item；本卡是其"相邻行块连续 → 窗口内部化 + NB 负载均衡 + 位级等价义务"的具体实例。
