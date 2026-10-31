# 技术卡片 scalar-01：Scalar bound 手段清单（占位）

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。 RegBase / MicroAPI / VF 部分仅适用于 arch35。
- **适用 bound**：Scalar / 小 case
- **一句话**：标量准备挡住计算时，减冗余 scalar、展开小循环、去封装开销、减标量-向量搬移。

> **占位卡**：手段清单已整理，`before/after` 代码与 msprof 实测数据**待增量补**（外部 VF 素材以向量为主，scalar 类需另采实测）。诊断特征见 `../scalar.md`。

## 何时用（诊断特征）

- `aiv_scalar_ratio` 高、SCALAR/SCALARLDST 泳道忙而 VECTOR 空闲；总计算量极小或 shape 很小。
- 指令发射率（IPC）低；SCALAR 与 VECTOR 间有不必要同步等待。

## 原理

小 case / scalar bound 下，标量地址计算、分支、封装开销成为关键路径。缩短标量关键路径、提升 IPC、减少 launch/封装开销即可提速。

## 怎么改（手段清单，before/after 待补）

| 手段 | 操作 | 效果 |
| --- | --- | --- |
| Scalar 优化 | 减少冗余 scalar 计算，合并条件分支 | 降低 scalar 指令数 |
| 循环展开 | 展开小循环减少分支代价 | 提升 IPC（配合 vec-10） |
| 减少循环轴 | 按 tiling 最简化循环轴 | 降低 scalar |
| 指令选择 | 用高效 scalar 指令替代低效序列 | 缩短关键路径 |
| 减少标量-向量转换 | 避免不必要的 Scalar ↔ Vector 搬移 | 减少搬移开销（配合 vec-04） |
| 性能友好 API | `set_flag`/`wait_flag` 代替 Queue、`LocalTensor` 代替 TBuf、去除多余 TPipe | 减少封装带来的 scalar |

## 性能数据

**待增量补**：需在 scalar bound 算子上采 msprof `aiv_time` + `aiv_scalar_ratio` 前后对比。

## 红线（Stop rules）

- 循环展开/减循环轴受 tiling 约束，改 kernel 须与 tiling 联动（见 `../../tiling/`）。
- 标量向量化（round-trip → 批量）见 vec-04，避免与本卡重复实现。

## 关联

- 源：`../scalar.md`。相邻卡片：vec-04（scalar round-trip 向量化）、vec-09（外层 for 下沉）、vec-10（小循环展开）。
