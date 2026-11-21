# 技术卡片 nobound-01：无 bound 手段清单（占位）

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。
- **适用 bound**：无 bound
- **一句话**：无单一瓶颈但吞吐未达标时，开 PingPong、UnitFlag 减同步、Preload/提早发射填流水气泡。

> **占位卡**：手段清单已整理，`before/after` 代码与 msprof 实测数据**待增量补**。诊断特征见 `../no-bound.md`。
>
> 若诊断指向**阶段先后顺序不当**（写回等刚发出的计算 / 批量搬运发射把计算发射反压住），先看 [nobound-02](nobound-02-stage-order-rescheduling.md)：本卡是在既定顺序内填气泡，nobound-02 改的是顺序本身。

## 何时用（诊断特征）

- 各硬件单元利用率均不高、无明显单一瓶颈，但总吞吐未达预期；trace 有流水气泡与等待区间、搬移-计算 overlap 不足。

## 原理

无 bound 通常是流水编排问题：搬移与计算未重叠、同步点过多、气泡未填满。通过 DoubleBuffer/PingPong 重叠搬算、UnitFlag 替显式同步、Preload/提早发射填气泡即可提吞吐。

## 怎么改（手段清单，before/after 待补）

| 手段 | 操作 | 效果 |
| --- | --- | --- |
| 开 PingPong / DoubleBuffer | 搬入与计算交替，隐藏搬移延迟；控制每次搬移量使计算时间 ≥ 搬移时间 | 搬算重叠 |
| UnitFlag 减同步 | MMAD→Fixp 用 UnitFlag 同步替显式 Barrier | 减同步开销 |
| 减少单次搬运量 | 仅搬当前所需数据，边算边搬不积压 | 降气泡 |
| Preload | 当前计算未完就启动下一批搬移；预取提前发射；调 Preload 窗口匹配计算耗时 | 填 MTE2 空闲窗口 |
| 指令提早发射 | 搬移/计算指令不等前一步完全结束就发射 | 减流水气泡 |
| Vec 融合减重复搬运 | 同一数据一次搬入多次复用（配合 vec-01） | 消中间搬出搬入 |

## 性能数据

**待增量补**：需采 msprof `aiv_time` + 各泳道 overlap 度前后对比（重点看 DoubleBuffer 是否生效，见 `../metrics.md` §3）。

## 红线（Stop rules）

- PingPong 使能时实际可用 UB 需除以 2，注意容量。
- Preload 窗口过大可能撑爆 UB；窗口须匹配计算耗时。
- 归约依赖链型空泡不属"无 bound"，属 VEC 依赖链，用 vec-08。

## 关联

- 源：`../no-bound.md`。相邻卡片：vec-01（Vec 融合减搬运）、vec-08（依赖链空泡）、vec-10（小定长循环展开）。
