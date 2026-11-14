# 跨核 R 分段前缀扫描 —— Core Partition Scan

代码：[dav310/cum_core_partition_scan.template](dav310/cum_core_partition_scan.template)
提取自：`ops-math/math/cumsum/op_kernel/arch35/cumsum_base/cumsum_core_sklansky.h` 及 `cumsum_core_ss_*.h`
实测数据来自 cann-bench Cummin case 13（`[1000003]` bf16 1D）。

## 解决的问题

当 **`M×N` 极小（切 A/行只能用满极少核）且 R 超长**（典型 1D `[1000003]`，`M=N=1`）时，
不管用流式还是 Kogge-Stone，都只有一个 work-item → 只有一个核在干活。
实测 2002 µs，而同规模的多行形态只要 30~60 µs。此时把**扫描轴 R 跨核切分**。

官方 cummin 也有同样的塌陷（`blockDim = 1`），所以这条路要参考 **cumsum** 而不是 cummin。

## Scan 特有：保序进位（与 Reduce 的可交换合并不同）

Reduce 的 group/BlockSplitR 因"部分和可交换"可任意分核合并；
**Scan 的跨核分段进位必须按 R 顺序传播**——前段的累积结果（carry）作为后段每个元素的初值
参与 `combine`。这是 Scan 相对 Reduce 的核心额外复杂度。

## 触发条件

参考 cumsum 的级联判据，**并行度不够就依次借轴：M → N → R**。
每一层都问同一个问题"M 够不够喂满核"，cumsum 在每个决策点都用 `M > coreNum/2`
而不是 `M >= coreNum`——半满的机器也比付出进位代价划算。

本实现用的判据（N == 1 分支内）：

```
M * 2 < coreNum  &&  R >= MIN_SEG * 2        // MIN_SEG = 8192
segs = min(coreNum / M, R / MIN_SEG)
segLen = ceil(R / segs);  segs = ceil(R / segLen)   // 回推一次，丢掉分不到数据的空核
```

段长下限保证每段仍摊得起固定的搬运/流水开销，也避免段数多到进位链变长。
"先由段数定段长、再用段长回推段数"这一步不能省，否则尾部会有拿不到数据的核。

## 执行流程：单次 launch + SyncAll

```
每核 blockIdx = w * segs + p：
  Phase A: ReduceRowRange(w, r0, rLen) 求段内 (min, argmin) → WriteSegTotal 写 workspace
  PipeBarrier<PIPE_ALL>() + SyncAll()    ← 核间全同步，轮空核也必须走到这里
  Phase B: ReadCarry 合并前序段进位 → ScanRowRange 带进位完整扫描并写出
```

**别让 Phase A 顺手把局部前缀写出去。** Phase A 只需要段内的 `(min, argmin)`，那是
**归约**而不是扫描——没有 log 因子，别复用扫描代码再丢掉输出（实测省 38%）。
让 Phase A 只读、Phase B 重扫一遍，多花的是算力而不是带宽。实测 `R=1000003` 从 2002 µs 降到 56 µs。

模板中已补充完整的 `ProcessBorrowR`、`ReduceRowRange`、`ReduceChunk`、`WriteSegTotal`、
`ReadCarry`、`InitCarry` 六个函数（类体内粘贴区），替代原先的 stub。

ReadCarry 里"读回前序 p 段结果并合并"是 p ≤ 段数（约 72）次标量比较，一次 DataCopyPad
搬进 UB 后在标量侧串行合并即可，开销可忽略。

## 三个必须踩的坑

**1. SyncAll 前要 `PipeBarrier<PIPE_ALL>()`**：保证段结果（MTE3 写出的 workspace）
已落 GM，否则其他核可能读到未刷新的数据。

**2. 轮空的核也必须走到 SyncAll**：判空写成 `bool active = ...` 而不是 `if (...) return`，
否则同步永远凑不齐（死锁）。轮空核在 SyncAll 之后才允许退出。

**3. 各段在 workspace 里独占一个 32B 槽位**：只用前 2 个 int32（值的位模式 + 下标），
撑到 32B 是为了让各核写各自的槽位时不会落进同一个 32B 块。`WS_SLOT_I32 = 8`。

## 三种进位实现的取舍

| 方式 | 屏障 | workspace | 适用 |
|------|------|-----------|------|
| **(A) 核内 SyncAll + 原地对数层 Sklansky** | 核内 `SyncAll` × `ceil(log2(segs))` | 无（进位直接读被扫核已写出的 y 行） | cumsum 生产实现；需要 `SetScheduleMode(1)`、blockDim 覆盖所有参与同步的核、host 预算 6 层 Sklansky 调度表（约 150 行 tiling 代码） |
| **(B) 两次 launch，进位在 kernel 内合成** | stream 顺序 | 小：`M × segs × 32B` | **推荐**。`<<<>>>` 单文件工程可直接用，无额外同步 API |
| **(C) 三次 launch，phase B 在 host 上算** | stream 顺序 + D2H/H2D | 同上 | 只有在 kernel 侧凑不出标量比较时才退到这里 |

(A) 还有个隐性约束：cumsum 为了让 Sklansky 调度表塞进 6 层，把核数**静默截断到 64**。
在 72 核的 A5 上这会白白丢掉 8 个核。方式 (B) 没这个限制。

## 两个实现细节

**各段在 workspace 里独占一个 32B 槽位。** 只用前 2 个 int32（值的位模式 + 下标），
撑到 32B 是为了让各核写各自的槽位时不会落进同一个 32B 块。

**16 位 dtype 在这条路上只能用提升后的 fp32 比较。** 段间进位要做标量比较，
而编译器不支持 16 位浮点的标量类型转换（`(float)bfloat16_t` 报 `not support bf16 type cast`）。
所以同一个算子里会出现两种比较类型：独占整行的路径用原生 16 位，借 R 的路径用 fp32。

## 关键技术

| 技术 | 要点 |
|------|------|
| **保序 exclusive 进位** | `carry[0]=identity`；`carry[p]=combine(totals[0..p-1])`。带索引时进位携带**全局** argmin/argmax，且并列语义与核内一致（last occurrence：更晚段/更晚元素胜） |
| **phase A 只读** | 段内合并结果就是"该段完整扫描后最后一个元素的值"，直接复用同一套扫描代码、跳过搬出即可，语义天然一致 |
| **段数回推** | `DoBlockSplit` 式的双 `CeilDiv`：先由长度定 factor，再由 factor 定 count，自动丢掉空核 |

## 适配

- Cumsum：`combine=Add`，去掉 idx 与并列语义（进位只带一个部分和）。
- Cummax：`combine=max`，identity `-inf`，并列 last-occurrence。
