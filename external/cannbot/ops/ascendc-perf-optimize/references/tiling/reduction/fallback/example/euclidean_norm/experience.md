# EuclideanNorm — Tiling 实践案例

> 以 EuclideanNorm 为例，展示归约算子在 **多轴交错布局** 下如何超出 [通用五模板](../../tiling-flow.md)，
> 走 [多轴交错连续载入模板](../../tiling-flow.md#11-多轴交错连续载入模板细节)。
> 平台：Ascend 950 / arch35（dav-3510），CANN 9.0.0，coreNum=72，UB≈192KB。数据全部 msprof 上板实测。

---

## 1. 算子概述

EuclideanNorm 沿 reduce 轴计算：

```
y = sqrt( Σ x²  over reduce 轴 )
```

等价于两步流水：

| 步骤 | 操作 | 沿 R 轴类型 |
|------|------|-----------|
| ① | Square + ReduceSum → 平方和 | 归约 |
| ② | Sqrt → output | 逐元素（仅作用于输出） |

**关键特征**：平方和是**可交换归约**——`Σ x²` 与累加顺序无关。因此**不必按 R 的逻辑顺序累加**，可按内存物理连续顺序载入、在 UB 内任意次序合并。这一性质正是交错模板成立的前提。

---

## 2. 形状准备（合轴 / 交错不可合）

EuclideanNorm 常见于多轴归约（`axes=[1]` / `[1,3]` / `[1,3,5]` / `[1,3,5,7]`）。合轴后分两类：

| 原始 shape | axes | 合轴结果 | 归类 |
|-----------|------|---------|------|
| `[64000, 32]` | [0] | R=64000, A=32 → RA（单 R） | 五模板 |
| `[4, 3200, 24]` | [1] | A=4, R=3200, A0=24 → ARA（单 R） | 五模板 |
| `[2,320,14,320]` | [1,3] | `A-R-A-R`（2 个 R，合不掉） | **交错** |
| `[4,2,20,49,32,9,15]` | [1,3,5] | `A-R-A-R-A-R-A`（7 轴 tail-A） | **交错** |
| `[4,2,2,640,32,32,2,2]` | [1,3,5,7] | `A-R-A-R-A-R-A-R`（8 轴 tail-R） | **交错** |

**为什么交错合不掉**：标记 A/R 后 A 与 R 严格交替出现，相邻轴类型不同 → `FuseAxis` 合不了；各轴 size>1 → `DropSizeOneAxes` 去不掉。最终仍是多个交替 A/R，**无法压成单个 R**，五模板的 `(A1,R,A0)` 前提不成立。host tiling 用 `axisNum`（严格交替后的轴数）与 `isTailR`（轴数偶=尾轴 R）刻画这类布局（`PreprocessPattern`，`euclidean_norm_tiling_host.h:251`）。

---

## 3. 模板映射

### 3.1 单 R case → 五模板（复用）

RA / ARA 等可合成单 R 的 case 直接走五模板（EuclideanNorm 的中间量只有一份平方和，UB 预算按标准归约，分母不追加中间 buffer）。实测这类 case 三种规约引擎（二分树 / Welford / adapt）性能接近（±10%），默认 Welford 顺序累加即可。

### 3.2 交错 case → 交错连续载入模板

只有满足窄门控的交错病态 case 才走新模板，否则回退五模板/group：

```
交错前置判定（合轴后仍多个交替 A/R）:
│
├─ dtype∈{fp32,fp16,bf16} + 偶数轴≥6 ARAR...AR（tail-R）+ 内块连续 + 32B 对齐
│        + R1×dtypeSize≤16B(burst 病态) + rowsSize≥8 + 内块无需分块(chunkA0==A0) + groupCnt≤coreNum
│     → 交错-tail-R（fast，两段式 + workspace；coresPerGroup==1 免 workspace）  见 tiling-flow §11.1 / §12.1
│
├─ dtype∈{fp32,fp16,bf16} + 奇数轴≥7 ARAR...A（tail-A）+ 内块连续 + 32B 对齐
│        + slabCnt×2≥coreNum + innerRowElems≥512
│     → 交错-tail-A（fast-RARA，无 workspace）          见 tiling-flow §11.2 / §12.2
│
└─ 其余（int32 / 内块小 / burst 中等 / 内块需分块 / rows 太小 / 非 32B 对齐）
      → 回退五模板（normal/group）
```

映射到 host 源码：`TryComputeFastTiling`（`euclidean_norm_tiling_host.h:709`）/ `TryComputeFastRARATiling`（同文件:808）；主入口在 `ComputeEuclideanNormTiling:940-946` 先试两条快路径，不命中落 `ComputeAUbFactor`→…→`ShouldUseGroup` 五模板/group。

### 3.3 单 R / 规则布局 case → R 轴分核 group（BlockSplitR，最主力）

单 R / 规则布局（RA/ARA/ARAR 等）选完五模板的 UB 切分后，再判 A 轴分核 normal 还是 R 轴分核 group：

```
ShouldUseGroup（euclidean_norm_tiling_host.h:566）:
│
├─ aLoopCntTotal > coreNum/2   → normal（切 A 已够满核）
├─ rLoopCntTotal <= 1          → normal（R 未分块，无跨核 R 可切）
└─ 否则（大 R + 小 A）         → group / BlockSplitR
      Phase1 各核部分和 → workspace[rGroupCnt,aTotal] → SyncAll → Phase2 RA 合并 + sqrt
      ComputeGroupSplit: rGroupCnt / usedCoreNum=aLoopCntTotal×rGroupCnt / workspaceBytes=rGroupCnt×aTotal×4
```

**group 是 20 case 里最主力的路径——#1–9 共 9 个走它**（见 §5 选型表 "group" 行）：这些 case 都是**大 R + 小 A**
（如 #1 `[64000,32]` axes[0]：R=64000 超单核容量、A=32≪coreNum=72，切 A 只能用 32 核 → 跨核切 R 用满 61 核）。
group 加速温和（~1.0–1.1×），价值是**把大 R 摊到多核用满核**，而非病态点大赢。详见
[tiling-flow §3.2](../../tiling-flow.md#32-r-轴分核-groupblocksplitr-大-r--小-a-的一等分支) 与
[group 模板 .md](../../../../../../ascendc-performance-best-practices/references/reduce/templates/euclidean_norm_group_block_split_r.md)。

host 自检（g++ 纯 host，coreNum=72, UB=192KB）：

| case | shape / axes | templateType | rGroupCnt | usedCoreNum | workspaceBytes |
|------|-------------|:-----------:|:---------:|:-----------:|---------------:|
| #1 ra_group_0001 | `[64000,32]` [0] | **1（group）** | 61 | 61 | 61×32×4 = 7808 |
| #6 ara_group_0001 | `[4,3200,24]` [1] | **1（group）** | 21 | 42 | 21×96×4 = 8064 |
| #8 arar_group_0001 | `[2,320,14,320]` [1,3] | **1（group）** | 2 | 56 | 2×28×4 = 224 |

---

## 4. 交错模板为何赢：一个 case 的推演（#18）

`[4,2,2,640,32,32,2,2]` axes=[1,3,5,7] → `ARARARAR`，内块=末 4 轴 `[A0=32,R0=32,A1=2,R1=2]`：

- **走五模板会发生什么**：CopyIn 边搬边转置把散落的 R 聚拢，最内 burst = 最内 R1 长度 = 2 元素 = **8B**。MTE2 打满（ratio≈0.995），有效带宽仅 ~145GB/s → 1128µs。
- **走 tail-R fast**：按内存物理连续做大 burst（sliceStride=R0×A1×R1=128 元素/面），A/R 分离移到 UB 内两步 AR 规约（step1 规约 R1，step2 逐(a0,a1)标量累加 R0），跨核部分和经 workspace 两段式合并。→ **136µs，8.29×**。
- groupCnt=轴0×轴2=4×2=8≤72 ✓，chunkA0 算出 ==A0=32（内块整块放得下，无需分块）✓ → 命中。

---

## 5. 20 case 实测选型表

> 单位 µs；加速 = baseline / merged；merged 模板即最终选型（数据源 `PERF_TUNING_REPORT.md §2`）。

| # | case | 合轴/布局 | baseline | merged | 加速 | 选型 |
|---|------|----------|--------:|------:|-----:|------|
| 1 | ra_group_0001 | RA 大R | 14.3 | 13.3 | 1.08× | group（五模板） |
| 2 | ra_group_0002 | RA fp16 | 13.3 | 12.2 | 1.09× | group |
| 3 | ra_group_0003 | RA bf16 | 13.4 | 12.5 | 1.07× | group |
| 4 | ra_group_0004 | RA | 17.3 | 17.4 | 1.00× | group |
| 5 | r_group_0005 | R fp16 | 9.9 | 10.9 | 0.91× | group |
| 6 | ara_group_0001 | ARA | 11.4 | 10.5 | 1.08× | group |
| 7 | ara_group_0002 | ARA fp16 | 10.9 | 10.9 | 0.99× | group |
| 8 | arar_group_0001 | ARAR（4轴） | 16.7 | 16.8 | 1.00× | group |
| 9 | ararara_0002 | 7轴 tail-A，slabCnt=8 | 61.8 | 60.4 | 1.02× | group（slab 太少回退） |
| 10 | ararara_0003 | 7轴，内块 5MB | 63.9 | 59.7 | 1.07× | normal |
| **11** | **ararara_0004** | **7轴 tail-A，内块 17KB** | **171.0** | **123.2** | **1.39×** | **交错-tail-A（fast-RARA）** |
| 12 | ararara_0005 | 7轴 tail-A，内块 15KB | 31.9 | 31.1 | 1.03× | **交错-tail-A（fast-RARA）** |
| 13 | ararara_0006 | 7轴，内块 320B | 72.9 | 75.2 | 0.97× | group（回退，见 §6） |
| 14 | ararara_0007 | 7轴 | 120.5 | 124.4 | 0.97× | group |
| 15 | ararara_0008 | 7轴，内块 32B，slabCnt=18000 | 24.4 | 22.8 | 1.07× | normal（回退，见 §6） |
| 16 | ararara_0009 | 7轴 | 61.1 | 62.0 | 0.99× | normal |
| 17 | arararar_0001 | 8轴，burst 64B | 100.9 | 99.4 | 1.01× | normal（回退，见 §6） |
| **18** | **arararar_0002** | **8轴 tail-R，burst 8B** | **1128.5** | **136.1** | **8.29×** | **交错-tail-R（fast）** |
| 19 | arararar_1007 | 8轴 fp16 | 17.0 | 16.5 | 1.03× | normal（fp16，门控未命中回退†） |
| 20 | ra_3004 | RA int32 | 7.8 | 7.5 | 1.04× | normal |

**结论**：20 个 case 里 **仅 #11 / #12 / #18** 真正需要交错模板（#18 8.29× 是唯一大赢家）；**#1–9 共 9 个走 R 轴分核 group（BlockSplitR，最主力路径）**，#10/#15/#16/#17/#19/#20 走 A 轴分核 normal——都是现成模板，三种引擎差异 ≤10%。选型两步：先过交错前置判定（不命中回退），再由 `ShouldUseGroup` 判 A 轴分核 normal 还是 R 轴分核 group。

> ⚠️ **dtype 门控已更新**：上表为 fp32-only 时代的实测（#19 等 fp16 case 因当时门控仅 fp32 而回退 normal）。当前 host 门控已放宽至 fp32/fp16/bf16（载入即升 fp32），fp16/bf16 交错 case 现已具备命中 fast 的资格，但其最终选型仍受 burst 病态 / rowsSize / 32B 对齐 / chunkA0 等硬阈值约束，**fp16/bf16 fast 路径需重新上板验证**。

---

## 6. gate 误判教训（#13 / #15，仅上板才能发现）

初版 tail-A（RARA）gate 太宽——只校验 `slabCnt×2≥coreNum`、未卡内块每行大小——**误命中 #13/#15 导致严重退化**：

| case | 内块/行 | innerRowElems | 误命中 RARA | 回退后 | 根因 |
|------|--------:|-------------:|-----------:|------:|------|
| #13 ararara_0006 | 320B | 80 元素 | 92.6µs（慢 **28%**） | 75.2µs | 内块每行太小，连续载入无 MTE2 收益，反被每 slab 的 step2 屏障+小规约拖垮 |
| #15 ararara_0008 | 32B | 8 元素 | **229µs（慢 10×!）** | 22.8µs | 同上，且 slabCnt=18000 → 屏障开销爆炸 |

**修复**：tail-A gate 增加 `innerRowElems ≥ 512`（2KB/行）阈值 → #13/#15 干净回退，#11/#12 保留（`euclidean_norm_tiling_host.h` 的 opt2 版本尚未加此阈值，合并 demo 已补）。

**另一类回退（#17）**：8 轴 tail-R 但内块最内 R=16 → burst 64B（中等），且内块需分块（chunkA0=14<A0=32）。fast 的分块屏障+逐平面规约吃掉 MTE2 增益，实测 **-9%** → `chunkA0==A0`（及 `R1×dtypeSize≤16B`）不满足，回退 normal。

**核心判据（务必牢记）**：

> `mte2_ratio` 单指标会误导。真正决定赢面的是 **基线最内 burst 大小**（tail-R：内块最内 R 长度；≤~16B 病态才大赢）或 **内块每行大小**（tail-A：innerRowElems ≥ 512 才值得）。中等 burst（64B）或小内块（≤320B）反被模板固定开销反超。**必须上板实测校准阈值。**

---

## 7. 选型自检

- [ ] 已合轴并判断是否可合成单个 R
- [ ] 交错（多个交替 A/R）→ 先过前置判定，再考虑五模板
- [ ] dtype 是否 ∈ {fp32, fp16, bf16}（int32 一律回退五模板）
- [ ] 内块整块字节是否 32B 对齐（连续 DataCopy 安全）
- [ ] tail-R：偶数轴≥6 + 内块连续 + `R1×dtypeSize≤16B`(burst 病态) + `rowsSize≥8` + chunkA0==A0 + groupCnt≤coreNum
- [ ] tail-A：奇数轴≥7 + 内块连续 + slabCnt×2≥coreNum + **innerRowElems≥512**
- [ ] 基线最内 burst 是否病态小（tail-R）/ 内块每行是否≥2KB（tail-A）
- [ ] 任一门控不满足 → return false，回退五模板/group（回退安全，零退化）
- [ ] 单 R / 规则布局：`ShouldUseGroup`——大 R（rLoopCntTotal>1）+ 小 A（aLoopCntTotal≤coreNum/2）→ R 轴分核 group，否则 A 轴分核 normal
- [ ] group 命中时 workspaceBytes = rGroupCnt × aTotal × 4（fp32）已回填
- [ ] used_core_num ≥ 1

---

## 8. 关联文档

- [多轴交错前置判定 + 五模板决策树](../../tiling-flow.md#31-多轴交错前置判定step-05)
- [R 轴分核 group（BlockSplitR）一等分支 + UB/多核公式](../../tiling-flow.md#32-r-轴分核-groupblocksplitr-大-r--小-a-的一等分支)
- [交错模板细节与 UB 预算](../../tiling-flow.md#11-多轴交错连续载入模板细节)
- [交错模板门控判据与 mte2 教训](../../tiling-flow.md#13-交错模板门控判据与-mte2-会误导的教训)
- [TilingData 字段语义（含 fast/fast-RARA 字段）](../../tiling-fields.md#35-多轴交错模板字段交错-tail-r--交错-tail-a)
- [Softmax 案例（单 R 五模板复用）](../softmax/experience.md)
