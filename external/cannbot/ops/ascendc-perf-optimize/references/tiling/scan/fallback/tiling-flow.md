# Scan / Cum 族 — 通用 Tiling 流程

> 前缀扫描类算子的模板选型决策树、UB 预算与**保序进位**多核切分。所有 Scan 算子共享同一套模板语言；不同算子仅在 combine/identity 与是否带索引上有差异。

---

## 1. 核心概念

| 术语 | 含义 |
|------|------|
| **合轴** | 将 shape + 扫描轴 dim 合并为 `(M, R, N)` 三元组 |
| **M** | 扫描轴 R **之前**所有维度乘积 |
| **R** | 扫描轴长度（前缀扫描沿此轴） |
| **N** | 扫描轴 R **之后**所有维度乘积（dim=-1 → N=1；dim=0 → M=1） |
| **combine** | 结合律二元运算：sum→+ / prod→× / min / max，带单位元 identity |
| **流式模板** | 沿 R 逐步串行更新常驻 running 状态；天然支持索引（argmin/argmax）与任意 combine |
| **oneway Sklansky** | R 全量入 UB，`ceil(log2(R))` 层对数深度并行前缀（无索引变体） |
| **twoway（折叠）** | 窄 N 时把长 R 折叠进空闲向量道，段内→段间两级 Sklansky |
| **ub chunk** | R 超单次 UB，切多个 chunk 串行，chunk 间用二进制进位 memo 合并 |
| **core partition（借 R）** | M×N 极小 + R 超长时**跨核切 R**，各核局部前缀 + **保序进位** |

---

## 2. 合轴（Step 0）

```
给定: shape, dim
1. dim 归一化（负数 += ndim）
2. M = ∏ shape[0:dim]；R = shape[dim]；N = ∏ shape[dim+1:]
输出: (M, R, N)
```

多维输入沿单一 dim 扫描：dim 前后各自乘积压成 M、N。扫描轴不与其他轴合并（前缀有序，不可交换）。

---

## 3. 模板选型决策树（Step 1–3）

```
输入: (M, R, N), dtype, op_type, ub_size, core_num

┌─ 带索引？(Cummin / Cummax，需输出 argmin/argmax)
│    ├─ N > 1（dim 非最后一维）
│    │    → 【tile 常驻扫描】整段 R 一次搬进 UB 按 [R][lane] 布局，
│    │      running-(val,idx) 常驻寄存器，循环体内零 DMA。lane 维用 N（N 够宽）
│    │      或借 M 轴折叠（N 窄）。**不要退化成逐 r 搬入搬出，实测慢 8~50 倍**
│    │      并且 **x/val/out 必须各开两份做 ping-pong**：块间 carry 串行、
│    │      计算无法重叠，但「搬入下一块 / 算当前块 / 搬出上一块」可以。
│    │      不做的话四条流水占比之和只有 101%（全串行），实测慢 12~31%。UB 预算见 §5.2
│    │      若 lane 数用不满（如 M=1 且 N 只切得出十几个元素宽的 tile）→ 转 §4.1
│    └─ N == 1（dim=-1，扫描轴在 GM 上连续）→ 见 §3.2
│      并行度判定见 §3.1；若 M×N 极小 + R 超长 → 转 §4 跨核借 R
│
└─ 无索引 (Cumsum / Cumprod / LogCumsumExp)
     ├─ N 宽（alignN > VREG/4=16 元素 fp32）
     │    → 【oneway Sklansky】沿连续 R 行整行向量 combine；R 全载 UB → key 类 1001
     │      R 超 UB → 叠加 ub chunk（key 类 1011）
     └─ N 窄（alignN ≤ VREG/4，且折叠 foldCount>1）
          → 【twoway 折叠】长 R 折叠进向量道，段内→段间两级 Sklansky；R 全载 → 1002；超 UB → 1012
     两分支再过 §3.1 并行度判定：M 小 + R 超长 → 跨核借 R（1101/1102/1111/1112）
```

### 3.1 并行度判定（切行是否用满核）

```
流式/oneway 先按切 M（行）或借 N 估 work-item 数 items：
  items ≥ core_num              → 直接切行，够满核
  items < core_num（不足）      → 先缩小分块补偿（减小 nTileLen/mBlk 提高 items，对齐 VL）
  仍 items ≪ core_num 且 R 超长 → 转 §4 跨核借 R（core partition）
```

Cumsum 生产 tiling 的等价判据（以促进 fp32 尺寸）：`M > core_num/2` 时切 M 即够（不借 R）；否则借 N 再借 R。

### 3.2 N == 1 的三级决策（带索引）

扫描轴连续，整行一次搬运只需 1 个描述符。三个正交的决策：

```
① 一批搬几行
   R ≤ rChunk（整行装得下）→ 行批处理：一次 DataCopyPad(blockCount=k) 搬 k 行，
                              k 行只付一对 flag，并给 x/val/out 各开两份做双缓冲
   R > rChunk               → 逐行分块，块间显式传进位

② UB 内用什么算法扫
   R > 2*VLC → lane 并行顺序扫描（指令条数与 log2(R) 脱钩）
   否则      → 行内 Kogge-Stone（log2(R) 级，R 小时它那趟固定的 log2(VLC) 更划算）

③ 比较类型取什么 —— 取决于 ② 选了哪个算法，不能一概而论
   走 lane 并行     → 一律用促进类型 P（4 字节）。16 位的 Gather/Scatter
                      没有全速路径，实测同元素数下 fp16 比 fp32 慢 1.6~1.9 倍；
                      升 fp32 后 gather 走 32 位快速路径，反超 1.6 倍
   走 Kogge-Stone   → 16 位可原生比较，每 vreg 128 lane，实测比升 fp32 快 1.5 倍；
                      但索引仍是 int32，需 UnPack<LOWEST/HIGHEST> 拆掩码驱动索引侧 Select
   需标量回读进位   → 必须用 P（编译器不支持 16 位标量比较）

   ↑ 索引宽度跟着比较类型走（等宽才不用拆掩码）。走 lane 并行时比较类型恒为 4 字节，
     故索引恒为 int32、出口一步 Cast 到 int64；int16→int64 一步 Cast 编译得过但结果错
```

**行批处理的 k**：`k = UB_BUDGET / perRow`，再夹到 `ceil(M/core_num)`（保证每核至少一批）。
`perRow = 2*(xStride*eb + (vStride+guard)*cb + oStride*8) + (vStride+guard)*cb`
（x/val/out 双缓冲各两份，idx 暂存一份）。`k == 1` 也走这条路——批处理没了，
但双缓冲带来的跨行重叠仍在，R 大时正是靠它。

**lane 并行的段长**（同时是 lane 间的 UB 地址步长，**host 与 kernel 必须用同一个公式**，
否则 UB 预算算错）：

```
S = ceil(R / VLC)
if (S * sizeof(C)) % 16 == 0:  S |= 1      # 步长是 16B 整数倍会撞 bank，实测慢 5.5 倍
数据区要按 VLC*S（而非 R）备量，尾部补齐成幺元
```

### 模板速查

| 模板 | 带索引 | N | R vs UB | M×N 并行度 | 关键策略 |
|------|:---:|---|---------|-----------|---------|
| **tile 常驻 + ping-pong** | ✅ | **>1** | 切 rChunk，块间 UB carry | 够满核 | `[R][lane]` 布局，running 常驻寄存器，三条流水重叠 |
| **lane 并行顺序扫描** | ✅ | **==1** | 整行入 UB | 够满核 | gather 虚拟转置，指令条数与 log2(R) 脱钩 |
| 行内 Kogge-Stone | ✅ | ==1 | 整行入 UB | 够满核 | log2(R) 级；R 小到一两个 chunk 时比 lane 并行划算 |
| 流式（streaming） | ✅ | 任意 | 逐 r 切片（不必全载） | 够满核 | 常驻 running-(val,idx)，仅作参考实现，**勿用于性能路径** |
| oneway Sklansky | ❌ | 宽 | 全载 | 够满核 | ceil(log2 R) 层整行 combine |
| twoway 折叠 | ❌ | 窄 | 全载 | 够满核 | 长 R 折叠进向量道，两级扫描 |
| +ub chunk | 视内层 | — | 超 UB | 够满核 | chunk 串行 + 二进制进位 memo |
| **core partition（借 R）** | 视内层 | — | 超长 | **不足** | 跨核切 R + 保序进位 |

### 3.3 Cummin/Cummax 带索引的 mode 选型（N==1 路径完整决策）

上述 §3.2 的三级决策（批几行 / 用什么算法扫 / 比较类型取什么）落到代码里是一个 **mode 字段**，
kernel 侧 `Process()` 按 mode 分发到不同路径。下面是以 Cummin 为蓝本的完整选型逻辑
（Cummax 只需把 combine 的 LE 换 GE、identity 换 -inf）：

```
输入: (M, R, N), elemBytes, ub_size, core_num
UB_BUDGET = ub_size - 8KB   // 留余量给框架与对齐

if N == 1:
    rChunkKs = UB_BUDGET / (elemBytes + 4 + 8)   // Kogge-Stone 单缓冲的 rChunk
    rChunkKs = min(rChunkKs, 8192)                // 上限：再大也不划算
    rChunkKs = min(rChunkKs, R)

    # ---- mode 5：行批处理 + 双缓冲（R 装得下 UB 时首选）----
    if R <= rChunkKs:
        cBytes = 4                          // 比较类型一律 4 字节（见 §3.2 ③）
        guardC = 256 / cBytes               // 哨兵区 = 1 vreg
        gsS = (R > 2*guardC) ? LaneSegLen(R, guardC, cBytes) : 0   // R 大走 lane 并行，小走 KS
        vStride = guardC + CeilAlign(gsS>0 ? guardC*gsS : R, 32/cBytes)
        xStride = CeilAlign(R, 32/elemBytes)
        oStride = CeilAlign(R, 4)           // int64 输出区按 4 元素对齐
        perRow = 2*(xStride*eb + (vStride+guardC)*cb + oStride*8) + (vStride+guardC)*cb
        k = UB_BUDGET / perRow              // 行批上限
        k = min(k, ceil(M / core_num))      // 每核至少一批
        if k >= 1:
            → mode 5, mFold=k, blockDim=min(M, core_num)
            # mode 5 内部：gsS>0 时走 lane 并行（GsScanPair 两行交错），gsS==0 时走 Kogge-Stone

    # ---- mode 3/4：跨核借 R（行数喂不饱核且 R 足够长）----
    MIN_SEG = 8192
    if M * 2 < core_num and R >= MIN_SEG * 2:
        segs = core_num / M
        segs = min(segs, R / MIN_SEG)
        if segs > 1:
            segLen = ceil(R / segs)
            segs = ceil(R / segLen)          // 回推丢掉空核
            → mode 3(Phase A) + mode 4(Phase B), wsSegs=segs, blockDim=M*segs
            # 单次 launch + SyncAll 时 mode 3/4 合一

    # ---- mode 1：逐行 Kogge-Stone（行批装不下时的保底）----
    → mode 1, rChunk=rChunkKs, blockDim=min(M, core_num)

elif N > 1:
    # ---- mode 0：tile 常驻 + ping-pong ----
    if N * elemBytes >= 32:                  // N 够宽
        nLen = 按 core_num 切 N 到目标 tile 数，CeilAlign 到 32B/eb
    else:                                    // N 窄，借 M 轴折叠
        nLen = N
        mFold = 目标 lane 宽 ~128，保证 rowGroups >= core_num
    slotT = CeilAlign(nLen * eb, 32) / eb
    computeLen = mFold * slotT
    bytesPerR = computeLen * (2*eb + 2*eb + 4 + 2*8)    // x双份 val双份 int32扫描区 int64输出区双份
    rChunk = UB_BUDGET / bytesPerR
    → mode 0, blockDim=min(totalWork, core_num)
```

**mode 速查表**：

| mode | 触发条件 | 路径 | 模板 |
|------|---------|------|------|
| 0 | N > 1 | tile 常驻 + ping-pong | cum_tile_resident_scan |
| 1 | N==1, R > rChunkKs（行装不下 UB） | 逐行 Kogge-Stone，块间显式传 carry | cum_row_kogge_stone |
| 3+4 | N==1, M*2 < core_num, R >= 16384 | 跨核借 R + SyncAll + 向量化 ReadCarry | cum_core_partition_scan |
| 5 | N==1, R <= rChunkKs | 行批处理 + 双缓冲；内部 gsS>0 走 lane 并行(GsScanPair)，gsS==0 走 Kogge-Stone | cum_row_kogge_stone (行批扩展) |

**mode 5 内部的算法选择**（gsS 字段）：
- `R > 2 * guardC`（guardC = 64 for fp32/int32）→ `gsS = LaneSegLen(R, guardC, 4)`，走 lane 并行顺序扫描
  + 两行交错（`LaneParallelScanPair`），奇数尾行退回单行（`LaneParallelScan`）
- `R <= 2 * guardC` → `gsS = 0`，走 Kogge-Stone（log2(R) 级，R 小时那趟固定的 log2(VLC) 更划算）

**host 与 kernel 的段长公式必须一致**：`LaneSegLen` / `GsSegLen` 在 host 侧算 UB 预算、
kernel 侧算 gather 步长时都要调同一个函数，否则 UB 预算算错。
host 侧函数名通常加 `Host` 后缀（如 `GsSegLenHost`），逻辑与 kernel 侧完全相同。

**workspace 预算**（mode 3/4）：`workspaceBytes = M * wsSegs * WS_SLOT_I32 * sizeof(int32_t)`，
其中 `WS_SLOT_I32 = 8`（每段独占 32B 槽位，只用前 2 个 int32 存 val+idx）。

---

## 4. 跨核 R 分段的保序进位（Scan 特有）

> **一等分支，不是可选增强。** 与 Reduce 的 group/BlockSplitR（可交换合并部分和）不同，Scan 跨核切 R 必须**保序传播进位**：前段的累积结果作为后段每个元素的初值参与 combine。

**触发**：`M×N` 极小（切行只能用满极少核，`items ≪ core_num`）且 R 超长（如 R≥4096）。典型 1D `[1000003]`（M=N=1）。

```
分段: R 切 rBorrow 段（≈ core_num），段 p 覆盖 [pStart,pEnd)，partLen=ceil(R/rBorrow)
Phase A（每核）: 段 p 局部前缀扫描 → 段内 y/idx（全局 r 索引），段末 (val,idx) 记为 totals[p]
Phase B: 对 totals[0..rBorrow-1] 做 exclusive 前缀 combine（保序）→ carry[p]
Phase C（每核）: 段 p 用 carry[p] 与段内局部前缀 combine → 最终 y/idx（p=0 无进位）
```

三种进位实现，推荐按 (B) → (C) 的顺序演进：

- **(A) 核内 SyncAll + 原地对数层 Sklansky**（cumsum 生产实现，`ceil(log2(rBorrow))` 层，
  无 workspace，进位读被扫核输出行）。需要 `SetScheduleMode(1)` 与 host 预算多层调度表，复杂。
- **(B) host 编排 2 次 launch**（先跑通用这个）：phase A 只读求段结果写 workspace，
  phase B 读回前序段进位重扫并写出，stream 顺序即跨核屏障，风险低。
- **(C) 单次 launch + `AscendC::SyncAll()`**（最终形态）：`kernel launch 的固定开销约 9 µs`，
  在小用例上和整个搬运时间同量级，值得用核间全同步换掉。

**(B) 的两个必须**：phase A 只需要段内的 `(min, argmin)`，那是**归约**不是扫描，
没有 log 因子，别复用扫描代码再丢掉输出（实测省 38%）；也别让 phase A 顺手写出局部前缀，
否则 phase B 要回读整个输出再写一遍，访存量从 `x + 输出` 涨到 `x + 2×输出`。

**(C) 的两个约束**：`SyncAll` 前要 `PipeBarrier<PIPE_ALL>()` 保证段结果已落 GM；
**轮空的核也必须走到 `SyncAll`**（提前 `return` 会死锁），所以判空要写成
`bool active = ...` 而不是 `if (...) return`。

`usedCoreNum = rBorrow`；`workspaceBytes = rBorrow × W × (sizeof(P)+4)`（val + int32 idx）。

### 4.1 推广到 `N > 1`（尚未实现，已定方案）

以上只覆盖 `N == 1`。`dim=0` 且 N 中等时（实测反例 `M=1, R=1009, N=1021` fp16）
tile 常驻那条路会卡住：`nLen` 被 32B 对齐压到 **16 个元素**（DMA burst 只有 32B），
切出 64 个 n-tile 喂 64/72 核，一个 vreg 只用 **16/64 条 lane**，沿 R 的串行链长 1009 步。
加了双缓冲后仍是 `vec 62%`，是**延迟受限**；`nLen` 调大也没用（N 固定，burst 与核数直接对冲）。

做法：先把 `nLen` 放大到 **128**（256B 的 burst）得到 8 个 n-tile，再把 R 切成
`core_num/8 = 9` 段凑满 72 核，串行链从 1009 步降到 113 步。每核领一个连续的
`(rSeg × nLen)` 切片，进位是长度 `nLen` 的向量。

> **更正**：先前记的「不要各核自己合并、必须三阶段」是错的。合并的是**向量**而非标量，
> `segs=9` 时只有 18 条向量 combine（约 100 条指令）；读放大
> `segs² × nLen × 8B = 83KB`，完全可忽略。**两阶段（各核归约 → SyncAll → 各核自己合并前序段
> 并重扫）就够了**。只有 `segs` 很大时才值得引入「派一个核算完前缀再写回」的三阶段。

---

## 5. UB 预算

### 5.1 流式模板（含索引）

```
work-item 向量宽 W = mBlk × nTileLen（目标 W_TARGET≈1024，按 VL=64 对齐）
每 work-item UB 常驻:
  xIn      = W × sizeof(T)        # 当前 r 切片
  curMin   = W × sizeof(P)        # 常驻 running value（promote）
  curIdx   = W × 4                # 常驻 running idx（int32）
  valOut   = W × sizeof(T)        # 输出降回缓冲（T==P 时可复用 curMin，省一份）
约束: 上述之和 ≤ ub_size（W_TARGET=1024 时约 12KB，远低于 UB，可增大 W 或多缓冲）
```

### 5.2 tile 常驻 + ping-pong（含索引，`N > 1`）

```
slotT      = CeilAlign(nCnt * eb, 32) / eb        # 每行槽位元素数（32B 对齐）
computeLen = mFold * slotT                        # 每个 r 步的 lane 数
bytesPerR  = computeLen * (2*eb + 2*eb + 4 + 2*8)
                           x双份  val双份 int32扫描区 int64输出区双份
rChunk     = UB_BUDGET / bytesPerR
```

相对单缓冲正好翻倍，于是 `rChunk` 减半、块数加倍——**这正是流水需要的，不必回调**。
块数至少 2 才有重叠（R 一块装得下时双缓冲自然失效）。

> ⚠️ **不要为省 UB 让两个缓冲区叠用。** 单缓冲时常把 int32 扫描区叠在 int64 输出区的
> 高半段上，靠 Cast「读指针 4B/元素、写指针 8B/元素」的错位就地展宽（完全成立）。
> 加了 ping-pong 后这会在 MTE3 搬出该输出区的同时被下一块的扫描写坏，**必须拆成独立分配**。
> 这个坑不报错，只出错误结果——所以上面公式里 int32 扫描区那项是单独的 `+4`。

### 5.3 oneway/twoway（无索引，参考 cumsum）

```
oneway: R 全载 [rFactor, alignN×mFold] fp32 + 输入×2/输出×2；rFactor 按 LastPow2 约束
ub chunk: 预留 memo（约 8KB）+ 进位数据（约 512B）+ 进位索引（约 8KB），再算 chunk 的 rBlockFactor
twoway: 原折叠 buf + fp32 折叠 buf + 展开 fp32 buf + 1 VL + 二级扫描数据/索引 ≤ ub_size
        窄 N 门控: alignN ≤ VREG/4；foldCount>1；fp16/bf16 foldCount≤4；foldLen≥256（向上取 2 的幂）
```

---

## 6. 多核切分

| 模板 | 切分轴 | 分配方式 |
|------|--------|---------|
| 流式 | M×N 的 work-item（mBlk 行 × nTileLen 列） | `used = min(items, core_num)`；items 不足时缩小分块补偿 |
| oneway/twoway | M（借 N） | 切 M；`M ≥ core_num` 只切 M，否则借 N |
| +ub chunk | M（借 N） | 同上，R 方向单核串行多 chunk |
| **core partition（借 R）** | **R（跨核切 R）** | `usedCoreNum = rBorrow`，保序进位（§4） |

`used_core_num = ceil(total_work / per_core_work)`，保证 ≥ 1。

---

## 7. 对齐与并列语义

| 场景 | 规则 |
|------|------|
| N/R 方向 DataCopyPad | 有效长度搬运（`PaddingMode::Compact` 打包成连续 W） |
| 向量宽 W / nTileLen | 按 `VL=64`（256B/4B）对齐 |
| 并列 tie（Cummin/Cummax） | **last occurrence**（对齐 torch）：min 直接用 `CMPMODE::LE`（`x <= cur` 即取更晚的），max 用 `GE`。写成 `GT`/`LT` 再回头修正只会多一条指令 |
| NaN | **必须显式判**，不能指望 `Compare` 的默认行为。见 §8.1 |

---

## 8. 常见踩坑

### 8.1 NaN 必须显式粘滞（精度失败的头号原因）

cann-bench 的 `value_range=["nan","nan"]` 生成的是 **NaN 与实数混合**的输入。
只写一路 `Compare<LE>(x, cur)` 不够：`(实数 <= NaN)` 为 false，会把已经取到的 NaN
"恢复"成后面的实数。正解是再加一路判 NaN 并 `Or` 成一个掩码：

```
leMask  = (x <= cur)        # LE 直接实现 last-occurrence
nanMask = (x != x)          # 判 NaN
takeB   = leMask | nanMask  # 合并成一个掩码，值和索引各省一条 Select
cur     = takeB ? x : cur
idx     = takeB ? r : idx
```

`Or` 合并掩码还有个附带好处：16 位比较配 32 位索引时只需拆一次掩码而不是两次。

1. **并列语义**：`LE`/`GE` 配合上面的掩码合并直接得到 torch 的 last occurrence。
2. **一核多 work-item 的 WAR 屏障**：本 item 末次 CopyOut(MTE3) 读 running-(min,idx)，下一 item InitRunning(V) 覆写；`MTE3_MTE2` 不约束 MTE3→V，须补 `MTE3→V` 屏障，否则每 work-item 最后一个 r 的 idx 被覆写成 0（值走独立缓冲正确，掩盖问题，仅"一核多 item + 大 R"暴露）。详见 best-practices 的 [usage_guide.md §4](../../../../../../ascendc-performance-best-practices/references/scan/templates/usage_guide.md)。
3. **1D 超长 R 的性能陷阱**：流式模板对 `M=N=1` 只出 1 个 work-item（W=1，单核，逐 r 微 DMA），实测 100 万点约 230µs。识别 `items ≪ core_num` 并转 §4 跨核借 R。
