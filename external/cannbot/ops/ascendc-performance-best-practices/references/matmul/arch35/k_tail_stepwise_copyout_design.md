# matmul K-Tail Stepwise Copyout（K 尾轮分步搬出）优化设计

## 1. 优化目标

解决 UnitFlag 开启后的遗留问题：**FIXPIPE 的搬出尾部延续到下一 Tile 时，下一 Tile 立即复用相同 L0C 地址仍可能产生读写冲突，造成 MMAD 流水空洞**。通过把 K 尾轮的输出在 N 方向近似对半拆成两个子块、调整循环顺序为「先算完并搬出第一个 N 子块，再算第二个」，使同一 L0C 子区域被下一 Tile 复用前获得更长的计算间隔，消除 FIXPIPE 读取与 MMAD 写入冲突。

适用场景（强门禁）：

- Cube Bound，且 MMAD 连续性对 kernel 总耗时有明显影响；
- **完整输出基本块无法放入半个 L0C**，需要在 K 尾轮沿 N 方向分步搬出；
- K 较大或单核计算轮数较多，FIXPIPE 引入的流水空洞会在连续 Tile 之间累积；
- A/B 为 MXFP4（或均为支持的 MXFP8 类型），Scale 为 FP8 E8M0，输出 FP16/BF16；
- 输入内轴满足硬件对齐：**MXFP4 对齐 256 B，MXFP8 对齐 128 B**。

## 2. 架构概览

### 2.1 问题背景（UnitFlag 的遗留冲突）

UnitFlag 让 FIXPIPE 在当前 Tile 最终累加期间提前搬出已完成的部分结果（512B 粒度）。但当输出块占满 L0C 时，FIXPIPE 搬出尾部与下一 Tile 的 MMAD 写入（复用相同 L0C 地址）在时间上重叠，产生 L0C 读写冲突 → MMAD 流水空洞。

### 2.2 未优化路径（`KL1 -> KL0 -> N Split` 顺序）

每个 KL0 轮次中切换 N 子块：第一个 N 子块触发 FIXPIPE 后，只间隔第二个 N 子块的一轮 MMAD 就可能进入下一 Tile——**较短的间隔无法稳定覆盖搬出尾部**。

```text
for kl1:                     # K 外层（L1 级）
    for kl0:                 # K 内层（L0 级）
        MMAD(N_sub0); MMAD(N_sub1)      # 每轮 K 内都在 N 子块间切换
    # 尾轮触发搬出后，间隔极短即进入下一 Tile → L0C 冲突
```

### 2.3 优化路径（`KL1 -> N Split -> KL0` 顺序）

保持 M 不变，沿 N 方向将 MMAD 近似对半拆成两个子块：

- **首个 KL1 轮次**：采用相同的 N 切分，分别初始化两个子块对应的 L0C 累加区域（`cmatrixInitVal` 分别置位）；
- **中间 KL1 轮次**：按完整 N 累加（不切分）；
- **最后一个 KL1 轮次（K 尾轮）**：再次按 N 切分，每个 N 子块先连续完成全部 KL0 累加，**在最终累加完成后立即通过 FIXPIPE 搬出到 GM**，再切换到下一个 N 子块。

```text
for kl1:
    if 首轮 or 尾轮:
        for n_sub in {N_sub0, N_sub1}:      # N 切分在外
            for kl0:                          # K 累加在内
                MMAD(n_sub)
            if 尾轮: FIXPIPE(n_sub) → GM      # 该子块全 K 累加完立即搬出
    else:
        for kl0: MMAD(完整 N)                 # 中间轮完整 N
```

第一个 N 子块完成全部 K 尾轮计算并触发搬出后，再执行第二个 N 子块的整段 KL0 计算——第二个子块的计算时间为搬出尾部提供了天然遮盖。无需拆分的输出 Tile 仍按完整 N 执行。

## 3. 关键参数配置

无新增 TilingData 字段；派生量在 Kernel 侧按 shape 计算：

| 量 | 计算 |
|----|------|
| `nSubNum` | K 尾轮 N 方向子块数（近似对半拆，典型为 2） |
| `nSubSize` | `DivCeil(baseN, nSubNum)` 对齐到 Cube 粒度 |
| `cmatrixInitVal` | 首个 KL1 轮次：两个 N 子块各自的第一次 MMAD 分别置 true（各自初始化 L0C 累加区域） |

## 4. 核心计算循环

```cpp
for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
    bool isKTail = (iter0 == kL1TileNum - 1);
    bool isKHead = (iter0 == 0);
    if (isKHead || isKTail) {
        // K 首/尾轮：N 切分在外，KL0 在内
        for (uint64_t nSub = 0; nSub < nSubNum; ++nSub) {
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                MmadParams para;
                para.m = actualCurM;
                para.n = nSubSize;                       // 子块 N
                para.cmatrixInitVal = (iter1 == 0);      // 每个子块独立初始化（首轮）
                if (isKTail && iter1 == kL0Iter - 1) {
                    para.unitFlag = FINAL_ACCUMULATION;  // 尾块：搬出最终结果
                } else {
                    para.unitFlag = NO_FINAL_ACCUMULATION;
                }
                Mad(...);
            }
            if (isKTail) {
                CopyL0C2GM(gmBlockC_[nSub 偏移], tensorL0C);  // 子块全 K 完成，立即搬出
            }
        }
    } else {
        // 中间轮：完整 N 累加
        for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            MmadParams para;
            para.n = actualCurN;                         // 完整 N
            para.cmatrixInitVal = (iter1 == 0);
            Mad(...);
        }
    }
}
```

## 5. 优化的关键修改点

| 修改项 | 优化前 | 优化后 |
|--------|--------|--------|
| K 尾轮循环顺序 | `KL1 -> KL0 -> N Split`（K 内切 N） | `KL1 -> N Split -> KL0`（N 外切，K 内连算） |
| 搬出时机 | 尾轮全部算完后统一搬出 | 每个 N 子块全 K 累加完成即搬出 |
| `cmatrixInitVal` | 每 Tile 首次置位 | 首 KL1 轮的两个 N 子块各自首次置位 |
| 首轮 N 切分 | 无 | 首 KL1 轮同样按 N 切分初始化两个累加区域 |

## 6. 注意事项与约束

1. **内轴对齐硬约束**：MXFP4 输入内轴须对齐 256 B、MXFP8 对齐 128 B，不满足时不可用本策略。
2. **仅尾轮/首轮切分**：中间 KL1 轮保持完整 N，避免额外 MTE1/L0 切分开销。
3. **与 UnitFlag 的配合**：本策略默认 UnitFlag 已开启（处理的就是 UnitFlag 的遗留冲突）；`FINAL_ACCUMULATION` 仅在尾轮每个 N 子块最后一次 MMAD 置位。
4. **输出 Tile 无需拆分时回退**：完整输出块能放入半个 L0C 的 Tile 保持完整 N 执行，不要为切而切。

## 7. 实施常见问题与解决方案

| # | 问题 | 根因 | 解决方案 |
|---|------|------|---------|
| 1 | 输出精度错误 | 首/尾轮子块 `cmatrixInitVal` 配置错（子块间互相污染或未初始化） | 首轮每个 N 子块第一次 MMAD 各自置 true；中间轮完整 N 从首轮子块结果续累加 |
| 2 | N 子块 GM 偏移错位 | 搬出偏移仍按完整 baseN 步长 | 子块搬出地址按 `nSub * nSubSize` 修正，外层 Tile 偏移不变 |
| 3 | 收益为负 | case 非 Cube Bound 或空洞不累积 | 按 §1 门禁复核；FIXPIPE Bound 主导时优先 UnitFlag 即可 |

## 8. 与其他优化的叠加关系

- **前置**：UnitFlag（[unitflag_design.md](unitflag_design.md)）——本策略处理其遗留的 L0C 复用冲突。
- **正交可叠加**：SWAT、pingpong、L1 Bank 冲突规避、Scale 合并载入（均作用于输入侧/调度层）。
- **互斥**：与「L0C Double Buffer」路径互为替代（能开 L0C 双缓冲时优先双缓冲；本策略面向 L0C 被完整输出块占满的场景）。

量化 matmul（MXFP8，SWAT 调度叠加 K-Tail Stepwise Copyout）实测：K 尾轮 MMAD 与 FIXPIPE 搬出的 L0C 读写冲突消除，尾轮计算与结果搬出并行度提升。
