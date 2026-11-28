---
type: "PyPTO Performance Optimization Card"
title: "SWAT 调度：Serpentine 行窗口 + 自适应尾块（M×N 分核调度）"
description: "面向 matmul 的 M×N 逻辑分块到有限 AIC 核的调度：机制 A 用按行窗口的蛇形（boustrophedon）遍历改善 B 侧访存复用；机制 B 把尾部薄片在 M/N 方向重新等分（不改 tile 数）；机制 C 在末轮把剩余原 tile 二维再切分，救援空闲核。"
status: "stable"
tags: ["pypto-pro", "cube", "matmul", "scheduling", "load-balance"]
item_id: "cube-11"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/gemm.md#count-the-complete-output-grid-before-assigning-workers"
      scope: "在完整输出 tile 网格上比较改变访问顺序、尾部工作分配与最忙 worker 路径。"
      boundary: "通用网格契约不提供 SWAT 蛇形映射或变尺寸尾块实现；三机制分别核验，未验证的 B/C 不能随 A 一并启用。"
bound_hint: "MAC、MTE2"
applicability: "机制 A：右矩阵 B 是访存大头且其工作集相对可共享 L2 偏大、B 侧 DRAM 流量高，固定列的 B tile 被多个 M 行重复读且复用距离超 L2；机制 B：M/N 尾薄片拖慢所在行窗口；机制 C：tile 数不整除核数致末轮空闲核。需先由 tiling 建模确认是调度/负载而非纯计算 bound"
target_api_gate: "仅限 Ascend 950PR 或 950DT；机制 A 仅用 pl.get_block_idx/get_block_num 与整数映射即可表达；机制 B/C 需要在 tiling 侧产出变尺寸 tile 或二维子块，DSL 是否支持同一 kernel 内混合 tile 尺寸须按目标版本复核，未验证前只用机制 A"
---

# 技术卡片 cube-11：SWAT 调度（Serpentine Window + Adaptive Tail）

- **适用 bound**：MAC/MTE2（核利用率不足或尾轮空闲核造成的长尾）
- **一句话**：matmul 的性能不只取决于单核 Tiling，还取决于 M×N 分块怎么落到核上；SWAT 用行窗口蛇形遍历 + 尾块自适应，处理"尾薄片拖慢窗口"和"末轮空闲核"两类调度损失。

## 何时用（诊断特征）

按机制分别判断；三者可叠加，但必须先分清是哪一类损失。

- **机制 A（行窗口蛇形）**
  - 右矩阵 B 是访存大头，且 B 的工作集**相对可共享的 L2 偏大**（放不下，或会被同时进行的其它访问挤出）；profiler 显示 B 侧 DRAM 流量偏高。
  - 固定列 `n` 的 B tile 会被多个 M 行重复读，而当前行主序下这些复用在时间上被"整行/整列"隔开，复用距离超过 L2 能覆盖的范围。
  - `mCnt`、`nCnt` 都足够大，窗口（如 4 行）有腾挪空间。
- **机制 B（尾部基块合并）**：`M % baseM` 或 `N % baseN` 产生明显小于基本块的边缘薄片，它与满块同处一个行窗口，窗口关键路径被满块拖到 `baseM`。
- **机制 C（尾块二维再切分）**：`totalTileCnt = mCnt × nCnt` 不是核数整数倍，末轮只有少数核有活；各核结束时间离散、最慢核/平均核比明显。

## 何时不适用

- MN 欠并行（`totalTileCnt < aicNum`，很多核无 tile 可分）——这不是 SWAT 的域，应先走 K 维切分的调度（见 cube-03 StreamK/Split-K）。
- **机制 A 无收益**：B 小、或关键路径的访存主要是 A、或 B 工作集能稳定驻留 L2——此时行主序与蛇形两侧的 B 都命中 L2，蛇形反而白牺牲 A 复用。
- 计算本身是纯 Cube bound、核间已均衡且无尾薄片时，任何调度改动都不会带来收益。
- 末轮空闲核很少（`totalTileCnt % aicNum` 接近 0）或每 tile 时间很短时，机制 C 收益低于噪声。

## 原理

SWAT 由三个相互约束的机制组成，共享行窗口长度（如 4）；三者分别针对 **B 侧 L2 复用、尾薄片关键路径、末轮空闲核**。

- **机制 A（Serpentine 行窗口）**：把核按行窗口分组，窗口内沿 N 走蛇形（偶行左→右、奇行右→左）。
- **机制 B（尾部基块合并）**：若 M 方向出现薄片，把最后若干 tile 的 M 尺寸重新等分成更均匀的尺寸（tile 数量不变）。
- **机制 C（尾块二维再切分）**：末轮只有 `endBlockIdx+1` 个核有活时，把该轮所有原 tile 在 M/N 方向切成 `mTailTile × nTailTile` 个子块，把活跃核数扩展到 `tailOriCnt × totalTailTile`。

### 机制 A 为什么有收益

matmul `C[M,N] = A[M,K] × B[K,N]`，算一个输出 tile `(mi, ni)` 需要 **A-tile(mi) + B-tile(ni)**；固定列 `ni`、遍历 M 时共用同一 B-tile，这就是 **B tile 复用**的来源。

- **行主序（before）**：每轮铺满一整 M 行。同一列 `ni` 的 B-tile 被 `m0..m_{M-1}` 各用一次，但两次使用之间隔着一整行（复用距离 ≈ 网格一整行的跨度）。B 大时等下一次用它已被挤出 L2，只能再从 DRAM 读 → **B 的 DRAM 重复读随 M 行数放大**。
- **蛇形行窗口（after）**：把 M 行按窗口分组、窗口内沿 N 推进，同一 B-tile 在窗口内被若干 M 行**在相邻时间**复用，复用距离从"跨整行"缩到"窗口内"；相邻窗口用蛇形（奇窗口反向）在边界列上共享 B-tile，进一步减少边界重读。B-tile 更可能驻留 L2 → **B 侧 DRAM 流量下降、MTE2 更接近流式带宽**。
- **代价——为什么收益是条件性的**：行主序下同一轮多个核共享同一个 A-tile；窗口化后一轮覆盖窗口内多行，A-tile 种类变多，**A 的复用被牺牲**。因此只有当 **B 侧访存是大头、且 B 工作集 > 可共享 L2** 时，B 省下的 DRAM 流量才大于 A 侧损失，才有净收益；B 落在 L2 内时两边都命中，收益趋近 0（实测中性的原因）。

机制 A 是纯整数重映射，机制 B/C 需要 tiling 侧提供变尺寸 tile 或原子块映射。

### 调度示意（图示）

以下为示意网格（`Rr` = 第 r 轮处理），实际行/列/轮次由 `mCnt/nCnt`、`MAIN_WINDOW`、核数决定。

**机制 A：Serpentine 行窗口**（8×8=64 个 tile，8 核，窗口=4 行）

```text
before —— 纯行主序：每轮铺满一整 M 行，同一列 B-tile 的复用距离 = 一整行
        n0    n1    n2    n3    n4    n5    n6    n7
 m0   [ R0 ][ R0 ][ R0 ][ R0 ][ R0 ][ R0 ][ R0 ][ R0 ]   ← 第0轮：整行 m0
 m1   [ R1 ][ R1 ][ R1 ][ R1 ][ R1 ][ R1 ][ R1 ][ R1 ]   ← 第1轮：整行 m1
 m2   [ R2 ] ...
 m3   [ R3 ] ...
 ...       同一列 n 的 B-tile 隔一整行才被复用到 → B 大时被挤出 L2

after —— 窗口={m0,m1,m2,m3}，窗口内沿 N 推进；奇窗口反向（snake）
 R0: 核0-3 → (m0,n0)(m1,n0)(m2,n0)(m3,n0)    核4-7 → (m0,n1)(m1,n1)(m2,n1)(m3,n1)
 R1: 推到 n2,n3        R2: n4,n5        R3: n6,n7
 窗口1={m4..m7} → n7→n0 反向
 同一列 n 的 B-tile 在窗口内被 4 个 M 行紧挨着复用（复用距离 ≪ 一整行）
```

**机制 B：尾部薄片等分**（tile 数量不变，只改尺寸）

```text
before:  [ 满块 baseM ][ 满块 baseM ][ 满块 baseM ][ 薄片 mTail ]   窗口关键路径被满块拖到 baseM
after :  [ 满块 baseM ][ 满块 baseM ][ newMain ][ newMain ]        末尾几块等分，薄片不再拖累窗口
```

**机制 C：末轮二维再切分**（把空闲核救回来）

```text
28 核，最后一轮只剩 5 个原 tile：
before:  [核0][核1][核2][核3][核4][ 空闲 x23 ]              活跃 5/28，长长尾
after :  每个原 tile 沿 M/N 切 mTailTile×nTailTile（例 2×2=4 子块）
         [核0-3:T0子块][核4-7:T1]...[核16-19:T4][ 空闲 x8 ]  活跃 5→20，尾轮被摊薄
```

**总览**

```text
        分核调度
          ├─ 机制A 蛇形行窗口      → 改善 B 侧 L2 复用（需 B 工作集 > L2 才有收益）
          ├─ 机制B 尾薄片等分      → 修窗口关键路径
          └─ 机制C 末轮二维再切分  → 救末轮空闲核
   三者共享同一 WINDOW，边界须对齐，不能互相打碎
```

## 怎么改（before / after）

SWAT 改动只在「`index → (mi, ni)` 的映射」，K 循环与 L0C 累加不变，因此 before/after 可直接对照。下面以 `kernel[None, NUM_CORES]` 启动、每核第 `r` 轮取 `index = core_id + r*num_cores` 处理一个 `TILE_M×TILE_N` tile 为背景；`M_TILES×N_TILES = TOTAL_CNT`。机制 A 已核验，机制 B/C 为能力门控。

```python
import pypto_pro.language as pl

num_cores = pl.get_block_num()
core_id = pl.get_block_idx() // pl.get_subblock_num()
rounds = (TOTAL_CNT + num_cores - 1) // num_cores

# ---- before：纯行主序（每轮从 0 核起，按 N 顺序推进）----
with pl.section_cube():
    for r in pl.range(0, rounds):
        index = core_id + r * num_cores
        if index < TOTAL_CNT:
            mi = index // N_TILES           # M 方向：慢变
            ni = index % N_TILES            # N 方向：快变
            i = mi * TILE_M
            j = ni * TILE_N
            # K 循环（两版相同）：load A/B -> move L0 -> matmul/matmul_acc
            acc_slot = acc.current()
            for ko in pl.range(0, K_OUTER):
                a_slot = a_wide.next(); pl.load(a_slot, a, [i, ko * KL1])
                b_slot = b_wide.next(); pl.load(b_slot, b, [ko * KL1, j])
                al = a_left.next();  pl.move(al, a_slot, offset=[0, 0])
                br = b_right.next(); pl.move(br, b_slot, offset=[0, 0])
                if ko == 0:
                    pl.matmul(acc_slot, al, br, phase=pl.AccPhase.Partial)
                else:
                    pl.matmul_acc(acc_slot, acc_slot, al, br, phase=pl.AccPhase.Partial)
            pl.store(out, acc_slot, [i, j], phase=pl.STPhase.Final)
```

```python
# ---- after：SWAT 机制 A（Serpentine 行窗口，MAIN_WINDOW 行内蛇形遍历）----
MAIN_WINDOW = 4
with pl.section_cube():
    for r in pl.range(0, rounds):
        index = core_id + r * num_cores
        if index < TOTAL_CNT:
            row_idx = index // N_TILES // MAIN_WINDOW            # 行窗口层
            mi = row_idx * MAIN_WINDOW + index % MAIN_WINDOW     # 窗口内取 M 行
            ni_normal = (index // MAIN_WINDOW) % N_TILES
            parity = row_idx % 2                                 # 奇窗口反向（branchless）
            ni = ni_normal + parity * (N_TILES - 1 - 2 * ni_normal)
            i = mi * TILE_M
            j = ni * TILE_N
            # K 循环与 store 同 before（略）
```

**唯一改动就是 `(mi, ni)` 的映射**：before 让同一 M 行的 N tile 连续推进（B 侧复用距离大），after 把核按行窗口分组、窗口内 N 蛇形，缩短 B 的复用距离。机制 B/C（尾薄片等分、末轮二维再切分）是 further 改动，见下。

能力门控（机制 B/C，伪码，**禁止直接复制**）：
- 机制 B：tiling 侧对最后 `mBaseTailSplitCnt` 个 tile 重算 `mTailMain`，让尾部尺寸均匀（tile 数量不变），kernel 按 `mNormalCnt/mBaseTailMain/mBaseTailLast` 重建形状；
- 机制 C：末轮 `curRoundIdx == round_ - 1` 时把每个原 tile 切 `mTailTile_ × nTailTile_`，`blockIdx_ % totalTailTile_` 定子块下标。
- 同 kernel 内混用不同 tile 尺寸/子块映射的合法性未在当前版本核验，采用前须先证 DSL 支持并补正确性证据。

## 性能与验证指标

- 预期变化：机制 A 预期 B 侧 DRAM 流量下降、MTE2 更接近流式带宽；机制 B 预期尾薄片所在窗口关键路径缩短；机制 C 预期末轮活跃核数与核利用率上升、最慢核时间下降。三者共同表现为总时间下降。
- 证据现状：机制 A 的蛇形行窗口已在 board 上核验正确，但在 `8192×8192×128`（B≈2MB）与 `2048×16384×512`（B≈16MB）两个形状上**未测出稳定收益**——两例 B 基本能驻留 L2，符合"B 工作集 ≤ L2 时收益趋近 0"的条件性判断；机制 B/C 未在当前版本实现验证。
- 资格门：必须先用 tiling 建模判明是"B 侧 L2 复用/尾薄片/末轮空闲核"哪一类损失，再选对应机制；机制 A 需先证明 B 工作集 > 可共享 L2 且 B 侧 DRAM 流量偏高。同 target、同形状 A/B，不能把纯调度的改动与 K 循环/双缓冲改动混在一起给一个总收益。

## 技术限制与风险

- 三机制共享窗口长度，边界必须对齐 `WINDOW_LEN`；打碎窗口会同时破坏 A 的复用与 B/C 的划分。
- 机制 B/C 改变 tile 尺寸分布，涉及 L0/L1 容量、L0C 复用与尾块 valid_shape，改动面大、风险高；未上板验证前不要启用。
- 若实测无收益（核间已均衡、B 侧已被 L2 覆盖），不要为"用了 SWAT"而保留额外复杂度。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/system_variables/get_block_idx.md`、`docs/zh/api/pro_api/SIMD-API/system_variables/get_block_num.md`（核索引与限核后的实际核数）
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/cube_computation/matmul.md`、`docs/zh/api/pro_api/SIMD-API/cube_computation/matmul_acc.md`（K 分块累加）
