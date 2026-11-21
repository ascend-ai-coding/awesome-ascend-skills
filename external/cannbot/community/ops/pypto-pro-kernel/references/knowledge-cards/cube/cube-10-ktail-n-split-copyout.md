---
type: "PyPTO Performance Optimization Card"
title: "matmul N 外 / K 内循环重构（cube-bound + 小侧操作数常驻 L1）"
description: "把逐 K 块交替累加两个 N 子块的写法改成 N 外 / K 内——一次算完一个 N 子块的整段 K 再算下一个；在 cube-bound 且操作数常驻 L1 时，降低逐块交替的标量/发射开销、改善 MMAD 供数。"
status: "stable"
tags: ["pypto-pro", "cube", "matmul", "loop-restructure", "n-split", "l1-resident", "phase"]
item_id: "cube-10"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/gemm.md#from-capacity-planning-to-performance-candidates"
      scope: "比较 N 外/K 内与交替 N 子块的重复搬运、发射开销和 L1 小侧驻留。"
      boundary: "本卡只保留已核验的循环结构机制，不把早搬出的历史设想当成当前重叠收益；MTE2-bound 场景需保留原对照。"
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/gemm-desugar.md#the-only-stated-reasons-to-hand-write-closed-list-source-verified"
      scope: "手写 K 累加和直接排水重排后，仍须核验 phase 与 Acc 生命周期。"
      boundary: "两组 Acc 的地址与初始化独立；该引用不把 unit_flag 或早搬出收益自动传播到重构后的路径。"
bound_hint: "MAC、SCALAR、MTE1"
applicability: "cube-only matmul，K 循环逐块交替累加两个 N 子块（每轮 2 个 MMAD、写两个不同 L0C 地址的 Acc）；profiler 显示 cube-bound（MAC 占用高、MTE2 占用低）；小侧操作数（如权重 B）可常驻 L1 复用（cube-04 Fullload 形态）"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 phase= 直排 store 与 pl.move 的 offset 切片（cube-04）；MTE2-bound / 操作数流式载入时该重构为负收益"
---
# 技术卡片 cube-10：matmul N 外 / K 内循环重构（cube-bound + 小侧操作数常驻 L1）

- **适用 bound**：SCALAR/发射与 MAC 供数（逐块交替两个 N 子块使每轮标量与依赖偏重）
- **一句话**：把「每 K 块交替累加两个 N 子块」改成「一次算完一个子块的整段 K 再算下一个」；前提是 cube-bound 且小侧操作数常驻 L1。

## 何时用（诊断特征）

- cube-only matmul，K 循环内逐块在两个 N 子块间交替累加（每轮 2 个 MMAD、分别写两个不同 L0C 地址的 Acc）。
- profiler 显示 **cube-bound**：`aic_mac_ratio` 高、`aic_mte2_ratio` 低；且 SCALAR/发射开销占比可观。
- **小侧操作数常驻 L1**：如 `C[M,N]=A[M,K]@B[K,N]` 里 B（权重）一次载入常驻 L1、沿 M 分块、A 按 M-tile 流式（cube-04 Fullload 形态）。
- 以上须从源码、生成物或 profiler 直接核对；ratio 只作线索。

## 何时不适用

- **MTE2-bound / 操作数流式载入**：重构会让 A 的 GM→L1 与 L1→L0 搬运翻倍，实测为负收益。
- 带宽 bound，或已开 L0C 双缓冲 / 已由其它手段消除流水空泡。
- 有 Acc→Vec epilogue（CV 融合）的 kernel：`phase=` 配对合法性须按 cube-01 复核。
- K 只有一轮、无可分开的「整段 K」。

## 原理

逐块交替写法里，每个 K 块要发 2 个 MMAD（分别写两个 Acc）并在两个 Acc 之间来回切换，循环/地址标量开销与依赖链偏重；把 N 放到外层、K 放到内层后，每个 K 块只需 1 个 MMAD、对单一 Acc 连续累加，per-iteration 的依赖链更短、标量与发射开销更低，MMAD 供数更顺。

> **与 AscendC 源设计的关系**：本卡源自 AscendC 的 *K-Tail Stepwise Copyout* 设计，原机制是「子块完成即搬出、用另一半计算遮盖 FIXPIPE 搬出尾部，消除搬出尾部与下一 MMAD 抢 L0C 的冲突」。**在 PyPTO-Pro 上该"搬出遮盖"机制是惰性的**：msprof 实测（同循环顺序、只改搬出时机）"早搬出 vs 末尾搬出"的 kernel duration 与各 pipe 占用完全一致——Acc 上无软件同步、`phase=` 只建立 MMAD→STORE 依赖、L0C 读写由硬件细粒度仲裁，不形成 MMAD 停顿。因此本卡只保留有效的「循环重构」，不再主张「搬出遮盖」。

> **与 cube-04 的区别（避免与"小侧常驻"混淆）**：cube-04 用"小侧操作数常驻 L1"削减重复 GM→L1（面向 **MTE2-bound**），其"何时不适用"列出的 `CUBE bound` 指的是"驻留本身"在 cube-bound 下不划算。本卡以"小侧常驻"为**前置**，收益来自其上的 **N 外 / K 内循环重构**（面向 **cube-bound**）：在已应用 cube-04 的基线上把逐块交替改成一次跑完一个子块的整段 K。二者机制不同、可叠加，不是同一手段。

## 怎么改（before / after）

以下 after 为嵌入片段（操作数先常驻 L1，再按 K 段 `pl.move` offset 切片进 L0；N 两半各跑一遍完整 K）。Tile 声明、地址与 K 段边界须按当前 DESIGN 核验；两个 Acc 须落在**不同 L0C 地址**（示例 `0x0000` 与 `0x20000`）。

```python
import pypto_pro.language as pl

# before（结构示意）：逐 K 块交替累加两个 N 子块
# for k in K 块:
#     matmul(ac0, al, b0); matmul(ac1, al, b1)     # 每轮 2 个 MMAD，在两个 Acc 间来回

# after：N 外 / K 内；操作数常驻 L1（示意 A、B 各 load 一次；真实 kernel 至少把 A 常驻）
with pl.section_cube():
    af = a_res.current()
    bf = b_res.current()
    pl.load(af, a, [0, 0])
    pl.load(bf, b, [0, 0])

    ac0 = acc0.current()
    for k in pl.range(0, K, TILE):                   # 子块 0：整段 K
        al = a_left.current(); br = b_right.current()
        pl.move(al, af, offset=[0, k])               # 从常驻 L1 切片进 L0
        pl.move(br, bf, offset=[k, 0])
        if k == 0:
            pl.matmul(ac0, al, br, phase=pl.AccPhase.Partial)
        elif k < K - TILE:
            pl.matmul_acc(ac0, ac0, al, br, phase=pl.AccPhase.Partial)
        else:
            pl.matmul_acc(ac0, ac0, al, br, phase=pl.AccPhase.Final)
    pl.store(c, ac0, [0, 0], phase=pl.STPhase.Final)

    ac1 = acc1.current()
    for k in pl.range(0, K, TILE):                   # 子块 1：整段 K
        al = a_left.current(); br = b_right.current()
        pl.move(al, af, offset=[0, k])
        pl.move(br, bf, offset=[k, HALF_N])
        if k == 0:
            pl.matmul(ac1, al, br, phase=pl.AccPhase.Partial)
        elif k < K - TILE:
            pl.matmul_acc(ac1, ac1, al, br, phase=pl.AccPhase.Partial)
        else:
            pl.matmul_acc(ac1, ac1, al, br, phase=pl.AccPhase.Final)
    pl.store(c, ac1, [0, HALF_N], phase=pl.STPhase.Final)
```

要点：

- `af`/`bf` 为常驻 L1 的操作数（`pl.load` 一次），循环内只做 L1→L0 的 `pl.move` offset 切片；offset 是元素级，切错段是静默精度错误。
- `acc0`/`acc1` 为两个独立单缓冲 Acc，`fractal=1024`，地址不同。
- `phase=` 配对遵循 cube-01：首块 `Partial`、中间 `Partial`、末块 `Final`，`store` 用 `STPhase.Final`。
- N 偏移按子块修正（`[k, ns*HALF_N]` / `[0, ns*HALF_N]`）；外层 Tile 偏移不变。
- 真实 kernel 若只能常驻 A（B 每 tile 不同），只对 A 做常驻+切片；此时重构的额外代价为 A 的 L1→L0 搬运翻倍。

## 性能与验证指标

- msprof 实测（**真实 cube-bound matmul**：B 常驻 L1、沿 M 分块、A 流式）：重构后 kernel cycles 明显下降，**SCALAR 占比大幅下降、MAC 窗口收缩**；同循环顺序下改变搬出时机对 duration 与各 pipe 占用无影响（印证"搬出遮盖"惰性）。
- **能力门控**：仅当 profiler 确认 **cube-bound（MAC 高、MTE2 低）且小侧操作数常驻**时成立；MTE2-bound/流式载入时该重构为负收益；MX 量化路径流式载入下未复现收益。
- 收益幅度不承诺；以 kernel cycles 与 mac/scalar/mte 占比为验收指标。

## 技术限制与风险

- **操作数常驻是硬前提**：`af`/`bf` + 对侧轮转 + 随路数据 ≤ L1 预算，超限先收缩深度或放弃；常驻区在 kernel 生命周期内不得被覆写。
- 两个 Acc 必须落在不同 L0C 地址；同地址或复用单缓冲 Acc 会退化为串行，收益消失。
- 重构让 A 被每个 N 子块各切片一次（L1→L0 翻倍）；A 常驻只去掉 GM 重载，L1→L0 翻倍仍在。
- 无软件同步的 Acc 依赖由 `phase=` 的硬件 unit_flag 承担；`phase=` 配错仍是卡死/精度事故（同 cube-01）。
- 本卡相对原 AscendC「K-Tail Stepwise Copyout」只保留操作结构，机制已按 PyPTO-Pro 实测修正（见「原理」注）。
- 本卡不改 dtype 与计算语义，精度与默认累加一致。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/cube_computation/matmul.md`、`docs/zh/api/pro_api/SIMD-API/cube_computation/matmul_acc.md`（`AccPhase`）
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/store.md`（`STPhase`）
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`、`docs/zh/api/pro_api/SIMD-API/memory_data_movement/move.md`（常驻与 offset 切片）
