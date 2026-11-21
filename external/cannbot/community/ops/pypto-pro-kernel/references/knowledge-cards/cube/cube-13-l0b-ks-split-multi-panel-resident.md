---
type: "PyPTO Performance Optimization Card"
title: "K 子切分让多个右矩阵分块同驻 L0B（消除多累加 matmul 的 Buffer 串行）"
description: "一次 K 迭代要对同一个左矩阵分块做 3 个输出 N-tile 的 matmul 时，把 K 维按 64 子切，令 3 个 [64,128] 右矩阵分块（共 48KB）同驻 64KB 的 L0B，避免复用 Buffer 把第 3 个 matmul 串到前一个之后。"
status: "stable"
tags: ["pypto-pro", "cube", "matmul", "l0b", "k-split", "pipelining"]
item_id: "cube-13"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/gemm.md#from-capacity-planning-to-performance-candidates"
      scope: "缩短 L0 K 子块以容纳多个独立 Right tile，联合核算 Left、Right、Acc 的字节与累加步数。"
      boundary: "本卡 3 个 B panel 的尺寸和平台容量不是全局常量；首块初始化、尾 K、所有路径地址及额外循环成本仍须验证。"
bound_hint: "MAC、MTE1"
applicability: "单个左矩阵分块要喂多个（≥3）输出 N-tile，即每个 K 迭代需 3 个 [128,128] B 操作数；而 L0B 容量只够 2 个 [128,128]（64KB ÷ 32KB），复用 Buffer 会让多出的 matmul 与前一个串行。"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖已核验的 pl.TileType(Mat, NZ, [K,N]=[64,128] / [M,K]=[128,64])、pl.matmul/matmul_acc 的 K 维分块累加（首块 matmul、后续 matmul_acc）、pl.move 的 L1→L0 搬运。"
---
# 技术卡片 cube-13：K 子切分让多个右矩阵分块同驻 L0B

- **适用 bound**：MAC/MTE1（多累加 matmul 因 L0B Buffer 复用而串行）
- **一句话**：把 K 按 64 切，3 个 `[64,128]` 右矩阵分块同驻 L0B，三个 matmul 不再抢同一 Buffer。

## 何时用（诊断特征）

- 一个工作项在一次 K 迭代里对**同一个左矩阵分块**做 3 个（或更多）输出 N-tile 的 `matmul_acc`（如 cube-12 的 N384 宽档）。
- L0B 只有 64KB：`[128,128] bf16` 右矩阵分块 32KB → 最多 2 个同驻；第 3 个只能复用某个已用 Buffer。
- 出现"复用 Buffer"写法时（`pl.move(buf, b2)` 紧跟在使用 `buf` 的 matmul 之后），第 3 个 matmul 被强制排在前面 matmul 之后才能开始 → 流水断流。

## 何时不适用

- 只有 1~2 个输出 N-tile（N256 / 单 N-tile）：2 个 `[128,128]` 正好同驻 L0B，无需子切分。
- **只切 K 但不做多个右矩阵分块同驻没有收益**：本卡实测中"N384 但复用 L0B Buffer"的写法在大 batch 只有 ~−0.3%（与噪声同级），收益基本被串行抵消——**必须配合子切分让右矩阵分块真正同驻**。
- 子切分使 K 循环迭代数翻倍，若 L1/L0 的搬运启动开销占比很高且无流水隐藏，反而回退。
- **MTE2 字节受限时本卡不改变总搬运字节**：KS 子切分减少的是 L0B Buffer 串行、不是字节，K 的总搬运字节不变；若瓶颈在 MTE2 搬运总量（`aic_mte2` 占比高、Cube 等 MTE2），应改用能真正减少字节的手段（[cube-12](cube-12-runtime-n-tile-width-dispatch.md) 的 A 复用、[cube-04](cube-04-fullload-resident-l1.md) 的小侧常驻、[cube-07](cube-07-weight-nz-prepack.md) 的 NZ 纯搬运）。

## 与相关卡片的区别

- **cube-03（StreamK / Split-K）**：同样叫“K 切分”，但 cube-03 是**核间**切 K、以原子部分和补核利用率（调度 bound），改变并行度；本卡是**核内**把 K 宏步按 `KS=64` 子切，只为让多个右矩阵分块同驻 L0B（MAC/MTE1 bound），不改并行度、不跨核、不引入原子归约与精度抖动。二者正交，可分别独立采用。
- **cube-12**：本卡是 [cube-12](cube-12-runtime-n-tile-width-dispatch.md) N384 宽档在 64KB L0B 上的必要实现，两者配套使用。

## 原理

- K=128 的 `[128,128]` 右矩阵分块 = 32KB，L0B 64KB 只放得下 2 个；3 个右矩阵分块必然复用 Buffer → 串行。
- 把每个 K 宏步按 `KS=64` 子切：A 取 `[128,64]`（16KB）、每个 B 取 `[64,128]`（16KB，ZN），**3 个右矩阵分块共 48KB ≤ 64KB**，可各自占一个 Buffer 同时驻留。
- 这样 3 个 `matmul_acc` 之间无 Buffer 依赖，可被硬件流水重叠；K 的总搬运字节不变（A 的重复读由 cube-12 的 R=3 降低），MMAD 总量不变。
- L0C 侧需 3 个 `[128,128] fp32` 累加器（192KB ≤ 256KB）。

## 怎么改（before / after）

结构示意（嵌入片段，截取自已上板验证的 N384 宽档内层；`al/r0/r1/r2` 分别来自 `[128,64]` L0A 与 3 个 `[64,128]` L0B 通道）。

```python
# before：K=128 整块；L0B 只有 2 个 Buffer，第 3 个 B 复用 r0 -> 第 3 个 matmul 串行
for k in pl.range(0, k_hidden, 1):
    ca = c_a_l1.next(); cb0 = c_b0_l1.next(); cb1 = c_b1_l1.next(); cb2 = c_b2_l1.next()
    pl.load(ca, hidden, [i*128, k*128]); pl.load(cb0, w, [k*128, j0*128])
    pl.load(cb1, w, [k*128, (j0+1)*128]); pl.load(cb2, w, [k*128, (j0+2)*128])
    al = c_a_l0.next(); r0 = c_b0_l0.next(); r1 = c_b1_l0.next()
    pl.move(al, ca); pl.move(r0, cb0); pl.move(r1, cb1)
    pl.matmul_acc(ac0, ac0, al, r0); pl.matmul_acc(ac1, ac1, al, r1)
    pl.move(r0, cb2)                       # 复用 r0：必须等上面的 matmul 用完
    pl.matmul_acc(ac2, ac2, al, r0)

# after：K 按 64 子切；A=[128,64]、B=[64,128]；3 个右矩阵分块各占 Buffer 同驻
for k in pl.range(0, k_hidden, 1):
    for s in pl.range(0, 2, 1):
        a64 = k_a64_l1.next(); b0 = k_b0_l1.next(); b1 = k_b1_l1.next(); b2 = k_b2_l1.next()
        pl.load(a64, hidden, [i*128, k*128 + s*64])
        pl.load(b0, w, [k*128 + s*64, j0*128])
        pl.load(b1, w, [k*128 + s*64, (j0+1)*128])
        pl.load(b2, w, [k*128 + s*64, (j0+2)*128])
        al = k_a64_l0.next(); r0 = k_b0_l0.next(); r1 = k_b1_l0.next(); r2 = k_b2_l0.next()
        pl.move(al, a64); pl.move(r0, b0); pl.move(r1, b1); pl.move(r2, b2)
        if k == 0 and s == 0:              # 首块 matmul 初始化 L0C，其后 matmul_acc
            pl.matmul(ac0, al, r0); pl.matmul(ac1, al, r1); pl.matmul(ac2, al, r2)
        else:
            pl.matmul_acc(ac0, ac0, al, r0); pl.matmul_acc(ac1, ac1, al, r1); pl.matmul_acc(ac2, ac2, al, r2)
```

三个 `[64,128]` L0B 通道各用**单缓冲**（`mutex_ids` 只给 1 个 id）；每个 16KB，3 个占 48KB。

## 性能与验证指标

观察 3 个 `matmul_acc` 之间的串行是否消除：以 MAC/MTE1 占用与 Task Duration 为准，KS 子切分后应看到 Cube 更连续、Buffer 复用造成的断流消失；只复用 Buffer 而不子切分时对照应无改善。K 的总搬运字节与 MMAD 总量均不变。正确性按常规精度回归，重点覆盖 K 段边界（`KS=64` 子步、首块 `matmul` 初始化、后续 `matmul_acc`）与尾块处理。

## 技术限制与风险

- L0B 容量是硬约束：`[64,128]` 右矩阵分块 16KB，3 个 48KB 可行；若右矩阵分块更大或数量更多需重新估算。
- 每个 K 宏步内部多一层 `s` 子循环，K 迭代数翻倍；须确认搬运开销被流水隐藏。
- 子切分改变 matmul 的 K 维分块，**首块必须用 `pl.matmul` 初始化 L0C，后续用 `pl.matmul_acc`**；漏改会读到未初始化累加器。
- 子切分的 L1/L0 tile group 需与窄档路径的地址按编译期不重叠规划（两条路径运行期互斥，但分配是编译期）。

## 参考资料

- 本仓 [mega_recurrent_kda.py](https://gitcode.com/cann/pypto-gym/blob/ccd64a44f843d077feab9a81d894a09bda3c5af2/ops/pro/kimi_k3/mega_recurrent_kda/op_kernel/mega_recurrent_kda.py)：`k_a64/k_b*` tile group 与 N384 内层。
- matmul 契约：`pypto_pro.language.matmul` —— lhs `[M,K]`（L0A）、rhs `[K,N]`（L0B，ZN）、dst `[M,N]`（L0C，NZ），K 维分块首块 `matmul`、后续 `matmul_acc`。
- 相关卡片：[cube-12](cube-12-runtime-n-tile-width-dispatch.md)（触发本实现的分档）。
