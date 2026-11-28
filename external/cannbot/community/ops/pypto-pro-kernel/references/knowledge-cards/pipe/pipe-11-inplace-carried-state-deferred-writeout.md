---
type: "PyPTO Performance Optimization Card"
title: "就地读写的循环携带状态延迟一拍写回（镜像快照）"
description: "循环每轮就地更新一个携带状态并必须写回 GM 时，单状态缓冲使写回与计算串行；把状态镜像到深度 2 快照并延迟一轮写回，令 DMA 与更新重叠。"
status: "stable"
tags: ["pypto-pro", "pipeline", "state", "writeback"]
item_id: "pipe-11"
bound_hint: "MTE3、PIPELINE"
applicability: "循环每轮就地读写一个携带状态（状态是读-改-写依赖，下一轮继续读它），且契约要求每轮把该状态写回 GM；普通输出可直接双缓冲但就地状态不能，且片上可容纳额外一份深度 2 镜像"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 make_tile_group 深度≥2 轮转、auto_mutex 与 pl.load/pl.store 整块搬运"
---

# 技术卡片 pipe-11：就地读写的循环携带状态延迟一拍写回（镜像快照）

- **适用 bound**：MTE3、PIPELINE
- **一句话**：状态被就地读写又必须每轮外写时主缓冲不能双缓冲；把它镜像进深度 2 快照、写回延迟一轮，使 DMA 与更新重叠。

## 何时用（诊断特征）

- 循环每轮对一个携带量做读-改-写（`S ← f(S, in_t)`），下一轮继续读同一个 `S`，故 `S` 只有一份、无法像普通输出那样直接双缓冲。
- 契约要求每轮把 `S` 写回 GM（硬性输出，不能按 `mem-06` 改成片上常驻）。
- trace 显示写回 MTE3 与计算 VEC 相加而非重叠：单缓冲上 store 与下一轮 update 构成写后读（WAR）依赖。
- 去掉每轮写回后总耗时明显下降，说明写回在关键路径上。

## 何时不适用

- 状态可片上常驻、无需每轮外写：用 `mem-06` 直接消除 GM 往返，优于本卡。
- 要写回的值是普通计算输出、可被直接双缓冲：用 `pipe-02`，无需额外镜像。
- 写回 DMA 时长显著大于一轮计算：深度 2 只留一轮余量，仍会阻塞；需更多快照槽或增大每轮工作量。
- 片上容量放不下额外一份状态镜像。

## 原理

单状态缓冲上，「update(S) → store(S)」与下一轮「update(S)」构成串行依赖：store 必须读完 `S`，下一轮才能改 `S`，故写回与计算不能重叠。把 `S` 同时镜像到深度 2 快照（`snap[t%2]`），本轮写 `snap[t%2]`，但 GM 写的是**上一轮**的 `snap[(t-1)%2]`：该槽本轮不写，与其在跑的 DMA 无关，因此写回可与本轮 update 并发；该槽要到两轮后才被复用，给 DMA 留出一轮完成窗口。代价是状态多占一份容量，且每个元素多一次 VF store（镜像与 state 同趟产出）。

## 怎么改（before / after）

结构示意：只展示写回位置与快照槽轮转，`st_g`/`snap_g`/`state_gm`/`update_vf*` 为占位符，缺少真实的 Tile 声明、地址、dtype 与同步上下文，片段不可直接编译或交付。

```结构示意
# before：就地更新后立即写回，单 state 上写后读依赖，写回与下一轮更新串行
state = st_g.current()
for t in pl.range(0, T):
    update_vf(state, in_ub[t])                 # state = f(state, in[t])
    pl.store(state_gm, state, [t * D, 0])      # 立即写回，需等 update 完成

# after：镜像到深度2快照，写回延迟一轮
state = st_g.current()
prev_buf = 0
for t in pl.range(0, T):
    snap = snap_g[t % 2]
    update_vf2(state, snap, in_ub[t])          # 同一趟产出 state 与新镜像 snap
    if t > 0:
        pl.store(state_gm, snap_g[prev_buf], [(t - 1) * D, 0])   # 写上一轮镜像
    prev_buf = t % 2
pl.store(state_gm, snap_g[prev_buf], [(T - 1) * D, 0])            # 补写最后一轮
```

## 性能与验证指标

- 消融确认收益上界：先**去掉每轮写回**测总耗时（输出作废，仅测时间），差值即写回串行成本；改后应收敛到接近该上界且仍保证写回。
- 对比 VEC 与 MTE3 的忙时/墙钟比（重叠度）与同条件 `Task Duration`。
- 正确性：完整套件回归；重点覆盖**最后一轮补写**与奇/偶轮槽位复用（末轮不被下一轮覆盖）。
- 经验规律：低核（写回占比更高）收益更明显，高核趋近无收益；具体幅度随实现与形态变化，须在目标形态 A/B 复测。

## 技术限制与风险

- 就地状态无法直接双缓冲，必须额外镜像一份：状态容量翻倍（如 `[128,128]` FP32 由 64 KiB 增至 2×64 KiB），可能需重排片上布局腾出空间；每个元素多一次 VF store（镜像与 state 同趟产出）。
- 深度 2 只留一轮余量；写回 DMA 慢于一轮计算时仍会阻塞，需加深快照或增大每轮工作量。
- 收尾补写最后一轮不能省；漏写则丢失末状态。
- 与 `pipe-02` 的关系：`pipe-02` 面向「输出可直接双缓冲」的通用延迟写回；本卡面向「输出是被就地读写的携带状态、无法直接双缓冲、须镜像」的变体，二者叠加项为「延迟一轮写回 + 收尾补写」。

## 参考资料（可选）

- 同库 `pipe/pipe-02-stage-order-rescheduling.md`（延迟一轮写回的通用形态）
- 同库 `mem/mem-06-resident-state-eliminate-gm-roundtrip.md`（可常驻时优先消除 GM）
- 验证案例：`ops/pro/kimi_k3/mega_recurrent_kda`（`op_kernel/mega_recurrent_kda.py` 的 `_kda_update_vf`、`RESULTS.md` 状态写回一节）。该算子大 batch 在 8/16/24/32 AIC 分别回收约 15%/14%/7%/2%，属于该实例，未在本卡泛化。
