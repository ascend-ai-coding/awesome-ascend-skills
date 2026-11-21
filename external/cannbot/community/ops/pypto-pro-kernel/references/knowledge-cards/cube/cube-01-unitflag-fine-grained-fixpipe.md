---
type: "PyPTO Performance Optimization Card"
title: "matmul `phase=` 细粒度 M↔FixPipe 流水（unit_flag 硬件握手）"
description: "在累加器经 pl.store 直接写回 GM 的 Cube kernel 中，给 K 循环的 matmul/matmul_acc 配置 AccPhase、给配对的 pl.store 配置 STPhase，用硬件 unit_flag 握手替代框架自动的整段 M↔FixPipe 软件同步，让 MMAD 与搬出细粒度重叠。"
status: "stable"
tags: ["pypto-pro", "cube", "matmul", "fixpipe", "pipeline"]
item_id: "cube-01"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/gemm-desugar.md#the-only-stated-reasons-to-hand-write-closed-list-source-verified"
      scope: "phase/unit_flag 属于必须保留手写 move/matmul/store 链的机制，生产和 drain 需成对核验。"
      boundary: "仅覆盖本卡直接 Acc→GM 排水路径；不把 phase 推广到未验证的 Acc→Vec，也不自动生成同步或 buffer 配对。"
bound_hint: "MAC、FIXPIPE"
applicability: "Cube kernel 的 K 循环累加器经 pl.store/store_tile 直接写回 GM（无 Acc→Vec epilogue），profiling 显示 FIXPIPE bound 或 MMAD 与搬出整段串行，且 L0C 被完整输出块占满无法开双缓冲"
target_api_gate: "仅限 Ascend 950PR 或 950DT；phase= 仅在 drain 为带 phase= 的 pl.store/store_tile 时合法；pl.move 是否携带 phase 形参随版本变化，Acc→Vec 排水的配对合法性须按目标版本重新验证，验证前禁用结论维持；AccPhase 末块 Final/其余 Partial 的序列须按目标 SDK 文档复核"
---
# 技术卡片 cube-01：matmul `phase=` 细粒度 M↔FixPipe 流水（unit_flag 硬件握手）

- **适用 bound**：FIXPIPE（L0C→GM 搬出瓶颈）或 MMAD 与 FIXPIPE 整段串行造成的流水空洞
- **一句话**：`phase=` 不是提示，是 Cube M pipe 与 FixPipe 之间的硬件 `unit_flag` 握手；在 store 直排的 Cube kernel 上用它替代框架自动整段同步，换 MMAD 与搬出的细粒度重叠。

## 何时用（诊断特征）

- kernel 为 cube-only 收缩，K 循环累加进单个 Acc tile，最终经 `pl.store` 直接写回 GM。
- msprof 显示 `aic_fixpipe_ratio` 偏高或 MMAD 段与搬出段明显串行（MMAD 完成后才开始整段搬出）。
- 生成代码中 M↔FixPipe 为框架自动插入的整段软件同步，且 L0C 被完整输出块占满、开不出双缓冲。
- 以上均须从当前源码、生成物或 profiler 直接核对；ratio 只作线索。

## 何时不适用

- **累加器经 `pl.move(..., acc_to_vec_mode=...)` 排入 Vector epilogue（CV 融合 kernel）：禁用。** `pl.move` 在部分版本中没有 `phase` 参数，`matmul(phase=Final)` 配 `pl.move` 会武装一个另一侧无法应答的协议——不写时硬挂（flag 卡 1，无报错无超时），错读时 fixpipe 读未写完的 L0C 触发 multi-bit ECC `error 171` 并经 RAS 路径打 `Alarm`（有实卡损失记录）。即使目标版本的 `pl.move` 已含 `phase` 形参，Acc→Vec 排水的配对合法性也须重新验证；在新证据出现前此类 kernel 仍必须删除 `phase=`，让框架自动同步。
- L0C 有空间开双缓冲时优先双缓冲；`phase=` 面向无法开缓冲的场景。。
- 测试形状的 K 循环如果只跑一轮（如L0C上有 2 个 buffer、但 `kv_tiles = 1`），每个 buffer 只用一次，`phase=` 配对错了也暴露不出来——这种"通过"不算证据。

## 原理

未配置 `phase=` 时，框架在 MMAD 与 FixPipe 之间插入整段软件同步：MMAD 全部完成后搬出才启动。配置 `phase=` 后改用硬件 `unit_flag` 握手（`matmul(phase=Final)` 置 flag，`store(phase=Final)` 读并清除），框架的自动 M↔FixPipe 同步随之关闭，搬出与计算的依赖粒度从整段细化到硬件握手的块粒度，MMAD 不必等整段搬出。这与 AscendC UnitFlag（Mmad `unitFlag=NO_FINAL_ACCUMULATION/FINAL_ACCUMULATION` + 删除 `SetFlag/WaitFlag<M_FIX>`）是同一件硬件机制的两种前端拼写；PyPTO-Pro 不暴露 512B 粒度参数，粒度由生成代码决定。

配对规则（缺一不可，违反即正确性事故而非性能回退）：

1. `matmul`/`matmul_acc` 带 `phase=`，drain 必须是带 `phase=` 的 `pl.store`/`store_tile`。
2. K 循环中只有最后一个 K 块用 `AccPhase.Final`，之前一律 `Partial`；全程 `Partial` + `store(Final)` 即使单块循环也以 `device error type 0xFFFF` 挂掉。

## 怎么改（before / after）

以下为嵌入片段；Tile 声明、TileGroup 轮转、dtype 与 K 切分上下文须按当前 DESIGN 核验。两个实现注意点：

- 循环内使用的 tile 句柄（`a_l1.current()` 等）须在 `pl.range` 循环**之前**定义，否则 parser 报 `F00004 Use of potentially undefined variable`。
- 片段中的 `if/elif/else` 按循环变量分支的形式在当前 DSL 可用。

```python
import pypto_pro.language as pl

# before：不带 phase，框架自动插入整段 M↔FixPipe 同步（正确，默认写法）
with pl.section_cube():
    for kb in pl.range(0, n_blk):
        # ... load L1/L0 操作数 ...
        if kb == 0:
            pl.matmul(ac, la, rb)
        else:
            pl.matmul_acc(ac, ac, la, rb)
    pl.store(out, ac, [0, 0])

# after：显式 unit_flag 握手；仅当 drain 是 pl.store/store_tile 时合法
with pl.section_cube():
    for kb in pl.range(0, n_blk):
        # ... load L1/L0 操作数 ...
        if kb == 0:
            pl.matmul(ac, la, rb, phase=pl.AccPhase.Partial)
        elif kb < n_blk - 1:
            pl.matmul_acc(ac, ac, la, rb, phase=pl.AccPhase.Partial)
        else:
            # 单块循环时首块即末块，首块直接用 AccPhase.Final + pl.matmul
            pl.matmul_acc(ac, ac, la, rb, phase=pl.AccPhase.Final)
    pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
```

注意 `n_blk == 1` 时首块即末块：首块用 `pl.matmul(..., phase=pl.AccPhase.Final)`，勿让循环只走 `Partial` 分支。

## 性能与验证指标

- 预期变化：`Task Duration(us)` 下降、`aic_fixpipe_ratio` 升高（重叠加深）、MMAD 断流减少；以 kernel 总时间和 MMAD 断流为验收指标。收益幅度随 fixpipe 在总时间中的占比缩放，fixpipe 占比小的形态（如深 K、MTE2 主导）预期收益不明显。
- 判读陷阱：开启后 profiling 中 fixpipe 段会变长——指令下发提前、实际搬运仍受末轮累加约束，不代表劣化；勿按 fixpipe 段长判断收益。

## 技术限制与风险

- 配对写错有两种挂法：一是死等（flag 没被清掉，无报错无超时）；二是 fixpipe 读到没写完的 L0C，报 ECC `error 171` 和驱动告警 `8C4BA00C`，要复位驱动，实卡上烧过。**怀疑是这类故障时别在共享机器上反复试**，直接查生成代码和错误码。
- 增加 buffer 数后挂死消失 ≠ 修好：buffer 变多只是让同一个 buffer 晚一轮被复用，如果新形状下每个 buffer 还是只用一次，只是没踩到 bug。想确认真的对了，就换一个 K 循环轮数超过 buffer 数的形状重测，通过才算数。
- 开启后 MMAD 还是断流的话，可能是搬出尾部和下一个 Tile 抢同一个 L0C 地址。PyPTO-Pro 没有对应写法，记录瓶颈证据后单独立项。
- 本卡不改计算顺序和 dtype，精度与默认写法一致；正确性风险只来自配对规则写错。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/basic_data_structures/AccPhase.md`、`docs/zh/api/pro_api/SIMD-API/basic_data_structures/STPhase.md`（`phase` 握手语义与配对约束）
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/cube_computation/matmul.md`、`docs/zh/api/pro_api/SIMD-API/cube_computation/matmul_acc.md`（`AccPhase` 参数）
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/store.md`、`docs/zh/api/pro_api/SIMD-API/memory_data_movement/store_tile.md`（`STPhase` 参数）
