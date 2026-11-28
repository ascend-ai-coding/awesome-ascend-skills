---
type: "PyPTO Performance Optimization Card"
title: "MTE2 预取（显式 ping/pong 组 + 消费后回填）"
description: "把 A/B 的 L1 缓冲声明为两组独立 tile group（ping/pong 各持独立 mutex），循环入口处先发射下一块的 pl.load 再消费当前块，使下一块的 MTE2 发射提前到当前块计算之前，保持搬运与计算流水连续。"
status: "stable"
tags: ["pypto-pro", "cube", "matmul", "mte2", "preload", "pipeline"]
item_id: "cube-05"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/memory-allocation.md#independent-buffers-and-slot-depth"
      scope: "用独立 buffer 地址与 mutex 允许 load 和当前计算交错，消费结束后再回填并测试 ring wrap。"
      boundary: "目标为局部槽位与生命周期原则，不是本卡奇偶 ping/pong 序列的正确性证明；首尾守卫与实际预取收益仍须验证。"
bound_hint: "MTE2"
applicability: "K 循环 matmul 已有双缓冲语义但流水仍见 MTE2 空泡（搬运发射滞后于数据依赖解除），且 k_blocks ≥ 2"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖独立 tile group + auto_mutex 的同组内 pipe 排序；动态奇偶分支选择句柄已在当前 parser 验证可用"
---
# 技术卡片 cube-05：MTE2 预取（显式 ping/pong 组 + 消费后回填）

- **适用 bound**：MTE2 发射空泡（非带宽 bound：带宽未满但段间有确定性 gap）
- **一句话**：ping/pong 两组独立句柄，当前块消费完后立刻回填再下一块，让 MTE2 始终错拍领先。

## 何时用（诊断特征）

- K 循环 matmul，已有双缓冲但仍见 MTE2 段间空泡；`k_blocks ≥ 2`。
- 空泡呈"发射滞后"形态而非带宽打满：MTE2 busy 与带宽利用率都不高。
- 单组深度轮转的写法中，下一块 load 的程序序被排在当前块计算之后，发射时机偏晚。

## 何时不适用

- MTE2 带宽真打满（真 bound）：瓶颈在带宽不在发射时机，应先做搬运减量（驻留/合并载入）。
- `k_blocks < 2`：无块可预取。
- L1 容不下两组缓冲：先收缩 tile。

## 原理

搬运指令的发射受程序序约束；把"下一块的 load"提前到"当前块的计算"之前发射，MTE2 与 MTE1/CUBE 形成稳定错拍：当前块在算时，下一块已经在搬。用两组独立的 tile group（各自独立 mutex）表达 ping/pong，auto_mutex 只保证同组内的读写次序，两组互不阻塞；消费完一个 buffer 后立刻用下下一块回填它，buffer 生命周期闭合。

## 怎么改（before / after）

以下为嵌入片段，截取自已上板验证的 K 循环 matmul kernel 对（每核一个输出 tile，k_blocks=16）；tile 尺寸、地址与 K 循环边界须按当前 DESIGN 核验。

```python
import pypto_pro.language as pl

# before：单组深度 2 轮转，下一块 load 的程序序排在本块计算之后
# for kb in pl.range(0, k_blocks):
#     am = a_l1.next(); bm = b_l1.next()
#     pl.load(am, a, [0, kb * KB])
#     pl.load(bm, b, [kb * KB, cid * NT])
#     pl.move(al, am); pl.move(br, bm)
#     if kb == 0:
#         pl.matmul(ac, al, br)
#     else:
#         pl.matmul_acc(ac, ac, al, br)
# pl.store(out, ac, [0, cid * NT])

# after：ping/pong 两组独立句柄，首轮双发，消费后立刻回填 kb+2
with pl.section_cube():
    pl.load(a_ping.current(), a, [0, 0])
    pl.load(b_ping.current(), b, [0, cid * NT])
    pl.load(a_pong.current(), a, [0, KB])
    pl.load(b_pong.current(), b, [KB, cid * NT])
    al = a_left.current()
    br = b_right.current()
    ac = acc.current()
    for kb in pl.range(0, k_blocks):
        if kb % 2 == 0:
            am = a_ping.current()
            bm = b_ping.current()
        else:
            am = a_pong.current()
            bm = b_pong.current()
        pl.move(al, am)
        pl.move(br, bm)
        if kb == 0:
            pl.matmul(ac, al, br)
        else:
            pl.matmul_acc(ac, ac, al, br)
        if kb + 2 < k_blocks:
            # 本轮刚消费的 buffer 立刻回填 kb+2（越界守卫不能省）
            pl.load(am, a, [0, (kb + 2) * KB])
            pl.load(bm, b, [(kb + 2) * KB, cid * NT])
    pl.store(out, ac, [0, cid * NT])
```

要点：回填目标是**本轮刚消费的 buffer**（不是另一个），且必须放在该 buffer 的 `pl.move` 之后；`kb + 2 < k_blocks` 守卫防止越界预取；按 `kb % 2` 动态选择句柄的写法已在当前 parser 验证可编译且结果正确。

## 性能与验证指标

观察 MTE2 段耗时与段间空泡、MMAD 连续性；搬运次数与字节数不应变化。正确性按常规精度回归，重点覆盖奇/偶 `k_blocks` 两种轮换对齐。注意：单组深度 2 + auto_mutex 的依赖调度本身已能重叠搬运与计算，手排预取只有在 trace 证实存在"发射滞后"形态空泡时才预期有效，否则为中性。

## 技术限制与风险

- 回填写错 buffer（写到尚未消费的另一个）是静默数据错乱；守卫漏写会越界读 GM。
- 两组缓冲的地址与 mutex 须成对独立；共用 mutex 会把两组串行化，预取失效。
- 循环内按 `kb % 2` 选择句柄引入少量标量分支开销，须被搬运段节省覆盖。
- 首轮回填语义要求 `k_blocks ≥ 2` 才有 PONG；`k_blocks == 2` 时循环内回填不触发，属预期。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/resource_management/make_tile_group.md`（独立组与 mutex）
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`
