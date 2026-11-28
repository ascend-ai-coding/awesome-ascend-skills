---
type: "PyPTO Performance Optimization Card"
title: "FullLoad：小侧操作数全量驻留 L1"
description: "一侧矩阵小且对侧循环次数 ≥2 时，把小侧（含随路数据）一次性载入 L1 驻留，对侧循环内用 pl.move 的 offset 从驻留区切片进 L0，消除小侧在对侧循环中的重复 GM→L1 搬运。"
status: "stable"
tags: ["pypto-pro", "cube", "matmul", "fullload", "l1-resident"]
item_id: "cube-04"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/gemm.md#from-capacity-planning-to-performance-candidates"
      scope: "把小侧跨循环不变操作数常驻 L1，减少重复 GM 读，并联合选择对侧轮转与 L0 分块。"
      boundary: "容量可容纳不等于当前瓶颈受益；驻留区、scale 偏移、L2 命中及对侧槽位须分别核算。"
bound_hint: "MTE2"
applicability: "matmul 一侧矩阵字节数 ≤ 可用 L1 预算（预留对侧流式与轮转空间），对侧循环次数 ≥2，且当前为 MTE2 bound"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 pl.load 单次大搬运与 pl.move 的 offset 切片语义（源 tile 大于目的 tile 时按元素偏移读）"
---
# 技术卡片 cube-04：FullLoad：小侧操作数全量驻留 L1

- **适用 bound**：MTE2（小侧矩阵被对侧循环重复搬运）
- **一句话**：小侧一次搬进 L1 驻留，对侧循环里只做 L1→L0 切片，不再碰 GM。

## 何时用（诊断特征）

- 一侧矩阵（含随路 scale 等）字节数可放入 L1 预算（并为对侧 ping/pong 预留空间）。
- 对侧循环次数 T ≥ 2：该小侧在基线中被重复搬 T 次，T−1 份是纯冗余。
- profiling 确认为真 MTE2 bound（对侧单次搬运量足够大，非小数据块密集型假 bound）。

## 何时不适用

- 两侧都放不进预算：物理不可行。
- T = 1：小侧本来就只搬一次，收益为零。
- CUBE bound：瓶颈不在搬运，驻留反而挤占 buffer。
- 已走 K 切分（StreamK/Split-K）：驻留语义失效，两者互斥。

## 原理

把"跨循环内容不变"的小侧数据从每轮流式搬入改为一次驻留：MTE2 总字节数减少 `(T−1)/T × 小侧字节`。驻留期间该 L1 区不被覆写，也就不需要对应的释放/轮转同步；对侧保持原有轮转不变。同一原则适用于任何跨外层循环只读且内容不变的数据（bias、mask、共享查表等）。

## 怎么改（before / after）

以下为嵌入片段，截取自已上板验证的 A[128, 512]、B[512, N] matmul kernel 对（对侧 n_tiles 按核轮转，T≥2）；tile 尺寸、地址预算与尾块处理须按当前 DESIGN 核验。

```python
import pypto_pro.language as pl

# before：A 的 pl.load 在双重循环内，每个 nt 都重复搬整份 A
# for nt in pl.range(cid, n_tiles, ncore):
#     for kb in pl.range(0, k_blocks):
#         am = a_l1.next(); bm = b_l1.next()
#         pl.load(am, a, [0, kb * KB])
#         pl.load(bm, b, [kb * KB, nt * NT])
#         pl.move(al, am); pl.move(br, bm)
#         if kb == 0:
#             pl.matmul(ac, al, br)
#         else:
#             pl.matmul_acc(ac, ac, al, br)
#     pl.store(out, ac, [0, nt * NT])

# after：小侧 A 全量驻留 L1（tile shape 覆盖整个 [M_tile, K]），循环内只做 L1→L0 切片
with pl.section_cube():
    af = a_full.current()
    pl.load(af, a, [0, 0])                        # 循环外一次性载入
    al = a_left.current()
    br = b_right.current()
    ac = acc.current()
    for nt in pl.range(cid, n_tiles, ncore):
        for kb in pl.range(0, k_blocks):
            bm = b_l1.next()
            pl.load(bm, b, [kb * KB, nt * NT])
            pl.move(al, af, offset=[0, kb * KB])  # 从驻留区切片进 L0
            pl.move(br, bm)
            if kb == 0:
                pl.matmul(ac, al, br)
            else:
                pl.matmul_acc(ac, ac, al, br)
        pl.store(out, ac, [0, nt * NT])
```

差异：A 的 GM→L1 只发生一次，循环内只剩 L1→L0 的 `pl.move`；`.current()` 句柄须在循环外取好。

## 性能与验证指标

观察 MTE2 总字节/段耗时与 Task Duration；小侧搬运次数应从 `T × k_blocks` 降为 `k_blocks`。正确性按常规精度回归，重点覆盖切片偏移（K 段错位会静默用错数据）。

## 技术限制与风险

- L1 预算是硬约束：驻留区 + 对侧轮转区 + 随路数据 ≤ 可用 L1，超限先收缩对侧深度再考虑放弃。
- `pl.move` 的 offset 单位是元素；切片起点写错是静默精度错误。
- 驻留区在 kernel 生命周期内不得被其他用途覆写。
- 量化变体的随路 scale 同理驻留时，切片偏移须按 scale 的粒度重算，不能与数据偏移混用。
- 小侧能被 L2 缓存住时，基线的重复读多命中 L2，本手段的时间收益可能被掩盖；它面向的是 L2 存不下或 MTE2 真饱和的场景，采用前以 profiling 确认。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`、`docs/zh/api/pro_api/SIMD-API/memory_data_movement/move.md`（offset 切片语义）
