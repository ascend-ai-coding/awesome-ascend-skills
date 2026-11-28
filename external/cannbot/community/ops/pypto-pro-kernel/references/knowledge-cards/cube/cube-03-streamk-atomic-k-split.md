---
type: "PyPTO Performance Optimization Card"
title: "StreamK / Split-K：K 维跨核切分 + 原子累加部分和"
description: "MN 方向 tile 数填不满核时，把 K 维切成多段分给多个核分别累加出 FP32 部分和，经 pl.store 的 AtomicAdd 写入同一输出块，用归约代价换核利用率。"
status: "stable"
tags: ["pypto-pro", "cube", "matmul", "streamk", "split-k", "atomic"]
item_id: "cube-03"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/constraints.md#atomicatomicadd-supports-only-six-destination-dtypes"
      scope: "K 维多核部分和写回前，核对目的 dtype 是否支持 AtomicAdd。"
      boundary: "dtype 支持不证明输出初始化、并发写者上限、确定性或 K 分段覆盖；这些条件仍按本卡和当前平台验证。"
    - kind: alternative_to
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/gemm.md#count-the-complete-output-grid-before-assigning-workers"
      scope: "完整 M×N 输出网格仍不足以提供并行工作时，再比较 K 维拆分与部分和合并。"
      boundary: "输出轴分工不需 K 合并；Split-K 增加原子或 merge 成本并改变累加顺序，不能因空闲核直接启用。"
bound_hint: "MAC"
applicability: "matmul 的 M/N tile 数明显小于可用核数且 K 足够长；输出 dtype 支持原子累加；FP32 部分和精度预算允许"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 pl.store/store_tile 的 atomic=AtomicAdd（目的区域须预先初始化）与 kernel[stream, block_dim] 启动"
---
# 技术卡片 cube-03：StreamK / Split-K：K 维跨核切分 + 原子累加部分和

- **适用 bound**：调度（核利用率不足：小 M、小 N、长 K）
- **一句话**：空闲核参与 K 维累加，部分和以原子加写入同一输出，用归约换并行。

## 何时用（诊断特征）

- `m_tiles × n_tiles` 明显小于可用核数（典型不足一半），kernel 被少数核的 K 向串行拖长。
- K 足够长，切分后每核仍有足够深的累加链摊销开销。
- 输出 dtype 与硬件原子加路径匹配；目的区域可在 launch 前初始化（清零责任唯一）。

## 何时不适用

- M/N tile 已填满核：切 K 只增加归约与同步开销。
- K 短、归约主导：部分和的写与累加成本超过切分收益。
- 输出 dtype 不支持原子累加，或精度合同不允许浮点原子顺序抖动——此时考虑 workspace + 显式归约的替代结构（不在本卡范围）。
- 每核重复清零输出是缺陷：初始化责任只能有一个。

## 原理

Data-Parallel 分核只在 M/N 维切任务；欠并行时大量核空闲。把 K 切成若干段，每核用独立 Acc 累加自己的 K 段得到 FP32 部分和，再以原子加累进同一输出块。并行度从 `min(mn_tiles, cores)` 提升到 `min(mn_tiles × k_parts, cores)`；代价是原子写的串行化与浮点累加顺序变化，部分和保持 FP32 可减少中间舍入。

## 与相关卡片的区别

- **cube-13（L0B KS 子切分）**：同样出现“K 切分”，但 cube-13 是**核内**把 K 宏步按 `KS=64` 子切，让多个右矩阵分块同驻 L0B，解决的是单核内 L0B Buffer 串行（MAC/MTE1 bound），**不改并行度、不跨核、不做原子归约**；本卡是**核间**切 K，用原子部分和补核利用率（调度 bound）。二者正交，不可互换。
- **FullLoad / 驻留类手段**：K 被本卡切开后“整段 K 驻留 L1”的语义失效，互斥。

## 怎么改（before / after）

以下为嵌入片段，截取自已上板验证的 M=128、N=128、长 K matmul kernel 对；tile 尺寸、地址与 K 段边界对齐须按当前 DESIGN 核验。

```python
import pypto_pro.language as pl

cid = pl.get_block_idx()

# before：Data-Parallel 只在 M/N 维切任务，mn_tiles=1 时仅 core 0 串行累加全 K
# with pl.section_cube():
#     mn_tiles = (a.shape[0] // 128) * (b.shape[1] // 128)
#     al = a_left.current(); br = b_right.current(); ac = acc.current()
#     for tid in pl.range(cid, mn_tiles, ncore):
#         for kb in pl.range(0, a.shape[1] // KB):
#             am = a_l1.next(); bm = b_l1.next()
#             pl.load(am, a, [0, kb * KB])
#             pl.load(bm, b, [kb * KB, 0])
#             pl.move(al, am); pl.move(br, bm)
#             if kb == 0:
#                 pl.matmul(ac, al, br)
#             else:
#                 pl.matmul_acc(ac, ac, al, br)
#         pl.store(out, ac, [0, 0])

# after：每个核负责一个 K 段，部分和原子加进同一输出
with pl.section_cube():
    k_part_len = a.shape[1] // K_PARTS      # 段边界须按 cube K 粒度对齐
    al = a_left.current()
    br = b_right.current()
    ac = acc.current()
    for kb in pl.range(0, BPP):
        koff = cid * k_part_len + kb * KB
        am = a_l1.next()
        bm = b_l1.next()
        pl.load(am, a, [0, koff])
        pl.load(bm, b, [koff, 0])
        pl.move(al, am)
        pl.move(br, bm)
        if kb == 0:
            pl.matmul(ac, al, br)
        else:
            pl.matmul_acc(ac, ac, al, br)
    # host 须在 launch 前把 out 清零；部分和以原子加累进共享输出
    pl.store(out, ac, [0, 0], atomic=pl.AtomicType.AtomicAdd)
```

启动侧按 `kernel[stream, k_parts](...)` 以 K 段数作为核数；`k_parts` 上限取 `floor(cores / mn_tiles)`。
`.current()` 句柄须在循环外取好（parser 要求变量先定义后使用）；K 段数与每段块数由同一份 host/kernel 布局计算。

## 性能与验证指标

观察 Task Duration、活跃核数与原子写占比；收益 = 关键路径缩短 − 归约成本，须逐 case 对比。精度关注浮点原子顺序带来的 run-to-run 抖动，验收口径须按合同明确。

## 技术限制与风险

- 原子加要求目的区域先初始化；忘记清零或重复清零都是正确性事故。
- 浮点原子加的顺序不确定，结果存在 run-to-run 抖动；确定性合同禁用。
- 并发原子写者数存在平台上限，超限会静默丢失部分和；`k_parts` 上限须按目标平台实测确认，宁保守勿超配。
- K 段边界须按 cube K 粒度对齐（`k_part_len` 为 KB 整数倍），不对齐会造成段间重叠或空洞；尾段用有效形状处理。
- 多核分工映射（如 `cid * tile`）必须以 `get_block_num()` 的实际返回为准：block_dim 超过物理核数时超出的 block 会被静默丢弃，按声明核数硬编码分工会产生未写区域。
- 与 FullLoad/驻留类手段互斥：K 被切开后"驻留全 K"的语义失效。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/store.md`、`docs/zh/api/pro_api/SIMD-API/memory_data_movement/store_tile.md`（atomic 参数）
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/basic_data_structures/AtomicType.md`
