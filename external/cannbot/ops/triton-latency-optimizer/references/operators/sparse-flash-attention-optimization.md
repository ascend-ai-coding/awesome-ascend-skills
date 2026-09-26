---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
category: operator_optimization
title: "Attention / Sparse Flash Attention（SFA）类算子优化点库（Triton-Ascend）"
description: "SFA（sp-index-given 随机索引 + MLA-absorb）在 Triton-Ascend 上的瓶颈判别、内核内 gather 连续化暂存主路径、有效方向与证伪方向全表、天花板估算。"
tags: [discrete_memory_access, device_gather, staged_scratch, bottleneck_analysis, profiling, online_softmax, precision_compensation, synchronization_bound, instruction_optimization]
keywords: [sparse_flash_attention, sparse_indices, topk, lightning_indexer, mla_absorb, gather_to_dot, staged_scratch, affine_consume, P_HI_LO, propagate_nan, msprof, MTE2]
created_at: '2026-09-20T00:00:00Z'
updated_at: '2026-09-20T00:00:00Z'
---

# Attention / Sparse Flash Attention（SFA）类算子优化点库（Triton-Ascend）

> **算子特定经验文档**（与 `lightning-indexer.md` 同类，按需加载）。
> 与 `flash-attention-optimization.md`（dense FA）/ `mla-paged-attention-optimization.md`（paged MLA）的区别：
> KV 访问是**逐 token 随机 gather**——没有块规则可跳过、没有页表可仿射化，**随机访存是主场景**。
> 设计期约束见生成期 template `block_sparse_attention.md`（triton-op-generator 插件内置模板，
> 不随本 skill 分发；L1.10 随机索引为主 / L1.14 gather 连续化暂存 / L1.16 随机主场景删 FAST 分支）。
> 证据基础：SFA 随机 topk 16 case（Ascend910B3, 20 cube / 40 vector core，
> triton-ascend 3.2.0 / CANN 9.1.0，msprof `kernel_details.csv` warmup2+active3 与 warmup10+active20 双口径复测）。

## 0. 一句话结论

**SFA 的性能 = «gather 连续化暂存的带宽» 与 «全仿射消费的布局转换开销» 的下界之和**；
在「unmasked gather→dot」lowering 缺陷修复之前，不要试图把 gather 喂进 `tl.dot`——
源码层的正确主路径只有一条：**gather→store 到 per-core scratch（L2 驻留）→ 全仿射 flash 消费**。

## 1. 形态判别（决定后续条目适用性）

| 特征 | 取值 | 影响 |
|---|---|---|
| `sparse_indices` 来源 | 算子输入（上游 lightning_indexer topk） | 无 host 侧建表；随机访存为默认场景 |
| 行内索引结构 | 无重复、升序（可能含 -1/越界填充） | 不可假设连续；需 clamp + score 层屏蔽 |
| KV 头数 | KV_N=1（全 Q 头共享同一稀疏集合） | 每 query token 一份索引，无头维展开 |
| 选块粒度 | token-wise（sparse_block_size=1） | 无块规则可利用；块稀疏 ≠ 块跳过 |
| 语义 | MLA-absorb：`q_nope@K̃_nope^T + q_rope@K̃_rope^T → softmax → @K̃_nope` | K̃ 的 nope/rope 两段都要 gather，KV 行被读两遍（QK 与 PV） |
| 输出 dtype | fp16（QK/PV 走 cube fp16，acc fp32） | `p` 需 hi+lo 二段拆分保精度 |

## 2. 先拆解，再优化：SFA 的瓶颈定位

**阶段拆解（stage/consume 分离测量，case16 B=64/QS=64/K=512/D=512/DR=64）**：

| 阶段 | 实测 | 说明 |
|---|---|---|
| stage（gather 选中行 → per-core scratch） | 3.47 ms | 读 2.36GB + 写 2.36GB ≈ **1.36 TB/s**（纯 gather→store 拷贝上限实测 1.4 TB/s） |
| consume（仿射读 scratch + dot/softmax/PV） | 4.79 ms | 有效带宽仅 ~0.53 TB/s → **布局转换受限，非带宽受限** |
| 融合单 kernel | 7.20 ms | 阶段间有 ~1.03 ms 自然重叠 |

**consume 细粒度拆解（BN=256，逐步叠加）**：

| 组成 | 累计耗时 | 增量 |
|---|---|---|
| loads + QK dots | 0.99 ms | — |
| + `tl.trans(st)`（[BN,16]f32→[16,BN]） | 2.94 ms | **+1.95 ms** |
| + online softmax（max/exp/sum/where） | 3.00 ms | +0.06 ms（**softmax 本身几乎免费**） |
| + 单次 PV dot | 4.39 ms | +1.39 ms |
| + hi-lo 第二段 PV dot | 4.76 ms | +0.37 ms |

**结论**：两个大头是 ① stage 的访存量（只能减流量，不能提速率）；② consume 的 `trans` + PV A 操作数布局转换（编译器 lowering 层，源码无法规避）。

## 3. 主路径结构（内核内 gather 连续化暂存）

```python
# phase1 stage: 每 program 一个 (b, s1)；按 STAGE_BN 分块 clamp 后 unmasked 2D gather
for k0 in range(0, K, STAGE_BN):
    sel = tl.load(idx_row + k0 + s_offs, mask=(k0 + s_offs) < K, other=0)
    selc = tl.minimum(tl.maximum(sel, 0), KVS - 1)
    tv = tl.load(kv_ptr + (kv_base + selc)[:, None] * D + d[None, :])          # unmasked gather
    tl.store(scr_v + (k0 + s_offs)[:, None] * D + d[None, :], tv, mask=...)
    tr = tl.load(kr_ptr + (kv_base + selc)[:, None] * DR + dr[None, :])        # 与 kv 同循环交错
    tl.store(scr_r + (k0 + s_offs)[:, None] * DR + dr[None, :], tr, mask=...)
tl.debug_barrier()                                        # store → load 可见性
# phase2 consume: 全仿射，与 dense FA 同构
for k0 in range(0, K, CONSUME_BN):
    fk_v = tl.load(scr_v + ...); fk_r = tl.load(scr_r + ...)   # 仿射（L2 命中）
    sel2 = tl.load(idx_row + k0 + c_offs, ...)                 # 重读 idx 重建 kv_valid
    st = tl.dot(fk_v, q_nope_t) + tl.dot(fk_r, q_rope_t)
    sc = tl.trans(st) * scale
    sc = tl.where(kv_valid[None, :], sc, NEG)                  # -1 填充屏蔽在 score 层
    ... online softmax (NEG=-1e30 有限掩码, propagate_nan=ALL) ...
    ph = p.to(fp16); pl = (p - ph.to(fp32)).to(fp16)           # p 二段拆分
    acc = acc * alpha[:, None] + tl.dot(ph, fk_v) + tl.dot(pl, fk_v)
```

要点：
- **scratch 尺寸 = NUM_CORES × K × (D+DR) × 2B**，每核一份、随任务覆盖复用（本例 20×512×576×2 ≈ 11.5MB，**可驻留 L2**）；
- stage 用 unmasked gather（满带宽，且规避编译器缺陷），consume 全仿射；
- 数值契约：fp16 dot + fp32 累加；有限掩码 `NEG=-1e30`；多块时 `tl.maximum(..., propagate_nan=ALL)`；PV 必须 hi+lo。

## 4. 已验证有效方向（按收益排序）

| 方向 | 判别依据 | 收益（实测） |
|---|---|---|
| **内核内 gather 连续化暂存**（本文件主路径） | gather 存在且 UB 放不下 fused；per-core working set 可驻留 L2 | 随机访存 vs 同形 CANN 算子 0.070 → 0.4110（**5.9×**） |
| **删 FAST/SLOW 双分支**（输入确定为随机索引时） | 索引来自 topk（非 arange）；组合 kernel 分支影响编译器调度 | case16：双路径 7.6404ms（check 0.157 + sfa 7.483）→ 单路径 7.1966ms（**-5.8%**）；同口径反事实 7.53→7.20（-4.4%）；随机 16 case 几何平均 0.4110→**0.4228** |
| **D 拆半 tiling 换大 BLOCK** | `[BN,D]` int32 地址张量是 UB 隐藏大项 | gather BN 32→64，迭代数减半，**+59%** |
| **PV hi+lo 二段拆分** | fp16 下 p 量化误差被 `Σp|v|/|Σpv|` 放大 | 去 pl 后 MERE 1.384e-3 > 2^-10（阈值），保留后 1e-6 级 |
| **STAGE/ CONSUME BLOCK 扫描** | — | `STAGE_BN=64 + CONSUME_BN=256 + multibuffer=False` 为最优组合（本例） |

## 5. 证伪方向全表（实测，勿重复尝试）

| 方向 | 实测数字 | 结论 |
|---|---|---|
| **Device-side gather + 全局 workspace**（拆独立 gather kernel，workspace 常驻 HBM） | case16：G 3.67~3.76 + A 5.42~6.06 = 9.13~9.87ms vs 现行单 kernel 7.198ms（**+27%~+37%**）；case1：G 7.83 + A 10.14~10.99 = 19.2~19.9ms vs 13.37ms（**+44%~+49%**）；输出 bitequal | **证伪**：per-core scratch 可驻留 L2 时，workspace 的 HBM 往返 + 两 kernel 串行无重叠，全面劣于内核内暂存 |
| kernel A 用 `grid=(tasks,)`（每 program 一任务，非持久化） | 11976 ms | 反模式：小任务 program 派发开销爆炸 |
| **masked gather 直喂 `tl.dot`** | 每迭代 ~12µs 固定开销（与 BN 无关）；K=512/BN=64 → 8 迭代 → ~20ms | 证伪（事务发起开销，BN 放大不摊薄） |
| **unmasked 2D gather 直喂 `tl.dot`** | 数值为垃圾值；极简形态编译失败 `Unknown core type: llvm.func @malloc` | 编译器 lowering 缺陷，源码层绕开 |
| PV 单段（去掉 pl） | MERE 1.384e-3 > 2^-10（max_abs 仅 6.1e-5，是 MERE 卡阈值） | 精度否决 |
| softmax 改在 `[BN,16]` 上做 axis=0 归约 + 转置 f16 | consume 4.66 → 9.87 ms | 证伪（axis=0 归约代价远超转置收益） |
| p 存 scratch 再转置仿射读回 | 6.72 ms | 证伪（全局往返劣于 UB 内 trans） |
| kr 独立循环 + 更大 BLOCK（128/256/512） | 7.69~7.89 ms（同循环交错 7.23） | 证伪（破坏 kv/kr 交错调度） |
| 任务级双缓冲软件流水（stage(t+1) ∥ consume(t)） | 7.44 ms，且与基线不完全 bitequal | 证伪 |
| stage unroll×2 / ping-pong / multibuffer / grid=40 / stage D 拆半 / `multiple_of` 提示 | 分别：更差 / 14.5ms 级 / BN=256 时 L1 溢出（5013504/4194304 bits）、BN=128 时 7.75ms / 7.71ms / 5.09ms / 零效果 | 全否决 |

**共同教训**：SFA 的 stage 瓶颈是**事务指令流**（MTE2 逐行小事务数量 = 任务数×K 行×2 张量，结构性不可合并），
不是流水缺失——所以 multibuffer/双缓冲类手段一律无效；
consume 的瓶颈是**布局转换**（trans + PV A 操作数），不是带宽或算力。

## 6. 天花板估算模板

```
单 case 下界 ≈ stage_floor + consume_floor − overlap
  stage_floor   = (K·(D+DR)·2B × tasks × 2) / 拷贝带宽        # 读+写；拷贝带宽按纯 gather→store 实测
  consume_floor = 迭代数 × (trans/PV 布局转换常数 + softmax + dot)   # 与 K/BN 成正比
  overlap       = 单 kernel 中两阶段的自然重叠（本例 ~1.0ms / 7.2ms）
若某方向声称能大幅提速：先确认它减少了上述三项中的哪一项；三项都不减的方案一律是噪声。
```

## 7. 与其他文档/约束的关系

- **优化点 4（离散访存）**：SFA 是"随机读"第三条路径（内核内暂存）的典型；三条路径选型见 `discrete_memory_access.md`。
- **优化点 22（Device-side gather）**：适用边界见 `device-side-gather.md`「反例与适用边界」——workspace 方案仅在 per-core working set 放不进 L2、或 gather 跨多 pass 重复时才可能占优。
- **multibuffer/双缓冲**：SFA 属"加缓冲无效"的典型形态，判据见 `multibuffer-and-double-buffering.md`。
- **生成期约束**：`block_sparse_attention.md` L1.10（随机为主场景）、L1.11（D 拆半）、L1.12（掩码缺陷）、L1.14（暂存骨架）、L1.15（禁拆 kernel）、L1.16（随机主场景删 FAST 分支）。
