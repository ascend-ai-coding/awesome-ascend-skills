---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
category: operator_optimization
title: "Attention / FlashAttention 类算子优化点库（Triton-Ascend）"
description: "FA 类算子在 Triton-Ascend 上的瓶颈判别、迭代数目标函数、有效方向与证伪方向全表。"
tags: [synchronization_bound, tiling_optimization, pipeline_optimization, bottleneck_analysis, profiling, launch_overhead, instruction_optimization]
keywords: [flash_attention, online_softmax, sync_block, pipe_barrier, propagate_nan, aic_scalar_ratio, MIX_AIC, tl.dot]
created_at: '2026-08-11T00:00:00Z'
updated_at: '2026-08-11T00:00:00Z'
---

# Attention / FlashAttention 类算子优化点库（Triton-Ascend）

> **优化点 #30 的参考文档。** 命中条件见 `SKILL.md` 索引表。
> 证据基础：**三个** FA 类算子各 50 case 的完整优化轨迹（Ascend910B2/910B2C，24 cube core / 48 vector core，
> triton-ascend 3.2.0~3.2.2 / CANN 8.5.1~9.1.0）：
> `FlashAttentionV2`（含 4 个投影 GEMM，3.68 / 换精度契约后 3.30）、
> `FlashAttentionFwd`（无投影、可变 mask，2.17 → 6.24）、
> `MultiQueryAttention`（含投影 + KV 共享，0.1052 → 2.3724，约 93 轮迭代）。
> 三者形态不同（有/无投影 GEMM、固定/可变 mask、不同精度契约、不同编译器版本），
> **独立收敛到同一瓶颈结论**——这是本文可外推的根据。
> ⚠️ 三者之间存在 **3 条直接冲突的结论**，判别前置条件见 template 卡的「跨算子结论冲突的统一判据」。
> 设计期与编码期的对应约束见 `@../../../../plugins-official/triton-op-generator/template/flash_attention.md`。

---

## 0. 一句话结论与目标函数

**FA 类算子在 triton-ascend 上的终局瓶颈是 Cube↔Vector 跨核同步，不是算术量、不是访存连续性、不是 dtype。**

正确的目标函数不是 FLOPs、也不是"向量指令条数"，而是：

```
代价 ≈ (循环体内互相有 UB 依赖的向量算子个数) × (KV 迭代总数)

iters = B*H*ceil(S_Q/BLOCK_Q) × ceil(scan_span/BLOCK_KV)
        └─ 放大 BLOCK_Q ──┘   └─ 收缩 span / 放大 BLOCK_KV / 消除对齐取整 ─┘
```

两个算子的**全部有效增益无一例外落在这个式子的三个因子上**；所有"减少算术量 / 改善访存 / 换 dtype / 加核"的尝试全部零收益（§3）。

### 0.1 为什么杠杆是 12 倍

指令级仿真（`msprof op simulator`）给出了 ratio 给不了的那一层：

| core | 流水占用 |
|------|---------|
| cubecore | FLOWCTRL 36.6%（`WAIT_FLAG_DEVI` 14 次 × **3400 cyc/次**）、**CUBE 仅 9.7%** |
| veccore | VECTOR 69.9%，**其中 `BAR` 就占 55.4%**（140 次 × **712 cyc/次**）、FLOWCTRL 20.5%、**真实算术合计仅 ~9%** |

`BAR` = IR 里的 `pipe_barrier[<PIPE_V>]`，编译器几乎在每条向量算子后面插一条，而一条真实向量指令平均只有 ~150 cycle。

⇒ **每减少一条循环内的向量指令，实际省下的是「指令 + 一条 712 cycle 的 BAR」，杠杆约 12 倍。**

第三个算子（MQA）的仿真进一步给出两条**只有逐指令数据才能得到**的结论：

| 观测 | 小 case（M=303） | 大 case（M=1794） |
|---|---|---|
| VEC / CUBE wall-clock 比 | 1.68x | **1.84x** ⇒ **关键路径在 Vector，不在 Cube** |
| CUBE 同步等待占比 | 69.0% | **73.6%** |
| VEC 同步等待占比 | 71.5% | **77.7%** |
| `BAR PIPE:VEC` | 38.2%（×49） | **45.0%（×715）** |
| `MMAD` | 3.4% | **2.6%** |

**规模越大 `BAR` 占比越高、`MMAD` 越低**——这正是大 M 档 speedup 最差的直接原因。
该 case 内层迭代总数仅 45 次却发射了 **715 个 `BAR`**，即**平均每次 KV 迭代插入约 16 次向量流水排空**：
这不是循环级同步，是**逐向量算子级的 drain**。
⇒ 同理，**把 BLOCK 翻倍使迭代数减半，等于整套 BAR 直接少一半**——这是本类算子最大的一笔。

---

## 1. 先判别形态：决定哪些条目适用

写第一行优化代码前先回答四问，答案决定后面哪些章节适用：

| # | 问题 | 影响 |
|---|------|------|
| Q1 | 有没有独立的投影 GEMM（`nn.Linear` 之类）？ | 有 → §2.6 / §2.7（权重预重排、分块分档）是最大的一类收益；没有 → 这两条完全不适用 |
| Q2 | mask 是固定的还是可变属性（causal / window / softcap 作为运行时参数）？ | 可变 → §2.2（区间收缩）+ §2.4（constexpr 特化）合计可达 +58%；固定 causal 拿不到这么多 |
| Q3 | 参考实现的 attention 在什么精度上算（有没有 `.float()`）？ | 决定第二个 dot 能否降精度，以及**趟数结构**（见 §4） |
| Q4 | 评测口径？ | 几何平均 ⇒ **每个 case 权重相同**，小 case 与大 case 同等重要，不能只优化大 case |

### 1.1 瓶颈类型判别：每次迭代成本反推法（建议列入标准流程）

从 `kernel_details.csv` 取每个 case 的 kernel `Duration`，按 tile 参数精确算出 KV 迭代总数，再算：

```
每次 KV 迭代成本 = Duration / ceil(iters / 核数)
```

| 观测 | 判定 | 后续 |
|------|------|------|
| 成本 **几乎与 BLOCK_Q / BLOCK_KV / D / dtype 无关**（实测 4~10us，最小 case 单次迭代 5.3us、最大 case 2464 次迭代平均 5.99us/次） | **固定开销（跨核同步）主导** | 目标函数 = 最小化迭代数，走 §2 |
| 成本随 tile 线性增长 | 算力/访存主导 | 本文 §2 的排序不适用，回到通用优化点（#2 tiling / #13 autotune） |

**"成本与 tile 大小无关"这一条，直接排除了算力瓶颈和访存瓶颈**，是最省事的一次性判别。

### 1.2 逐核负载不均衡（只有 simulator 能看见）

cubecore 耗时呈**两档**分布时，直接对应尾波不满：

```
43 us × 18 核        30 tasks ÷ 24 cores
83 us × 6  核   →    6 核拿 2 个 task，18 核拿 1 个
wall-clock 由 6 个双任务核决定 ⇒ 有效核利用率仅 30/(24×2) = 62.5%
```

全 50 case 统计（`maxT = ceil(tasks/24)`）：100% 效率 30 个、80~96% 11 个、**62~75% 9 个**。

⚠️ **on-device 聚合 profiling 完全看不到这个**——它只给全核平均 ratio，尾部空转被平均掉了。
由此还导出一条：**固定 tile 在 `S < BLOCK` 时纯空转**（某算子 27/50 case 中招，最小 case 的 `S=3` 在 `SQ=256` 下空转 85 倍）
⇒ tile 必须按实际 shape 收缩（`min(BLOCK, ceil(S))`），与"开到 UB 上限"是同一条规则的两面。

### 1.3 `msprof op simulator` 采集要点

- `--soc-version` **必须用本机 SoC**（如 `Ascend910B2`）。参考命令里的 `Ascend910_9372` 属 910_95 系列，soc 不匹配会得到**另一套架构的流水**，结论不可用。
- 应用退出阶段常报 `Child process killed by signal 11` + `Running task failed`，但只要 `Profiling data parse finished` 正常、CSV 已落盘，**数据可用**，不是采集失败。
- 产物 `*_instr_exe.csv`（逐指令逐 pipe 周期）+ `trace.json`（按 pipe 分 lane 的时间线，可直接求 Cube/Vector 交集算真实重叠度——实测只做到 **35~43%**，离满重叠差一半以上）。

---

## 2. 有增益的方向（按收益排序，每条含判别依据与失效边界）

> 标注含义：**【通用】**=两算子一致，可直接套；**【条件性】**=判据与条目同行；**【算子特定】**=实测值，非普适。

### 2.1 全面 `tl.dot` 化 —— 前置条件，不算优化 【通用】

- **判别**：profiling 里 `aic_mac_ratio < 10%` 且代码里有 `for d in range(head_dim)` ⇒ 直接重写，不用扫参。
- **现象**：把 `scores` 写成逐维外积（`for d: scores += q[:,None]*k[None,:]`）、用 `tl.where(d_offs==d, ...)` 逐维散写 `acc`，Cube 完全空转，benchmark 直接超时无有效数据。
- **改法**：`tl.dot(q, tl.trans(k))`，**操作数保持原生 dtype、累加器 fp32**（`tl.dot` 默认 `out_dtype=fp32`）。
- **实测**：不可测量 → 2.3554；`aic_mac_ratio` 7.5% → 17.3%。【算子特定】

> ⚠️ **不要把操作数升 fp32 再 dot**。"fp32 路径禁用 `tl.dot`"这个流传的约束，指的是**升精度后走异构路径**的写法；
> 原生 dtype 操作数 + fp32 累加器在 fp32 输入下实测精度 50/50 全过。两者是不同的事，别混。

### 2.2 ★ KV 扫描区间按 causal / window 收缩 【条件性：仅当 mask 可裁剪】

把 mask 条件折叠成一个区间，据此收缩循环上下界；区间外整块被 mask 成 `-inf`，对 online softmax 无贡献，跳过**数值等价**。

```python
# host 侧：causal 与 window_right 折叠成单一上界，window_left 折叠成下界
upper = min(0 if causal else INF, window_right if window_right >= 0 else INF)
lower = -window_left if window_left >= 0 else -INF
# kernel 内：对 q-block [q0, q0+BQ) 取并集
kv_lo = max(q0 + delta + lower, 0)              # delta = S_K - S_Q
kv_hi = min(q0 + BQ - 1 + delta + upper + 1, S_K)
for kv_start in range(kv_lo, kv_hi, BLOCK_KV): ...
```

- **判别**：算冗余倍数 = `S_K / (每行真正需要的 key 数)`。实测最高 70x（`causal + wl=5`, `S_K=423`, 每行真需 6）。
- **实测**：3.047 → 3.9691（**+30%**）；单 case `#48` 5.87 → 0.78 ms。【算子特定】
- **必做对照组**：留一个**不可裁剪**的 case（无任何 mask 约束），它改动前后应当**不变**——立刻确认收益来源正确。
- **失效**：无 mask 或 mask 不可表达为区间（数据相关的稀疏 mask）时不适用。

### 2.3 ★ `kv_lo` 不对齐 + (BLOCK_Q, BLOCK_KV) 联合放大 【条件性：两者必须成对做】

```python
kv_lo = max(q0 + delta + lower, 0) // BLOCK_KV * BLOCK_KV   # ❌ 平均多扫 BKV/2
kv_lo = max(q0 + delta + lower, 0)                          # ✅
```

扫描区间通常只有 1~3 个 block，对齐平均多出 `BKV/2` 个 key ⇒ **凭空多一整次迭代**，而每次迭代要付 8 个跨核握手：

```
对齐:   iters/bh = ceil(S_Q/BQ) × (1 + ceil((BQ+W)/BKV))   ← "+1" 会吃掉放大 BKV 的全部收益
不对齐: iters/bh = ceil(S_Q/BQ) ×      ceil((BQ+W)/BKV)
```

| 配置 | 子集几何平均 |
|------|--------------|
| 128x64（对齐） | 3.87 |
| 64x128（**对齐**） | 3.92（**+1.2%，看起来这条路没用**） |
| 64x128（**不对齐**） | **5.00（+29%）** |

- **实测**：正式闭环 4.8343 → 5.8316（**+21%**）。【算子特定】
- ⚠️ **方法论教训**：若先单独试"放大 BLOCK_KV"，会得到 +1.2% 并放弃这个方向。**两个改动互相掩盖，必须一起测。**
  遇到"理论上该有收益但实测几乎为零"，先检查是不是有另一个因子把收益吃掉了。

### 2.4 ★ mask 分支 constexpr 特化 + 上下界折叠 【条件性：mask 为运行时标量时】

```python
# ❌ 3 次二维比较 + 3 次 or，分支无法 DCE
mask = full([BQ,BKV], False)
if causal:            mask |= (rel > 0.0)
if window_left >= 0:  mask |= (rel < -window_left)
if window_right >= 0: mask |= (rel > window_right)

# ✅ constexpr 分支 + 上下界折叠 ⇒ 2 次比较 + 1 次 and
keep = kv_ok[None, :]
if HAS_UPPER: keep &= (rel <= upper)
if HAS_LOWER: keep &= (rel >= lower)
```

- `HAS_UPPER` / `HAS_LOWER` / `HAS_SOFTCAP` 作为 `tl.constexpr` 传入，让编译器把不适用的整段 DCE 掉。代价是每种组合一个特化版本（实测 ~8 种），编译只发生一次且不计入 benchmark。
- **上下界折叠**：`causal=True` 时恒有 `window_right=0`，`rel>0` 与 `rel>wr` 完全重合，折叠后少一次二维比较。
- **实测**：2.5115 → 3.047（**+21%**）。【算子特定】

### 2.5 ★ BLOCK 开到 UB 上限 【通用】

内层迭代总数 = `B*H*ceil(S/BLOCK_Q) × ceil(span/BLOCK_KV)`，两个 BLOCK 同时翻倍 ⇒ 迭代数变 1/4。

| SQ/SKV | 64/64 | 128/64 | 64/128 | **128/128** | 256/* |
|--------|-------|--------|--------|-------------|-------|
| 子集几何平均 | 0.697 | 1.129 | 1.155 | **1.42** | 编译失败 |

**UB 上限由编译器直接报数字**，看错误信息就知道差多少：

```
error: ub overflow, requires 2646528 bits while 1572864 bits available!   # 1572864 bit = 192KB (UB)
error: cc overflow, requires 2097152 bits while 1048576 bits available!   # 1048576 bit = 128KB (local buffer)
```

tile 选取规则（35 case × 5 组扫描验证）：

```python
BLOCK_KV = min(128, ceil16(S_K))                    # 尽量大，但不超过实际 key 数
BLOCK_Q  = 满足 BQ*BKV <= 8192 的最大 2 的幂         # 面积上限 = scores tile fp32 32KB，不是拍脑袋
BLOCK_Q  = min(BLOCK_Q, max(32, ceil32(S_Q)))
while BLOCK_Q * BLOCK_D * elem_size > 64KB: BLOCK_Q //= 2
BLOCK_D  = ceil16(D)                                # ⚠️ 不是 next_pow2
```

三条容易踩错的点：

- **`BLOCK_D` 用 `ceil16(D)` 不用 `next_pow2(D)`**：triton-ascend 支持非 2 的幂 `tl.arange`。D=96→96 vs 128、160→160 vs 256、192→192 vs 256，`next_pow2` 白白浪费 30~60% tile。
- **`BLOCK_KV` 不要限定在 {32,64,128}**：曾限制成 2 的幂，`S_K=33/37/47` 三个 case 的 BKV 从 48 掉到 32，分别回退 20%/27%/33%。
- **UB 上限不可用简单阈值预测**：同为 `128x128`，D=96 fp16 编得过、D=128 fp16 `PlanMemory Failed`、**D=160 fp16 反而编得过且比 64x128 快 22%**。占用还受 constexpr 分支组合（活跃变量数）影响，**任何基于 `BQ*BKV*BD*esz` 的代理阈值都会误判**（这也直接导致 §3.6 的自动回退方案失败）。

⚠️ **结构性改动之后必须重扫 tile，不要查历史清单**：一个算子在区间收缩后最优解从 `128x64` 变成 `64x128`（与旧结论相反）；另一个在权重重排后重扫，结论未变——但那是**验证过**的，不是假设。

### 2.6 ★ 权重按 tile 形状预重排成连续布局 【条件性：仅有投影 GEMM，且 `MTE2>60% 且 MAC<50%`】

投影权重是常量，可以在 `__init__` 里用 CPU 任意重排，运行期零开销。

```python
# nn.Linear 的 [out, in] → 转置成 [in, out]（数值逐位不变）
# → 补零到 [K_pad, NBN*BLOCK_N] → 拆成 [NBN, K_pad, BLOCK_N] 连续
t = torch.zeros(kp, nbn * blk_n, dtype=dtype)
t[:d_model, :d_model] = w.t()
packed = t.view(kp, nbn, blk_n).permute(1, 0, 2).contiguous()
```

kernel 侧 b tile 基址 `w_ptr + block_n * (K_pad*BLOCK_N)`，行跨度恰为 `BLOCK_N` ⇒ 整个 tile 是一整段连续内存，且补零后**不再需要掩码**。

- **实测**：+5.0%，**无退化 case**；画像上 linear 从访存受限（MTE2 62~90%）变成算力受限（MAC 58%）。【算子特定】
- **两级递进**：先做 `[out,in] → [in,out]`（最内轴 stride 从 `in_features` 变 1，**数量级收益**），再做按 tile 重排（+5%）。
- ⚠️ **同一思路搬到 attention 的 k/v 上完全无效**，见 §3.3。

### 2.7 ★ 投影分块：dtype 分档 + 并行度保护（两半缺一不可） 【条件性：仅有投影 GEMM】

只放大不保护 = **负收益**：

| 轮次 | 改动 | 正式 speedup |
|------|------|--------------|
| 基线 | 16-bit/fp32 都用 (64,128,256) | 3.2632 |
| 只放大 | 16-bit 放到 (128,256,256) | **3.1412（−3.7%，拒绝）** |
| 放大 + 并行度保护 | 同上 + 块数保护 | **3.3503（+2.7%）** |

根因是**块数塌陷**：`M=89` 时 `BLOCK_M=128` 下只剩 3 个块，24 个核只用了 3 个。

```python
par_thr = num_cores // 2            # 阈值扫参：24→3.6633、16→3.8283、12→3.8576
while nblocks(blk_m, blk_n) < par_thr and (blk_n > 64 or blk_m > 16):
    交替 blk_n //= 2 / blk_m //= 2  # 先缩 N 再缩 M
```

- **fp32 必须单独一档**：元素大一倍，同样的 BLOCK 撑爆 UB（实测 fp32 最优 (64,128,256)，16-bit 最优 (128,256,256)）。

### 2.8 ★ `tl.maximum(propagate_nan=ALL)` —— 必须按 KV 块数分档 【通用】

`tl.maximum(a,b)` 默认 `propagate_nan=NONE`，在 IR 里被展开成 **7 条向量指令**：

```
vcmp_eq_1d_float(a,a) → vnot_1d_bool     # isnan(a)
vcmp_eq_1d_float(b,b) → vnot_1d_bool     # isnan(b)
vmax_1d_float
vsel_vv_1d_bool_float × 2
```

指定 `ALL` 后直接对应硬件单条 `vmax`。

```python
if MULTI_BLOCK:      # host 侧算出的 max_kvblk > 1
    m_new = tl.maximum(m_i, tl.max(scores, axis=1), propagate_nan=tl.PropagateNan.ALL)
else:
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
```

- **语义等价性论证（必须写进注释）**：`scores` 永不含 NaN——输入有限，`m_i` 用**有限值**初始化（不是 `-inf`），被 mask 的位置经 `exp(-3e38 - m_finite)` 直接下溢成 0，不存在 `inf-inf`。两种语义结果完全一致。
- ⚠️ **全局开启是负收益，必须分档**（两个算子独立复现）：

| 配置 | 结果 |
|------|------|
| 不开（基线） | 6.073 |
| **全局开启** | **5.5302（−9%）** |
| **按 `max_kvblk > 1` 分档** | **6.2362（+2.7%）** |

  另一算子上同样现象：多块 case 快 4.4%（最大 −21.5%），单块 case **退化 25~45%**（两次重复一致，机理未查明）。
- ⚠️ **它主要是性能开关，不是正确性要求**——迁移文档把它讲成正确性项，实测在本平台上首先是性能项，且在部分 shape 上是负收益。**按数据分档，别照抄。**
- ⚠️ 判定采纳靠的是**逐 case 的 impl 时延比**（生效组 0.958 / 关闭组 1.004），不是总分——只有 11/50 走多块路径时，总分只 +2.7%，落在噪声带边缘。

### 2.9 kernel 内 fp32 张量除法 → host 侧倒数乘法 【条件性：仅当向量核在关键路径】

```python
# ❌ 每个 KV 迭代对 [BLOCK_Q, BLOCK_KV] 做一次 fp32 除法
scores = tl.dot(q, tl.trans(k)) / tl.sqrt(tl.cast(Dh, tl.float32))
# ✅ host 侧算好 head_dim ** -0.5，kernel 里只剩一次 FMA
scores = tl.cast(tl.dot(q, tl.trans(k)), tl.float32) * inv_sqrt
```

- **一个算子上 −36%**（AIV 的 fp32 `vdiv` 是多拍迭代实现，且这是循环内对整个 tile 的操作）；
- **另一个算子上两次证伪**（打平）。差异原因：后者基线的 scale 本来就是乘法，剩下的除法只在 softcap 路径且只覆盖 40/50 case；且其 `aiv_vec_ratio` 只有 0.25~0.32，**向量核有 2/3 时间在等 Cube，减少向量指令不落在关键路径上**。
- **统一判据**：有仿真数据就比 `veccore 总 cycle` vs `cubecore 总 cycle`；只有 ratio 就看 **`aiv_vec_ratio ≳ 0.35`**。低于此值时，**任何"减少向量指令条数"的优化都不要做**。
- **失效边界（重要）**：**不能外推到循环外的除法**。末尾 `o_acc / l[:,None]`（每 task 只做一次）改成倒数广播乘，实测 2.8710 vs 2.8821 —— **无收益**。判据是「该除法是否在内层循环里、作用于二维 tile」。

### 2.10 CV 指令重排（把 Cube 的 `tl.dot(p,v)` 提前发射） 【条件性：成本低，值得每次试一次，但不要预设收益】

```python
m_new = tl.maximum(m_i, tl.max(scores, axis=1))
p     = tl.exp(scores - m_new[:, None])
pv    = tl.dot(p, V_tile)          # Cube 提前发射
p_sum = tl.sum(p, axis=1)          # Vector 延后
alpha = tl.exp(m_i - m_new)
acc   = acc * alpha[:, None] + pv
l_i   = l_i * alpha + p_sum
```

**三个同类算子三个结果：0（1.4503 vs 1.4676）/ +1.9% / +6.4%。** 纯重排、数值等价，改动成本极低——试一次，但**幅度不可预期，不要写进设计假设**。

### 2.11 低精度输入：`p` 二段拆分替代 V 升 fp32 【条件性：仅 fp16/bf16 输入】

`tl.dot(p_fp32, V.to(fp32))` 需要在 UB 里物化 `[BLOCK_KV, BLOCK_D]` 的 fp32 副本（64x128 时 **64KB，是 UB 里最大的一块**），而 V 本身只有 11(fp16)/8(bf16) 位有效位，升 fp32 一位不增。

```python
if PV_SPLIT:                       # = (dtype is not float32)
    p1 = p.to(in_dtype)
    p2 = (p - p1.to(tl.float32)).to(in_dtype)
    pv = tl.dot(p1, V_tile) + tl.dot(p2, V_tile)     # ~22 bit，两次同构 dot
else:
    pv = tl.dot(p, V_tile.to(tl.float32))            # fp32 输入：本就是 no-op，别动
```

- **实测 +2.7%，精度 50/50 全过。**【算子特定】
- **失效**：fp32 输入**不要开**——没有 buffer 可省，且 fp32 的判据阈值最紧，22 位没有余量。
- ⚠️ 注意区分两个实验：**直接降精度**（`p` → 输入 dtype，~11 bit）实测 27/50 越阈 ✗；**二段拆分**（~22 bit）✅。前者的结论不能用来否定后者，反之亦然。

### 2.12 其它小项

| 改法 | 结果 | 备注 |
|------|------|------|
| K 直接读成转置态（去掉 `tl.trans`） | +0.7% | ⚠️ **机理假设被 IR 证伪**：原以为省一对握手，改完循环体内 `sync_block` 仍是 8 个、一个没少。收益来自 load 路径。**收益真实但机理写错会误导下一个人，故如实记录** |
| grid 收缩到 `min(核数, tasks)` + 核内步长循环 | +0.5% | 基础写法而非优化项；多一层 `scf.for` 的代价几乎把省下的抵消 |
| 删除 host 端 permute/contiguous/pad | +16%（保守值） | 见 §5.2 口径说明；profiling 时间归属里出现 `Transpose`/`ViewCopy`/`ZerosLike`/`Slice` 就直接删，不用扫参 |

---

## 3. 无增益 / 负收益 / 编译器阻塞的方向（**这一节比第 2 节更值钱**）

所有条目都是**单变量对照**。噪声水平见 §5.3。

### 3.1 编译开关：14 个逐个单测，命中率 0 ✗ 【通用】

两个算子分别测了 5 个和 9 个开关：

| 开关 | 结果 |
|------|------|
| `enable_ubuf_saving` / `ops_reorder` / `sync_solver` / `enable_auto_bind_sub_block` / `enable_vf_fusion` | 全部落在 ±2% 噪声内 |
| `limit_auto_multi_buffer_of_local_buffer="no-limit"` | 早期看似 +3.4%，换基线包夹协议复核后落回 +0.7%（噪声内）；**换精度契约后变成 −9%，明确有害** |
| `multibuffer=True` | **−10%** |
| `enable_mixed_cv=True` | **−9%**（文档明确只在 910_95 生效） |
| `enable_flatten=False` | 编译失败（CANN 8.5.1 未知参数） |

第三个算子逐一实测约 **30 个参数**，正收益仅 3 个（`enable_ubuf_saving` +4.0%、`limit_auto_multi_buffer` +0.8%、`ops_reorder` +0.7%），其余无效或有害（`enable_dynamic_cv_pipeline` 0/50、`enable_preload` 15/50 RuntimeError、`num_stages=3` −1.4%、`sync_solver` −2.0% …）。

两条比"命中率低"更重要的经验：

- ⚠️ **官方 FA 的参数组合不能整体照搬**（实测 0/50 编译失败），必须逐个拆分验证；
- ⚠️ **增益不叠加**：`enable_ubuf_saving`(+4.0%) 与 `BLOCK_SQ=256`(+3.2%) 单独都是正收益，**组合后反而降到 1.97 —— 二者争同一块 UB**。正确做法是把释放出的 UB **定向给最缺的那一档 case**；
- 同一参数对不同 kernel 效果相反：`ubuf_saving`/`ops_reorder` 对**标量/串行受限**的 attention kernel 有效，加到**访存受限**的投影 kernel 上反而 −1.1%。

**结论**：**最多花 20 分钟逐个测一遍然后放弃，不要穷举。**

### 3.2 "减少向量指令条数"整类 —— 三次证伪 ✗ 【条件性判据见 §2.9】

| 改法 | 减少的指令 | 结果 |
|------|-----------|------|
| 掩码链合并成单次二维比较（12 条 → 4 条） | 8 条 | **0%**（2.9239 vs 2.9448） |
| `BLOCK_D == D` 时 constexpr 消掉 d 方向掩码（45/50 case 适用） | 每次 load 一个二维布尔量 | **−1.3%（噪声内）** |
| `libdevice.tanh` 消掉 softcap 路径的两处 fp32 张量除法 | 2 条 | **两次测量均打平** |
| bool 掩码换 fp32 加性 bias | — | **编译失败**（多一个 `[SQ,SKV]` fp32 中间量，需 1656832 bit） |
| 运行时 `scf.if` 只在对角/边界块做掩码 | — | 噪声内 |

**证伪了什么**：`aic_scalar_ratio = 73%` **不等于**「标量算术过多」。把向量指令砍掉 1/4 一点不动。
**判据应该是「这条指令值多少拍」，不是「有几条指令」**——fp32 `vdiv` 一条抵几十条普通向量指令；掩码链虽有 12 条，但都是单拍的比较/选择。

> 例外并不矛盾：在**两趟结构**（每块做两遍掩码）下把掩码合并成一次比较**有 +2%**，因为那时掩码链的 BAR 不再与相邻算子的 BAR 合并。同一改动在不同结构下结论相反 —— 这正是 §2.5 末尾"结构性改动后必须重扫"的另一个侧面。

### 3.3 布局连续化（attention 侧）✗ 【通用】—— 与 §2.6 形成对照

权重连续化对投影 GEMM 有 +5%，于是自然想到把 q/k/v permute 成连续布局。**用一个 5 分钟的 `torch.contiguous()` 诊断实验直接证伪**：

| | 跨步视图 | 连续布局 |
|---|----------|----------|
| 算子 A（4 个 case） | 96.5 / 54.3 / 199.0 / 67.8 us | **106.9 / 57.3 / 201.9 / 70.5 us**（一个都没变快） |
| 算子 B（17 个 case 几何平均） | **3.9755** | 3.8047（**15/17 变慢**） |

**方法论价值**：想验证"布局是否是瓶颈"，**不要先去实现零拷贝的正确方案**，先用 `.contiguous()` 花几分钟测一次——它引入的额外 kernel 不影响你要看的那个 kernel 的 `Duration`。这条实验省掉了一整轮（1~2 小时 + 有回退风险）的工作。

### 3.4 循环结构改动 ✗ 【通用】

| 改法 | 结果 |
|------|------|
| KV 循环拆「无掩码整块」+「对角/尾块」 | **编译失败**（循环体复制使 UB 翻倍，需 2646528 bit / 可用 1572864 bit；两轮都试，向量指令减少后仍失败） |
| `S ≤ BLOCK_KV` 时去掉 kv 的 `scf.for` | 小 case 子集 7.3508 → **6.6056（−10%）**；去掉循环后**编译器失去 multi-buffer 流水机会** |
| 同上，另一算子上三种写法 | **全部触发 aicore exception 507015**；逐 shape 定位无可靠规避谓词（`(64,32)`/`(64,64)` 崩，`(64,80)`/`(32,48)` 正常）⇒ 判为 codegen 缺陷 |
| 保留 `for`、只把 online softmax 的累加改成赋值（单块路径 `alpha≡0`） | `error: 'scf.for' op 0-th result ...`，24/50 编译失败（破坏 loop-carried 值结构） |

⇒ **「少一层 `scf.for` 必然更快」这个直觉在本平台上不成立。**
⇒ 单块特化是一笔"确定存在但拿不到"的收益（某算子 39/50 case 适用），两种独立写法都被编译器阻塞。

> 👉 遇到同类场景：先用单 case 脚本快速试崩 2~3 个 shape（每次 ~1 分钟）再决定是否投入。
> ⚠️ aicore exception 会**毒化设备上下文**，一个进程只能测一个配置，必须每次重启进程。

### 3.5 `al.scope(core_mode)` 手写 CV 流水线 —— 编译器崩溃，方向关闭 ⛔

`cv-fusion-pingpong.md` 把它列为跨核同步瓶颈的**唯一出路**。**门槛测试（10 分钟，强烈建议投入前先做）**：

| kernel | 结果 |
|--------|------|
| 单 scope（对照组） | ✅ OK，maxdiff 7.6e-06 |
| **两段 scope（cube + vector）** | ✗ **`bishengir-compile` LLVM stack dump** |

另一算子在 910B2 上逐个原语实测，给出更明确的证据：

| 原语 | 910B2 结果 |
|---|---|
| `al.scope(core_mode=…)` | 玩具 kernel ✅ / 真 kernel 内 **0/50 MLIR 失败** |
| **`al.fixpipe`** | ❌ `RuntimeError: only supported on Ascend910_95` |
| `bl.alloc` / `bl.to_tensor` | ❌ 编译失败 |
| `al.sync_block_set/wait` | ❌ 运行时 vector core exception |

`fixpipe` 是 Cube→Vector 的片上通道、是整套方案的核心 ⇒ **在 910B 上此路不通，不必尝试**。
> **规则：读 CV / 同步类文档前先确认目标架构**——910_95 是 Reg-based、910B 是 Mem-based，该区分同时决定 CV API 的可用性。

910B2 + CANN 8.5.1 上不可用。API 符号齐备（`scope` / `sync_block_set` / `copy_from_ub_to_l1` / `fixpipe` 都能 import），**但编译不过**——注意 extension 里有 `is_compile_on_910_95`，该特性很可能是 910_95 专属。

配套的 `al.multibuffer(tensor, size)`（按张量给 K/V 开双缓冲）实测 **40/50，失败集合恰好是全部多块 case**，`NaN 位置不匹配: Framework=0/183296, Implementation=48000/183296` —— 手动多缓冲破坏了 K/V tile 的依赖跟踪。

> 👉 **方法论**：这类"文档指明但工作量巨大"的方向，一定先用最小 kernel 做门槛测试。本次 10 分钟关闭了一个原本要 1~2 天的方向。

### 3.6 其它已证伪 ✗

| 改法 | 结果 |
|------|------|
| grid 超订（24 → 48） | 无收益（MIX kernel 超订不会带来同步延迟的重叠） |
| `exp2` + log2e 折进 scale | 无收益（`tl.exp` 本就走 exp2 路径） |
| scale 与 mask 合并成一次 FMA | **编译失败**（UB 超限，需 1656832 bit） |
| 波次量化的单目标代价模型（`ceil(块数/核数) × BM × BN`） | **1.90 vs 2.41（−21%）**——模型不含访存项，会选出 16×64 这类小 tile。经验扫参得到的分档已隐含"算力/访存/波次"三者折中，别用单目标模型替换 |
| tile 按代价排序 + 编译失败自动回退 | **6.1009 vs 6.1012 完全打平**——代价模型能选对"最优的那一个"（13/14），但次优候选之间排序不可靠，最优项因 UB 失败退档后可能比固定规则更差 |
| **纯 Vector 版 attention**（用 3D 广播乘 + 归约代替两个矩阵乘，期望编成 `AI_VECTOR` 以消掉 MIX 固定开销） | **双重证伪，见 §3.7** |
| 非融合三段式（QK → softmax kernel → PV） | **算账否决，未实现，见 §6.3** |

### 3.7 ⛔ 结构性下限：含向量计算的 kernel 必为 `MIX_AIC` 【通用·最重要的一条负知识】

**动机**：小 case 的 attention kernel 贴着一个 **~4.5us 的地板**（最小 case `S=3`、计算量近似为零，attention 仍要 4.9us，而同 shape 的纯 GEMM kernel 只要 1.4us）。`Accelerator Core` 列显示 attention 是 `MIX_AIC`、GEMM 是纯 `AI_CORE`。

**做法**：写一个不含 `tl.dot` 的 attention kernel，期望它编译成 `AI_VECTOR`，免掉 MIX 的固定开销。

**结果：证伪，而且是双重的。**

| case（S / D） | Cube 版 | 纯 Vector 版 |
|---------------|---------|-------------|
| 3 / 24 | 4.9 us | **5.6 us** |
| 43 / 64 | 6.2 us | 11.6 us |
| 101 / 64 | 14.1 us | 44.6 us |

16 个小 case 子集几何平均 7.04 → 4.73（**−33%**）。

**根因（IR + profiling 交叉确认）**：纯 Vector 版**仍然被编译成 `MIX_AIC`**（`Block Num=4 / Mix Block Num=8`，与 Cube 版完全一致）。对照 IR：纯 Cube kernel 里**没有任何 `mix_aiv` 函数**，而 attention kernel 同时有 `..._mix_aic` 和 `..._mix_aiv`。

⇒ **triton-ascend 是按「有没有向量侧计算」决定 kernel 类型的，与用不用 `tl.dot` 无关。**

**可复用的通例**：
1. 任何含 softmax / elementwise 的 kernel 在 triton-ascend 下**必然是 `MIX_AIC`**，每次发射带 **~4.5us** 固定成本，是工具链的结构性下限。
2. 推论：**小 shape 上绝对不要为了模块化把融合算子拆成多个 kernel**——每多一个 MIX kernel 就多 ~4.5us，而小 case 的整个 impl 时延也才 6~8us。
3. 判别：看 `kernel_details.csv` 的 `Accelerator Core` 列（`MIX_AIC` vs `AI_CORE`），或看 IR 里有没有 `_mix_aiv` 函数。**改写数学表达改变不了这个分类。**

---

## 4. 精度：先过闸门，再谈性能

性能优化**不得**以放宽精度判据为代价：正式结论一律走 `verify.py`，不改阈值、不用 `--verify_not_required`。

- `tl.dot` dtype 契约、低精度四条逐位契约、1-ulp 匹配率诊断法：见
  `@../../../triton-precision-debug/references/attention-lowprec-contract.md`
- **掩码用有限极小值**（`-3.0e38` / `-1e30`）**不用 `-inf`**：`-inf` 会在整行被掩时产生 `exp(-inf-(-inf))=NaN`，并逼着 `tl.maximum` 保留 NaN 处理（与 §2.8 冲突）。
- **判据推论**：若失败集合**严格与 dtype 相关**（如"全部且仅仅是 fp16/bf16 case"），基本可直接判为**算术路径不匹配**，不用去搜 tiling。
- ⚠️ **精度契约会倒逼结构**：当参考要求"先用最终的 `l` 归一化再舍入"时，online softmax 只能拿到滚动的 m/l ⇒ 必须多趟。最终形态：单块（`S ≤ BLOCK_KV`）一趟、多块两趟。
  **不要试图用"减小 BLOCK_Q 换单趟"**：总迭代数 = `ceil(S/BQ)·ceil(S/BKV)·趟数`，BQ 减半使第一项翻倍、趟数 2→1，**正好抵消**。已算账否决。

---

## 5. 测量方法论（不做这一步，上面所有数字都是噪声）

### 5.1 环境闸门必须用双探针

torch 参考实现代码恒定，其时延是天然环境探针。**但只用全量几何平均会漏检**：

| 探针 | 基准 | 带宽 |
|------|------|------|
| 全量 50 case 几何平均 | 0.2166 ms | ±8% |
| **大 shape 子集（case 41-50）** | 0.7752 ms | ±12% |

补第二探针的直接原因：某轮全量探针 0.2299 **尚在带内**，但大 shape 探针 1.0176 比基线 **高 31%**——全量探针被 30 个小 case 主导，对只打大 shape 的漂移完全不敏感。

**实测代价**：某轮首测 **4.6678**、闸门判无效、重测有效值 **4.3826** —— 直接采信首测会**高估 6.5%**。

> 比值型指标不能自证有效：分子分母同步变慢时 speedup 反而"看起来变好"。
> 同机器上其他会话在跑 benchmark 是常态，不要试图独占，用**闸门 + 自动重测**应对。

### 5.2 ⚠️ benchmark 时延口径（写报告时必须注明）

```python
for name, group in df.groupby("Name"):
    avg_us = group.iloc[-active_count:]["Duration(us)"].sum() / active_count
total_avg_us = sum(operator_avg_times.values())
```

**按 kernel Name 分组，同名 kernel 一次 forward 内发射 L 次只被计成 1 次**（低估 L 倍）。实例：

| | 真实设备时间 | 口径计入 |
|---|---|---|
| torch | 343 us | 184 us |
| 本实现 | 187 us | 128 us |
| speedup | **1.84x** | 1.43x |

另一算子：benchmark 口径 6.24，**真实设备时间 9.24x**（torch 侧被低估 1.503x）。

两个必须显式声明的取舍：

- ✗ **不要**把多次发射合并进同一个 `@triton.jit`（用 `MODE: tl.constexpr` 分支）——共用一个 Name 会让测得时延直接降到 1/L，**不改变任何真实性能，纯粹利用口径缺陷**。
- ✗ 也**不要**因为口径而放弃真实优化：三个投影融合成一次 GEMM 是真实收益，但会让口径分数变差。

**做天花板估算前务必先确认分数口径**，否则会把一个已经接近目标的实现误判为差 40%。

### 5.3 噪声水平决定了你能分辨多小的改动

| 协议 | 分辨率 |
|------|--------|
| 20 case 单次，直接比 | **±2%**（同一配置两次 2.9448 / 2.8821） |
| 同一份代码复测两次 | **2.2%**（6.178 / 6.0442） |
| 10 case **基线包夹**（base → A → base） | **±0.3%** |

**教训**：某次有 4 个候选在单次协议下看起来"略优于基线"，换包夹协议后**全部落回噪声内**。
⇒ **≤3% 的改动必须用包夹协议复测，否则会把噪声当成收益采纳。**

### 5.4 profiling / IR / 仿真器的分工

| 只有 profiling 能给 | 只有 IR 能给 | 只有指令级仿真能给 |
|---|---|---|
| 时间归属（哪个 kernel 占多少） | 结构根因（`scf.for` 层数、同步指令数量与位置） | 单条指令值多少 cycle（决定优化杠杆） |
| 资源维度（MTE2/MAC/VEC 谁饱和） | **证伪**（推翻由 ratio 得出的错误假设） | `aic_scalar_ratio` 高是真标量算术还是同步等待 |
| 串行度 `(aic+aiv)/Duration` | 哪些内建被展开成几条指令 | 小 shape 固定开销由什么构成 |

**建议：优化早期用 ratio 选主攻 kernel，卡住之后（尤其 SCALAR/VEC 高但改不动时）立刻上仿真器**，不要靠一串证伪实验反推。

---

## 6. profiling 字段 → 优化点（含误导案例）

### 6.1 每个字段实际指向了什么

| 字段 | 它**建议**的优化点 | **实测是否命中** | 正确的读法 |
|------|-------------------|-----------------|-----------|
| `Duration(us)` 分项占比 | 先打占比大的那个 | ✅ **完全可靠** | 唯一无条件可信的字段。每轮重看一次，主攻目标会换 |
| `aic_mac_ratio` **低**（<10%） | Cube 空转 → 检查是否没用 `tl.dot` | ✅ 命中 | `<10%` + 代码里有 `for d in range(head_dim)` ⇒ 直接重写，不用扫参 |
| `aic_mac_ratio` **高**（>55%） | 已算力受限 | ✅ 命中（后续所有尝试均无收益） | 把精力转移走 |
| `aic_mte2_ratio` 高 | 访存受限 → 减流量 / 加大连续跨度 | ✅ **命中且指标同步移动**（72%→57%、MAC 44%→58%） | `MTE2>60%` **且** `MAC<50%` 才算访存受限；两个都高说明只是重叠得好 |
| `(aic+aiv)/Duration` >1.5 | CV 串行 → 需流水 | ⚠️ **指标正确、药方错误** | 解法是**减少同步次数**（放大 BLOCK / 收缩区间），**不是**重排指令 |
| `aic_scalar_ratio` 高 | 「标量受限」 | ❌ **五次改法全部落空** | **不能直接采信**，见 6.2(a) |
| `aiv_vec_ratio` 高 | Vector 受限 → 减 elementwise | ⚠️ **半对** | 按**单条指令的代价**判断，不是按**指令条数**；且 `<0.35` 时整类不做 |
| `cube_utilization(%)` | 「Cube 很忙」 | ❌ **纯误导** | 见 6.2(b) |
| `Block Num` / `Mix Block Num` / `Accelerator Core` | grid 是否生效、是否 MIX kernel | ✅ 用于**排除**假设 | 每次改 grid 后核对，防止"静默算少" |
| 同一 `Name` 的**行数** | — | ✅ 暴露测量口径问题 | `行数/(warmup+repeats)` = 每次 forward 的发射次数 |

### 6.2 三个会误导的字段

**(a) `aic_scalar_ratio` — 最大的坑。** 提示"标量受限（地址/循环/整数除模）"，据此做了五次改动**全部落空**（§3.2）。dump IR 才看清真实构成：kernel 里有 **113 条同步指令**（`set_flag` 31 + `wait_flag` 31 + `pipe_barrier` 27 + `sync_block_set` 12 + `sync_block_wait` 12），AIV 内层循环 32 条向量调用里 **22 条后面各跟一个 `pipe_barrier`**。**那 73% 是等待被计入了标量管线，不是标量算术。**

> ⚠️ 这已经在**三个算子**上重复出现（0.55 / 0.654 / 0.736，三次判定"标量运算过多"全落空），可以当成通例：
> **`aic_scalar_ratio` 高 ⇒ 先 dump IR 数同步指令，不要直接改标量代码。**

**(b) `cube_utilization(%)` — 名字骗人。** 同一行里 `cube_utilization = 88.383%` 与 `aic_mac_ratio = 0.084` 同时出现。88% 的"cube 利用率"对应 8.4% 的 MMAD 占比。**判断 Cube 忙不忙一律用 `aic_mac_ratio`。**

**(c) ratio 是占比，分母会变，必须同时看 `dur` 绝对值。** 某轮 SCALAR 从 59.7% 涨到 71.0%，看起来"更标量受限了"，实际 `scalar_time ≈ 0.9 × dur × ratio` 从 8609us 降到 8152us（**同步时间基本没动**），而总耗时从 16023 降到 12757（−20%）。真实结论是"被砍掉的全是向量算术时间，同步开销一点没少"——这正是判定"瓶颈是 CV 同步"的第一个量化证据。

**(d) 天然对照组**：只动 A kernel 的那几轮，B kernel 的 profiling 应当**纹丝不动**。如果"没改的那个 kernel"指标也变了，说明测量被环境污染，该轮作废。

### 6.3 决策流程

```
1. 看 Duration 分项占比 ─────────────► 选主攻 kernel（每轮重看，会换）
2. aic_mac_ratio < 10% ？───── 是 ──► 检查是否没用 tl.dot ⇒ 重写（数量级收益）
                              否
3. (aic+aiv)/Duration > 1.5 ？── 是 ──► CV 串行
                              │        ✗ 不要先做单次迭代内的指令重排
                              │        ✓ 先按 §1.1 做每次迭代成本反推法确认
                              │        ✓ 再按 §2.2/2.3/2.5 压迭代数（区间收缩 → 去对齐 → BLOCK 开到编译失败为止）
                              否
4. MTE2 > 60% 且 MAC < 50% ？── 是 ──► 访存受限
                              │        ✓ 放大 tile（必须配并行度保护）
                              │        ✓ 常量张量按 tile 形状预重排成连续布局
                              │        ⚠ 用含大/中/小三档的子集验证，别只看大 case
                              否
5. MAC > 55% ？──────────────── 是 ──► 已算力受限，只剩减少无效计算
                              │        ⚠ 波次别用单目标代价模型（实测 −21%）
                              否
6. SCALAR / VEC 高 ────────────────► 【必须先 dump IR 或上仿真器】
                                     ├ 同步指令占大头 ⇒ 回到第 3 条
                                     ├ 某内建被展开成多条 ⇒ 换写法（真收益，如 §2.8）
                                     └ 只是普通向量指令多 ⇒ 先看 aiv_vec_ratio 是否 <0.35，
                                       是则整类不做；否则先做单变量实验再动手
```

---

## 7. 天花板估算（提出大改造前先算账）

### 7.1 灵敏度分析（5 分钟，决定要不要继续投入）

用实测逐 case 时延代入，假设整体加速 f 倍再取几何平均（含 ~4.5us 的 MIX 发射地板）：

| f | 无地板 | 含 4.5us 地板 |
|---|--------|---------------|
| 1.00（现状） | 6.24 | 6.24 |
| **1.62** | 10.01 | **9.92** |
| 2.00 | 12.36 | 12.04 |
| 4.00 | 24.71 | 21.13 |

⇒ 结论可以非常具体："要到 10x，只需整体再快约 1.6 倍，卡点是迭代数要再降 1.6 倍 ⇒ tile 面积要放大 ~2.6 倍 ⇒ 被 UB 挡死"。

### 7.2 硬件效率对照：先看自己离峰值多远

| 算子 | 实测算力 | 占峰值 | 结论 |
|---|---|---|---|
| A（fp32 主导） | 融合 attention 13.9 TFLOPS（torch 的 bmm 只有 9.2 TFLOPS） | 已超过 torch | fp32 matmul 实测天花板就是 ~10 TFLOPS 量级 ⇒ **10x 不可达，早说** |
| B（fp16 主导） | 有效计算 5.0 TFLOPS / 含 padding 25.8 TFLOPS | 1.3% / 6.9% | **算力余量 14 倍以上，完全不是算力瓶颈** ⇒ 10x 在算力上可达，卡在同步与 UB |

⇒ **同一类算子、不同 dtype，天花板结论可以完全相反。必须自己算，不能抄。**

### 7.3 访存账模板（用来否决大改造）

「非融合三段式」的否决计算：

```
最大 case 物化 [B*H, S_Q, S_K] fp32 分数矩阵 = 370 MB
写出 + 读入 + 写回 + 再读入 ≈ 1.5 GB ≈ 900 us（按 1.6 TB/s）
而当前整个 attention kernel 才 400 us     ⇒ 否决，不实现
```

融合 attention 的存在意义就是避免物化 S×S，**提出拆开之前先做这个乘法**。

---

## 8. 推荐的落地顺序

1. **先建测量基础设施**（闭环脚本 + 环境双探针 + 诊断子集扫描器），并**量出噪声水平**（同一份代码复测两次），再写第一行优化代码。
2. **精度优先**：按 §4 跑通全量 verify。精度不过直接 exit，不进入性能测量。
3. **看时间归属**：出现 `Transpose`/`ViewCopy`/`ZerosLike`/`Slice` 就直接删掉 host 端搬移（§2.12），不用扫参。
4. **`tl.dot` 化**（§2.1），取得第一个可测量 baseline + profiling 画像。
5. **用每次迭代成本反推法判定瓶颈类型**（§1.1）：与 tile 无关 ⇒ 走第 6 步；随 tile 线性 ⇒ 本文排序不适用。
6. **按迭代数公式逐项压**：`mask constexpr 特化`（§2.4）→ `KV 区间收缩`（§2.2）→ `去对齐 + tile 联合放大`（§2.3，**必须成对**）→ `BLOCK 开到编译失败为止`（§2.5）。**每次结构性改动后重扫 tile。**
7. **dump IR**：数 `scf.for` 层数与循环体内 `sync_block` 数，看有没有内建被展开成多条（§2.8），并用它**证伪**由 ratio 得出的假设。
8. **低成本试一次**：CV 指令重排（§2.10）、低精度 PV 二段拆分（§2.11，精度必须实测）。
9. **有投影 GEMM 才做**：权重预重排（§2.6）+ 分块分档与并行度保护（§2.7）。
10. **算天花板**（§7）。目标不可达就早点说，把剩余时间投在"把每一档 case 推到各自的硬件上限"，而不是追一个不存在的数字。
11. **以下不建议投入**，除非画像与本文明显不同：编译开关穷举（§3.1）、减少向量指令条数（§3.2，先看 `aiv_vec_ratio` 是否 <0.35）、布局连续化（§3.3，要测也先用 `.contiguous()` 花 5 分钟）、放大 tile 面积超过 UB（§2.5）、单块特化（§3.4）、`al.scope` 手写流水（§3.5，先做 10 分钟门槛测试）。

---

## 9. 与其它文档的关系与冲突声明

| 文档 | 关系 |
|------|------|
| `cv-fusion.md` / `cv-fusion-tiling.md` | 上位方法论（CV 融合通用）。本文是其 FA 子类的实测细化 |
| `cv-fusion-pingpong.md` | ⚠️ **冲突**：它把 `al.scope` + ping-pong 列为唯一出路，实测在 910B2 + CANN 8.5.1 上崩编译器（§3.5）。投入前先做门槛测试 |
| `docs_triton_IR/docs_for_triton_agent/17-flash-attention-migration.md` | ⚠️ **逐条复核结果**：Diff 2（grid 收缩）= 基础写法非优化项，+0.5%；Diff 4（block pointer）未单独复现收益；Diff 6（exp2）**✗ 无收益**，（scale+mask 合并 FMA）**✗ 编译失败**；Diff 7（指令重排）三算子 0/+1.9%/+6.4%，**不可预期**；Diff 8（`propagate_nan`）**是性能开关不是正确性项，且必须分档**；Diff 9（7 个编译参数）**✗ 全在噪声内** |
| `06-scalar-degradation-avoidance.md` | ⚠️ 其"整数比较降级"现象在本类算子上**成立**，但据此做的优化**结论不成立**（§3.2）——现象 ≠ 可优化 |
| `template/flash_attention.md` | 设计期（Layer 1/2）与编码期（Layer 3）的对应约束 |
| `triton-precision-debug/references/attention-lowprec-contract.md` | 精度闸门与低精度逐位契约 |

<!-- okf:related:start -->

## 相关

- 设计与编码期约束: [FA 类算子优化经验（template）](../../../../plugins-official/triton-op-generator/template/flash_attention.md) — 同一知识的 Layer 1/2/3 形态，Phase 2/3 入口
- 精度闸门: [Attention 低精度逐位契约](../../../triton-precision-debug/references/attention-lowprec-contract.md) — 性能优化的前置闸门，§4 的展开

<!-- okf:related:end -->
