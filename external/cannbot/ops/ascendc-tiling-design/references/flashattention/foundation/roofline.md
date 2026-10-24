# FlashAttention 类算子 — 基本块与 Roofline 分析

> 本文档定义 FA 类算子**基本块(Bq, Bk)选取**的 Roofline 方法论,承接 [`design/shape.md` §3](../design/shape.md)(N3)。
> **前置**:已完成 [`fundamentals.md`](./fundamentals.md) 分析产出(选型结论 + shape 边界 + 性能精度目标)。
> **产出去向**:本文推导出的 `mEff` / `Bk` / 性能区制,是 [`design/shape.md` §5 多核](../design/shape.md)、[`design/resources.md` §1 内存](../design/resources.md)、[`design/execution.md` §3 流水/同步](../design/execution.md) 的输入,不可在完成本文推导前凭经验选取。
> **命名说明**:`mEff` / `curG` 等为方法论抽象名,对应的公开 API 见 [`fundamentals.md` §0](./fundamentals.md)。

---

## §1 为什么 FA 需要专门的 Roofline 模型

FA 类是 **mix kernel**(AIC 算 GEMM + AIV 算 softmax,跨核 stage 化),它的性能瓶颈**不能用单一 AI(算术强度)概括**。三个独立的吞吐通道各自可能成为瓶颈:

1. **Cube 算力**(AIC 的 Mmad 吞吐)
2. **HBM 带宽**(K/V/Q/O 的 GM 加载与写回)
3. **跨核/片上吞吐**(AIV 的 softmax 吞吐 + Cube↔Vector 片上握手 + O_acc streaming 往返)

> ⚠️ **常见建模错误**:把片上跨核握手(S/P/PV 的 L0C↔UB↔L1 movement)与 HBM 加载**混进同一个 AI 分母**,再用"混合带宽"求单一 AI。这会:
> - 把 AI 人为封顶在 `D/sizeof(T)`(建模 artifact),让人误以为 FA 永远到不了 compute-bound;
> - 掩盖真正的第三瓶颈(AIV/片上吞吐),而它恰恰是 mix kernel 最常见的瓶颈。
>
> **正确做法**:三个通道各自独立建模,性能 = 三个 ceiling 取 **min**;瓶颈 = 限制项。

---

## §2 三通道 Roofline 模型

### §2.0 GQA 多头合并:mEff 的定义

FA 类 Mmad 的有效 m 维不是 Bq 本身,而是:
```
mEff = Bq × curG
```
curG = 同一 kvHead 下合并到一个 task 的 qHead 数(≤ G = Hq/Hkv,见 [`fundamentals.md` §3.2 I2](./fundamentals.md))。

> 记号约定:`Bq` 指名义 Q 块大小,`curBq` 指某 task 的实际处理量(尾块 `curBq < Bq`);同理 `curG ≤ G`。容量校验用实际量 `mEff = curBq × curG`,块尺寸规划用名义 `Bq × G`。下文在不涉及尾块时统一写 `mEff = Bq × curG`。

- **Prefill**(Sq 较大):Bq 本身足够大,mEff 主要受 L0A 容量约束,curG 可能 < G
- **Decode**(Sq=1):Bq=1,`mEff = curG`,应最大化 curG = G

**mEff 是 FA Roofline 的核心自变量**——它同时决定 AI 和三通道的相对松紧。

### §2.1 单 task 的 FLOPs(遍历全 Sk)

两个 GEMM,各遍历全 Sk:
```
C1 (Q·K^T):  2 × mEff × Sk × D
C2 (P·V):    2 × mEff × Sk × D
FLOPs_total = 4 × mEff × Sk × D
```

> **causal 的影响不在 AI**:causal / 下三角 mask 下,被完全 mask 的 tile 被跳过——**FLOPs 与对应的 K/V HBM 加载同比例减少**(平均有效 KV 长度约 Sk/2)。由于 §2.2 的 AI 推导中 Sk 会约掉,**AI ≈ 不变**;causal 真正影响的是**绝对耗时**与**核间负载均衡**(见 [`design/shape.md` §5.5](../design/shape.md)),不是算术强度。跳过完全屏蔽 tile 时的**正确性要求**(C1、V1 都要跳过,mask 值用大有限负数防 NaN)见 [`design/execution.md` §1](../design/execution.md)(N4 softmax)。

### §2.2 通道 2:HBM 带宽与 AI_HBM(主 Roofline)

**单 task 的 HBM 流量**(用户张量,遍历全 Sk):

| 张量 | 流量 | 说明 |
|------|------|------|
| K 加载 | `Sk × D × sizeof(T)` | 全序列加载一次 |
| V 加载 | `Sk × D × sizeof(T)` | 全序列加载一次 |
| Q 加载 | `mEff × D × sizeof(T)` | 驻留(L1),摊薄 |
| O 写回 | `mEff × D × sizeof(T)` | 末块写出 |
| **O_acc streaming**(仅 streaming UB 模式)| `2 × mEff × D × sizeof(acc) × (Sk/Bk)` | 每个 s2 chunk 读回+写出 O_acc,共 Sk/Bk 次 |

`Sk ≫ mEff` 时(典型),K/V 主导:
```
Bytes_HBM ≈ 2 × Sk × D × sizeof(T)  +  2 × mEff × D × sizeof(acc) × (Sk/Bk)
```

**算术强度(对 HBM)**:
```
AI_HBM = FLOPs_total / Bytes_HBM
```

忽略 O_acc streaming 项(Bk 足够大时该项小)时的**清晰上界**:
```
AI_HBM ≈ 4·mEff·Sk·D / (2·Sk·D·sizeof(T)) = 2·mEff / sizeof(T)
```

**关键结论**(与 [`base_design.md` §2.2-§2.3](../subfamilies/base_design.md) 一致):
- **AI_HBM ∝ mEff**(不封顶在 D/sizeof(T))
- Decode(mEff = G):`AI_HBM = 2G / sizeof(T)`,故 **decode 的 AI ∝ G**,GQA group 合并是提升 AI 的唯一杠杆
- **Bk 通过 O_acc streaming 项影响 AI**:streaming 模式下增大 Bk 减少 O_acc 往返次数(Sk/Bk),从而降低 HBM 流量、提升 AI。非 streaming(O_acc 驻 UB)下 Bk 对 AI 无直接影响,仅减少 s2 loop 切换开销

**HBM ceiling**:
```
Perf_HBM = Peak_HBM_BW × AI_HBM
ridge_HBM = Peak_CUBE_FLOPS / Peak_HBM_BW   (FLOPs/Byte)
AI_HBM < ridge_HBM  → 该通道可能是瓶颈(HBM-bound)
```

### §2.3 通道 3:跨核/片上吞吐(AIV-bound 校核)

FA 是 mix kernel:AIC 算 GEMM 的同时 AIV 算 softmax + 做片上握手。若 **AIV 端吞吐 < AIC 端吞吐**,即使 HBM 与 Cube 都不饱和,算子仍被 AIV 拖住(cross-core-bound)。

**AIV 端每 tile 的工作量**:
- softmax over `[mEff, Bk]`:ReduceMax / Exp / ReduceSum,∝ mEff·Bk
- 片上握手 movement(不占 HBM,占片上 SRAM 带宽):
  - S:L0C→UB(Fixpipe),`mEff × Bk` 元素
  - P:UB→L1,`mEff × Bk` 元素
  - PV:L0C→UB,`mEff × D` 元素

**校核判据(吞吐平衡)**:
```
T_AIC(tile) ≈ FLOPs(tile) / (Peak_CUBE_FLOPS × cube_util)
T_AIV(tile) ≈ softmax_ops(mEff·Bk) / AIV_vector_throughput
              + onchip_bytes(mEff·Bk) / Peak_onchip_BW

cross-core-bound  ⟺  T_AIV > T_AIC
```

**主杠杆**(见 [`design/execution.md` §3.2](../design/execution.md) 跨核同步):
- 提高 AIC:AIV 比例(如 `__mix__(1,2)` → 更多 AIV)
- 减小握手粒度、per-stage 同步、双缓冲让 AIC/AIV 重叠
- 减少 PipeBarrier 冗余(< 30%,见 [`design/execution.md` §3.4](../design/execution.md))

> **与旧模型的关系**:旧模型的"L2 流量项"实际是本通道的片上握手 movement,不应记为 HBM/L2 带宽下的 AI 分母项,而应作为**独立的 AIV/片上吞吐校核**。

### §2.4 通道 1:Cube 算力(compute ceiling)

```
Perf_compute = Peak_CUBE_FLOPS × cube_util
```
cube_util 主要由 Mmad m/n/k 对齐到 16 的倍数决定(见 §4 杠杆表)。

### §2.5 三通道取 min

```
Perf ≈ min( Perf_compute,  Perf_HBM,  Perf_crosscore )
瓶颈 = 三者中最小的那个通道
```

| 区制 | 判据 | ceiling | 主杠杆 |
|---|---|---|---|
| **Compute-bound** | AI_HBM ≥ ridge_HBM 且 AIC 吞吐最低 | `Peak_CUBE × util` | Mmad 对齐、Cube/Vector 重叠、减 PipeBarrier |
| **HBM-bound** | AI_HBM < ridge_HBM 且 HBM 吞吐最低 | `Peak_HBM × AI_HBM` | 增大 mEff(∝AI)、Q 驻 L1、增大 Bk 减 O_acc 往返 |
| **Cross-core-bound** | T_AIV > T_AIC | AIV/片上吞吐 | 提 AIC:AIV 比、减握手粒度、双缓冲、减 PipeBarrier |

FA 类在常见 D 档位、mEff 较小时通常先撞 **HBM-bound 或 cross-core-bound**;通过增大 mEff 提升 AI 后,可能转入 compute-bound 或 cross-core-bound。

---

## §3 基本块可行域:下界(性能)与上界(硬件)

### §3.1 下界 — Roofline 反推最小 mEff

HBM-bound 下 `Perf ∝ AI_HBM ∝ mEff`,需要 mEff 足够大才能达到可接受性能。从 `AI_HBM = 2·mEff/sizeof(T)` 反推:
```
目标 AI = α × ridge_HBM   (α 为可接受的 ridge 占比,如 0.3~0.5)
→ mEff_min = ceil(α × ridge_HBM × sizeof(T) / 2)
```

Decode 场景 mEff = curG ≤ G:若 `mEff_min > G`,说明该 shape 下 HBM-bound 无法靠增 mEff 突破 → 依赖流水隐藏延迟(见 §4)或 split-KV reduce(见 [`design/shape.md` §2](../design/shape.md) N2 启用判据)。

### §3.2 上界 — 硬件容量约束最大 mEff / Bk

基本块受三层硬件约束,取三者交集:

1. **UB 约束**(主约束,见 [`design/resources.md` §1.1](../design/resources.md)):所有 UB buffer 总和 ≤ UB 容量。V1-chunk 模式下 softmax 状态 buffer(max/sum/expMax)是全 mEff 常驻项,随 mEff 线性增长(每行一个 datablock,见 §4.1 说明)
2. **L1 约束**(见 [`design/resources.md` §1.3](../design/resources.md)):A1 端口 `mPad × D × sizeof(T) × rotation_slots ≤ L1_A1_port`;B1 端口 `Bk × D × sizeof(T) × rotation_slots ≤ L1_B1_port`
3. **L0 约束**(见 [`design/resources.md` §1.4](../design/resources.md)):C1 输出 `S = [mEff, Bk]` 与 C2 输出 `[mEff, D]` 都落 L0C,但两个 GEMM **时间复用**同一 L0C(C1 的 S 经 Fixpipe 撤出后 C2 才写 PV),故 L0C 需满足二者中较大者:`max(mEff × Bk_pad, mEff × D_pad) × sizeof(float) ≤ L0C`。Bk 常大于 D,故 C1 项常为 L0C 的绑定约束

**V1 两模式与上界的关系**:

| 模式 | 触发条件 | v1In 占用 | v1Out 占用 |
|------|---------|----------|----------|
| V1-fit | `mEff × Bk_pad × sizeof(float) ≤ UB 单槽容量` | `mEff × Bk_pad × 4` | `mEff × Bk_pad × 2` |
| V1-chunk | 上述不满足 | `chunkRows_v1 × Bk_pad × 4` | `chunkRows_v1 × Bk_pad × 2` |

### §3.3 硬件常量来源

所有 Peak 值与容量必须**运行时查询**,禁止硬编码:
- `Peak_CUBE_FLOPS` / `Peak_HBM_BW`:平台规格(见 `/npu-arch`)
- UB / L1 / L0 容量:`PlatformAscendC::GetCoreMemSize(...)` 系列

---

## §4 场景特化:Prefill / Decode / Causal

| 场景 | mEff 来源 | 典型瓶颈 | 首要杠杆 |
|------|----------|---------|---------|
| **Prefill**(Sq 大)| Bq 大,mEff 受 L0A 约束,curG 可 < G | compute 或 cross-core | 增大 Bq 提 AI;Mmad 对齐;流水重叠 |
| **Decode**(Sq=1)| mEff = curG ≤ G | HBM 或 cross-core(AI 天花板低)| 最大化 curG=G;核数富余时 split-KV reduce |
| **Causal**(叠加)| 同上 | 同上 + 负载不均 | AI 近似不变(FLOPs 与 K/V 字节同比例减);zigzag/剪枝均衡(见 optimization §4.2)|

---

## §5 量化路径的 AI 修正

量化 dtype 下 K/V 的 HBM 流量按量化字节宽计:
```
Bytes_HBM ≈ 2 × Sk × D × sizeof(quant_T)
AI_HBM = 2·mEff / sizeof(quant_T)
```

**fp16(2B)→ fp8(1B)**:`sizeof` 减半 → **AI_HBM 翻倍(≈ +100%)**,不是旧混合模型给出的 +48%。+48% 是把片上握手混进 AI 分母的 artifact。

量化路径的完整契约(scale 轴、c1v1Loop)见 [`quantization_design.md`](../subfamilies/quantization_design.md)。

---

## §6 产出清单

进入 §3 多核切分前,DESIGN.md 必须展示以下推导(**每项展示算式,禁止只给结论**):

- [ ] **AI_HBM 公式** + 代入 mEff/D/sizeof 的数值 + **ridge_HBM**
- [ ] **三通道 ceiling** 各自估算 + **min 取哪个 → 瓶颈区制**
- [ ] **cross-core 校核**:T_AIV vs T_AIC 的定性/定量判断
- [ ] **mEff_min**(Roofline 下界)+ **mEff_max / Bk_max**(硬件上界)
- [ ] **V1 模式判定**(V1-fit / V1-chunk)
- [ ] **最终选定** mEff, Bk 值 + 在可行域中的相对位置
- [ ] **场景**:prefill / decode /(是否 causal)
- [ ] **性能上限估算** = min(三通道 ceiling)

---

## §7 参考资源

| 主题 | 入口 |
|---|---|
| 设计流程(WHAT)| [`design/shape.md`](../design/shape.md) §3(N3)|
| 数学模型与不变量 | [`fundamentals.md`](./fundamentals.md) |
| 内存容量约束公式 | [`design/resources.md`](../design/resources.md) §1 |
| 优化杠杆(流水 / 同步 / 负载)| [`design/execution.md` §3](../design/execution.md)、[`design/shape.md` §5.5](../design/shape.md) |
| 变体 mEff/AI 特化 | [`base_design.md` §2](../subfamilies/base_design.md)、[`quantization_design.md` §2](../subfamilies/quantization_design.md) |
| 平台规格(Peak 值)| `/npu-arch` |
