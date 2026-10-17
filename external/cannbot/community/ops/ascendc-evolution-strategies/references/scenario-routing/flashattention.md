# FlashAttention 优化工作流（MHA / GQA / MLA 分支 + 正交 trait）

> 适用：**QK^T → softmax → PV** 的 softmax 注意力及其 head 结构变体（MHA / GQA / MQA / MLA），以及正交 trait（causal / sparse / paged / decode / 量化 / phase）。
> ⚠️ 迁移注记：旧 S0 形态判定已并入 Step 2 分支路由表；历史观测文档若引用 S0-S6 阶段名，以本文 Step 2-4 为准。
> FA 族策略收益与依赖均有序：**结构改写 > CV 流水 > 内存搬运 > mask/稀疏特化 > 调度均衡**。从未完成的最早阶段选策略，每轮落地一个阶段。

## Step 0 — 族判定确认

命中 [README.md](README.md) 判定表第 1 行（QK^T 指纹 / online softmax 状态机 / head 数关系）即归本族。
注意：**纯 MLP 通道注意力**（SE/SK：global pool→FC→scale，**无 conv**）**不归本族**（落 reduction/elementwise）；**无 QK^T 的递归/scan**（GDN-KDA/GLA）归 [ssm-scan.md](ssm-scan.md)。
**conv-based CV-Attn**（Triplet/BAM/CBAM：global pool → **conv 2→1** → BN(training)/FC → 激活 → gate 广播，**分支 ≥ 2**）归本族 **`cv-attn` 分支**（见 Step 2 与下文 CV-Attn 快速路由节）。

## Step 1 — 通用前置

打开 [_common-prerequisites.md](_common-prerequisites.md) 过 C1-C5。FA 族常见命中：**C1**（老版纯向量点积实现 → 直接转 S1 的 P120，它就是 P102 在 FA 的落地配方）、**C5**（softmax 必配 A5）。

## Step 2 — 分支路由（先判 head 结构 / 小类，按序路由）

**小类映射**（源自 [attention_family.md](../../../ascendc-evolution-knowledge/references/a3/algorithm_insights/attention_family.md) 分类，小类 → 分支 + 推荐策略）：

| 小类（计算模式） | 分支 | 推荐策略 |
|---|---|---|
| **MHA / FA**（CP1 稠密 / CP2 流式分块） | standard | S1 **P119**（online softmax）+ **P120**（vec-dot→cube mmad）→ S2 P121（CV 流水）→ S3 P122（KV 大块化）+ **P142**（UB 常驻 O 累加器，与 X38 GM ring 二选一）；中小 shape host-bound → **P143**（launch 合并 + per-shape tiling 缓存） |
| **GQA / MQA**（CP1+CP3 / CP1，head 共享） | gqa | **P115**（GQA tiling）+ **P116**（G→M 合并）+ **X5/X4**（decode GEMV→GEMM）+ **P126**（变长调度）；流式骨架同 MHA/FA → **P142**；投影 GEMM 侧 X33（group A-reuse）、decode bounded-logits X31 |
| **MLA**（CP3+CP4+CP11，latent 压缩） | mla | **P124**（nope/rope 拼 K）+ P120 + **P140**（M-tile 放大）+ P125（跨任务预取）+ P126；流式骨架 → **P142** |
| **Sparse-掩码型**（CP1，全量算+掩码） | standard + sparse:causal-window | **P123**（skip + 边界 mask）+ **P118**（mask 预生成独立 AIV）；中小 shape host-bound → **P143** |
| **Sparse-块跳过型**（CP3，子集遍历） | standard + sparse:block_table | **P137**（紧凑化走 Cube）+ P138/P139（L0 守卫/多槽零填充）+ P76/P77；流式累加骨架 → **P142** |
| **Sparse-展开型**（CP9，unfold 邻域） | standard | 特殊：unfold 邻域骨架，通用前置 + 瓶颈驱动 |
| **Paged**（CP4，分页 KV） | standard + paged/decode | **P60/P61**（KV NZ/L2）+ **P126** + P57/P58（decode 分区）；流式骨架 → **P142** |
| **SageAttention**（CP10，量化） | standard + 量化 | P48/P49/P70（量化支路，QK int8 / PV fp16） |
| **HSTU**（CP12，去 softmax） | standard | 无 online-softmax：跳过 S4，直接 P120 mmad + P121 流水 |
| **CV-Attn-空间自注意力**（CP1，视觉语义） | standard | 同 standard 优化；按视觉特征图语义处理 |
| **CV-Attn-双分支/通道/空间门控/聚合分发**（CP8，gate 广播，无 QK^T；Triplet/BAM/CBAM） | cv-attn | **P144**（host 端 zpool/BN-stats 计算 + 标量传参）→ **P145**（多 launch 免同 launch GM 写后读）→ **P146**（910B2C 32B 对齐/标量通道加固）→ **P147**（NPU conv 12-sig-bit 舍入模拟）；H/W 小 → **P104/X58**（(b,cb,h) 多核切分）；conv 计算量大 → **P128/P129** |

> ⚠️ 小 S launch-bound 优先 host 侧（**P143** launch 合并 + tiling 缓存 / P54 / X28），勿机械套 tiling/流水（P-GQA-7 教训）。

按 head 结构的精确判定（`num_q_heads` vs `num_kv_heads`）与 mla latent 判据见下方基础链前的前置确认；正交 trait（sparse/paged/量化）在 Step 4 追加。

## Step 3 — 基础链（S1→S6 按序推进）

| 阶段 | 目标 | 策略（按优先级） | 跳过条件 |
|---|---|---|---|
| S1 结构改写 | 消除算法级浪费（收益最大） | P119（two-pass→online softmax）、P120（向量点积→cube mmad）、P124（仅 mla 分支） | 已是 online softmax + cube 路径则整阶段跳过 |
| S2 CV 流水 | AIC/AIV 生产者-消费者重叠 | P121（SplitFuse 流水，依赖 P120）、P14（CV 预发射）、P125（跨任务预取，依赖 P121）、P29 | 仍在纯 Vector 路径时先回 S1 完成 P120 |
| S3 内存与搬运 | 摊薄握手/启动开销，L1/L2/UB 常驻 | P122（KV 大块化 + L1 预算）、**P142**（O tile UB 常驻，与 X38 GM ring 二选一）、P18/P53（L1 常驻）、P38（softmax 状态常驻）、P60（decode KV NZ）、P61（L2 cache）；host 侧 **P143**（launch 合并 + per-shape tiling 缓存，中小 shape） | mte2_stall 非主瓶颈且 L1/UB 预算已达标 |
| S4 mask/稀疏特化 | 消除无效计算与搬运 | P123（causal 跳过 + 边界 mask）、P118（mask 预生成独立 AIV）、P76/P77（sparse KV 预合并）、P80（Sink task） | 非 causal/sparse/sink 形态 |
| S5 调度与负载均衡 | 多核利用率与变长调度 | P116（GQA G→M 合并）、P117（host MatmulApiTiling 配方）、P126（GM 任务表变长调度）、P57/P58（FlashDecode 分区） | tiling_imbalance 非瓶颈且核已打满 |
| S6 通用收尾 | 基础数据通路与数值稳定 | D1、P1、P5、P7、A4、A5、P90；Bwd 另加 P91 | — |

## Step 4 — 正交 trait 追加表（基础链之上叠加；phase 切换阶段组，其余不重排链序）

| trait | 判定条件 | 追加卡 | 跳过条件 |
|---|---|---|---|
| phase=fwd（默认） | 正向传播 | 走正向基础链 S1-S6 | — |
| phase=bwd | 反向传播（FlashAttentionBwd / SDPA-Bwd / BlockSparseAttnBwd） | **切换阶段组**为反向链：重算 softmax P → dS=dO@V^T、dV=P^T@dO → dP=softmax 反向 → dQ=dS@K、dK=dS^T@Q → P91；mask/稀疏 trait 照常叠加 | 非反向算子 |
| causal | 下三角 mask | S4 P123 + P118 | 全 attention 无 mask |
| sparse:block_table | block_table/block_size 分块稀疏加载（SFA/FlashSparse 形态，7_SparseFlashAttention） | **P137**（紧凑化走 Cube：top-k 经 AIV gather 搬成紧凑 tile 再喂 AIC mmad，不再回退 Vector）+ P76/P77 + 稳定性 **P138**（L0 容量守卫）/ **P139**（多槽零初始化） | 稠密 attention |
| sparse:causal-window | band/窗口/stride mask skip | S4 P123 + P118 | 稠密 attention |
| paged/decode | Q len=1 + paged KV cache | S3 P60 + P61；S5 P126 + P57/P58 | prefill 全量 |
| 量化 | dtype∈{int8/fp8}、有 scale | 转 [quantization.md](quantization.md) 支路（P48/P49/P70），量化是 trait 不换族 | 无量化 |

## CV-Attn（CP8 门控广播）分支快速路由

> 适用：**conv-based 视觉/通道注意力模块**——`out = x ⊙ gate(x)`，gate 来自 N（≥2）条分支，
> 每条分支 = 全局/局部 pool(mean/max) → **conv 2→1（或 1×1）** → BN(training)/FC → ReLU/SiLU → Sigmoid。
> 代表算子：TripletAttention（ch/cw/hw 三分支）、BAM、CBAM（conv 版）、EMA。
> 分类依据见 [attention_family.md](../../../ascendc-evolution-knowledge/references/a3/algorithm_insights/attention_family.md) CP8。
> **判定分界**：分支含 **conv + BN(training)** → 本分支；纯 `pool→FC→scale`（SE/SK，无 conv）→ reduction。

**瓶颈预判（fast triage，按频率排序）**

| 观测 | 主瓶颈 | 首选策略 |
|---|---|---|
| host 做 zpool/BN stats + D2H 回读 + 每 launch 后 stream sync | host-CPU 往返 + 同步 | **P144**（host 算 zpool + fp64 归约 BN 统计 → 6 标量传参） |
| msprof 显示多 launch 固定开销 / GM workspace 往返 | launch-bound / MTE3 | **P144**（削 launch）+ **P145**（保多 launch 但去掉冗余相位） |
| `npuSynchronize error` / 偶发 NaN / UB 越界 | 同 launch GM 写后读 / 非 32B 对齐 DMA | **P145** + **P146** |
| H/W 小（如 7×11、15×15）→ AIV 利用率低 | tiling 负载不均 | **P104**（(b, cb, h) 三维切分填满核） |
| scalar pipe 占比高（40%+） | 标量通道 async | **P146 §4**（热路径去 `GetValue`，1-elem 向量 op） |
| K=3 大 C case 相对误差 ~1e-3 稳定超桶 | conv 输入量化未复现 | **P147**（RoundTo12Sig） |
| conv 计算量大（大 C × 大 kernel） | cube 未利用 | **P128**（bf16 hi/lo）/ **P129**（fused img2col） |

**策略顺序（默认路由）**：P144 → P145 → P146 → P147（四张为「跑通 + 到 1x」的充分集，
建议一轮内串完）；随后按 profiling 证据追加 P104/X58（多核）/ P128/P129（Cube conv）。

**两份实证基线**（直接作为 next-best 起点，勿从零写）：
- `64_TripletAttention_v2` round_3/parallel_2（L4/64_TripletAttention）：
  4-launch + host zpool/stats，geomean **1.28x**、50/50 精度。
- `36_BAM_evo` round_8/parallel_2（L4/36_BAM_evo）：0.98x、50/50。
  两者瓶颈均非纯计算，而是 host 往返 / scalar / launch 开销（BAM msprof：scalar 40%+）。

## 规则

1. **分支 + trait 顺序**：先 Step 2 定分支（head 结构 / CV-Attn 小类），再 Step 3 走基础链，最后 Step 4 按命中 trait 追加策略；`workflow_ref.stage_order` = 基础链 + branch 专属 + traits 追加。
2. **顺序不可颠倒**：cube 路径（S1）→ CV 流水（S2）→ 大块化（S3）是硬依赖链，跳阶段会产生编译失败或负收益变体。
3. **同轮组合必须过 [../strategy-compatibility.md](../strategy-compatibility.md) §1/§2** 的依赖/互斥检查（如 P121 依赖 P120；P123 与逐元素 mask 兜底互斥须删除旧路径）。
4. **workflow 分支首步读 baseline profiling 判定瓶颈域**：launch-bound → host 侧（P54/X28）；compute-bound 长 S → S2/S5 流水深化（P-GQA-7/MQA 观测教训）。
5. **Refine 阶段若 profiling 证据明确指向某瓶颈**，以 strategy-index.md "按瓶颈类型查表"优先，本工作流顺序让位于证据。
