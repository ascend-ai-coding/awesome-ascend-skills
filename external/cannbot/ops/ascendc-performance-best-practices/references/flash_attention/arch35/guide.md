# FlashAttention 家族性能优化策略索引

> 适用架构：arch35（A5 / DAV_3510）；具体型号、SDK 与验证范围以正文为准。

按瓶颈和数据复用模式查找策略：既包括SCALAR_BOUND与启动头，也包括Q驻留、MTE流水、tile预算、在线softmax与KV分片归约。

适用于 `flash_attention` 族变体：`fused_infer_attention_score`（FIA，已实测）、`kv_quant_sparse_flash_attention`（QSFA，A5 MLA 路径已验证）、sparse attention / SCFA `GetValue` DataCache 批量读取案例、`flash_mla_with_kvcache`的M96/S112流程、优化和源码阅读资料（记录两个长KV主例的历史归约实验摘要）、`mla_prolog`（MLA 前处理 Cube+Vector 大融合算子）、`compressor`（MLA 风格 KV 压缩融合，decode T=1 访存受限）；其它FA变体需按布局、容量、同步和精度边界迁移。

> **使用前置**：调优时先确认瓶颈与复用模式。标量策略参考`bound_type`、`SCALAR%`、`SCALARLDST%`或启动头；矩阵/搬运策略查看Q复用次数、MTE等待和多层容量。学习FA计算与流水时可直接进入分层文档与Markdown图，不要求先具备profiler报告。

| 策略 | 子类判定 | 适用场景 | 核心手段 | 详细文档 |
|------|---------|---------|---------|---------|
| **scalar_general_techniques**（4 手段）| 任意 FA SCALAR_BOUND（含 SCALAR / SCALARLDST / MIXED，FIA 已实测） | 通用 Scalar 减载：head 开销 / Init 开销 / 主循环 I-Cache miss | icache 预取 + 静态 LocalTensor + constParam 局部化 + tilingdata 直读入参 | [scalar_general_techniques_design.md](scalar_general_techniques_design.md) |
| **scalar_precompute**（S1–S5）| `SCALAR% ≥ 80%` 且 `SCALARLDST% < SCALAR%` | SCALAR ALU 是真瓶颈：softmax 掩码位运算、KV 偏移成员预乘、循环不变量外提、pow2 div/mod、工作空间基址预计算 | 把 ALU 重计算移到 Host / kernel 入口预计算，热循环只读结果 | [scalar_precompute_design.md](scalar_precompute_design.md) |
| **scalarldst_subbound**（MR1–MR5）| `SCALARLDST% ≥ 95%` 或 `SCALARLDST% > SCALAR%` | SCALARLDST 访存通路饱和：lazy-init 守卫每轮多 1 次 load、`*_Pow2_/*_Shift_/*_Mask_` 三成员配对、热循环内 `this->member` 反复 load | 守卫清理 / 准常量分支折叠 / 局部 const 捕获 / switch 折叠 / 成员局部化，直接减 load | [scalarldst_subbound_design.md](scalarldst_subbound_design.md) |
| **tilingdata_cv_ssbuf_forwarding** | 无明显流水 bound，但 A5 mixed AIC/AIV kernel 启动头开销明显 | A5 MLA mixed AIC/AIV：cube/vector 重复 full TilingData 拷贝，cube 只需共享字段子集 | vector full copy TilingData，抽取 `CVSharedParams` 后经 `__ssbuf__` + flag 转发给 cube，cube 跳过 full copy | [tilingdata_cv_ssbuf_forwarding_design.md](tilingdata_cv_ssbuf_forwarding_design.md) |
| **getvalue_cacheline_batching** | sparse attention / SCFA hot loop 中连续或近连续读取 topk index、block id 等 GM metadata | sparse KV：每 1-2 个 `GetValue` 后穿插 offset / block table / DataCopyPad，DataCache 局部性不足 | 先 cache-line friendly 批量读取连续 metadata 到局部数组，再按原语义粒度消费 | [getvalue_cacheline_batching_design.md](getvalue_cacheline_batching_design.md) |
| **query_l0a_resident** | 本核同一Q跨KV复用；区分已有路径未选中与核内驻留不足 | 同核同Q跨多个KV块；按目标D、dtype和布局重算预算 | 部分Q驻留、临时Q与P分时复用；MLA例：Q96占18 KiB、总L0A 63 KiB | [Q驻留与使用时段复用](common-optimizations.md) |
| **tile_pipeline_budget** | 增大M受L1/UB限制；等待KV或完整PV结果 | 混合AIC/AIV的QK→softmax→PV流水 | M/S2/槽数联合预算与分段消费；MLA例：三槽预取、PV两段N256 | [分块预算与流水](flow-to-pipeline.md) |
| **online_softmax_splitkv** | KV长、Q并行不足；需要流式统计与跨核合并 | 先核对分片输出归一化格式和无效行语义 | first/last特化、FP32累计、归一化Oj加权FD、FA/FD阶段复用UB | [在线更新与FD](semantics-and-decomposition.md) |
| **mask_visibility_pipeline** | 因果长KV/短Q，多数tile已可证明全可见 | 带mask路径的候选；随库阅读源码未集成，需迁移复验 | 全可见tile选择无mask VF；当前块mask在等QK前准备；检验关键路径收益 | [mask与流水](common-optimizations.md) |
| **mla_prolog_fusion** | Cube+Vector 大融合算子（多路 Matmul + RMSNorm + RoPE + 量化 + cache 写融进单 kernel），非 SCALAR_BOUND 场景 | MLA 前处理及同形态注意力前/后处理融合：需要 CV 流水互相掩盖、多量化模式选路、L1/L0/UB/workspace 内存规划 | 算子融合 + 权重拼接 + 反量化/量化融合 + CV 流水同步（点对点/全核 barrier/稀疏同步）+ Double Buffer + NZ 搬运 + TilingKey 位域编译期特化 | [mla_prolog_fusion_design.md](mla_prolog_fusion_design.md) |
| **compressor_fusion** | Cube+Vector 大融合的 KV 压缩算子，**decode 单 token(T=1) varlen、权重载入访存受限 + 核欠订**（与 mla_prolog 同族、瓶颈相反：非算力/流水而是带宽+同步）| MLA 风格 KV 压缩（投影 MM + softmax 加权和 + RMSNorm + 半旋 RoPE + cmpKv 写）；小 batch/decode 下核欠订、跨核 K 规约走 GM+SyncAll | K1 shape-adaptive core（消空转核）+ K2 MTE2 权重合并载入 + K3 FULL_LOAD K 轴核并行（摊薄载入）+ K4 静态 tiling 低同步；证伪护栏（权重留 Cube、只看 MTE2 会误判）+ CV-UB 跨核规约 GM+SyncAll 架构下界 | [compressor_fusion_design.md](compressor_fusion_design.md) |

## CANNBot-DSL 实现知识（语言子目录 cannbot-dsl/）

本族的 CANNBot-DSL 实现知识统一收在 [cannbot-dsl/](cannbot-dsl/index.md)，分两层：**optimize/**（数学语义 → 流水建模 → 容量方程，讲 DSL FA 算子怎么优化，SFA 举例）与 **assemble/**（积木模块 + 胶水，讲怎么组装出高性能 FA）。方法链是把优化手段落到 DSL 代码的通用方法论，不绑定具体算子（跨族方法论、编程模型要点与证据纪律见 [CANNBot-DSL 公共实现知识](../../common/arch35/cannbotdsl/index.md)）；SparseFlashAttention（SFA）是当前唯一端到端验证过的完整实例（1.85×、seed 逐位一致、37 步消融），仅作贯穿示例——同构链可拓展，异构链逐层重建。按信号进入对应层：

| 层 | 回答的问题 | 进入信号 | 文档（现以 SFA 实例贯穿） |
|---|---|---|---|
| 1. 数学语义 | 算子算什么，哪些数值语义必须保留 | 新变体/新计算链建模起点 | [1-math-semantics](cannbot-dsl/optimize/1-math-semantics.md) |
| 2. 流水建模 | 数据在 Cube/Vector 两侧如何流动、节拍如何编排 | CUBE busy < AIC 窗口（出洞）或 AIV 断流；AIV SCALAR 贴满但 CUBE <97% | [2-pipeline-modeling](cannbot-dsl/optimize/2-pipeline-modeling.md)（含 AIC+AIV 延迟线通用模板） |
| 3. 容量方程 | tile 形状、缓冲深度、滞后如何从台账解出 | 需为 AIC+AIV 混合核定 tile/环深/滞后，或调参前缺容量证明 | [3-capacity-equations](cannbot-dsl/optimize/3-capacity-equations.md)（四层法） |
| 4. 积木模块 | 整类投放的参考实现模块（胶水随类走） | 机制级切片拼装面不可控，需要整类参考实现 | [assemble/](cannbot-dsl/assemble/index.md)（积木 M0-M4 + 胶水，SFA 整类投放件逐模块独立文件） |
| 5. 胶水模块 | 表达层规则与拼接件 | 拆块拷贝数值错位、双代被同步校验器拒绝、动态 region 报错、链式发射 unit_flag 不明 | [glue-idioms](cannbot-dsl/assemble/glue-idioms.md) / [glue-fragments](cannbot-dsl/assemble/glue-fragments.md) |

上方策略索引只收共同（语言无关/AscendC）优化手段；DSL 条目不再混入策略表，统一由本节与 [cannbot-dsl/index.md](cannbot-dsl/index.md) 索引。

## 从语义到流水的分层参考资料

在线softmax、Q复用、S2拆分及流水依赖属于FA前向的共同知识。本节直接按计算问题组织阅读；具体数字和代码以当前MLA为实例，按当前case缺少的信息选读：

| 层次 | 文档 | 内容 |
|---|---|---|
| 1. 算子语义到流程分解 | [语义与计算任务](semantics-and-decomposition.md) | FA共同语义、矩阵轴、核内在线归约与跨核FD；MLA的K/V共享特例 |
| 2. 流程分解到流水编排 | [通路与流水](flow-to-pipeline.md) | MTE2/MTE1/M/FIX/V/MTE3三轮、生产消费通知、缓冲生命周期 |
| 3. 常见优化技巧 | [优化条件与选型](common-optimizations.md) | Q驻留完整推导、实际派发、复用分布、tile/预取/PV/mask/FD |
| 4. 当前工程例子 | [MLA：BSND查询与分页BBND缓存的M96/S112实例](example-mla-m96-s112-bsnd-paged-kv.md) | 完整流程、函数入口、优化触发点、L1/L0/UB/FD容量表 |
| 4. SFA工程例子 | [SparseFlashAttention token-wise MLA实例](example-sparse-flash-attention-tokenwise-mla.md) | Q/K 576维语义、每token离散gather、64×128基本块、V0→MM1→V1→MM2→V2、504 KiB L1、split-G及10个真机case |
| 5. SFA DSL 双核流水实例 | [SFA DSL 延迟线流水](cannbot-dsl/optimize/2-pipeline-modeling.md) + [容量方程](cannbot-dsl/optimize/3-capacity-equations.md)（更多见 [DSL 最佳实践库](cannbot-dsl/index.md)） | CANNBot-DSL AIC+AIV 混合核：AIV 直推 L1（无 GM workspace 往返）、延迟线滞后 1/3/4、O 原地累加、四层 tiling 建模、1.85× 实测阶梯与 37 步消融账 |

> **`mla_prolog_fusion` 与本节的关系**：本节四层资料聚焦 FA **前向注意力主体**（QK→softmax→PV 的在线归约与流水）；`mla_prolog_fusion` 聚焦注意力**前处理**（多路 Matmul + RMSNorm + RoPE + 量化 + KV/KR Cache 写的 Cube+Vector 大融合），两者互补，可分别参阅。

迁移边界：QK→softmax→PV、在线状态重缩放和正确分片加权可用于相同数学语义的FA；K/V同源、D576/DV512、Q96地址、三槽及FD16/FD8属于当前实例条件。独立K/V须保留各自搬运，容量、mask、头映射及同步按目标实现核验。详细条件见[优化篇](common-optimizations.md)。

收到case先确认SoC/SDK、shape/layout/dtype、mask/LSE与实际派发，按需选读。实施前核对容量和最后消费者，沿用正确baseline；检查Out/LSE和完整耗时，只做分析时保留待验证项，用例数遵守用户范围。

区分源码已有实现、当前case是否选中、运行时是否有效复用；优化可以是已有路径的派发调整，也可以是核内改造。四层文档的图均在Markdown内，源码和脚本为本地普通文件，历史实验仅保留文档摘要。

`sources/`保留6个M96/S112关键源码阅读文件，不包含完整依赖/构建工程。
BSND/BBND无mask例子用于流程教学；历史设备归约A/B来自TND/PA_NZ带mask主例，
不能混用覆盖声明。SFA的10个精度case、value输入探针与K/S2/G扫描是2026-09-16
新增的Ascend950PR实测，覆盖范围、调试要点和必要代码片段均放在
[token-wise实例](example-sparse-flash-attention-tokenwise-mla.md)，不另建源码阅读索引。

## 子类选型规则（强制）

本节针对已经判定为SCALAR_BOUND的场景；不是Q驻留、MTE流水或矩阵分块策略的前置步骤。

| `SCALAR%` | `SCALARLDST%` | 子类 | 加载策略 |
|---|---|---|---|
| 任意 SCALAR_BOUND | 任意 | **通用候选** | 先检查 scalar_general_techniques 4 手段的触发条件，仅实施命中热点且满足前提的项，再按下方子类深化；不要求四项全部执行 |
| ≥ 80% | < SCALAR% | **SCALAR** | scalar_precompute（S1–S5） |
| 任意 | ≥ 95% **或** > SCALAR% | **SCALARLDST** | scalarldst_subbound（MR1–MR5） |
| ≥ 95% | ≥ 95% | **MIXED** | 先 MR1–MR5（对 SCALAR 无副作用），重采 trace 后重判 |

> 当 SCALARLDST 是真正主瓶颈时，禁止仅按 scalar_precompute 推进。S1/S2/S5 的"成员预计算"会把 SCALAR ALU 压力转嫁给 SCALARLDST 访存通路，可能导致 case cycle 劣化。

## 效果摘要

| 策略 | 已验证效果 |
|------|------------|
| `scalar_general_techniques` | 8 组 case Scalar 耗时平均下降 33.22%，端到端耗时平均提升 14.74%，SCALAR_BOUND 跳出 |
| `scalar_precompute` | 稳健减少 decode 热循环地址计算、除法和公共前缀重复计算；预期 SCALAR 下降 12% - 30% |
| `scalarldst_subbound` | 最长核 case cycle 从 18353 降至 18076，约 -1.51% |
| `tilingdata_cv_ssbuf_forwarding` | 减少 A5 mixed AIC/AIV kernel 中 cube 侧 full TilingData 拷贝和启动 Scalar 指令 |
| `getvalue_cacheline_batching` | 改善 sparse index / GM metadata 连续 `GetValue` 的 DataCache line 利用率 |
| `query_l0a_resident` | 每个满Q块处理T个KV块时，相比重复装Q96理论减少(T−1)×18 KiB的L1→L0A搬运。SFA的Q跨S2驻留与L1A释放配对，不能只改加载侧 |
| `sfa_v0_gather_chain` | SFA离散gather：16行UB聚合减少workspace握手；两token合并DataCopyPad在G≤64常见、split-G下要按shape选型。见[token-wise实例](example-sparse-flash-attention-tokenwise-mla.md) |
| `tile_pipeline_budget` | 当前M96/S112多层容量与生命周期有源码依据；未提供每个分块/预取技巧的独立耗时消融 |
| `online_softmax_splitkv` | 数学正反对照验证重缩放/归一化与正确FD权重；历史报告记录同binary允许S2拆分+FD相对禁止拆分，两长KV主例约4.33×且两模式精度通过；不是纯FD耗时或局部VF技巧消融 |
| `mask_visibility_pipeline` | 保留全可见直通和提前准备的候选方法及边界；本地阅读源码未集成，无新增上板验证 |
| `mla_prolog_fusion` | 方法论层（源自 MlaPrologV3 设计与实现）：减少 kernel 下发与中间结果 GM 往返、CV 并行掩盖搬运；落地具体 case 后需 profiling 复测 |



## 扩展新 FA 变体

当新增 sparse attention / FAS-grad 等变体时：

1. 在 `references/flash_attention/arch35/` 内增列同类 design 文档，文件名使用 `<strategy>_design.md`，具体FA变体写入适用场景或正文；共同计算机制补充到本层语义、流水与技巧文档，具体实现放入按含义命名的实例文档，不额外建立算子子目录
2. 通用方法论（S1–S5 的预计算思路、MR1–MR5 的 load 减载思路）可直接迁移；变体特定的代码位置 / 字段名应在 design 文档里单独沉淀
3. 更新本 guide.md 表格 + [SKILL.md](../../../SKILL.md) 算子族表格
