---
name: external-cannbot-ops-ascendc-performance-best-practices
description: Ascend C 算子性能优化最佳实践库：按算子族组织优化经验与参考代码，供实施阶段查询。 当前覆盖 MatMul / Reduction
  / Scan(Cum) / Gather / Elementwise / Broadcast / Scatter / Conversion / Scalar /
  RadixSort / SIMT 等算子族； MatMul 与 FlashAttention 同时收录 arch22（A2/A3）和 arch35（A5）知识，
  以及 FA 家族 fused_infer_attention_score 通用 4 手段（icache 预取/静态 LocalTensor/类成员局部化/TilingData
  直读）、 SCALAR（S1-S5）与 SCALARLDST（MR1-MR5）scalar bound 优化、QSFA A5 启动头开销优化、 sparse attention/SCFA
  批量读取优化，及 FA 前向设计（含 MLA）、Q驻留选型与 FlashDecode 归约。 matmul 族含统一性能建模与 Bound 判定（真/假 MTE2、VF
  Bound）及 unitflag/k_tail_stepwise_copyout/l1_bank_conflict/weightnz/grouped_matmul
  等策略； 公共能力含 CV 融合（Cube-Vector 协同）编程模型与数据通路选型（__mix__(1,2)、GM 中转、L0C→UB 直通、CrossCore
  同步纪律）。 Gather 路径选择必须区分框架层物理 Transpose、kernel 内 Transpose、 PER_ELEMENT small-GD tiled
  full-load 和 Scalar/SIMT，并同时执行能力门控、UB 容量建模、搬运成本门控与代表 case A/B； `dim != last` 不是进入
  Transpose 的充分条件，Transpose 放不下也不代表 Scalar 一定优于 SIMT。 触发：查询某类算子的性能优化参考实现、实施某项优化，或判断
  Gather 等路径时需加载对应优化经验； arch35 目标编写或改造 Reg VF 前应先查询跨算子族 VF 范例库。
original-name: ascendc-performance-best-practices
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-17'
synced-commit: fe65c77485d42a93c7e45f3e67ab96d95504b9dc
license: UNKNOWN
---

# Ascend C 算子性能优化最佳实践

## P0：CANN SDK 安装目录只读

Agent 不得修改 `$ASCEND_HOME_PATH`、`$ASCEND_OPP_PATH` 及任意 `Ascend/cann-*` SDK 安装目录，包括编辑、覆盖、删除、移动、重命名、修改权限或创建文件。

DAV_3510 目标的 Reg/MicroAPI 实施先按 [VF 范例库](references/common/arch35/regbase/README.md) 与 `ascendc-regbase-best-practice` 的开源算子表检索相似实现。命中任意一边即可，但必须实际打开代码；两边均无时记录检索结果后再实现。改代码前按 [模板与构建契约](references/common/template_usage.md) 完成真实构建预检。首次编译/集成失败才完整读取 [CANN / Reg VF 编译问题 FAQ](references/common/arch35/regbase/cann_version_compat_faq.md)；API 或精度问题按需查 `ascendc-regbase-best-practice` 的 API 或 pitfalls 索引，不预读其 design/paradigm 部分。不得修改 SDK。

## 先按目标代际筛选

按下表定位算子族和目标架构：`arch22` = A2/A3（DAV_2201 / 220x），`arch35` = A5（DAV_3510）。多代际实现分别放在族内的 `arch22/`、`arch35/`，只加载目标目录；纯通用知识不分目录。MatMul、FA 已收录两代知识；arch22 的样例/源码提供 A2/A3 实现与 API 路由证据，但现成收益数据主要来自 A2，A3 仍须按目标型号复验。族级 guide 以策略路由和适用范围概览为主，主题分册说明实现依据；arch22 文档不固化外部工程的完整源码路径，使用时按目标源码取证。适用范围不等于实测范围，详细边界见 [代际与验证范围](references/common/architecture-scope.md)。

**RegBase / Reg / MicroAPI / VF 知识及范例是 arch35 独占**；arch22 不进入该实现路线。其他路线也须核对精确 SoC 与目标 SDK 能力，不能凭算法名称推断兼容。

## 算子分类体系

按 **算子族（operator family）** 组织优化知识，同一族内所有变体共享同一份文档（如 `matmul`、`matmul_mxfp4`、`batch_matmul`、`matmul_a16w16` 等均归入 `matmul` 族；`matmul_all_reduce`、`allgather_matmul`、`matmul_reducescatter`、`alltoall_matmul` 等均归入 `mc2` 族）。

| 类别 | 典型算子 | 适用架构 | 优化设计指南 |
|------|---------|---------|------------|
| MatMul 矩阵乘类 | MatMul, BatchMatMul, GroupMatMul, GroupedMatmul, MatMul_MXFP4, MatMul_A16W16 | arch22（DAV_2201 / A2/A3）；arch35（DAV_3510 / A5） | ✅ [A2/A3 性能优化指南](references/matmul/arch22/guide.md)（含检视判据） · [A5 性能建模](references/matmul/arch35/performance_modeling.md) · [A5 基本块](references/matmul/arch35/basic_block_design.md) · [A5 性能优化指南](references/matmul/arch35/guide.md) · [A5 Blaze 路线评估](references/matmul/arch35/blaze_design.md) |
| MC² 通算融合类 | matmul_all_reduce, allgather_matmul, matmul_reducescatter, alltoall_matmul | arch35（DAV_3510 / A5） | ✅ [性能优化指南](references/mc2/guide.md) |
| FlashAttention 注意力类 | FusedInferAttentionScore, FlashAttention, FlashMlaWithKvcache, KvQuantSparseFlashAttention, SparseFlashAttention, SCFA | arch22（DAV_2201 / A2/A3）；arch35（DAV_3510 / A5） | ✅ [A2/A3 性能优化指南](references/flash_attention/arch22/guide.md)（含检视判据） · [A5 性能优化指南](references/flash_attention/arch35/guide.md)（含 FIA 跨子类通用 4 手段与 SCALAR / SCALARLDST 子类决策、QSFA 启动头、稀疏注意力元数据，以及 FA 语义→流水→优化→实例；附源码阅读材料、数学验证脚本和历史实验摘要） |
| RadixSort 基数排序类 | TopK, KthValue, Sort, ArgSort, ArgMax/Min | arch22：A2 实测、A3 待验证；arch35：TopK | ✅ [A2 性能经验](references/sort/arch22/radix_sort.md) · [A5 TopK](references/sort/arch35/templates/usage_guide.md) |
| Scalar 编码与诊断 | 任意 ScalarBound 算子 | 通用（arch22 / arch35）；硬件参数按目标核验 | ✅ [性能优化指南](references/common/scalar/guide.md) |
| Reduction 归约类 | ReduceSum, Softmax, LayerNorm, RMSNorm, ArgMax | arch35（DAV_3510 / A5） | ✅ [实现索引](references/reduce/guide.md) · [模板代码与使用指南](references/reduce/templates/usage_guide.md) · [State Resident](references/reduce/templates/state_resident_design.md) |
| Scan / Cum 前缀扫描类 | Cumsum, Cummin, Cummax, Cumprod, LogCumsumExp | arch35（DAV_3510 / A5） | ✅ [实现索引](references/scan/guide.md) · [模板代码与使用指南](references/scan/templates/usage_guide.md) · [流式扫描(含索引)](references/scan/templates/cum_streaming_scan.md) · [跨核借R分段](references/scan/templates/cum_core_partition_scan.md) |
| Gather 索引类 | GatherV2, Gather, torch.gather, Embedding | arch35（DAV_3510 / A5） | ✅ [实现索引](references/gather/guide.md) · [模板代码与使用指南](references/gather/templates/usage_guide.md) |
| Elementwise 逐元素类 | Sin, Cos, Abs, Exp | arch35（DAV_3510 / A5） | ✅ [双缓冲设计](references/elementwise/double_buffer_design.md) · [向量化效率优化](references/elementwise/vector_efficiency_design.md) |
| Broadcast 广播类 | Add, Mul, Sub 等含广播轴算子；**Cdist/距离归约类在 reduce 维=1 时退化为广播** | arch35（DAV_3510 / A5） | ✅ [性能优化指南](references/broadcast/broadcast_design.md)（选型型，见结构 B） |
| Scatter 按轴散射类 | Scatter, ScatterAdd, ScatterReduce（对应 CANN ScatterElementsV2 / ONNX ScatterElements；**不含** ScatterNd/ScatterUpdate/ScatterList） | arch35（DAV_3510 / A5）的 SIMT 实现 | ✅ [性能优化指南](references/scatter/scatter_design.md)（六条 SIMT 并行/仲裁路径 + 三份范式代码）；跨代际的重复目标写入先读下方公共能力 |
| Conversion 数据转换类 | Transpose, Concat, Split | arch35（DAV_3510 / A5） | ✅ [Transpose融合设计](references/conversion/transpose_fusion_design.md) · ✅ [性能优化指南](references/conversion/guide.md) |
| Convolution 卷积类 | Conv2D, DepthwiseConv, Conv3DDX（反卷积输入梯度 stride≥2）, Conv3DDW（权重梯度） | arch35（DAV_3510 / A5） | ✅ [性能优化指南](references/conv/guide.md) · [Conv3DDX kernel split](references/conv/templates/conv3ddx_kernel_split.md) · [Conv3DDW StreamK + 确定性模板](references/conv/templates/conv3ddw_streamk_deterministic.md)（五文件 direct-launch 与流程说明） |
| NN 神经网络类 | GroupNorm | — | 📋 规划中 |
| Random 随机类 | RandomUniform, Dropout | — | 📋 规划中 |
| SIMT 进阶优化 | 条件分支、离散索引等不规则操作（**编程模型层优化**；仅在算子族选定实现仍使用 SIMT 后叠加） | arch35（DAV_3510 / A5） | ✅ [性能优化指南](references/common/arch35/simt/optimization-guide.md) |

“典型算子”只用于映射算子族，不代表每个变体在两代都有同等实现证据。arch22 MatMul 当前收录基础 MatMul/预加载/Split-K/缓存类策略，不提供 A5 MXFP4/MXFP8 实现；arch22 FA 当前以 FIA nonquant/MLA 源码为主要事实源，其他 FA 变体必须先证明共享 ABI、模板和 feature 路径，否则记录 `architecture_reference_gap`。

> 未收录的算子族返回「该算子族优化知识暂未收录」。各族详细的优化类型、叠加关系、选型决策见该族 `references/<family>/` 目录。
>
> **"数据依赖寻址"是一整类，先分读侧还是写侧**：访存地址由运行期数据（indices / offsets 张量）决定的算子，**读侧（聚集）→ [Gather 族](references/gather/guide.md)、写侧（散射）→ [通用写入归约选路](references/common/indexed_write_reduction.md)**；目标为 arch35 且语义匹配按轴 Scatter 时，再读 [Scatter 族 SIMT 实现](references/scatter/scatter_design.md)。
> 这类算子外形逐元素，**极易误判进 Elementwise 族**，但两者优化范式相反：Elementwise 求"高效搬进 UB 用向量指令算"，而地址随运行期数值跳变时 UB tile 化不成立。判据是**地址是否数据依赖**，不是计算长得像不像 elementwise。
> 两侧的取舍也不同——读侧可以靠 Transpose / full-load 把不规则读变规则；写侧的原始落点由 index 决定，但可在数值语义与成本允许时先按目标分组，再由目标所有者规则写回。直接写入时则要选择并行划分与冲突消解策略。
>
> **先完成主算子族及数据依赖写入选路，再决定是否进入 SIMT 进阶优化**：融合算子保留主族；只要存在多个源写同一目标，就叠加读取公共写入归约指南。语义匹配按轴 Scatter 且目标为 arch35 时，才加载 Scatter 族的 SIMT 范式代码。只有选定实现仍包含 SIMT 时，才继续读取 SIMT 指南优化线程数、寄存器压力、VF 内分支和访存。
>
> **融合算子的优化**：融合算子必须按内部计算单元拆解，而不是按外层算子名称排除基础族。例如 AddRelu 可拆解为 Broadcast（Add）+ Elementwise（Relu）。如果融合计算流程包含矩阵乘单元，则每个矩阵乘单元都必须独立命中 MatMul 族，再把结果与外层 Vector/Reduction/通信路径合并分析；外层名称不能把它判为“非 MatMul”。
>
> **内嵌 MM 的计算引擎边界**：保留 Cube MMAD、改用低阶接口重建 `L1 → L0 → Mmad → Fixpipe` 流水，与把 MM 计算替换为 AIV 向量指令是两条不同路线。`register-resident-vector-chain` 只适用于已经确认是向量计算链的 AIV 路径，不能作为 MM 替换或内嵌 MM 的默认依据。
>
> **arch35 SCALAR_BOUND 子类选择规则**：DAV_3510 上 FIA（及其他 FA 变体）命中 SCALAR_BOUND 时，必须按 Step 3 报告的 `SCALAR%` 与 `SCALARLDST%` 判定子类（详见 [references/flash_attention/arch35/guide.md](references/flash_attention/arch35/guide.md) 的「子类选型规则」表）：`SCALAR% > SCALARLDST%` 走 SCALAR 子类；`SCALARLDST% ≥ 95% 或 > SCALAR%` 走 SCALARLDST 子类；两者都 ≥ 95% 先做 MR1–MR5 再重判。arch22 不使用这些 A5 阈值，改读 [A2/A3 Scalar 与启动开销](references/flash_attention/arch22/scalar_overhead_design.md) 并按目标 profiling 重判。

## 公共优化能力（跨算子族）

FA按[分层参考资料](references/flash_attention/arch35/guide.md#从语义到流水的分层参考资料)定位语义、流水、优化技巧与工程实例。迁移共同机制时，另行核对K/V共享、布局和缓冲预算等实例条件。区分“源码有实现、当前case选中、运行时有效复用”，再决定改派发还是改核内实现；验证以目标case和记录的证据范围为准。

| 优化类型 | 适用架构与场景 | 文档 |
|---------|---------|------|
| **数据依赖写与重复目标归约** | arch22 / arch35 的通用算法选路；具体 API 和收益按目标核验 | [目标所有权、暂存分组与数值顺序](references/common/indexed_write_reduction.md) |
| **尾块处理（Tail Block）** | arch35；数据量不能被 tile 大小整除 | [尾块处理指南](references/common/arch35/tail_block_design.md) |
| **数据搬运（DataCopy）** | arch22；910B 搬运经验，A3 需复验 | [数据搬运](references/common/arch22/datacopy_optimization_design.md) |
| **UB/TBuf 常驻复用与 Bank 冲突** | 两代分别使用各自 bank 规则；容量与 padding 按目标取证 | [arch22 Bank 冲突](references/common/arch22/ub_bank_conflict.md) · [arch35 常驻复用与 Bank 冲突](references/common/arch35/ub_resident_design.md) |
| **CV 融合（Cube-Vector 协同）** | arch35；Cube（MMAD）与 Vector（激活/Cast/量化/归约）需协同的融合算子：选数据通路（分离/GM中转/L0C→UB直通）、`__mix__(1,2)` 编程模型、CrossCore 同步纪律、双 AIV 分工 | ✅ [CV融合设计](references/common/arch35/cv_fusion_design.md) |
| **Reg VF 编译与兼容** | arch35 独占；构建模式、模板适配、版本兼容或编译诊断 | [CANN / Reg VF 编译问题 FAQ](references/common/arch35/regbase/cann_version_compat_faq.md) |
| **Reg VF 范例库** | arch35 独占；按计算模式、前置条件及 readiness 查询 | [VF 范例库与机器可读目录](references/common/arch35/regbase/README.md) |

## 查询方式

| 输入 | 必需 | 说明 |
|------|------|------|
| 精确 SoC / 架构 | 是 | 从请求和目标工程取证；多个目标分别查询 |
| 算子名 | 是 | 如 `matmul` / `matmul_mxfp4` / `batch_matmul` |
| 优化类型 | 否 | 如 `pingpong` / `swat` / `streamk` / `fullload` / `scale_coalescing` / `mte2_preload`；不提供则加载目标代际适用部分 |

查询流程：**精确目标与代际 → 算子名 → 拆解内部计算单元 → 映射到算子族 → 读取目标适用的指南与模板 → 完成族级选路、模板和 Tiling 方案 → arch35 按计算模式查询 VF 范例库 → 判断选定实现是否使用 SIMT → 必要时进入 SIMT 进阶优化**。算子族映射规则：精确匹配族名直接命中；以族名为前缀或核心词（如 `matmul_mxfp4`、`batch_matmul`）归入该族；其他形态由调用方按功能显式指定。融合算子要为每个内嵌计算单元记录独立的族、数据流和实现路线。DAV_3510 case 的热点若是向量计算、归约、寄存器重排或 Cast/Pack，即使当前源码没有 Reg/MicroAPI，也必须查询一次 [VF 范例库](references/common/arch35/regbase/README.md)；查询不等于选中，仍须满足条目前置条件和 readiness 门禁。

### 融合算子内嵌 MatMul 路由（arch35）

目标为 arch35 且源码、伪代码或 profiling 显示融合算子包含 `mm`/`matmul`/`mmad` 单元时，对每个单元分别执行以下步骤，并在报告中保留 `mm_route`、`mm_shape`、`mm_layout` 和证据路径：

1. 先读取 MatMul 族的性能模型、基本块切分、流水和模板，记录 `M/N/K`、dtype、L1/L0 容量、`k-tail`、`unitflag`、`Fixpipe` 和同步约束。低 AIC MAC 利用率、较小 K 或外层融合本身都不能单独推出“AIV 替换 MM”。
2. 候选顺序固定为：Blaze 路线（仅 DAV_3510，按「MatMul Blaze 确定性路由」判定）→ 保留 Cube MMAD 的低阶流水重建 → 现有高阶 MatMul API 的数据流/tiling 优化 → 有独立实测对照后才允许评估 AIV 替换。低阶流水重建应核验 `A2/B2/CO1` LocalTensor、`LoadData2DParamsV2`、`MmadParams`/`Mmad`、`Fixpipe`、`DataCopyCO12DstParams` 和 `MTE1`/`MTE2`/`MTE3` 事件配对。
3. 必须把“替换 MM 计算引擎”和“用低阶接口重建 MM 内部流水”分别记录为不同路线；不能把前者的性能假设套在后者上。只有在 Blaze 路线、低阶 Cube 路线、Cube 高阶路线和 AIV 路线有同 case、同环境、同协议的可比较证据时，才可选择 AIV 路线。

主算子族确定后，如源码显示多个源可能写同一目标，还须叠加[公共写入归约选路](references/common/indexed_write_reduction.md)，即使主族不是 Scatter；该页只提供算法候选，不能将 arch35 Scatter 的实现 API 用到其他代际。

**RoPE 确定性路由**：仅 DAV_3510 目标执行以下 VF 范例匹配；A2/A3 保留目标原实现并另行取证，不进入 A5 catalog。源码/算子名出现 `rope`、`rotary`、`rotary_pos_emb`，或计算语义为 `x*cos + rotate(x)*sin`、前后半交换乘加时，必须命中 catalog 的 `paired-segment-rotate` 候选并完整读取 `references/common/arch35/regbase/paired_segment_rotate.md` 与其代码。不得因为基线未使用 Reg/MicroAPI、没有 RoPE 专属目录或先归入 Elementwise 族而跳过；只有逐项比较布局模式和前置条件后才能排除。

**量化 VF 确定性路由**：DAV_3510 热点包含量化、Cast 或 Pack 时，先从公式和数据流提取统计量、scale 与物理布局，再按以下优先级查询 catalog；`int8`/`fp8` dtype 或算子名中仅出现 `quant` 不是选中依据。

1. 存在 MX/microscaling、shared exponent、block scale、E8M0 scale，且 reciprocal、FP4/FP8 Pack 与 shared exponent 构造保持同一寄存器流水 → `shared-exponent-pack`。即使同时计算局部 amax，也优先走此条。
2. 同一 UB 驻留行先以 amax 或 min-max 生成动态 scale，再第二遍完成 symmetric/asymmetric/smooth quant 的 Cast/Pack → `two-pass-amax-cast`。
3. 只有外部静态 scale 的 quant/requant/dequant、量化 MatMul 的 Cube 主路径，或无法证明上述数据依赖 → 不强行套用这两个条目，记录 `no_matching_vf_example` 并继续算子族选路。

每个 DAV_3510 量化 case 都须在方案中记录 `quant_vf_route`：命中信号、选中 ID、被排除 ID 及理由、`preconditions` 判定、readiness 和目标工程验证方案。两个量化条目当前均为 `annotated_excerpt`，只能补齐目标上下文并重验，不能表述为可直接复制的完整模板。

**MatMul Blaze 确定性路由**：DAV_3510 的 matmul 族选路必须先完成 Blaze 路线判定（`blaze_native` / `blaze_custom` / `blaze_migration` / `unsupported`，见 [Blaze 路线评估](references/matmul/arch35/blaze_design.md)），再进入 AscendC 策略选型；判定为 `unsupported` 时记录 `unsupported_points` 证据后继续。ops-tensor 只读且 commit 进证据身份；不得因源码未用 Blaze 而跳过评估，环境不可得时记 `environment_blocked` 降级。命中 Blaze 路线后跳转 `ascendc-blaze-best-practice`（迁移路线为 `ascendc-blaze-migration`）执行，不在本知识库复述其方法。

### SIMT 进阶优化门控（算子族优先）

SIMT 是编程模型层的进阶优化，不能替代算子族选型。按以下顺序执行：

1. 初次读取源码时只记录 `simt_candidate`：出现 `__simt_vf__` 或 `Simt::VF_CALL` 可直接记录；出现 `asc_vf_call` 时检查目标函数是否由 `__simt_vf__` 修饰，或是否使用 `threadIdx` / `blockIdx` / `blockDim` / `gridDim` 等 SIMT 内置变量。`asc_vf_call` 也可调用 SIMD VF，不能仅凭调用名判定；`__launch_bounds__` / `LAUNCH_BOUND` 等弱特征只作为辅助证据。
2. 无论是否记录 `simt_candidate`，先按精确目标筛选算子族，读取 `references/<family>/` 中目标适用的指南与模板，完成实现路径、模板、数据流和 Tiling 选型。禁止先按 SIMT 指南打磨现有实现。
3. 族级选定实现不使用 SIMT 时，清除 `simt_candidate`，不加载 SIMT 指南。即使原始 kernel 使用 SIMT，也应按族级结果改走选定路径。
4. 族级选定实现使用 SIMT 时，再加载 [SIMT 性能优化指南](references/common/arch35/simt/optimization-guide.md)，把适用 Rule 作为族级方案之后的进阶优化项；族级指南始终具有更高优先级。
5. 报告先记录算子族、选定路径、模板和 Tiling，再单列可选的 SIMT 进阶项及证据（文件、函数或调用点、profiling 瓶颈、参考 Rule）。实施阶段保持同一顺序，不能只凭源码命中标记提前进入 SIMT 优化。

### 模板代码查找链路

按 [模板源码查找、复制与构建契约](references/common/template_usage.md) 完成族级模板选型、源码阅读、候选副本适配和构建验证。仅 arch35 的 Reg VF 路线还须查 [VF 范例库](references/common/arch35/regbase/README.md)；具体代码入口、预检命令与编译诊断保留在上述参考文档中，不在本入口重复。

## 性能优化文档结构（两类，均为性能优化，按算子族性质选其一）

> 两类都是"让算子更快"的最佳实践，区别只在优化的形态：A 是对单一实现调参，B 是在多种实现间选对路径。

### A. 单实现调优型（对单一实现做某项性能优化，如 matmul pingpong/swat）

每份 `<优化类型>_design.md` 的章节组织：

**必选章节：**

1. 优化目标 —— 效果与量化收益（kernel μs / MTE2 段 / CUBE busy 等）
2. 架构概览 —— 存储层级、数据流、buffer 布局、事件同步模型
3. 关键参数 —— 新增 / 调整字段与 Host 侧计算
4. 核心计算循环 —— 改造前后对照（含事件同步）
5. 优化的关键修改点 —— 表格形式

**可选章节：**

6. 注意事项 / 约束 —— 前置条件、L1/L0 预算、边界与兼容性
7. 实施常见问题与解决方案 —— 高频踩坑与根因
8. 实测性能、选型决策、与其他优化的叠加关系、自检清单

### B. 多实现选型型（族内有多种实现路径，选对即性能最优，如 `broadcast`）

这类族的性能优化**本质就是按场景选对实现**——`broadcast` 的 OneDim 前置快路径 + tile 内 NDDMA / DataCopyPad / UB 三类实现，选错直接慢甚至出错（nan）；选对才快。这同样是性能最佳实践，只是优化点在"选型"而非"调参"。结构：

1. `<family>_design.md` 总览 —— 速查表、**选型决策树**（决定性能的核心）、tiling/切分逻辑、约束
2. 每种实现一份分册（如 `nddma_design.md`）—— 适用条件、Kernel 写法、注意
3. 可直接参考/裁剪的 AscendC 范式源码，由族级指南索引
4. 进阶/边界单列（如 `advanced_tiling.md`）

> 选 B 时"收益"体现为**选对实现避免劣化**，可不强制量化 μs，但必须给出**选型依据**（什么场景用哪种实现、为什么快/为什么错）。

## 扩展新算子族

1. 创建 `references/<family>/` 目录（以族名命名，非单个变体）
2. 判断族性质：优化型 → 按 A 写优化设计文档；实现指导型 → 按 B 写总览、分册和必要的范式源码
3. 更新本 SKILL.md 分类表格，并按代际范围参考登记适用架构、来源与实际验证范围

## 内容形态

知识以 Markdown 为入口；实现指导型算子族（如 `broadcast`）可提供 AscendC 范式源码，并在族级指南中给出链接和适用范围。
