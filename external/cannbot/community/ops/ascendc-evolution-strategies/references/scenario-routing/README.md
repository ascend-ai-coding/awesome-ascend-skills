# Scenario Routing — 算子归族与工作流入口

> 进化优化的第一步：**先归族，再进对应族的工作流（workflow），按工作流的阶段顺序调用策略卡**。
> 不要直接在 strategy-index.md 平铺选卡——那里只是注册表，没有主次顺序。

## 使用协议

```
Step 0 — 归族：用下方判定表确定算子族（结构判据，不靠算子名）
Step 1 — 通用前置：打开 _common-prerequisites.md，逐项判定（命中用卡，未命中跳过）
Step 2+ — 族内阶段：打开对应族的 workflow 文件，按阶段顺序推进
Refine 阶段 — profiling 证据可用时，strategy-index.md 的瓶颈查表优先于工作流顺序
```

## 族判定表（按特异性从高到低判定，命中即停）

**不要靠算子名字判族**（EMSA / BAM / CoT / CrissCross / Window / Axial / BigBird / Linformer 等变体名字不含标准词）。判据来自三处可观察证据：`model.py` 数学结构 / kernel 与 host 代码特征 / shape 特征。

| 顺序 | 族 | 结构判据（命中任一即归族） | 易混淆点 | 工作流 |
|---|---|---|---|---|
| 1 | **Attention 系（QK^T 指纹）** FA(MHA/self/cross) / GQA / MLA / **CV-Attn** | ① 计算主干 `Q@K^T → (scale+mask) → softmax → @V`（含 online softmax 变体）；② kernel 含 online softmax 状态机（running max/sum/rescale）或 FA 特征参数（`preTokens`/`nextTokens`/sink/paged KV cache）；③ head 结构（分类判据见 [../../../ascendc-evolution-knowledge/references/a3/algorithm_insights/attention_family.md](../../../ascendc-evolution-knowledge/references/a3/algorithm_insights/attention_family.md)：线性/稀疏/压缩 大类别 + head 结构）：`num_q_heads == num_kv_heads` → **FA(MHA/self/cross)**；`num_q_heads == G×num_kv_heads (G>1)` → **GQA**（`num_kv_heads==1` → MQA 为 GQA 特例）；MLA = **latent c 压缩**（KV 经 latent 向量重建，decode 特征）；④ **conv-based CV-Attn**（Triplet/BAM/CBAM：global pool → **conv 2→1** → BN(training) → 激活 → gate 广播，分支 ≥ 2，CP8） | softmax 夹在两个 matmul 中间是 FA 指纹，**必须先于 matmul/conv 判**；**纯 MLP 通道注意力（SE/SK：global pool→FC→scale，无 conv）有 sigmoid 但无 QK^T、无 conv、无 head 结构 → 不归本行**（落 reduction）；SSM 线性递归无 QK^T → 下行 | [flashattention.md](flashattention.md)（branch=standard/gqa/mla/cv-attn） |
| 2 | **SSM/Scan 系**（GDN-KDA / GLA） | ① 沿序列维**递归/chunk-scan**（`h_t = f(h_{t-1}, x_t)`），**无 QK^T、无 softmax**；② delta rule 形态 `S ← β⊙(k⊗v − corr) + S`（含 corr 修正项）→ **GDN-KDA**；③ 门控衰减形态 `S = α⊙S + k⊗v`（含衰减门 α，无修正项）→ **GLA**；④ kernel 含 chunked/并行 scan 或 state 寄存器证据 | k⊗v 是**外积**易与 matmul 混——但递归状态更新是串行依赖非纯 GEMM，**先于 matmul 判**；名字含 Attention/注意力但无 QK^T 即非 FA 行；判据须读 `model.py` 数学结构（α/β/corr 项）而非仅 kernel 符号 | [ssm-scan.md](ssm-scan.md)（branch=delta-rule/gla） |
| 3 | **Convolution** | ① 滑窗局部加权和（im2col+GEMM 形态）；② kernel/host 含 `kH`/`kW`/`stride`/`pad`/`dilation` 卷积参数或 `im2col`/`LoadData3D` 证据；③ 变体：depthwise（`groups==Cin`）、ConvTranspose（col2im/分数步长形态） | im2col 之后就是 GEMM——**先于 matmul 判**，滑窗参数是 conv 指纹 | [convolution.md](convolution.md) |
| 4 | **Matmul 系**（cv_fusion / moe） | ① 主干为（批量）矩阵乘 `A[M,K]@B[K,N]`，或 matmul + 向量后处理融合；② kernel 用 `MatmulImpl`/`Mmad`/L0A/L0B/L0C 或 host 侧 `MatmulApiTiling`；③ MoE 变体加判 expert 分组/门控（GroupedMatmul、SwiGLU）；④ 正交 trait：transA/transB（布局）、batch、grouped、量化（见 matmul.md M0/M4/M9/M10） | 纯向量/标量模拟的 matmul 也算（正是 P102 的猎物）；**无 QK^T 的递归/scan 归 SSM/Scan 系** | [matmul.md](matmul.md) |
| 5 | **Quantization** | 存在 scale/zero_point、quant/dequant 算子对、int8/fp8 张量 | **量化通常不是独立族而是 trait**：命中后回主干族 workflow 的量化支路；仅当算子本体就是 quant/dequant 时才进本族 | [quantization.md](quantization.md) |
| 6 | **Reduction 系**（normalization/softmax） | ① 沿轴归约（sum/mean/max/min）；② normalization = 减均值除方差 + scale/shift；③ kernel 含 `ReduceSum`/`WholeReduceSum`/Welford 状态 | 独立的 softmax 归此；夹在 QK^T 与 PV 之间的归 Attention 系；**纯 MLP 通道注意力（SE/SK：global pool→FC→scale，无 conv）归此**；**conv-based CV-Attn（Triplet/BAM/CBAM，分支含 conv 2→1+BN）归 Attention 系 cv-attn 分支** | [reduction.md](reduction.md) |
| 7 | **Sort** | 比较-交换/归并结构、TopK、argsort 索引跟踪 | — | [misc-families.md](misc-families.md) |
| 8 | **Conversion** | **输出元素集合 = 输入元素集合，值不变只变位置/布局**（transpose/permute/split/concat）——纯搬运 + 地址重排，无算术运算 | vs index_scatter：conversion 地址映射编译期静态，gather/scatter 索引是运行时数据 | [conversion.md](conversion.md) |
| 9 | **小族** | index_scatter（gather/scatter 运行时索引）、pooling（滑窗池化）、broadcast（输入 shape 不同需广播对齐）、optimizer（参数+梯度+状态多路更新） | pooling 的滑窗是池化（max/avg），不是 conv 的加权和 | [misc-families.md](misc-families.md) |
| 10 | **Elementwise** | **排除法兜底**：输入输出 shape 相同、逐元素独立、无跨元素依赖，以上均不命中 | 放最后判——它是"无结构"的族 | [elementwise.md](elementwise.md) |

## 兜底规则

1. **多族特征同时命中**（如量化 GQA、conv+bias+激活融合）→ 拆"主干族 + trait"：主干决定进哪个 workflow，trait（量化/稀疏/融合后处理）走 workflow 内的对应支路。与 `ops/ascendc-tiling-design` 的"基础形态 + 正交 trait"同构。**trait 记录进 workflow_ref.traits 字段**（可空数组，如 `["causal","paged/decode"]`），workflow_guided 节点生成时一并填写。
2. **全部不命中** → 只做 [_common-prerequisites.md](_common-prerequisites.md) 通用前置 + strategy-index.md 瓶颈查表，workflow 缺失不妨碍进化。
3. **workflow 内标注"覆盖缺口"的阶段** → 该场景尚无沉淀卡，走通用前置 + 瓶颈驱动，优化完成后按 X 系列流程沉淀新卡。
4. **判据顺序提示**：`FA 指纹（QK^T+softmax）→ 递归外积（无 QK^T 的 SSM/Scan）→ 独立 softmax（通道注意力/归约）`——三处陷阱按此顺序排除，避免 name-based 误判。
