# 融合算子兼容性与量化收益判断

本卡覆盖两类辅助决策：(A) 量化与融合算子交互时的兼容性判断，(B) 量化收益评估口径与主线参考路径，(C) 参考文件职责。

> 回退规则统一在 `SKILL.md` 第六步 6.5，本卡不复述。

---

## A. 融合算子兼容性

### A.1 兼容性分级

| 等级 | 含义 | 默认动作 |
| --- | --- | --- |
| A | 量化主线算子 | 保留融合算子，并接入量化输入和 scale |
| B | 可与量化共存，但通常不承载主量化语义 | 输入 dtype/layout 契约仍满足时保留 |
| C | 需要量化友好替身或 post-load 结构融合 | 先改造周边链路，再判断性能 |
| D | 大融合子图或模型专属融合契约 | 先检查契约，不预设全量量化兼容 |

### A.2 算子映射

| 算子或融合家族 | 等级 | 量化判断 |
| --- | --- | --- |
| `npu_grouped_matmul` | A | MoE 量化专家计算主线；可通过 scale/per-token-scale 等输入支持 int8/int4 权重路径 |
| `npu_moe_init_routing_v2` | A | 可保留在量化 MoE 路线中；路由初始化通常不是主要阻塞点 |
| `npu_moe_distribute_dispatch_v2` | A | EP 通信可支持动态量化通信模式 |
| `npu_moe_distribute_combine_v2` | A | 需要与 dispatch 输出和通信量化模式约束配套 |
| `npu_mla_prolog_v3` | A | MLA/KVCache/query 量化关键契约点；检查 `weight_quant_mode`、`kv_cache_quant_mode`、`query_quant_mode`、cache mode |
| `npu_fused_infer_attention_score(_v2)` | B | 通常可保留；key/value/cache 可为 int8/int4，compute 是否 BF16 取决于路线 |
| `npu_kv_rmsnorm_rope_cache(_v2)` | B | MLA-like cache 路径通常可复用，但依赖 cache layout 和 scale 来源 |
| `npu_rms_norm` | B | 通常保留，处理浮点激活 |
| `npu_add_rms_norm` | B | 通常保留，处理浮点 residual/activation 链路 |
| `npu_moe_gating_top_k` | B | router 通常是浮点或元数据链路，不是量化收益中心 |
| `npu_moe_gating_top_k_softmax` | B | 同 gating top-k |
| `npu_moe_finalize_routing` | B | 通常在量化专家计算后保留 |
| `npu_swiglu` | C | 浮点 activation 算子；W8A8 常需要 `npu_dequant_swiglu_quant`，W4A8 常需要 `npu_swiglu_clip_quant`，或先做 post-load gate/up 融合 |
| `IndexerProlog` / `LightningIndexer` / `Sparse Flash Attention` / 模型专属 mega-kernel | D | 量化必须服从融合子图契约，尤其是输出 dtype、cache layout、必需 side tensor |

### A.3 路线级判断

#### MLA + MoE

大多数 MoE 和 MLA 融合算子在量化下兼容性较高。主要变化通常发生在 expert activation 或权重 layout 处理，不在 attention core。

检查：`npu_mla_prolog_v3` 输入/cache/scale 契约；`npu_grouped_matmul` 权重位宽和 scale layout；TP/EP 下 `smooth_scale` 是否完整；`npu_swiglu` 是否必须替换为 `npu_dequant_swiglu_quant` 或其它量化友好 activation 链。

#### Indexer / 长序列 + MoE

融合算子通常定义量化边界。不要假设 indexer 周围所有 Linear 都能量化。

检查：prolog/indexer 输出是否必须保持 FP16/BF16；KV cache 和 LI cache 是否有显式 C8 契约；sparse attention 是否是 int8 存储、BF16 计算；`MoEGMM` 是否使用与普通 `Linear` 不同的位宽。

#### Attention BF16、MLP/MoE 量化

当 attention 侧融合较脆弱时，这是有效的保守路线。保留 attention 融合和 BF16，只单独评估 MLP/MoE 收益。

检查：`ignore` 必须显式保留 attention 模块；量化收益只能按 MLP/MoE 口径评估；如果只改变 cache 存储或相邻 Linear，不要宣称 attention 已量化。

#### Dense Decoder

主要融合风险不是复杂 NPU attention kernel，而是原本融合的大 matmul 被拆碎。

检查：导出的 `gate/up/down` 是否拆散原 fused gate-up 路径；导出的 `q/k/v` 是否拆散原 fused QKV 路径；post-load 融合是否能恢复更大的量化 matmul；小模型 decode 下，动态量化开销是否吞掉收益。

### A.4 初评估分析规则

- 对 MoE，先乐观判断 `npu_grouped_matmul + routing + dispatch/combine` 可兼容量化，再验证 scale/layout 细节。
- 对 MLA/Indexer，从 fused prolog、cache、sparse attention 契约反推量化范围。
- 对 Dense SwiGLU，将 `npu_swiglu` 标记为需要 post-load 融合或量化友好替身。
- 对 FA，区分"FA core 可保留"和"QKV/cache/prolog 契约仍满足"。
- 对任何融合算子冲突，记录是否可以回退非融合路径并继续产出量化基线。

---

## B. 量化收益判断

### B.1 五维口径

量化 Skill 只沉淀量化收益的判断方法和已确认结论，不沉淀 AFD、多流、权重预取、SuperKernel、graph cache、部署形态调整等非量化优化手法。

量化收益必须拆成五个口径，不要只用一个 benchmark 数字下结论：

| 口径 | 主要证据 | 判定方式 |
| --- | --- | --- |
| 理论预期 | 量化对象、权重/Cache 体积、计算访存比、batch/seq/并行形态 | 判断量化方案本身是否具备权重带宽、KVCache 带宽、MoE/MLP 计算或访存收益空间 |
| 端到端收益 | 关闭 profiler 的 benchmark JSON 或日志 | 记录 Prefill、Decode、吞吐；分别对比量化前最新基线、原始基线和同条件 BF16 补跑，输出阶段收益、累计收益和同条件收益 |
| 显存收益 | benchmark 元数据、NPU 显存日志、部署卡数 | 记录峰值显存、权重显存和部署卡数变化；显存收益可以独立成立，不等价于低时延收益 |
| 算子级收益 | `op_statistic.csv`、`operator_details.csv` | 观察 `GroupedMatmul`、`QuantBatchMatmul`、`DequantSwigluQuant`、`DynamicQuant`、Attention/Indexer kernel 的增减，只用于证明局部收益或定位损耗来源 |
| 非量化瓶颈解释 | `communication.json`、trace、rank tail、编译缓存状态 | 只用于解释端到端收益为何与理论或算子收益不一致；不在本 Skill 沉淀非量化优化方案 |

### B.1.1 收益拆解模型（端到端 decode 收益的可解释口径）

量化收益是模型属性 × 部署方案的复合产物，**不能脱离 exec_mode / 并行形态 / batch / context 给单一数字**。decode 时间可拆成四段：

```
T_decode ≈ T_attn + T_FFN_GEMM + T_dispatch + T_overhead
```

量化对每段的影响方向不同：

| 段 | 量化作用 | 占 T_decode 比（典型） |
| --- | --- | --- |
| `T_attn`（MLA / softmax / KV 读写） | 通常不量化 -> **0** | 30-50%（context 长占比上升） |
| `T_FFN_GEMM`（dense MLP + MoE expert int8 GEMM） | int8 GEMM ~0.6x BF16 -> **节省 ~40% 这段** | 30-45%（batch 大占比上升） |
| `T_dispatch`（alltoall + 路由 + finalize） | 是否生效取决于 dispatch_v2 是否走 W8A8 路径 -> **0 或减半 alltoall I/O** | 5-15%（EP 度高占比上升） |
| `T_overhead`（dynamic_quant 等额外 host launch） | **增量**；ge_graph 编进图约等于 0，eager 单 step 数十 ms | 0-60%（exec_mode 决定） |

**预估 decode 收益**（符号约定：Δ < 0 = 加速/节省，Δ > 0 = 变慢；右侧 `Δ_dispatch_saving`、`Δ_overhead_cost` 均为非负幅度）：

```
Δ_decode ≈ -0.4 × (T_FFN_GEMM / T_decode) - Δ_dispatch_saving + Δ_overhead_cost
```

其中 `Δ_dispatch_saving ≥ 0`（dispatch 走 int8 省下的 alltoall I/O 占 decode 比），`Δ_overhead_cost ≥ 0`（dynamic_quant 等额外 host launch 占 decode 比；eager 下可能主导，使 Δ_decode 转正、decode 变慢）。

实测验证（LongCat-Flash-Lite, A3, ge_graph）：

| 配置 | T_FFN/T_decode 估值 | 预测 Δ | 实测 Δ |
| --- | --- | --- | --- |
| EP=8 b=2 1K | ~30% | -12% | -10.8% |
| EP=8 b=8 4K/1K | ~38% | -15% | -13.3% |
| EP=16 b=2 1K | ~30% + dispatch 减半 | -15% | -13.6% |
| eager + EP=8 b=2 1K | ~30% 但 host overhead 大 | +20% | **+23%（反向）** |

**评估方法**：在写收益结论前用 BF16 profiler 拿到 T_FFN 占比，能 ballpark 量化收益上限。实测与预测偏差 >5pp 时回头查 dispatch 路径或 host overhead，**不要直接归因为“量化效果不行”**。

**强诊断信号**：同一 commit / 同卡 / 同 prompt 下 W8A8 vs BF16 在不同 exec_mode 测出不同方向的 Δ_decode，说明 host launch overhead 是主因，量化的纯计算收益约等于 ge_graph 测得的那个数。eager 数据不进收益结论，仅用于功能验证。

### B.1.2 显存收益分账（量化只动权重，不动激活和中间 buffer）

显存收益要拆成五栏，每栏量化作用不同：

| 栏 | 量化作用 | 典型量级（LongCat-Flash-Lite EP=8） |
| --- | --- | --- |
| 模型权重 | **覆盖 `targets` 内的层** -> bit_ratio 减少；scale 元数据增加 | BF16 19.74 / W8A8 14.34 GiB（**-27%**） |
| KV cache | 不量化时为 0；C8 KV 时减半 | b=8 4K/1K 约 10 GiB（W8A8 deploy 不动） |
| Activation / Hidden states | **完全不变**（GEMM 输入仍 BF16，只在量化点临时转 int8） | 约 5 GiB |
| MoE / GMM 中间 buffer | int8 GEMM 输出 int32 -> dequant 后 BF16；**尺寸由 token 数决定，量化不影响** | b=8 4K/1K 单次约 9 GiB（GMM workspace） |
| 通信预分配 | 不变 | HCCL 约 3-5 GiB |

**理论权重公式**（拆成「量化后体积 W_quant」与「相对 BF16 的节省 Δ_weight」两个量，避免混用）：

```
W_quant  = model_size × (coverage × bit_ratio + (1 - coverage)) + scale_metadata   # 量化后权重体积
Δ_weight = W_quant - model_size                                                    # 相对 BF16 的节省（负=省）
```

LongCat-Flash-Lite W8A8 deploy 实测对账：

- coverage（量化覆盖率）约 80%（dense MLP + MoE expert，剩余约 20% 是 MLA + embed + lm_head + ngram 全 BF16）
- bit_ratio = 0.5（int8 / bf16）
- scale 元数据约 0.7-1 GiB（per-channel weight_scale + per-expert smooth_scale）
- 量化后体积 `W_quant ≈ 19.74 × (0.8 × 0.5 + 0.2 × 1.0) + 0.8 = 12.6 GiB`，对应节省 `Δ_weight ≈ 12.6 - 19.74 = -7.1 GiB`
- 实测 14.34 GiB 比理论 `W_quant` 多 1.7 GiB（MLA prolog 的 NZ 权重副本 + allocator 对齐）；实测净省约 -5.4 GiB（19.74 -> 14.34，即 -27%），与理论方向一致、量级略小

**关键分账规则**：写显存收益时强制说“权重 -X GiB / 卡，但 b=N seq=M 配置仍可能 OOM 在 [激活 / GMM 中间 buffer / KV cache]”，不要简单写“显存 -27%”就结束。

**OOM 不一定是量化能力问题**：当瓶颈在 KV cache、MoE alltoall buffer、GMM 中间 tensor、通信预分配时，量化无法绕过，需要的是正交手段（MoE prefill chunk、paged attention、batch 调整、KV cache 量化等）。**评估时遇到 OOM 必须先分账，再判断是不是量化路径问题**。

对 LongCat 这类 EP prefill 路线，还要先确认是否已有 `moe_chunk_max_len` / chunked prefill 这类框架侧缓解手段；如果 chunk 后 OOM 消失或峰值显存明显回落，应优先归因到调度 / 中间 buffer 峰值，而不是误判为量化适配本身失效。

### B.1.3 部署依赖性（roofline 边界条件）

B.1.1 的 4 段加和回答“收益由什么组成”，B.1.2 的显存分账回答“显存收益落到哪些 buffer”；本节回答“每段能加多少由部署决定”。同一 ckpt 在不同部署下实测收益从 +20%（变慢）到 -15%（节省）都观察到过，因此**理论收益估算必须固定部署 4 维**。

#### 量化加速的边界

量化只降低两类硬件资源消耗：

- HBM <-> 计算单元的字节搬运量（int8 vs bf16 -> 减半）
- 计算单元峰值 FLOPs/cycle 需求（int8 MAC 通常比 bf16 高 1.5-4x，硬件相关）

它不降低 host launch、跨卡通信（除非 dispatch 显式走 int8 payload）、同步阻塞这三类开销。所以一个算子从量化拿到的实际加速由它在 roofline 上的位置决定：

| 算子位置 | 量化加速上限 |
| --- | --- |
| Memory-bound | 约等于精度比（int8 vs bf16 约 2x） |
| Compute-bound | 约等于硬件 int8 / bf16 peak TFLOPS 比 |
| Launch / Comm / Sync-bound | 0 或负（量化插入算子是净开销） |

#### 部署 4 维 -> roofline 位置 -> B.1.1 哪段加速

| 部署维度 | 调整方向 | 主要影响 B.1.1 哪段 |
| --- | --- | --- |
| **exec_mode**（eager / graph） | eager 下每个 quant / dequant op 是独立 host launch，量化插入算子从接近 0 变主导；graph 把 forward 合成连贯调度链，摊薄到接近 0 | `T_overhead`，可决定 Δ 整体方向 |
| **batch / 实例 token 数** | 小 batch 时单 op 太小、launch-bound；batch 增大后会从 memory-bound 向 compute-bound 过渡 | `T_FFN_GEMM` 加速幅度 |
| **并行切分**（TP / EP / DP） | EP 大时每专家 token 少，MoE GMM 更容易落 launch-bound；TP / DP 不改变加速比但改变绝对节省 | `T_FFN_GEMM`（尤其 MoE GMM） |
| **KV / activation 量化策略** | decode 阶段 attention 多为 KV bandwidth-bound；不量化 KV 则 attention 段加速约等于 0；attn 投影也不量化时 `T_attn` 整段不变 | `T_attn` |

#### 应用规则

写理论收益分析时强制：

1. **报告固定 4 维并给出收益矩阵**：写“W8A8 省 X%”时，必须同时给出 `(exec_mode, batch, parallelism, KV 策略)`，否则结论不可移植。
2. **B.1.1 拆完每段补 roofline 标签**：memory-bound / compute-bound / launch-bound，从而给出该段加速上限。预测与实测偏差 > 5pp 时，优先核对 roofline 判断，而不是简单归因为“量化效果不行”。
3. **拐点 scenario-specific**：具体阈值（MoE GMM 进入 memory-bound 的 token/expert 拐点、batch 进入 compute-bound 的拐点）依赖硬件 + kernel 实测，**不能从其他平台或模型硬抄**。
4. **profile 优先于 wall-clock 单点**：单点是 4 维 × 算子分布的复合结果；只有 profile 的 `op_statistic` 才能定位“收益从哪里来 / 哪里没拿到”，是排除 scenario-specific 误差的唯一手段。

### B.2 仓内文档证据强度分级

- **可作为明确量化收益证据**：同一文档中给出 BF16 与量化同场景对比，或给出同等规模下不同量化位宽对比。例如 DeepSeek-V3.2-Exp 文档中的 BF16 64 卡 128K TPOT < 30ms、W8A8C8 TPOT < 20ms，以及 MoE W4A8 相比 W8A8 的同规模对比。
- **可作为量化绝对性能证据**：README 只给量化 benchmark，或结果同时包含 MTP、perfect eplb、AFD、服务框架、图缓存等前提时，只能说明量化版本在该组合配置下达到绝对性能目标，不能单独宣称量化 speedup。
- **不作为 Skill 经验沉淀**：尚未正式确认方案、收益未稳定复现、或主要结论依赖非量化优化的案例。此类内容可以留在模型 `progress.md` 或专项报告中，等收益确认后再抽象进参考卡。

### B.3 主线参考路径

| 路线 | 结构 | 典型证据 | 主量化路径 | 复用价值 |
| --- | --- | --- | --- | --- |
| A1 | MLA + MoE + MTP | `deepseek-r1` | 统一 W8A8，可叠加 KVCache C8 | 适合 MLA+MoE 结构，细节见结构卡 C |
| A2 | Indexer / 长序列 + MLA + MoE | `deepseek-v3.2-exp`、`glm-5` | Linear W8A8 + MoEGMM W4A8/W8A8 + KVCache C8 + LI cache | 适合混合量化，细节见结构卡 C 经验 C2 |
| A3 | MoE + attention 大多保持 BF16 | `longcat-flash` | MLP/MoE W8A8，attention BF16 | 适合 attention 融合约束强于 projection 量化收益的保守路线 |
| A4 | Dense Decoder | （本仓暂无 Dense Decoder 集成示例） | 外部 compressed-tensors 产物输入 infer，通常 Linear W8A8 | 结构通用，命中时套用结构卡 A |

不要把 `qwen3-next` 这类外部 patch/service 路线作为 infer 主线 `compressed-tensors` 参考路径。

---

## C. 参考文件职责

| 文件 | 职责 | 不写入 |
| --- | --- | --- |
| `quantization-contract.md` | 产物最小交付物 / 配置字段 / 张量语义 / 运行对象映射 / 本仓 9 步运行机制 / 关键代码入口 | 结构经验、模型 benchmark、回退/验证/决策 |
| `quantization-structure-cards.md` | 三类结构卡（A/B/C）、每卡含已确认模型经验、post-load 例外汇总、新结构补卡规则 | 产物契约总表、收益结论、回退规则 |
| `quantization-fusion-and-benefit.md`（本卡） | 融合算子兼容性 / 收益判断口径 / 主线参考路径 / 文件职责 | 产物契约、模型实施细节、回退规则 |

新增内容优先补到职责最匹配的参考文件。仅在以下情况新增参考内容：

- 现有结构卡无法覆盖当前模型结构，或出现新的 infer runtime object。
- 量化产物引入新的张量语义、scale/cache/layout 规则或 post-load 规则。
- 出现可复用的新融合回退模式。
- 有正式确认并可复现的量化收益结论。
