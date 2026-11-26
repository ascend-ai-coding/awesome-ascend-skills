# 算子融合机会、可行性判定、风险与排除条件

> 本文是 SKILL.md 第 5、6 步的展开：怎么判断"值得融合、能不能融合、该不该做"。
>
> 配套文件：`npu-fused-op-catalog.md`（torch_npu 已有融合算子清单）——评估任何候选前**必读**，先查重再谈自研。

## 目录

1. [融合为什么快：收益的四个来源](#1-融合为什么快收益的四个来源)
2. [大模型推理常见融合机会模式表](#2-大模型推理常见融合机会模式表)
3. [可行性判定维度（自研前逐项过）](#3-可行性判定维度自研前逐项过)
4. [排除条件（出现即不推荐融合）](#4-排除条件出现即不推荐融合)
5. [风险清单](#5-风险清单)
6. [实现成本分级](#6-实现成本分级)
7. [决策流程](#7-决策流程)

---

## 1. 融合为什么快：收益的四个来源

1. **减少中间张量的读写**（最主要）：相邻算子 a→b 的中间张量 t 原本要"写一次 HBM + 读一次 HBM"。融合后 t 留在片上（Unified Buffer / 寄存器），省掉约 2×bytes(t) 的 HBM 流量。访存受限的序列，收益直接按省下的字节数折算。
2. **减少 kernel 下发（launch）次数**：每个 kernel 有固定的下发/启动开销。小 shape（decode 低并发）场景下一次融合省掉 N−1 次 launch，收益占比可能超过访存收益。
3. **消除布局转换**：TransData / Transpose 若能通过融合消除、或让上游算子直接产出下游需要的布局，收益 = 该转换的大部分耗时。
4. **更好的数据复用与流水**：融合后可在同一 kernel 内做 Cube+Vector 流水、提高 UB 数据复用率。

**收益不来自哪里**（防止错误预期）：

- 融合不改变数学计算量——计算受限的大 GEMM 相邻融合，几乎无收益
- "算子数量减少"本身不是收益，收益要落实到省下的访存量 / launch 次数 / 转换耗时

## 2. 大模型推理常见融合机会模式表

先在时间线上找下列**未融合序列模式**（模式只是线索，是否真实存在以时间线与源码为准），再对照第三列查现成算子：

| 模式 | 时间线上的典型未融合序列 | torch_npu 现成融合算子（见 npu-fused-op-catalog.md） | 主要收益来源 |
|---|---|---|---|
| 残差 + 归一化 | Add → (Mul→Rsqrt→Mul→Mul)（RMSNorm 展开）；LayerNorm 展开 | npu_add_rms_norm 系列（含 cast / quant / dynamic quant 变体） | 访存 + launch |
| 激活门控 | Sigmoid→Mul→Mul / Silu→Mul（SwiGLU）；GELU→Mul | npu_swiglu / npu_swiglu_quant / npu_gelu_mul 等 | 访存 |
| 位置编码 RoPE | sin/cos 生成 → Mul/Add → rotate 序列 | npu_apply_rotary_pos_emb | 访存 + launch |
| 量化/反量化 | Cast→MatMul、DynamicQuant→MatMul→Dequant | npu_weight_quant_batchmatmul、npu_quant_matmul_gelu 等 | 访存 + 低精度计算 |
| 布局转换 | 连续 TransData / Transpose / Permute | npu_confusion_transpose（beta）、npu_transpose_batchmatmul | 访存 |
| Attention 展开 | BatchMatMul → Add(mask) → Softmax → BatchMatMul 长序列 | npu_fused_infer_attention_score 系列（全量+增量）、npu_fusion_attention 系列 | 访存（softmax 中间矩阵） |
| MoE 路由 | Softmax→TopK → Gather/Scatter → 多个 MatMul → 加权求和 | npu_moe_gating_top_k_softmax、npu_grouped_matmul 系列、npu_ffn | 访存 + launch |
| MoE 通信 | AlltoAll(v) 与专家计算完全分离 | npu_alltoallv_gmm 系列 | 通信计算流水 |
| TP 通信 | AllGather→MatMul / MatMul→AllReduce | npu_all_gather_base_mm、npu_mm_all_reduce_base 等 | 通信计算流水 |
| KV Cache 写入 | Pad / ScatterUpdate / 切片拷贝序列 | npu_kv_rmsnorm_rope_cache（MLA 场景）等 | 访存 + launch |

**第一原则：先查现成的。** 表中第三列任何算子若能满足语义、shape、dtype、精度与版本要求，方案就应当是"改调用"，而不是自研 AscendC——两者实现成本差一个量级。只有确认现成算子不可用（语义不符 / shape 或 dtype 不支持 / 版本不含 / 性能已知不佳）之后，才进入自研评估，且要在报告里写明"为什么现成的不行"。

## 3. 可行性判定维度（自研前逐项过）

| 维度 | 要回答的问题 | 常见否决点 |
|---|---|---|
| 数学语义 | 融合后能否保持等价（或误差在验收标准内）？计算顺序能否重排？ | 非线性与量化点顺序不可交换；累加顺序改变导致精度不达标 |
| 数据依赖 | 序列是否真是一条链？中间张量是否被分支消费？ | 中间张量被其他消费者使用 → 要么把分支一起融进来，要么那次写出省不掉，收益缩水 |
| Shape / 对齐 | 尾轴是否满足对齐要求（向量算子常见 32B 对齐）？shape 是否静态？ | 尾轴不对齐需 padding；动态 shape 范围大导致 tiling 难以覆盖 |
| 数据类型 | 各算子输入输出 dtype？归约是否需要升 FP32 累加？ | 混合 dtype 转换点在融合后仍存在，收益缩水；累加精度不达标 |
| 布局 | 输入是 ND 还是 NZ（5HD 类）？上下游期望什么布局？ | 融合算子被迫在入口/出口做 TransData，收益被吃掉 |
| 精度要求 | 归一化/softmax 的累加精度？量化插拔点位置？ | 用户精度验收标准（atol/rtol 或端到端评测分）未确认前，不能承诺精度 |
| 硬件限制 | 中间张量放得下 UB 吗（UB 为数百 KB 量级，以芯片手册为准）？Cube 与 Vector 能否在一个 kernel 内流水？ | 中间张量超大需分块，分块后复用率下降；跨引擎融合工程复杂度高 |
| 版本兼容 | 目标 CANN / 芯片 / torch_npu 支持所需特性吗？现成算子在该版本可用吗？ | 需查证（见 `ascend-operator-reference.md` §4），查不到就标 ⚠ |

## 4. 排除条件（出现即不推荐融合）

- **已有等价融合实现**：查 catalog 与 CANN 算子库后确认存在 → 改用现成算子，不重复造轮子
- **算子含随机性或强副作用**：dropout（RNG 状态与随机种子对齐问题）、与下游读取顺序强绑定的原地更新——融合会改变语义
- **强 host 交互**：序列中间需要 host 决策（动态分支、同步回读）
- **中间张量过大**：分块后 UB 放不下、片上复用率骤降，估算收益趋近于零
- **序列已与其他流重叠**：候选算子的时间段本来就被通信/其他计算覆盖，融合节省不在关键路径上（见 `performance-estimation.md` §5）
- **占比过低**：序列总耗时占比 <1%~2% 且绝对耗时小（阈值结合绝对收益与工程成本定）→ 不划算
- **相邻两个大 GEMM**：都计算受限，只省一次中间写出，占比极小——除非是量化 GEMM 融合等有额外收益来源的特殊情形

## 5. 风险清单

| 风险类别 | 典型内容 | 报告中的呈现要求（对应报告第 8 节） |
|---|---|---|
| 精度 | 累加顺序改变、量化点移动、BF16/FP16 舍入差异 | 给出与未融合实现的精度对比验证方案；量化场景明确量化点位置变化 |
| 兼容性 | CANN / 驱动 / 芯片型号 / 框架版本支持矩阵；beta 接口可能变动 | 列出依赖的版本组合，逐项标注"已核实 / ⚠ 需核实" |
| 工程 | 动态 shape 覆盖、维护成本、回归测试范围、上线流程 | 成本分级（§6）+ 回归清单 |
| 性能 | 小 shape 下无收益甚至劣化；与图模式、量化等其他优化冲突 | 预测中写明适用条件与失效条件 |

## 6. 实现成本分级

- **低**：改调用现成 torch_npu 融合接口 / 打开框架已有开关（天级）
- **中**：AscendC 自研纯 Vector 型融合算子（访存型序列）、shape 较静态（周级）
- **高**：Cube+Vector 混合、通信+计算融合、动态 shape、需同步修改框架调度逻辑（周~月级）

> 时间为量级感受，仅供候选排序参考，随团队 AscendC 经验浮动。

## 7. 决策流程

```
发现未融合序列（profiling-analysis.md §6 的模式识别）
  │
  ▼
查 npu-fused-op-catalog.md 是否有现成融合算子
  ├─ 有 → 核对语义 / shape / dtype / 精度 / 版本兼容
  │        └─ 匹配 → 建议"改调用"（低成本方案，同时给自研作为备选）
  │        └─ 不匹配 → 写明不匹配点，继续向下
  ▼
逐项过 §3 可行性八维
  ├─ 命中 §4 任一排除条件 → 不推荐融合，写明理由（这不是失败结论，是有价值的结论）
  ▼
用 performance-estimation.md 估算收益（含不确定性）
  ├─ 收益明确 > 成本 → 进入候选清单，参与排序
  ├─ 收益不明确 / 数据不足 → 列待补充数据清单，给条件化估计，不下结论
  └─ 当前实现已较优 → 明确建议保留现有实现
```
