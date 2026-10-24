# FlashAttention 类算子 — 基础理论层

> 本文档定义 FA 类的**通用数学模型、不变量与分析阶段产出**。所有 FA 变体(GQA / MLA / 量化 / 稀疏)共享本节内容。
> **选型入口** → [`patterns.md`](../patterns.md);**设计流程** → [`design/`](../design/_governance.md)(节点 G→N12)。

---

## §0 抽象名与公开 API 对照

> 本目录方法论用**抽象名**表达设计概念(如 `mEff`、`K_BASE`),便于跨平台复用。下表把这些抽象名映射到 **asc-devkit 公开 API**(有文档/头文件),供实现时定位。
> **纪律**:本 skill 只引用公开资料(asc-devkit 开源社区、CANN 官方文档)。抽象名承载"该决策什么 / 该满足什么不变量";**其具体变量命名由实现自行决定**,方法论不绑定任何一份实现的私有符号名。

**抽象概念 → 公开 API**:

| 方法论抽象名 | 含义 | 对应的公开 Ascend C API / 概念 |
|---|---|---|
| `mEff` = Bq × curG | Mmad 有效 m 维 | Mmad 的 `m` 维参数(`MmadParams.m`);数值由 Tiling 决定,变量名自定 |
| `curG` / `gMaxPerTask` | 合并到一个 task 的 qHead 数 / 其上限 | Tiling 派生量,无固定 API 名;由 Host Tiling 决定 |
| `K_BASE` / `M_BASE` / `S2_BASE` | L0/L1 分块基础块 | Mmad `k`/`m`/`n` 维分块步长(`MmadParams`)+ LoadData 分块参数;数值 Tiling 决定,变量名自定 |
| `CACHE_SIZE` / `PRELOAD_N` | 流水 ring buffer 深度 / task 环形缓冲区大小 | 算子自定义常量(非 API);由流水编排设计决定,见 `design/execution.md §3`(N10 流水/同步)|
| V1 softmax | online softmax V1 阶段 | **`SoftmaxFlashV2<float>`**(asc-devkit `include/adv_api/activation/softmaxflashv2.h`,有文档与示例);`T` 仅支持 half/float,bf16 走 `<float>`+Cast |
| bf16 Mmad | fm/filter=bf16, acc=fp32 | `Mmad<float, bfloat16_t, bfloat16_t>`(公开 Mmad 模板;见 `Mmad.md`)|
| V 转置加载 | L1→L0B 转置 | `LoadDataWithTranspose`(公开 `LoadDataWithTranspose.md`;950PR 仅支持 L1→L0B 通路)|
| V2 归一化 1/sum | 求倒数 | `Reciprocal<float, config>`(950PR 仅提供带 config 的重载;config 有默认值 `DEFAULT_RECIPROCAL_CONFIG`,无需显式传参;`Reciprocal.md`),或 `Div` 替代 |
| 跨核同步 | AIC↔AIV 握手 | `CrossCoreSetFlag<modeId,pipe>` / `CrossCoreWaitFlag<modeId,pipe>`(公开 ISASI;modeId 语义见 implementation_ref.md §1)|
| L0C→UB 搬运 | Fixpipe | `Fixpipe` + `FixpipeParamsArch3510`(950PR)/ `FixpipeParamsM300`;见 implementation_ref.md §4 |

**跨核同步 modeId(公开文档)**:950PR/950DT 支持 modeId 0/1/2/4;A2/A3 等上一代仅支持 0/1/2。FA 混合 kernel 的 AIC↔AIV 握手在 950PR 用 modeId 2 或 4,详见 [`implementation_ref.md` §1](../implementation_ref.md)。

> **重要**:某些 FA 参考实现内部有私有融合函数(如把 V1 softmax 融合成单函数)或私有的跨核同步封装类。这些**不是** Ascend C 公开 API,走 asc-devkit 公开 API 的开发路径**无法直接调用**。本 skill 不推荐、不依赖任何私有符号——默认路径一律用上表的公开 API。

---

## §1 FA 类算子定位

### §1.1 算子定义

FA(FlashAttention)类算子是分块化、不显式实例化 attention score 矩阵的 attention 计算族。三个输入 Q / K / V,一个输出 O,沿 KV 维度分块流式累积 softmax 与加权和,**不在显存中构建完整的 [Sq, Sk] score 矩阵**。

输入语义(以维度名表示,具体 layout 由设计阶段声明):
- **Q**:query,含 batch / Sq / qHead / D 维(典型 BSND layout `[B, Sq, Hq, D]`)
- **K, V**:key / value,含 batch / Sk / kvHead / D 维(典型 BSND layout `[B, Sk, Hkv, D]`)
- **O**:输出 attention,维度与 Q 同

D 为 head dim,Hq / Hkv 为 query / kv head 数。FA 类内不同变体在 Hq 与 Hkv 的关系、KV 是否经压缩、Sq 是否退化为 1 等维度上特化。其他 layout(BNSD / SBHD 等)需要在设计阶段额外声明 stride 计算。

### §1.2 与其他算子族的边界

FA 类与其他算子族的关键区分:

| 区分维度 | FA 类 | 其他族 |
|---|---|---|
| Kernel 类型 | `__mix__(N, M)` AIC+AIV 协同 | 多数为纯 vector / 纯 cube |
| 计算流 | 跨核 stage 化(Cube 算 GEMM / Vector 算 softmax)| 单核内完成 |
| 跨 KV 分块状态 | 必须在线 softmax 累积(max/sum/O_acc)| 多数无在线状态 |
| 任务维度 | s2(KV 分块)**禁入**任务分发维度 | 多数所有维度均可任务分发 |
| Workspace | 多段跨核握手 + 自读自写 GM 段 | 多数无 / 单段 |

因此 FA 类不能套用 reduction / elewise / matmul 的设计模式,必须按本目录的方法论。

---

## §2 数学模型(FA-2 Online Softmax)

### §2.1 计算公式

对每个 KV 分块 `j = 0, 1, ..., (Sk/Bk)-1` 顺序累积:

```
S_j   = (Q · K_j^T) * scale                                  # [Bq, Bk]
m_j   = max(m_{j-1}, rowmax(S_j))                            # [Bq]   (m_{-1} = -∞)
sum_j = exp(m_{j-1} - m_j) * sum_{j-1} + rowsum(exp(S_j - m_j))  # [Bq]
O_j   = exp(m_{j-1} - m_j) * O_{j-1} + exp(S_j - m_j) · V_j  # [Bq, D]

最终: O = O_last / sum_last
```

**关键性质**:
- `S_j`(Q·K^T 结果)仅 `[Bq, Bk]` 大小,与 Sk 无关 —— FA 的"不实例化全矩阵"由此实现
- 跨 KV 分块累积时需要 **rescale 旧状态**(`exp(m_{j-1} - m_j)`),这是 online softmax 的本质
- 末块归一化(除以 `sum_last`)只做一次;**中间步骤的 P 不做归一化**——V1 阶段输出的是 `exp(S_j - m_j)`(未归一化指数),归一化推迟到 V2 末块(见 §2.2 与 `design/execution.md §1` N4 softmax)

### §2.2 数据流四阶段语义

FA 类把每个 KV 分块的计算切成 4 个 stage,跨 AIC / AIV 协同执行:

| 阶段 | 核 | 操作 | 输入 → 输出中间矩阵 | 跨核同步语义 |
|---|---|---|---|---|
| **C1** | AIC | Q·K^T GEMM(Nd2Nz Q/K → L1 → L0 → Mmad → Fixpipe)| → `[Bq, Bk]` 中间矩阵到 GM | 完成后通知 AIV |
| **V1** | AIV | scale + softmax + cast | `[Bq, Bk]` → 未归一化的 P(`exp(S-m)`)到 GM | 等 C1 完成 / 完成后通知 AIC |
| **C2** | AIC | P·V GEMM(Nd2Nz P/V → L1 → L0 → Mmad → Fixpipe)| P + V → `[Bq, D]` 中间矩阵到 GM | 等 V1 完成 / 完成后通知 AIV |
| **V2** | AIV | 跨 KV 分块累积(默认 task 级模式下做 online rescale + Add;末块归一化)| `[Bq, D]` → 写回 attentionOut(末块)/ GM 自读自写段(非末块,在线累积)| 等 C2 完成 |

C1 → V1 → C2 → V2 是同一 KV 分块内的严格数据流方向(见 §3.1 不变量 I1)。不同 KV 分块之间如何交错(流水排布、loop body 内 stage 顺序、几级流水)由设计阶段决定,不在本不变量约束内。

> **注**:V2 阶段的"跨 KV 分块累积"语义因并行策略而异:
> - **默认 task 级**(s2 在 task 内顺序):V2 跨 s2 块在线累积 max / sum / O_acc(stateful)
> - **Split-KV reduce**(s2 跨核切分):V2 只算自己切片的 partial 值,最终的 max / sum / O 在跨核 combine 阶段产生
> 详见 [`design/shape.md §2`(N2 并行策略选择)](../design/shape.md)。

### §2.3 跨 KV 分块状态

数学上跨 KV 分块需要持有三类量:

- **max**:`[Bq]` 维,每行当前最大 score
- **sum**:`[Bq]` 维,每行归一化后的 sum
- **O_acc**:`[Bq, D]` 维,累积的(未归一化)输出

**两种并行策略下,这三类量的物理实现与生命周期不同**:

| 并行策略 | max / sum 持有方式 | O_acc 持有方式 |
|---|---|---|
| **默认 task 级**(详见 `design/shape.md §2` N2)| task 内跨 s2 在线更新(stateful);通常 UB 常驻(尺寸 `[Bq]` 小)| 设计核心决策点(见 `design/shape.md §1.2` N1 UB 模式决策):大 D 场景 ∝ m × D 常驻 UB 无法承受,必须改存 GM workspace 每 chunk 流式 rescale/accumulate |
| **Split-KV reduce** | 每核计算 partial(无 stateful 累积);最终 max / sum 在 cross-core combine 阶段产出 | 每核计算 partial O,落 partial reduce workspace 段;最终 O 在 combine 阶段加权归并 |

---

## §3 不变量 I1-I5

任何 FA 类实现都必须满足以下五个不变量。**违反必崩**(silent zero / 精度漂移 / 多核死锁)。

### §3.1 I1 — 数据流方向不变量

同一 KV 分块内,stage 依赖方向严格为 **C1 → V1 → C2 → V2**。

禁止:
- 把 stage 重排成 C1 → C2 → V1 → V2(让 AIC / AIV 角色分段)
- 在 stage 之间用错跨核同步方向(如 V1 完成只设置 AIC 等待但不设置 V1→C2 通知)

流水的 loop body 执行顺序(无论是 2 级还是 3 级)与本不变量不冲突 —— 它讲的是不同 KV 分块如何交错,同一分块内的依赖方向仍是 C1→V1→C2→V2。

**违反后果**:Mm2 Fixpipe ADDR_MISALIGN / 跨核数据未就绪 → 输出全零或乱值。

### §3.2 I2 — GQA 同 kvHead 同任务不变量

GQA 形态中,同一 kvHead 下的 G 个 qHead **必须在同一个任务内合并处理**(`mEff = curBq × curG`,合并到 Mmad m 维度)。

禁止:
- 把 G 个 qHead 拆到 G 个独立任务,每个任务各自搬运同一份 KV

**违反后果**:KV 重复搬运 G 次,Mte2 / L1 利用率塌陷。

### §3.3 I3 — 跨 task 状态隔离不变量

跨 task 持有的状态(默认 task 级模式下的 online softmax max / sum / O_acc / exp;split-KV reduce 模式下的 partial 段)**必须按任务分槽隔离**,不同任务的状态不能共享同一物理位置。

禁止:
- 用单一 buffer 跨任务共用 stateful 状态(默认 task 级模式)
- 用单一段跨核共用 partial 输出(split-KV reduce 模式)
- 在任务级 PRELOAD 下,用同一 task slot 公式覆盖跨 stage handshake buffer(混淆 task-level 与 loop-level slot 语义,见 [`design/execution.md §3.3`](../design/execution.md))

**违反后果**:被上一任务的尾状态污染 → 输出乱值。

### §3.4 I4 — s2 状态累积正确性不变量

跨 KV 分块的 online softmax 状态(`max` / `sum` / `O_acc`)必须被**完整、正确地累积**,这是 FA 数学正确性的硬要求。

满足此不变量的两种合法切分方案:

1. **默认 task 级切分(主流场景)**:s2 loop 在 task 内部顺序执行,`s2` 索引**不进入** taskIdx 任务分发维度。
2. **Split-KV reduce(适用 Sq=1 + 大 Sk + 核数富余场景,详见 [`design/shape.md §2`](../design/shape.md) N2)**:s2 沿 KV 维拆给多核并行,每核计算自己切片的局部状态,完成后通过专门的 cross-core combine 阶段归并出最终结果。

**禁止**:
- 在默认 task 级模式下把 `s2` 放入 taskIdx
- 启用 split-KV reduce 但**不补**partial reduce + cross-core combine
- 在同一算子内混用两种切分方案

**违反后果**:输出乱值 / 多核死锁 / 精度崩塌

### §3.5 I5 — MX 类格式 scale 轴对齐不变量

MX 类(mxfp8 / mxfp4 / mxfp6 等)块量化格式的 scale **必须沿被消费 matmul 的 reduction (K-mmad) 轴量化**。

**根源**:硬件 Tensor Core 对 MX 类格式的硬约束—— "data must be consecutive over the reduction dimension"。

**attention 中的推论**:

| 张量 | 被消费 matmul | K-mmad 轴 | scale 量化轴 |
|------|------------|---------|-----------|
| Q | Q·K^T | D | D |
| K | Q·K^T | D | D |
| P(kernel 内构造)| P·V | S_k | S_k |
| V | P·V | S_k | **S_k**(不是 BSND innermost 的 D)|

**值得警惕的反直觉**:V 在 BSND `[B, Sk, Hkv, D]` 布局下 D 是 innermost 连续维度。按"沿连续维度量化"的直觉会得到"V scale 沿 D",但这与 PV Mmad 的 K-mmad = S_k 不兼容。**业界标准把 V scale 沿 S_k 量化**。

**违反后果**:Mmad ScaleB 轴错位 → 输出 NaN / inf(Mmad 直接产出非数,不是数值漂移)。

> 量化 trait 的完整契约见 [`quantization_design.md`](../subfamilies/quantization_design.md)。

---

## §4 分析阶段必须产出清单

设计阶段开始之前,必须先回答以下问题。产出落到 DESIGN 文档(或同等阶段产物)。

### §4.1 选型结论

回答:**这个 attention 算子的选型结论是什么(基础 FA 形态 + 正交 trait)?**

判定依据:
- Sq、Hq、Hkv、是否有 KV 压缩、是否纯 decode、是否稀疏
- 选型决策表见 [`patterns.md` §2](../patterns.md)

**产出**:选型结论(基础 FA 形态: GQA / MHA / MLA;正交 trait: 量化 / 稀疏)+ 选定理由。

### §4.2 输入 shape 边界与合法性约束

回答:**用户输入 shape `(B, Sq, Sk, Hq, Hkv, D)` 的合法范围与边界如何处理?**

必须区分:
- **合法性约束**(必须校验并拒绝):如 `Hkv ≠ 0`、`Hq % Hkv == 0`、`D` 落在支持档位
- **数值上限**(**不应硬性拒绝**):如 B、Sq、Sk、Hq 的实际上限应由内部基本块设计承担

通用约束(适用于所有 Ascend C 算子)见 `development-guide §1`。本节只标注 **FA 类特有**的合法性条件:
- `Hq % Hkv == 0`(GQA group 整除)
- `D` 必须落在 Tiling 支持的对齐档位
- causal / sliding window 等可选模式的合法组合

**产出**:合法性校验清单 + 内部基本块如何承担数值上限的设计思路。

### §4.3 dtype / causal / 可选特性枚举

回答:**需要支持哪些可选特性的组合?**

常见特性维度:
- **dtype**:fp16 / bf16 / fp32 / 量化(int8/FP8)
- **causal mask**:on / off / 子模式(下三角 / sliding window / sink)
- **sparse pattern**:on / off
- **alibi / RoPE 等位置编码**:是否在算子内集成
- **feature flags**:PSE / PostQuant / Prefix / ChunkedPrefill 等(详见 [`feature_flags.md`](../specialization/feature_flags.md))

**产出**:特性维度笛卡尔积清单 + 必须支持的组合 / 可延后的组合。

此清单决定 `design/execution.md §2`(N8 编译期特化)编译宏分类(哪些必须编译宏分离独立 target)与精度门禁规划(每个组合需要的 atol)。

### §4.4 性能与精度目标声明

回答:**达成什么样的精度门禁?达成什么样的性能预算?**

- **精度门禁**:各 dtype × 各模式组合的 atol / rtol;来源指向 `/ops-precision-standard`,FA 类暂无独立条目,按 attention / softmax 通用区间取
- **性能预算**:典型场景的 Task Duration 目标(若已知);若有对标实现,声明对标方与差距 acceptance(如不超过 X%)
- **基本块 Roofline 目标**:声明目标性能区制(compute-bound / HBM-bound / cross-core-bound);若对标实现已知,声明基本块尺寸与算术强度的对标值。[`roofline.md`](./roofline.md)(经 `design/shape.md §3` N3 进入)将据此通过三通道 Roofline 分析确定最优 Bq/Bk

**产出**:精度目标表 + 性能目标声明(可为开放型,无具体数字)。
