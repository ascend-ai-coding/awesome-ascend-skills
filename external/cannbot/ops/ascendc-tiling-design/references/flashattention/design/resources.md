# 内存分配(节点 N7 + N9)

> **关注点**:片上/片外内存如何划分 + Host Tiling 如何汇总。
> 节点序列:N7 内存划分(UB/L1/L0/GM)→ N9 Host Tiling 汇总。
> **核心阶段**——FA 类算子设计的难点集中在内存划分。每个层级(UB / L1 / L0 / GM)的问题清单都必须逐项回答。所有 buffer 公式中只允许出现:运行时查询的硬件常量、Tiling 决策出的基本块、dtype 字节宽——**禁止**直接写数字。

---

## §1 内存划分(节点 N7)

> 依赖:N3(基本块)、N6(核数)。正确性约束(容量校验公式)在本节定死;可选优化方案(O_acc 位置 / 分块策略)与之合一。

### §1.1 UB 划分

#### Q1:列出所有 UB buffer

回答:**算子需要哪些 UB buffer?**

每个 buffer 必须明确:

| 字段 | 说明 |
|---|---|
| 名称 / 用途 | 该 buffer 在算子流程中承担什么角色 |
| 单槽容量公式 | 公式中只允许出现:运行时查询的硬件常量、Tiling 决策出的基本块、dtype 字节宽。**禁止**直接写数字 |
| depth | TQue 槽数(决定 buffer 物理大小 = 单槽容量 × depth)|

典型 UB buffer 职责类别(以 streaming UB 模式的 FA 为例):
- **输入接收 queue**:接收跨核握手段读回的 chunk
- **输出 queue**:V1 / V2 写出
- **softmax 状态 buffer**(默认 task 级模式下):跨 s2 块在线累积的 max / sum / exp
- **softmax 临时 buffer**:由 Ascend C SoftmaxFlashV2 类 API 的临时 buffer size 查询函数返回
- **causal mask buffer**:**默认不预分配完整 `[mEff, Bk]` 矩阵**——优先在 S 矩阵上原地生成 mask(逐行按行号公式 + Compare/Select 写回),或仅用单行模板(见 [`base_design.md` §10.2 清单 5](../subfamilies/base_design.md))。仅当 mask 确实无法原地、且必须整块常驻时才分配独立 buffer,并计入 UB 总量

#### Q2:单次 DataCopy 量是否 ≤ 单槽容量?

回答:**所有 `DataCopy(UB, GM, ...)` / `DataCopy(GM, UB, ...)` 调用,单次搬运的字节数是否都不超过对应 TQue 的单槽容量?**

若不满足,**必须实施 chunk loop 切分**,公式:`mSplit = 单槽字节容量 / (cols × sizeof(elem))`。V1 / V2 必须**对称**做 chunk loop,不能只切一边。

**踩坑警告**:`mEff × cols × sizeof(elem) > 单槽容量` 时单次 DataCopy 会溢出 UB,破坏邻接 buffer。

**产出**:每个 DataCopy 调用点的 `单次量 ≤ 单槽` 校验;不满足的位置实施了对称 chunk loop。

#### Q3:UB 总占用 ≤ `GetCoreMemSize(UB)` 校验

回答:**所有 UB buffer 占用之和是否 ≤ 硬件查询所得 UB 容量?**

校验对象:`Σ (单槽容量 × depth) ≤ PlatformAscendC::GetCoreMemSize(CoreMemType::UB, &ubSize)`。

**Q3 校验公式(含 V1 chunking)**:
```
UB_total = v1In(min(mEff, chunkRows_v1) × Bk_pad × sizeof(compute_T))
         + v1Out(min(mEff, chunkRows_v1) × Bk_pad × sizeof(P_T))
         + softmaxStateBuf(slots × mEff × DATABLOCK_BYTES)   // max/sum/expMax 状态
         + softmaxTmp(临时空间大小查询函数返回)
         + v2In(chunkRows × D_pad × sizeof(compute_T))
         + v2Out(chunkRows × D_pad × sizeof(compute_T))
         + v2Rescale + maskBuf(causal, 若未原地)
```

> ⚠️ **softmax 状态 buffer(max/sum/expMax)每行是"一个 datablock",不是"元素数 × sizeof"**。broadcast 模式下每行有效值只有 1 个,但按一个 datablock(`DATABLOCK_BYTES`,当前架构为 32B)对齐存储、datablock 内全部填同一值。因此每行占 `DATABLOCK_BYTES`(与 dtype 无关),**不要**按 dtype 元素数算成 `16 × sizeof` 之类的值(那是把 half 的每-datablock 元素数误用)。确切数值以目标版本 softmax API 文档为准。

**产出**:UB 总占用清单 + 与硬件查询值的比对。

#### Q4:∝ m × D 的 UB buffer 消除判定

回答:**是否存在常驻 UB、尺寸 ∝ m × D 的 buffer?**

详见 §1.2 与 shape §1.2 UB 模式决策。

**产出**:已消除的 ∝ m × D buffer 清单 + 替代方案。

#### 自检清单(进入审查前必跑)

- [ ] **清单 1(streaming UB 容量)**:所有 TQue 单槽容量 × depth 列清;所有 kernel 内 DataCopy 量 ≤ 单槽;超容位置已实施 chunk loop;V1 / V2 chunk loop 对称

### §1.2 UB 优化:O_acc 存放位置 + Chunk 行数(chunkRows 权威公式)

#### O_acc 存放位置

| 可选方案 | 说明 | 优势 | 代价 |
|---------|------|------|------|
| UB 常驻 | `[mPad, D_align]` 的 O_acc 常驻 UB | 读写快 | 大 D 时 UB 溢出 |
| GM workspace + streaming | O_acc 存 GM,按 chunk 读回 | 与 D 解耦 | 每次 chunk 有 GM↔UB 搬运 |

**推荐条件**:`mPad × D_align × sizeof > UB 余量` → **必须** GM streaming。

> **中间段落地位置(片上 L0C→UB vs L0C→GM)是结构决策,默认优选片上 L0C→UB**;其 Fixpipe `isToUB=true` 契约(静默出错陷阱)见 execution §3.5。

#### Chunk 行数自适应(chunkRows 权威公式)

| 决策 | 公式 | 说明 |
|------|------|------|
| Chunk 行数 | `chunkRows = chunkBufferBytes / (D_align × sizeof(compute_T))` | 小 D 时 chunk 行数大、loop 少;大 D 时 chunk 行数小、loop 多 |

> **本公式为权威定义**。shape §1.2(streaming 原则第 3 条)与 §2 Host Tiling §6.3(决策维度)只保留原则 + 前指本处,不重复公式。

> 📊 **数值标定项**:`chunkRows` 由 `chunkBufferBytes` 预算派生,**是自适应数值**——设计期给出公式与默认 `chunkBufferBytes` 预算即可,精确值随 shape 运行时自适应/profiling 微调。但"O_acc 是否 streaming(不常驻 UB)"这一**结构决策**必须设计期定死。

> **结构决策必须落成 TilingData 字段(不能只停在文档)**:"O_acc 是否 streaming" 定死后,**必须**在 TilingData 里体现为一个 kernel 可读的显式信号(角色名 `oAccInUb` / `streamingMode`,命名由实现自定,见 [`base_design.md` §7.3](../subfamilies/base_design.md)),kernel **必须**按该字段选择 O_acc 布局(常驻 UB vs GM streaming),**不得**在 kernel 内自行硬编码另一种布局。
>
> ❌ **反例(GQA 实测踩坑)**:DESIGN.md 定死 streaming、TilingData 里却没有对应字段,kernel 遂自选"O_acc 常驻 UB"——结构决策停在文档、没变成强制信号,是 UB 溢出的直接一环。凡设计期定死的**结构**决策(非数值标定项),都必须有 TilingData 字段承载并被 kernel 读取。

### §1.3 L1 划分

#### Q1:A1 / B1 端口分离

回答:**Q / P 放在哪个 L1 端口?K / V 放在哪个 L1 端口?**

**硬约束**:Cube `Load2D` 仅支持 A1→L0A、B1→L0B 两条通路。因此:
- **Q、P**(均要进 L0A 做 fm)→ **必须 A1**(`TBuf<TPosition::A1>`)
- **K、V**(均要进 L0B 做 filter)→ **必须 B1**(`TBuf<TPosition::B1>`)

**违反后果**:K/V 放 A1 时 Load2D 无法把它们搬到 L0B,P 段输出全零。

**产出**:每个 L1 buffer 的 TBuf 位置声明 + 端口选择理由。

#### Q2:单槽容量 vs 一次 Nd2Nz 全量 / M 轴分块判定

回答:**Q / K / V 的一次 Nd2Nz(`mPad × D`、`Bk × D`)是否会超出 L1 单槽容量?**

判定公式:
```
perSlotElems = L1_PORT_SIZE / rotation_slots / sizeof(elem)
若 一次 Nd2Nz 元素数 > perSlotElems → 必须实施 M 轴(或 N 轴)L1 分块
```

**M 轴分块**:外层 `for (m = 0; m < mPad; m += M_BASE)` 只搬 M_BASE 行到 L1。

**N 轴分块**(P·V 阶段):外层 `for (nOff = 0; nOff < D_align; nOff += N_BASE)` 只搬 N_BASE 列到 L0B。

**分块选项对照**:

| 可选方案 | 说明 | 优势 | 代价 |
|---------|------|------|------|
| 一次性搬入 | `mPad × D` 全量 Nd2Nz 到 L1 | Nd2Nz 次数少 | L1 单槽可能放不下 |
| M-axis 分块 | `for (m = 0; m < mPad; m += M_BASE)` 分批搬 | L1 容量与 D 解耦 | Nd2Nz 次数增加 |
| N-axis 分块(P·V 阶段)| `for (nOff = 0; nOff < D_align; nOff += N_BASE)` | L0B 容量与 D 解耦 | Mmad 次数增加 |

**产出**:Q / K / V 每段的 `单次 Nd2Nz 量 ≤ perSlotElems` 校验 + 不满足时的 M / N 轴分块方案。

#### Q3:rotation 深度依据

回答:**Q/P 与 K/V 各自的 L1 rotation 槽数是多少?依据是什么?**

L1 rotation 深度**因 FA 变体而异**,见 [`extension_points.md` §1.2](../foundation/extension_points.md)。

**禁止**:把 shape §1.4 决策的跨核握手槽数和 L1 rotation 槽数混用同一常量(语义不同)。

**产出**:Q/P 与 K/V 各自的 rotation 槽数 + 变体查表依据。

### §1.4 L0 划分

#### Q1:D 触发 K-axis 分块的判定门槛

回答:**`D > X` 时触发 L0 K-axis 分块,X 是多少?依据是什么?**

X 来自 L0A / L0B 单 fragment 容量 / dtype 字节宽。具体公式:
```
L0A 容量 ≥ m × K_BASE × sizeof(fm_dtype)
L0B 容量 ≥ K_BASE × n × sizeof(filter_dtype)
```

**关键**:K-axis 分块**单一代码路径**就够,不分 D 档。

**分块选项对照**:

| 可选方案 | 说明 | 优势 | 代价 |
|---------|------|------|------|
| 单次 Mmad | `Mmad(.k=D)` | Mmad 次数少 | L0A/L0B 可能放不下 |
| K_BASE 步长累加 | `Mmad(.k=K_BASE)` | L0 与 D 解耦 | Mmad 次数 = D/K_BASE |

**产出**:K-axis 分块判定公式 + K_BASE 取值依据。

#### Q2:L0A / L0B / L0C 容量验算

回答:**配合 K_BASE / M_BASE / N_BASE,L0A / L0B / L0C 容量都满足吗?**

校验公式:
```
L0A: m × K_BASE × sizeof(fm_dtype)  ≤  L0_A 查询值
L0B: K_BASE × n × sizeof(filter_dtype)  ≤  L0_B 查询值
L0C: m × n × sizeof(accumulator_dtype)  ≤  L0_C 查询值
```

**产出**:L0A / L0B / L0C 各自容量验算结果。

### §1.5 GM Workspace 划分

#### Q1:列出所有 workspace 段

回答:**算子需要哪些 GM workspace 段?**

典型段(按职责分类):
- **跨核握手段**(Cube 写 → Vector 读 / Vector 写 → Cube 读)
- **跨 loop 自读自写段**(若需要)
- **任务级状态段**(若需要 stateful 跨 s2 累积)
- **变体扩展段**:见 [`extension_points.md` §1.4](../foundation/extension_points.md)(如 MLA 的 int32 中间段)
- **并行策略扩展段**:若 N2 选 split-KV reduce → 增 partial O / partial logsumexp 段

#### Q2:槽数选择依据(语义分离)

回答:**每段 workspace 的轮转槽数是几?各槽语义是什么?**

**强制要求**:不同语义的槽用不同常量定义、用不同公式取模,**不可共用**(详见 execution §3.3)。

| 段类别 | 对应槽语义(shape §1.4)| 公式形态(计数器 / 常量与 execution §3.3 一致)|
|---|---|---|
| 跨核握手段 | 跨 stage 握手槽 | `handshakeSlot = loop & (DB-1)`(loop=全局 task 迭代器,DB 为 2 的幂)|
| 跨 loop 自读自写段 | 跨 loop 自读自写槽 | `selfRefSlot = loop % (PRELOAD_N+1)`(loop=全局 task 迭代器)|
| 任务级状态段 | 跨 task 状态槽 | `stateSlot = mloop % (PRELOAD_N+1)`(mloop=行计数器,softmax max/sum 为行级状态)|

> ⚠️ 计数器不可混用:跨核握手 / 跨 loop 自读自写用 **loop**(全局 task 迭代器),softmax 行级状态用 **mloop**(行计数器)。详见 shape §1.4 与 execution §3.3。

#### Q3:workspace 总量的外层乘子 = 并发槽位数,不是 totalTasks

回答:**per-slot 段大小算出来后,乘以什么得到 workspace 总量?**

**workspace 总量 = per-slot 段大小之和 × 并发槽位数**。默认 task 级并行下,同一 AI Core 顺序执行分到的多个 task,workspace 在 task 间**天然复用**(task 结束即释放),因此:
```
并发槽位数 = min(totalTasks, usedCoreNum)     // ≈ 核数,不是总任务数
workspaceTotal = Σ(per-slot 段大小) × 并发槽位数
```
kernel 侧按 `aiCoreIdx` 索引 per-core 偏移。

**禁止**:按 `totalTasks` 分配(把 totalTasks 当外层乘子)。跨核握手段的生命周期仅一个流水节拍(一次 s2 内的 C→V 交接),更不该为每个 task 各留一份——按 totalTasks 分配会放大数百倍(典型 6144 vs 32),推理小 batch 场景易 OOM。

**产出**:workspace 外层乘子 = `min(totalTasks, usedCoreNum)` 的确认 + per-core 偏移索引方式。

#### Q4:按运行时 Sk 分配原则

回答:**workspace 是按 `Sk_max` 静态预留,还是按运行时 `Sk` 动态分配?**

**正确答案永远是动态**:Host 暴露 `GetXxxWorkspaceSize(...)` API,按运行时 Sk 计算返回 size。

**禁止**:按 Sk_max 静态预留。

**产出**:`GetXxxWorkspaceSize(...)` API 设计 + 按运行时 Sk 的计算公式。

#### 自检清单(进入审查前必跑)

- [ ] **清单 9(长 Sk workspace 槽轮转)**:所有 workspace 段按运行时 Sk 分配;shape §1.4 列出的所有槽语义类别数量与公式分清楚
- [ ] **清单 9b(workspace 外层乘子)**:workspace 总量按 `min(totalTasks, usedCoreNum)` 分配,**不是** totalTasks;kernel 侧按 `aiCoreIdx` 索引 per-core 偏移

> GQA 特有的 **清单 4-8**(M 轴尾块 padding 行 / causal mask 边界 / bf16 V Cube 直通 / 输出 BSND 写回 / mPad>16+多 s2 块)定义见 [`base_design.md` §10.2](../subfamilies/base_design.md)——这些内容随变体(mEff / Bq-major / BSND 写回)特化,以 base_design.md 为准。设计 GQA/MLA 时须一并校核。

### §1.6 内存层自检清单 O10-O12

- [ ] **清单 O10**:O_acc 存放位置(常驻 UB / GM streaming)按 §1.2 推荐条件定死,且**已落成 TilingData 字段**(`oAccInUb` 类)、kernel 按字段读取——不是仅停在文档、由 kernel 自选
- [ ] **清单 O10b(UB 预算单一事实源)**:host UB 预算算术与 §1.1 Q3 `UB_total` 公式逐项一致(buffer 项 / 单槽尺寸 / depth);host、kernel、Q3 三者**不存在第三套** UB 算术(见 §2.3 绑定约束)
- [ ] **清单 O11**:L1 分块校验通过
- [ ] **清单 O12**:L0 K-axis 分块校验通过

---

## §2 Host Tiling 决策(节点 N9)

> **产出汇总阶段**——把 N1-N8 的所有设计决策汇总成 Tiling 函数的输入 / 输出 / 决策维度。本节不引入新决策。依赖:N1-N8。

### §2.1 输入

Host Tiling 函数的输入:
- 用户输入 shape(`B, Sq, Sk, Hq, Hkv, D` 及变体特有维度)
- 硬件查询结果
- 变体选型 + 可选特性枚举
- 并行策略选择(承接 N2)

**产出**:Host Tiling 函数签名 / 入参列表。

### §2.2 输出字段清单

Host Tiling 函数返回的 TilingData 必须按以下**职责类别**覆盖:

| 字段类别 | 内容 |
|---|---|
| **基本块尺寸** | m 维 task 上限 / m 维基本块 / s2 维基本块 / 流式 chunk 行数 等 |
| **派生尺寸** | GQA group 合并维度数、Sq 维分块数 等 |
| **任务分发参数** | 任务总数 / 实际使用核数 / 每核任务数 |
| **Workspace 段尺寸** | 按 §1.5 列出的各段 |
| **硬件容量回写** | UB / L1_A / L1_B / L0 容量等 |
| **API Host Tiling 数据** | SoftmaxFlashV2 等 Ascend C 复合 API 的 Host Tiling 数据透传 |

**产出**:`TilingData` 结构体字段清单。

### §2.3 决策维度

回答:**给定用户输入与硬件容量,各基本块尺寸如何决定?**

典型决策维度:

- **Roofline 基本块可行域**(承接 N3)
- **prefill / decode 分流**
- **m 维 task 上限**
- **s2 维基本块**
- **流式 chunk 行数**(streaming UB 模式下,真·通用公式):`chunkRows = chunkBufferBytes / (D_align × sizeof(compute_T))`(权威定义见 §1.2)

> **绑定约束(UB 预算单一事实源)**:Host Tiling 在反推 `gMaxPerTask` / `chunkRows` / 校验是否放得下时,所用的 UB 占用算术**必须逐项等同** §1.1 Q3 的 `UB_total` 公式(相同的 buffer 项、相同的单槽尺寸、相同的 depth)。**禁止**在 host 侧另立一套简化/近似模型(例如"每行 2×fp32 + 1×kv 的 per-group reuse 估算")。Q3 公式是 UB 预算的**唯一事实源**,host 反推与 kernel 布局都从它派生;三者出现第三套算术即为契约违反。
>
> ❌ **反例(GQA 实测踩坑)**:host 用 `ubPerG = Bq × (2×maxBd×4 + maxBd×2)` 估算(假设 buffer 复用),而 kernel 实际按 no-sharing(每行 3×fp32 + 2×kv)且 O_acc 常驻 UB。host 预算判定通过(≈160KB),kernel 实际 ≈256KB **溢出 UB**——根因就是 host 预算算术偏离了 Q3 公式,Q3 沦为无人调用的校验式。

**产出**:每个基本块的决策依据 + **host UB 预算算术与 §1.1 Q3 公式逐项一致的核对**。

### §2.4 维度打开纪律

> **门禁耦合项**:本节纪律是 §_governance §2 设计即最优门禁的实现期延伸,不是内存决策。列于此仅因它汇总 N1-N8 的 lever 接入顺序。

回答:**新算子开发时,各特性 / lever 按什么顺序接入?哪些能逐步放开、哪些必须一开始就按最优设计?**

接入顺序服务于"精度回归可定位"(每接入一项跑一次全场景回归,出错能定位),但**接入顺序只约束实现期 build 的先后,不放松设计目标**——设计阶段必须一并承诺全部最优配置(见 [`_governance.md` §2](./_governance.md) 设计即最优门禁)。按对核内计算方案的影响分三类:

| 类别 | 实现期能否逐步放开 | 典型项 | 纪律 |
|---|---|---|---|
| **核外 / 正交层** | ✅ 可逐步放开 | 多核并行(单核→多核)、FlashDecode / split-KV reduce | 叠加在核内方案之上,开 / 关不改核内计算方案;后接入不需重构设计。但设计阶段仍须承诺(核数 / 负载均衡、split-KV 启用 regime 都在设计里定死)|
| **正确性覆盖维度** | ✅ 可逐步放开(为精度回归可定位)| dtype(fp16→bf16)、causal on / off、对齐 / 非对齐、D 档(小→中→用户上限)| 编译宏 / 参数分离;逐维打开只为定位精度问题,非性能分阶段 |
| **核内性能方案(不可退化)** | ❌ 模块级构建,但目标恒为最优 | curG(GQA 合并)、mEff、流水级数 / double buffer / 软流水重叠、PipeBarrier 粒度、Cube 分块 / K_BASE(与 shape §1.3 / resources §1.4 内存划分绑定)| 设计一开始即锁定 Roofline + 容量最优点;核内可模块级 bring-up,但退化中间态只是调试脚手架 |

**边界判据**:把某项后接入,是否需要改动已承诺的核内计算方案(mEff / curG / 流水 / barrier / Cube 分块)?否 → 归前两类,可逐步放开;是 → 它是核内性能方案,属设计目标,**不得**列为"以后放开的步骤"。

**脚手架纪律**:MVP / bring-up 允许临时退化配置(单 head、无流水)先打通正确性,但——
1. 必须显式标注为**调试脚手架,非设计 / 交付物**;
2. 设计文档承诺的始终是最优配置;
3. 算子标记完成前,所有核内 lever 必须收敛到设计承诺的最优形态。**不存在"S5 / 后续迭代再优化"**——最优是本次开发内的 in-scope 项。

**典型接入顺序**(核内性能方案自身按模块级 bring-up,不在此逐步"打开"列):

```
MVP(单核 + 单 head bring-up 脚手架 + fp16 + non-causal + 对齐 + 小 D)
  → 非对齐 / bf16 → causal              (覆盖维度)
  → 多核 / split-KV                      (核外正交层)
  → D 维度扩展(小 D → 中 D → 用户上限)   (覆盖维度)
```

其中 curG / 流水 / DB / K_BASE 等核内性能方案**不作为"打开步骤"**:它们从设计起即为最优目标,核内以模块级 bring-up 逐步构建到该目标形态,单 head / 无流水的中间态仅为脚手架。

**禁止**:多维度同时打开(精度回归出错无法定位);把核内性能方案的退化中间态当作设计目标或交付物。

**强制门禁**:用户需求中 D 上限较大时,MVP 之后必须在多核 + 流水线到位后立即打开"D = 用户上限"的精度回归。

**产出**:接入顺序声明(标明各项所属类别) + 各维度的精度回归覆盖 + 核内性能方案均已按最优设计(非"后续放开")的确认。
