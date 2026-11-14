# 问题形态与分核(节点 N1 + N2 + N3 + N5 + N6)

> **关注点**:问题形态如何映射到硬件 + 如何切分到多核。
> 节点序列严格等同依赖链:N1 宏观形态 → N2 并行策略 → N3 基本块/Roofline → N5 平台依赖 → N6 多核/负载。
> 宏观决策定义算子的整体形态——Kernel 类型 / UB 模式 / Cube 分块层次 / 流水线深度(N1 架构选型 4 项)+ 并行策略(N2)+ 基本块与 Roofline(N3)。这层决策定下后,后续 N6-N10 的所有细节都基于这些选择展开。
> **依赖链核对**:N2(并行)先于 N3(Roofline);N5(平台依赖)依赖 N3(macro 早置,标记供 N8/N10 消费)。

---

## §1 宏观形态(节点 N1)

回答:**这个 FA 算子用哪种 kernel 类型 / UB 模式 / Cube 分块层次 / 流水级数?**

### §1.1 Kernel 类型决策标准

回答:**这个 FA 算子用哪种 kernel 类型?**

**推荐**:FA/GQA 类算子应使用 `__mix__(1, 2)`(1 AIC + 2 AIV)。Cube 算 GEMM(Q·K^T 与 P·V),Vector 算 softmax 与归一化。这是 FA 类在 AIC+AIV 混合架构上的天然形态。

**禁止**:使用 `__vector__`(AIV_ONLY)做 FA 类算子——纯 Vector 实现 GEMM 性能比 Cube 低 10-100×,不可能达到性能目标。仅在 MVP 调试阶段可临时使用 AIV_ONLY 做精度基线。

决策维度:
- AIC : AIV 比例(典型 `__mix__(1, 2)`,即每 AIC 配 2 个 AIV;若 AIV 端是瓶颈考虑提比)
- 是否需要额外子 kernel(如 device 端 V 转置,见 §1.4 L0 分块 fallback,resources §1.4)

**产出**:kernel 类型 + AIC/AIV 比例 + 是否含子 kernel。

CANN 版本相关的入口属性差异(如 `extern "C"` / `TPipe` / `WaitFlag` 模板参数)见 `development-guide §1.2 代码结构` 与 `/npu-arch`。

### §1.2 UB 模式决策(streaming 与 D 解耦判定标准)

回答:**UB 上是否存在常驻 ∝ m × D 的 buffer?**

如果有(典型如把 `[m, D]` O_acc 常驻 UB、把 P·V 中间矩阵常驻 UB、把 sum 广播展开后常驻 UB 等),大 D 场景下 UB 必然溢出。判定:

- **答案是"有"** → 必须改架构,把 `[m, D]` 移到 GM workspace,UB 改用固定大小 ping-pong buffer 分 chunk 处理(称为 **streaming UB 模式**)
- **答案是"没有"** → 已经走 streaming,继续设计

**streaming 模式的核心原则**:
1. UB 用**固定大小**(与 D 无关)的 ping-pong buffer
2. `O_acc` 改存 GM workspace,跨 chunk 通过输入接收 queue 流转(读回 + rescale + 累加 + 写出)
3. m × D 矩阵在 UB 内按行分 chunk 处理,chunk 行数公式为权威定义见 resources §1.2(运行期参数自适应,非编译期分支)
4. Cube Mmad 配合 k-axis 分块(见 §1.3),L0 / L1 也与 D 解耦
5. **单一代码路径**覆盖全 D 范围

**决策依据**:
- streaming 在小 D 下退化为单 chunk loop,实测开销小(典型 ≤ 5%)
- 不按 D 分支:维护两套 codebase 的成本远高于 streaming 单路径的小开销
- 扩展性:未来更大 D / 新平台 UB 容量变化不需要重做决策

**禁止**:把 Tiling 决策做成 `if D ≤ X: 朴素模式 else: streaming` 两套代码。

### §1.3 Cube 分块层次决策

回答:**Cube 端 L0 / L1 是否需要按 K 轴或 M 轴分块?**

L0 / L1 容量有限,FA 类的 Cube GEMM 在大 D 场景下必然超容。判定层次:

- **L0 K-axis 分块**:把 Mmad 的 k 维度切成 `K_BASE` 步长,`Mmad(.k=K_BASE, .cmatrixInitVal=(k==0))` 让 Mmad 自身累加。**触发条件**:`D > L0 单 fragment 容量`(典型场景几乎总是触发)。**作用**:L0A / L0B 容量与 D 解耦
- **L1 M-axis 分块**:Q Nd2Nz 不一次性搬入全 `mPad × D`,只搬 `M_BASE` 行到 L1。**触发条件**:`mPad × D × sizeof(Q_T) > L1 单槽容量`(具体公式见 resources §1.3 L1 划分)。**作用**:L1 容量与 D 解耦
- **L1 N-axis 分块**(P·V 阶段):V LoadData 按 N 方向切;**触发条件**:`mStep × kStep × fractal_size > L0B InitBuffer 容量`(大 D 场景触发,A5 平台尤为关键)。实现时对所有 D 值**统一走 N 轴分块代码路径**,循环次数 = `ceil(D_align / baseN)`(baseN 由 L0B 容量查询定),小 D 时循环 1 次(无实际切分但代码路径统一)。不应根据 D 值分支

**产出**:每层(L0 K / L1 M / L1 N)的分块判定与决策门槛,门槛取值来自硬件查询(`PlatformAscendC::GetCoreMemSize` 系列,不硬编码)。

#### 基本块取值:按推导,不照抄

基本块(K_BASE / M_BASE / N 轴步进等)**必须由 Roofline(§3)与容量约束推导**,随平台/shape/dtype 变化,**不设规范数值**。以下是推导要点(非固定值):

- **K_BASE 尽量取到 D**(在 L0A/L0B 容量允许时):`K_BASE = D` 时单次 k-pass 完成 Mmad,无需跨 k-pass 累加;`K_BASE` 越小,k-pass 次数 = `ceil(D/K_BASE)` 越多,每次都带 SetFlag/WaitFlag/PipeBarrier 同步开销。**选 K_BASE 必须代入 k-pass 次数评估同步开销**——这是 K_BASE 决策的核心方法论。
- **M_BASE / N 轴步进**:在 L1/L0 单槽容量约束下取尽量大(减少分块循环次数),上界由 resources §1.3/§1.4 的容量校验公式给出。
- 若需要初始试验值,可先按"K_BASE=D、M/N 步进取容量允许的最大 16 倍数"起步,再按目标平台实测调整。

#### L0 数据加载 API 选型

L1→L0 加载选型取决于数据布局和平台能力(公开 API 见 asc-devkit `矩阵数据搬入至L0-Buffer/`):

- **不转置**(左矩阵 MK 布局):`LoadData`(`Load2D` / `Load2DV2` 系列),不转置
- **需转置**(右矩阵 KN 布局,如 V):`LoadDataWithTranspose`(参数结构体 `LoadData2dTransposeParamsV2`)。**平台约束**:950PR/950DT 上该 API **仅支持 L1→L0B 通路**(不支持 L1→L0A)。优先选用支持目标 dtype(如 bf16)直接转置的形态,避免 ReinterpretCast workaround

> ⚠️ **非转置 load2dV2 是 Nz2Nz——L1 源须已按 NZ 连续排布多个分形,`mStep` 才能跨分形批量装载**。非转置 load2dV2 不重排分形(Nz2Nz),按分形步长在 L1 上寻址相邻分形。若上游 Nd2Nz 每次只把 1 个 M-fractal 搬到 L1 同一起点,L1 上只有单个有效分形,此时 `mStep≥2`(读 ≥2 连续分形)会读到相邻位置的无效/错位数据,`mStartPosition` 的多分形选段也因步长不匹配而失效(现象类似"硬件读错行",根因是布局契约未对齐,非硅片 bug)。规则:要用 `mStep` 跨分形批量装载 L0A/L0B,Nd2Nz 的 dst NZ 布局必须让多个 M-fractal 按 load2dV2 的分形步长**连续排布**;否则只能单分形装载(`mStep=1`)。退到单分形是正确性 workaround,但牺牲 Cube 批量装载效率,不应作为默认。非转置 load2dV2 的 Nz2Nz 分形语义以目标版本「矩阵计算输入搬运约束」为准。

**设计检查点**:确认目标平台对选定 LoadData API 的类型支持(特别是 bf16 转置)与通路限制(L0A/L0B),在架构承诺表标记"平台依赖"(见 §4)。

### §1.4 流水级数与槽语义分类

回答:**loop body 内几级流水?各级 stage 之间靠什么槽数交错?槽语义有几类?**

#### Q1:流水级数

由两件事决定:
- 数据流方向 stage 数(同一 KV 分块内的 stage 数,如 [`fundamentals.md` §2.2](../foundation/fundamentals.md) 的 4 stage:C1 / V1 / C2 / V2;变体扩展可能更多,见对应 `subfamilies/` 文件)
- 跨 KV 分块之间的交错深度(loop body 内同时活跃的 task 数)

**级数选型**(承接上,确认该级数下的具体编排方案):

| 可选方案 | 说明 | 优势 | 代价 |
|---------|------|------|------|
| 2 级(C1+V1 / C2+V2 配对)| C1+V1 配对,C2+V2 延后 | 实现简单、buffer 少 | Cube/Vector 交替空闲,隐藏延迟能力有限 |
| 3 级(C1→V2→V1+C2 交错)| C1 当前轮,V2 上上轮,V1+C2 上一轮 | Cube/Vector 同时工作 | buffer 多、slot 管理复杂 |

**推荐条件**:2 级是常见首选(实现简单、buffer 少)。3 级在需要最大化 Cube/Vector 重叠时有优势。

**产出**:流水级数 + 该级数下 loop body 内 stage 执行顺序(详见 execution §3.1)+ 选择理由。

#### Q2:槽数的语义分类(枚举)

不同语义的槽**必须用不同常量定义,公式分离**。FA 类常见的槽语义类别(枚举):

- **跨 stage 握手槽**:跨核 / 跨 stage 的中间矩阵 buffer 轮转,用于让上一 stage 输出与下一 stage 输入错开
- **跨 task 状态槽**:跨 task 持有的 stateful 状态(默认 task 级模式下的 softmax max/sum/exp)
- **跨 loop 自读自写槽**:同一段在不同 loop 迭代中被自身读写(如默认 task 级模式下 V2 跨 s2 累积的 O_acc GM 段)
- **L1 rotation 槽**(变体特定深度,见 [`extension_points.md` §1.2](../foundation/extension_points.md))
- **task 环形缓冲区槽**:task 元信息的环形缓冲区

每类槽**数量独立**——可能相同也可能不同;**取值由各自所对应的"并行度需求"决定**,不可一并而论。

> **权威 slot 分类标准表 + 取模计数器 + 索引方式 + `mloop`/`loop`/`PRELOAD_N`/`CACHE_SIZE` 规范块见 execution §3.3**(唯一权威表,本节仅列语义枚举)。

**禁止**:把不同语义的槽数写在同一个常量里——混用会导致 silent 数据污染或 race(详细规范块见 execution §3.3)。

**产出**:本算子涉及的所有槽语义类别 + 每类的槽数取值 + 决策依据。

> **一种典型实例化**(GQA 2 级流水的槽数配置):跨 stage 握手槽 2(双缓冲)+ 跨 task 状态槽 `PRELOAD_N+1`(如 3)+ 跨 loop 自读自写槽 `PRELOAD_N+1`(V2 阶段同时活跃的 task 数)+ L1 rotation 槽 = 2(默认形态;MLA 因 latent history 需更深,见 [`extension_points.md` §1.2](../foundation/extension_points.md))。该实例采用 2 级流水(C1+V1 配对 → C2+V2 延后)。这些是方法论层面的槽数关系,常量命名由实现自定。

---

## §2 并行策略选择 + split-KV 启用判据(节点 N2)

回答:**默认切分方式能否吃满核数?吃不满时是否启用 split-KV reduce(FlashDecoding 技术)?**

| 策略 | 任务维度构造 | 满足 I4 的方式 | 适用场景 |
|---|---|---|---|
| **默认 task 级**| `taskIdx → (batch, kvHead, gS1Block, gBlock, ...)`,s2 在 task 内顺序执行 | task 内顺序累积 online softmax 状态,天然满足 | 默认情况;`totalTasks ≥ usedCoreNum` |
| **Split-KV reduce(FlashDecoding 技术)** | 同一 KV 序列沿 s2 维拆给多核并行,完成各自局部 softmax + PV 后做跨核 combine | 各核计算 partial 状态后,通过 cross-core combine 阶段归并出最终 `max` / `sum` / `O`,等价满足 | `totalTasks << usedCoreNum`,典型 Sq=1 纯 decode + 极大 Sk(> 2K)的"核数富余"场景 |

**Split-KV reduce 是 FA 算子的一种实现技术,不是变体**——它可以叠加在任意 FA 变体之上。

### §2.1 Split-KV reduce 启用时的额外契约

- **新增 partial reduce workspace 段**:partial O 段 + partial logsumexp max 段 + partial logsumexp sum 段
- **Softmax 变体**:每核处理自己的 s2 切片,不走 stateful 累积,改为每切片独立计算后再 cross-core combine
- **新增跨核 combine 阶段**:所有切片完成后,从 partial 段读回各切片局部结果,做加权归一化得到最终 O
- **任务结构变化**:任务模型本质不同,task 描述结构精简

### §2.2 Split-KV reduce 启用判据

**启用条件**(全部满足):
1. `totalTasks < usedCoreNum`(核数富余)
2. `Sq == 1`(典型纯 decode 场景)
3. `Sk` 足够大(> 2K)
4. Kernel 支持 partial reduce + cross-core combine 路径

**禁止**:在 `totalTasks ≥ usedCoreNum` 时启用 Split-KV reduce。

**产出**:并行策略选择 + 启用 split-KV reduce 时的 workspace 段、combine 阶段、任务结构说明 + 不支持时的替代策略。

#### 注意

设计文档应同时声明:**选型结论**(基础 FA 形态 GQA / MHA / MLA + 正交 trait 量化 / 稀疏)+ **并行策略**(默认 / split-KV reduce)两个独立维度。

---

## §3 基本块与 Roofline 分析(节点 N3)

回答:**基本块(Bq, Bk)选多大?依据是什么?**

基本块是最外层切分粒度(Bq = Q 序列块大小,Bk = KV 序列块大小),直接决定单次 tile 的算术强度与三通道吞吐平衡。

**GQA 多头合并**:FA 类 Mmad 的有效 m 维不是 Bq 本身,而是 `mEff = Bq × curG`(见 [`fundamentals.md` §3.2 I2](../foundation/fundamentals.md))。curG ≤ G = Hq/Hkv。decode(Bq=1)时 `mEff = curG`,必须最大化 curG 才能提升 Cube m 维利用率与 AI。

**核心方法**:FA 是 mix kernel,性能瓶颈**不能用单一 AI 概括**——需按 **三通道 Roofline**(Cube 算力 / HBM 带宽 / 跨核片上吞吐)各自建模,性能取三者 min。完整模型(AI_HBM = 2·mEff/sizeof(T) ∝ mEff、cross-core-bound 校核、可行域上下界、prefill/decode/causal 场景特化、量化 AI 修正)见 **[`roofline.md`](../foundation/roofline.md)**。

> ⚠️ **禁止**:把片上跨核握手(S/P/PV movement)与 HBM 加载混进同一 AI 分母求"混合带宽 AI"——会把 AI 人为封顶、掩盖真正的 AIV/片上瓶颈。详见 [`roofline.md` §1](../foundation/roofline.md)。

**产出**(每项展示推导算式,禁止只给结论;完整清单见 [`roofline.md` §6](../foundation/roofline.md)):
- **AI_HBM** + **ridge_HBM** + 三通道 ceiling 与 min 判定
- **mEff_min**(下界)+ **mEff_max / Bk_max**(硬件上界)+ **V1 模式判定**
- **最终选定** mEff, Bk + 性能区制(compute / HBM / cross-core-bound)

---

## §4 平台依赖识别(节点 N5)

回答:**设计中选用的硬件特化 API 是否依赖特定平台行为?**

FA 类算子大量使用硬件特化 API,这些 API 在不同平台上的行为可能存在差异。常见平台敏感类别:

- **跨核同步原语**:modeId、PIPE 类型、多 AIV 通知方式因平台而异
- **Cube 数据加载**:转置支持的数据类型因平台而异
- **Fixpipe 输出参数**:参数结构体因平台而异
- **硬件加速计算 API**:语义和模板参数支持因平台而异

**原则**:设计阶段对每个硬件特化 API 评估其是否依赖特定平台行为。有依赖 → 在架构承诺表中标记"平台依赖"。macro 早置,标记供 N8(编译特化,execution §2)/ N10(流水同步,execution §3)消费。

**产出**:架构承诺表中每个硬件特化 API 的平台依赖状态已确认。

---

## §5 多核切分 + 负载均衡(节点 N6)

> 承接 N2 已选的并行策略展开。默认 task 级与 split-KV reduce 的任务维度构造路径不同,但本节问题清单对两种策略都适用。

### §5.1 任务维度构造

回答:**单个 task 对应输入空间中的哪一段?taskIdx 如何解码到 (batch, kvHead, gS1Block, gBlock, ...)?**

FA 类的任务维度通常是 `batch × kvHead × Sq 分块 × G 分块`(GQA;其他变体见对应 `subfamilies/` 文件)。映射到 taskIdx 可用闭式 decode 或 metadata range-walker,二者等价(见 [`base_design.md` §3.1](../subfamilies/base_design.md))——本节只要求产出任务空间的正确覆盖,不规定具体映射写法。

**产出**:task 解码方式(闭式或 range-walker)+ 每个 task 对应的输入/输出空间范围 + 任务总数公式。

### §5.2 不变量校核(I2 / I4 在切分上的体现)

设计切分方案时必须显式校核(见 [`fundamentals.md` §3](../foundation/fundamentals.md)):

- **I2(GQA 同 kvHead 同任务)**:任务维度构造必须让同一 kvHead 下的 G 个 qHead 在同一 task 内合并到 mEff
- **I4(s2 状态累积正确性)**:已在 N2 完成"默认 task 级 / split-KV reduce"二选一声明;在本节复述该选择并校核切分实现

**产出**:任务维度构造对 I2 / I4 的满足性说明。

### §5.3 核数获取

回答:**用多少核?核数从哪来?**

核数必须运行时查询,**禁止**硬编码。FA 类(mix kernel)查询的是 AIC 核数:

```
aicNum 来源 = PlatformAscendC::GetCoreNumAic()
usedCoreNum = min(aicNum, totalTasks)
```

**产出**:核数查询 API + 实际使用核数公式 + 多余核的处理方式。

### §5.4 负载均衡与空闲核处理(基础)

回答:**任务在核间如何分配?尾核怎么办?**

- 任务在核间的分配方式(典型为均匀切片)
- 当 `totalTasks % usedCoreNum != 0` 时,尾核的处理
- 空闲核(`aiCoreIdx >= usedCoreNum`)的入口跳过

**产出**:任务分配公式 + 尾核处理 + 空闲核跳过。

### §5.5 负载均衡策略(结构决策)+ 自检清单 O8-O9

#### §5.5.1 核数利用率

回答:**`totalTasks / usedCoreNum` 的比值是多少?能否吃满所有核?**

- `totalTasks ≥ usedCoreNum`:每核至少 1 个 task → 核数利用率 100%
- `totalTasks < usedCoreNum`:存在空闲核 → 需要考虑 Split-KV reduce(见 §2.2)

**产出**:典型 shape 下的 totalTasks 估算 + 核数利用率。

#### §5.5.2 变长序列 + Causal Mask 均衡

回答:**变长序列和 causal mask 下,各核的工作量差异有多大?怎么均衡?**

| 策略 | 说明 |
|------|------|
| Zigzag 分配 | 偶数轮正序、奇数轮反序,让大 task 和小 task 交替分配 |
| 动态 work-stealing | 维护全局 task 计数器,空闲核主动领取下一个 task |
| Sparse 剪枝 | 计算每个 KV tile 的实际有效范围,跳过被 mask 完全覆盖的 tile |

> 📊 **数值标定项**:zigzag 与 work-stealing 在**结构上等价可选**(都需设计期决定 task 映射如何构造),但"哪个在具体 shape 上更均衡"可 profiling 对比后定;设计期选定一个默认策略(causal 场景常默认 zigzag,实现简单)即可,精确收益标定归性能验收。**是否需要均衡策略**(即 causal/变长下是否存在负载倾斜)必须设计期判定。

**产出**:负载均衡策略(标注默认选择)+ 预期均衡效果。

#### §5.5.3 负载层自检清单

- [ ] **清单 O8(核数利用率)**:典型 shape 下核数利用率已估算;空闲核有处理策略
- [ ] **清单 O9(均衡策略)**:causal/变长场景下的均衡策略已选定
