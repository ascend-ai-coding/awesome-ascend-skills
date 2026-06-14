# FlashAttention 类算子设计层

> 本文档列出设计阶段**必须回答的问题**(WHAT)。**不给具体数值、不给完整代码**——具体取值与可行实现由后续单独维护的参考代码承接。
> 进入本阶段的前置:已完成 [`overview.md` §4 分析阶段产出](./overview.md)(子族选定 + shape 边界 + 可选特性枚举 + 性能精度目标)。

---

## §1 设计阶段必须产出清单

进入实现之前,DESIGN 文档(或同等位置)必须显式产出以下条目。每条对应本文一节。

| # | 产出 | 对应章节 |
|---|---|---|
| 1 | 子族确认(承接 overview)| 进入 §2 前 |
| 2 | 宏观决策(架构选型 4 项 + 并行策略选择)| §2 |
| 3 | 多核切分方案(任务维度构造 / 不变量校核 / 核数获取 / 负载均衡)| §3 |
| 4 | 内存划分(UB / L1 / L0 / GM workspace 各自的 buffer 清单与公式)| §4 |
| 5 | 编译宏分类(条件性功能识别 + 编译宏 vs 运行期 if 判定)| §5 |
| 6 | Host Tiling 决策结果(输入、输出字段、决策维度、维度打开纪律)| §6 |
| 7 | 流水线编排(节拍数 / 同步事件 / slot 命名 / slot 公式分离 / PipeBarrier 契约)| §7 |
| 8 | 开发起点(基线验证步骤)| §8 |
| 9 | 校验门禁规划(atol 来源指向 `/ops-precision-standard`,本阶段不重复)| 见 overview §4.4 |

任一条目未产出 → 阻塞,不得进入实现。

**设计依赖链**(自顶向下,无前向依赖):

```
§2 宏观决策(架构 + 并行策略)
    ↓
§3 多核切分(在 §2.5 已选并行策略下)
    ↓
§4 内存划分(在 §2.1-§2.4 架构下)
    ↓
§5 编译宏(在 §4 内存划分下,判定依据 = TBuf 变化)
    ↓
§6 Host Tiling 决策(把 §2-§5 所有决策汇总成 Tiling 函数产出)
    ↓
§7 流水线编排(基于 §2.4 + §3 + §4 的节拍 / 同步 / barrier 一体)
    ↓
§8 开发起点(进入实现的最后准备)
```

---

## §2 宏观决策:确立算子整体形态

宏观决策定义算子的整体形态——Kernel 类型 / UB 模式 / Cube 分块层次 / 流水线深度(§2.1-§2.4 架构选型 4 项)+ 并行策略(§2.5)。这层决策定下后,后续 §3-§7 的所有细节都基于这些选择展开。

### §2.1 Kernel 类型决策标准

回答:**这个 FA 算子用哪种 kernel 类型?**

FA 类的天然形态是 `__mix__(N, M)` AIC + AIV 协同:Cube 算 GEMM(Q·K^T 与 P·V),Vector 算 softmax 与归一化。决策维度:

- AIC : AIV 比例(典型 `__mix__(1, 2)`,即每 AIC 配 2 个 AIV;若 AIV 端是瓶颈考虑提比)
- 是否需要额外子 kernel(如 device 端 V 转置,见 §4.3 fallback)

**产出**:kernel 类型 + AIC/AIV 比例 + 是否含子 kernel。

CANN 版本相关的入口属性差异(如 `extern "C"` / `TPipe` / `WaitFlag` 模板参数)见 `development-guide §1.2 代码结构` 与 `/npu-arch`。

### §2.2 UB 模式决策(streaming 与 D 解耦判定标准)

回答:**UB 上是否存在常驻 ∝ m × D 的 buffer?**

如果有(典型如把 `[m, D]` O_acc 常驻 UB、把 P·V 中间矩阵常驻 UB、把 sum 广播展开后常驻 UB 等),大 D 场景下 UB 必然溢出。判定:

- **答案是"有"** → 必须改架构,把 `[m, D]` 移到 GM workspace,UB 改用固定大小 ping-pong buffer 分 chunk 处理(称为 **streaming UB 模式**)
- **答案是"没有"** → 已经走 streaming,继续设计

**streaming 模式的核心原则**:
1. UB 用**固定大小**(与 D 无关)的 ping-pong buffer
2. `O_acc` 改存 GM workspace,跨 chunk 通过输入接收 queue 流转(读回 + rescale + 累加 + 写出)
3. m × D 矩阵在 UB 内按行分 chunk 处理,chunk 大小 = `bufferSize / D_align`(运行期参数自适应,非编译期分支)
4. Cube Mmad 配合 k-axis 分块(见 §2.3),L0 / L1 也与 D 解耦
5. **单一代码路径**覆盖全 D 范围

**决策依据**:
- streaming 在小 D 下退化为单 chunk loop,实测开销小(典型 ≤ 5%;decode 长 KV 等少数场景可能略高,具体取决于 chunk loop 调度开销)
- 不按 D 分支:维护两套 codebase 的成本远高于 streaming 单路径的小开销
- 扩展性:未来更大 D / 新平台 UB 容量变化不需要重做决策

**禁止**:把 Tiling 决策做成 `if D ≤ X: 朴素模式 else: streaming` 两套代码。

### §2.3 Cube 分块层次决策

回答:**Cube 端 L0 / L1 是否需要按 K 轴或 M 轴分块?**

L0 / L1 容量有限,FA 类的 Cube GEMM 在大 D 场景下必然超容。判定层次:

- **L0 K-axis 分块**:把 Mmad 的 k 维度切成 `K_BASE` 步长,`Mmad(.k=K_BASE, .cmatrixInitVal=(k==0))` 让 Mmad 自身累加。**触发条件**:`D > L0 单 fragment 容量`(典型场景几乎总是触发)。**作用**:L0A / L0B 容量与 D 解耦
- **L1 M-axis 分块**:Q Nd2Nz 不一次性搬入全 `mPad × D`,只搬 `M_BASE` 行到 L1。**触发条件**:`mPad × D × sizeof(Q_T) > L1 单槽容量`(具体公式见 §4.2 L1 划分)。**作用**:L1 容量与 D 解耦
- **L1 N-axis 分块**(P·V 阶段):V LoadData 按 N 方向切;**触发条件**:`mStep × kStep × fractal_size > L0B InitBuffer 容量`(典型 D ≥ 256 触发,A5 平台尤为关键)

**产出**:每层(L0 K / L1 M / L1 N)的分块判定与决策门槛,门槛取值来自硬件查询(`PlatformAscendC::GetCoreMemSize` 系列,不硬编码)。

### §2.4 流水级数与槽数决策

回答:**loop body 内几级流水?各级 stage 之间靠什么槽数交错?槽语义有几类?**

#### Q1:流水级数

由两件事决定:
- 数据流方向 stage 数(同一 KV 分块内的 stage 数,如 §2.2 的 4 stage:C1 / V1 / C2 / V2;子族扩展可能更多,见 subfamilies)
- 跨 KV 分块之间的交错深度(loop body 内同时活跃的 task 数)——通常 = AIC 与 AIV 等待延迟所能掩盖的最大并行度

**产出**:流水级数 + 该级数下 loop body 内 stage 执行顺序(详见 §7.1)。

#### Q2:槽数的语义分类

不同语义的槽**必须用不同常量定义,公式分离**(见 §7.4)。FA 类常见的槽语义类别:

- **跨 stage 握手槽**:跨核 / 跨 stage 的中间矩阵 buffer 轮转,用于让上一 stage 输出与下一 stage 输入错开
- **跨 task 状态槽**:跨 task 持有的 stateful 状态(默认 task 级模式下的 softmax max/sum/exp)
- **跨 loop 自读自写槽**:同一段在不同 loop 迭代中被自身读写(如默认 task 级模式下 V2 跨 s2 累积的 O_acc GM 段)
- **L1 rotation 槽**(子族特定深度,见 [`subfamilies.md §2.2`](./subfamilies.md))

每类槽**数量独立**——可能相同也可能不同;**取值由各自所对应的"并行度需求"决定**,不可一并而论。

**禁止**:把不同语义的槽数写在同一个常量里——混用会导致 silent 数据污染或 race。

**产出**:本算子涉及的所有槽语义类别 + 每类的槽数取值 + 决策依据。

> **一种已落地的实例化**(GQA 三级流水排布,具体配置见参考代码):跨 stage 握手槽 2 + 跨 task 状态槽 = 跨 stage 握手槽 = 2 + 跨 loop 自读自写槽 3(V2 阶段在三级流水中同时活跃的 task 数)+ L1 rotation 槽 = 2(默认子族)/ 3(MLA)。

### §2.5 并行策略选择:默认 task 级 vs split-KV reduce

回答:**默认切分方式能否吃满核数?吃不满时是否启用 split-KV reduce(FlashDecoding 技术)?**

并行策略分两类:

| 策略 | 任务维度构造 | 满足 I4 的方式 | 适用场景 |
|---|---|---|---|
| **默认 task 级**| `taskIdx → (batch, kvHead, gS1Block, gBlock, ...)`,s2 在 task 内顺序执行 | task 内顺序累积 online softmax 状态,天然满足 | 默认情况;`totalTasks ≥ usedCoreNum` |
| **Split-KV reduce(FlashDecoding 技术)** | 同一 KV 序列沿 s2 维拆给多核并行,完成各自局部 softmax + PV 后做跨核 combine | 各核计算 partial 状态后,通过 cross-core combine 阶段归并出最终 `max` / `sum` / `O`,等价满足 | `totalTasks << usedCoreNum`,典型 Sq=1 纯 decode + 极大 Sk(> 2K)的"核数富余"场景 |

**Split-KV reduce 是 FA 算子的一种实现技术,不是子族**——它可以叠加在任意 FA 子族(GQA / MHA / MLA / 稀疏)之上,本质是替换并行模型(task 级 → s2 切片 + cross-core combine),不改变算子的数学形态。

#### Split-KV reduce 启用时的额外契约

- **新增 partial reduce workspace 段**:partial O 段 + partial logsumexp max 段 + partial logsumexp sum 段(具体段名见参考实现)
- **Softmax 变体**:每核处理自己的 s2 切片,不走 stateful 累积,改为每切片独立计算后再 cross-core combine
- **新增跨核 combine 阶段**:所有切片完成后,从 partial 段读回各切片局部结果,做加权归一化得到最终 O
- **任务结构变化**:任务模型本质不同,task 描述结构精简(只需 batch / kvHead / Sq 块标识 + 实际合并的切片数),与默认 task 级的 RunInfo 不兼容

**产出**:并行策略选择 + 启用 split-KV reduce 时的 workspace 段、combine 阶段、任务结构说明。

#### 注意

Split-KV reduce 是**叠加在某个 FA 子族上的实现选择**,不会让算子变成新的"子族"。设计文档应同时声明:**子族**(GQA / MHA / MLA / 稀疏 / 量化)+ **并行策略**(默认 / split-KV reduce)两个独立维度。

---

## §3 多核切分:必须回答的问题

> 承接 §2.5 已选的并行策略展开。默认 task 级与 split-KV reduce 的任务维度构造路径不同,但本节问题清单对两种策略都适用。

### §3.1 任务维度构造

回答:**单个 task 对应输入空间中的哪一段?taskIdx 如何解码到 (batch, kvHead, gS1Block, gBlock, ...)?**

FA 类的任务维度通常是 `batch × kvHead × Sq 分块 × G 分块`(GQA;其他子族变体见 subfamilies)。

**产出**:
- task 解码公式(`taskIdx → 各维度`)
- 每个 task 对应的输入/输出空间范围
- 任务总数公式 `totalTasks = ...`

### §3.2 不变量校核(I2 / I4 在切分上的体现)

设计切分方案时必须显式校核(见 overview §3):

- **I2(GQA 同 kvHead 同任务)**:任务维度构造必须让同一 kvHead 下的 G 个 qHead 在同一 task 内合并到 mEff
- **I4(s2 状态累积正确性)**:已在 §2.5 完成"默认 task 级 / split-KV reduce"二选一声明;在本节复述该选择并校核切分实现:
  - 若默认 task 级 → 校核 s2 不在 taskIdx 维度,只在 task 内顺序执行
  - 若 split-KV reduce → 校核 partial reduce + cross-core combine 实现路径完整

**产出**:任务维度构造对 I2 / I4 的满足性说明。

### §3.3 核数获取

回答:**用多少核?核数从哪来?**

核数必须运行时查询,**禁止**硬编码(通用规则见 `development-guide §1.3 硬件适配`)。FA 类(mix kernel)查询的是 AIC 核数:

```
aicNum 来源 = PlatformAscendC::GetCoreNumAic()
usedCoreNum = min(aicNum, totalTasks)
```

**产出**:核数查询 API + 实际使用核数公式 + 多余核的处理方式(典型为 kernel 入口跳过)。

### §3.4 负载均衡与空闲核处理

回答:**任务在核间如何分配?尾核怎么办?**

- 任务在核间的分配方式(典型为均匀切片)
- 当 `totalTasks % usedCoreNum != 0` 时,尾核的处理
- 空闲核(`aiCoreIdx >= usedCoreNum`)的入口跳过

**产出**:任务分配公式 + 尾核处理 + 空闲核跳过。

---

## §4 内存划分:必须回答的问题

> **核心阶段**——FA 类算子设计的难点集中在内存划分。每个层级(UB / L1 / L0 / GM)的问题清单都必须逐项回答。所有 buffer 公式中只允许出现:运行时查询的硬件常量、Tiling 决策出的基本块、dtype 字节宽——**禁止**直接写数字。

### §4.1 UB 划分

#### Q1:列出所有 UB buffer

回答:**算子需要哪些 UB buffer?**

每个 buffer 必须明确:

| 字段 | 说明 |
|---|---|
| 名称 / 用途 | 该 buffer 在算子流程中承担什么角色 |
| 单槽容量公式 | 公式中只允许出现:运行时查询的硬件常量、Tiling 决策出的基本块、dtype 字节宽。**禁止**直接写数字 |
| depth | TQue 槽数(决定 buffer 物理大小 = 单槽容量 × depth)|

典型 UB buffer 职责类别(以 streaming UB 模式的 FA 为例,具体命名见参考实现):
- **输入接收 queue**:接收跨核握手段读回的 chunk(streaming 模式下也用于读回自读自写段的上一轮中间结果)
- **输出 queue**:V1 / V2 写出
- **softmax 状态 buffer**(默认 task 级模式下):跨 s2 块在线累积的 max / sum / exp,按 task slot 隔离
- **softmax 临时 buffer**:由 Ascend C SoftmaxFlashV2 类 API 的临时 buffer size 查询函数返回
- **causal mask buffer**(仅当 causal,且 mask 不能原地操作时)

#### Q2:单次 DataCopy 量是否 ≤ 单槽容量?

回答:**所有 `DataCopy(UB, GM, ...)` / `DataCopy(GM, UB, ...)` 调用,单次搬运的字节数是否都不超过对应 TQue 的单槽容量?**

若不满足,**必须实施 chunk loop 切分**,公式:`mSplit = 单槽字节容量 / (cols × sizeof(elem))`。V1 / V2 必须**对称**做 chunk loop,不能只切一边。

**踩坑警告**:`mEff × cols × sizeof(elem) > 单槽容量` 时单次 DataCopy 会溢出 UB,破坏邻接 buffer。单 task / 单 s2 小用例可能蒙混过关(恰好落邻接 buffer 没人覆写),multi-task / multi-s2 时被并发 task 覆写 → 表现为"奇/偶 task FAIL"或周期性 chunk 错位。

**产出**:每个 DataCopy 调用点的 `单次量 ≤ 单槽` 校验;不满足的位置实施了对称 chunk loop。

#### Q3:UB 总占用 ≤ `GetCoreMemSize(UB)` 校验

回答:**所有 UB buffer 占用之和是否 ≤ 硬件查询所得 UB 容量?**

校验对象:`Σ (单槽容量 × depth) ≤ PlatformAscendC::GetCoreMemSize(CoreMemType::UB, &ubSize)`。

校验结果不满足 → 下调 Tiling 决策的基本块(m 维 task 上限 / chunk 行数)直至满足。

**产出**:UB 总占用清单 + 与硬件查询值的比对。

#### Q4:∝ m × D 的 UB buffer 消除判定

回答:**是否存在常驻 UB、尺寸 ∝ m × D 的 buffer?**

详见 §2.2 UB 模式决策。本节是 §2.2 决策的逐 buffer 落地校验:
- O_acc(主嫌疑):是否已移到 GM workspace?
- P·V 中间矩阵 / 上一轮 O_acc 缓存 / sum 广播展开 等所有 ∝ m × D 的尾巴 buffer:是否已全部消除?
- sum 广播是否走 `BinaryRepeatParams(src1RepStride=1)` 在 Div 内自动消费,而非分配中间 buffer?

**产出**:已消除的 ∝ m × D buffer 清单 + 替代方案。

#### 自检清单(进入审查前必跑)

- [ ] **清单 1(streaming UB 容量)**:所有 TQue 单槽容量 × depth 列清;所有 kernel 内 DataCopy 量 ≤ 单槽;超容位置已实施 chunk loop;V1 / V2 chunk loop 对称(切分公式 / softmax stat slot 偏移 / SoftMaxShapeInfo.srcM / causal mask 行映射四方面对称)

### §4.2 L1 划分

#### Q1:A1 / B1 端口分离

回答:**Q / P 放在哪个 L1 端口?K / V 放在哪个 L1 端口?**

**硬约束**:Cube `Load2D` 仅支持 A1→L0A、B1→L0B 两条通路。因此:
- **Q、P**(均要进 L0A 做 fm)→ **必须 A1**(`TBuf<TPosition::A1>`)
- **K、V**(均要进 L0B 做 filter)→ **必须 B1**(`TBuf<TPosition::B1>`)

**违反后果**:K/V 放 A1 时 Load2D 无法把它们搬到 L0B,P 段输出全零(GQA 早期历史 bug)。

**产出**:每个 L1 buffer 的 TBuf 位置声明 + 端口选择理由。

#### Q2:单槽容量 vs 一次 Nd2Nz 全量 / M 轴分块判定

回答:**Q / K / V 的一次 Nd2Nz(`mPad × D`、`Bk × D`)是否会超出 L1 单槽容量?**

判定公式:
```
perSlotElems = L1_PORT_SIZE / rotation_slots / sizeof(elem)
若 一次 Nd2Nz 元素数 > perSlotElems → 必须实施 M 轴(或 N 轴)L1 分块
```

`L1_PORT_SIZE` 是 A1 / B1 端口的预算分配(`PlatformAscendC::GetCoreMemSize(CoreMemType::L1_A/L1_B)` 查询所得 ≥ 该预算)。`rotation_slots` 由 §2.4 流水线深度决策与 [`subfamilies.md` §2.2 L1 rotation 子族差异](./subfamilies.md)决定。

**M 轴分块**:外层 `for (m = 0; m < mPad; m += M_BASE)` 只搬 M_BASE 行到 L1。

**N 轴分块**(P·V 阶段):外层 `for (nOff = 0; nOff < D_align; nOff += N_BASE)` 只搬 N_BASE 列到 L0B(N 方向 fractal 数受 L0B 容量约束)。

**易掩盖陷阱**:
- 假设"L1 容量 512KB 充裕"而不做分块——512KB 是 A1+B1 总容量,单 buffer 分不到这么多
- 只做 L0 K-axis 分块而忽略 L1 溢出 → 大 D 时精度崩塌或 kernel 静默错乱(典型 D≥512)

**产出**:Q / K / V 每段的 `单次 Nd2Nz 量 ≤ perSlotElems` 校验 + 不满足时的 M / N 轴分块方案。

#### Q3:rotation 深度依据

回答:**Q/P 与 K/V 各自的 L1 rotation 槽数是多少?依据是什么?**

L1 rotation 深度**因 FA 子族而异**,见 [`subfamilies.md` §2.2](./subfamilies.md)。设计时按子族查表选值,不要凭直觉拍。

**禁止**:把 §2.4 决策的跨核握手槽数和 L1 rotation 槽数混用同一常量(语义不同)。

**产出**:Q/P 与 K/V 各自的 rotation 槽数 + 子族查表依据。

### §4.3 L0 划分

#### Q1:D 触发 K-axis 分块的判定门槛

回答:**`D > X` 时触发 L0 K-axis 分块,X 是多少?依据是什么?**

X 来自 L0A / L0B 单 fragment 容量 / dtype 字节宽。具体公式:
```
L0A 容量 ≥ m × K_BASE × sizeof(fm_dtype)
L0B 容量 ≥ K_BASE × n × sizeof(filter_dtype)
```

其中 m / n / K_BASE 是 Tiling 决策出的基本块。L0A / L0B 容量从 `PlatformAscendC::GetCoreMemSize(CoreMemType::L0_A/L0_B)` 查询。

**关键**:K-axis 分块**单一代码路径**就够,不分 D 档。D 小 k 循环少、D 大 k 循环多,自动适配。

**产出**:K-axis 分块判定公式 + K_BASE 取值依据。

#### Q2:L0A / L0B / L0C 容量验算

回答:**配合 K_BASE / M_BASE / N_BASE,L0A / L0B / L0C 容量都满足吗?**

校验公式:
```
L0A: m × K_BASE × sizeof(fm_dtype)  ≤  L0_A 查询值
L0B: K_BASE × n × sizeof(filter_dtype)  ≤  L0_B 查询值
L0C: m × n × sizeof(accumulator_dtype)  ≤  L0_C 查询值
```

P·V 阶段 n = D 较大,L0C 容易吃紧,可能需要 N 轴 L1 分块配合(见 §4.2 Q2)。

**产出**:L0A / L0B / L0C 各自容量验算结果。

### §4.4 GM Workspace 划分

#### Q1:列出所有 workspace 段

回答:**算子需要哪些 GM workspace 段?**

典型段(按职责分类,具体段名见参考实现):
- **跨核握手段**(Cube 写 → Vector 读 / Vector 写 → Cube 读):承载 `[Bq, Bk]` Q·K^T 中间矩阵、`[Bq, Bk]` P 矩阵、`[Bq, D]` P·V 中间矩阵 等
- **跨 loop 自读自写段**(若需要):默认 task 级 + streaming UB 模式下,V2 跨 s2 块在线累积的 O_acc 落地段(设计在 GM 而非 UB,见 §2.2)
- **任务级状态段**(若需要 stateful 跨 s2 累积):softmax max / sum / exp 在默认 task 级模式下的持有
- **子族扩展段**:见 [`subfamilies.md`](./subfamilies.md)(如 MLA 的 int32 中间段)
- **并行策略扩展段**:若 §2.5 选 split-KV reduce → 增 partial O / partial logsumexp max / partial logsumexp sum 三段

#### Q2:槽数选择依据(语义分离)

回答:**每段 workspace 的轮转槽数是几?各槽语义是什么?**

**强制要求**:不同语义的槽用不同常量定义、用不同公式取模,**不可共用**(详见 §7.4 slot 公式分离原则)。本节给出 §4.4 涉及到的各段所属语义类别:

| 段类别 | 对应槽语义(§2.4 的分类)| 公式形态(详见 §7.4)|
|---|---|---|
| 跨核握手段 | 跨 stage 握手槽 | `loopSlot = loopCounter % handshakeSlots` |
| 跨 loop 自读自写段 | 跨 loop 自读自写槽 | `loopSlot = loopCounter % selfRefSlots`(可能与上式不同模) |
| 任务级状态段 | 跨 task 状态槽 | `taskSlot = taskCounter % preloadSlots` |
| 子族扩展段 / 并行策略扩展段 | 按其具体语义归类(可能跨多类)| — |

#### Q3:按运行时 Sk 分配原则

回答:**workspace 是按 `Sk_max` 静态预留,还是按运行时 `Sk` 动态分配?**

**正确答案永远是动态**:Host 暴露 `GetXxxWorkspaceSize(...)` API,按运行时 Sk 计算返回 size;caller 按返回 size 分配 workspace。

**禁止**:按 Sk_max 静态预留——例如 `跨核握手段 size = B × Hq × Bq × Sk_max × sizeof(elem)`,Sk_max 较大(如 32K)时单块可达数百 MB,实际运行时浪费严重且限制用户上限。

**产出**:`GetXxxWorkspaceSize(...)` API 设计 + 按运行时 Sk 的计算公式。

#### 自检清单(进入审查前必跑)

- [ ] **清单 9(长 Sk workspace 槽轮转)**:所有 workspace 段按运行时 Sk 分配;§2.4 列出的所有槽语义类别(跨 stage 握手 / 跨 loop 自读自写 / 跨 task 状态 / L1 rotation)数量与公式分清楚
- [ ] **清单 4(M 轴尾块 padding 行)**:m 维对齐后的 padding 行 mask / softmax 处理已明确;`SoftMaxShapeInfo.srcM` 取 m 有效维(不是 m 对齐维);见 subfamilies §3.1 GQA 契约
- [ ] **清单 7(GQA 输出 BSND 写回)**:输出按 (g, i) 二维循环逐行 `DataCopyPad`,不做批量 `DataCopy(GM, UB, curBq × D)`(Bq 行不连续);见 subfamilies §3.1
- [ ] **清单 8(m 对齐维 > 单 fractal + 多 s2 块组合)**:GQA 维度首次打开时跑 3 类用例(单 s2+m 对齐 > fractal / 多 s2+m 对齐 = fractal / 两者同开);见 subfamilies §3.1

---

## §5 编译宏分类:必须回答的问题

> 紧贴 §4 内存划分——编译宏判定标准的核心依据是"该特性是否引入 / 删除 TBuf 成员或改变片上 buffer slot 数"。

### §5.1 条件性功能识别

回答:**算子的可选特性中,哪些是条件性功能?**

特性维度来自 overview §4.3 枚举(dtype / causal / sparse / 量化 / ...)。其中**条件性功能**指:启用 / 关闭会改变算子内部结构,而不只是改变数值参数。

### §5.2 编译宏 vs 运行期 if 判定标准

回答:**每个条件性功能用编译宏分离独立 target,还是运行期 if 跳过?**

**判定标准**(通用规则见 `development-guide §6 性能调优纪律`):

| 条件 | 处理 |
|---|---|
| 改动**会**引入 / 删除 TBuf 成员或改变片上 buffer slot 数 | **必须**编译宏分离独立 target |
| 改动只是数值参数变化 | 运行期 if(或 Tiling 透传)|

**理由**:TBuf 成员存在性本身影响编译器 UB 布局,运行期跳过 InitBuffer 不能避免布局漂移。

**FA 类常见编译宏分离场景**:
- `dtype`(fp16 / bf16):影响 typedef 与 Mmad 模板参数同型链,**必须**编译宏分离
- `causal` on / off:影响 causal mask TBuf 是否声明,**必须**编译宏分离
- `sparse` on / off:同上(若涉及 TBuf 变化)

按 overview §4.3 列出的特性笛卡尔积组合,各自一个 target。

**产出**:CMakeLists 中的 target 列表 + 各 target 的编译宏定义清单。

---

## §6 Host Tiling 决策:必须回答的问题

> **产出汇总阶段**——把 §2-§5 的所有设计决策汇总成 Tiling 函数的输入 / 输出 / 决策维度。本节不引入新决策,只把前面已定的决策映射到 Tiling 函数字段。

### §6.1 输入

Host Tiling 函数的输入:
- 用户输入 shape(`B, Sq, Sk, Hq, Hkv, D` 及子族特有维度)
- 硬件查询结果(`PlatformAscendC` 系列查询的 UB / L1_A / L1_B / L0 / AIC 核数 / AIV 核数)
- 子族选型 + 可选特性枚举(causal / dtype / sparse 等)
- 并行策略选择(承接 §2.5)

**产出**:Host Tiling 函数签名 / 入参列表。

### §6.2 输出字段清单

Host Tiling 函数返回的 TilingData 必须按以下**职责类别**覆盖(具体字段名见参考实现):

| 字段类别 | 内容 |
|---|---|
| **基本块尺寸** | m 维 task 上限 / m 维基本块 / s2 维基本块 / 流式 chunk 行数 等 |
| **派生尺寸** | 由基本块 + 输入 shape 推导出的派生量(GQA group 合并维度数、Sq 维分块数、kvHead 维分块数 等;子族特有派生字段见 subfamilies)|
| **任务分发参数** | 任务总数 / 实际使用核数 / 每核任务数 |
| **Workspace 段尺寸** | 按 §4.4 列出的各段(供 caller 分配)|
| **硬件容量回写** | UB / L1_A / L1_B / L0 容量等,供 kernel 端容量校验,避免 kernel 内再次查询 |
| **API Host Tiling 数据** | SoftmaxFlashV2 等 Ascend C 复合 API 的 Host Tiling 数据透传 |

**产出**:`TilingData` 结构体字段清单(按类别落到具体字段名)。

### §6.3 决策维度

回答:**给定用户输入与硬件容量,各基本块尺寸如何决定?**

典型决策维度(供回答时参考):

- **prefill / decode 分流**:Sq = 1 时优先吃满 GQA group 合并维度,m 基本块取 1;Sq 较大时 m 基本块吃满 task 上限、GQA 合并维度退化;中间 Sq 介于两者
- **m 维 task 上限**:在 streaming UB 模式下与 D 解耦,只需取定 Mmad m 维支持的上限即可(典型为硬件 m 上限或更小的调度粒度)
- **s2 维基本块**:受 L1 / L0B 容量 + 用到的 Ascend C 复合 API 约束(如 SoftmaxFlashV2 half 重载 srcK 下限)
- **流式 chunk 行数**(streaming UB 模式下,真·通用公式):

```
chunkRows = chunkBufferBytes / (D_align × sizeof(compute_T))
```

> 该公式是 streaming UB 模式下"chunk 与 D 解耦"的关键数学依据(不是 GQA 特例):chunk 行数由"UB 单 chunk 容量"与"每行字节数"的比值决定,从而让小 D 时 chunk 行数大、loop 少,大 D 时 chunk 行数小、loop 多,**单一代码路径自适应**。

**产出**:每个基本块的决策依据(不必给具体数字,但要说清楚函数依赖)。

### §6.4 维度打开纪律

回答:**新算子开发时各特性维度按什么顺序打开?**

逐维度打开,每打开一维跑一次全场景精度回归。典型顺序:

```
MVP(单核 + 单 qHead 单 kvHead + fp16 + non-causal + 对齐 shape + 小 D)
  → 非对齐 / GQA / bf16
  → causal
  → 多核
  → 三级流水
  → D 维度扩展(逐档:小 D → 中 D → 用户上限)
```

**禁止**:多维度同时打开,精度回归出错时无法定位维度。

**强制门禁**:用户需求中 D 上限较大时,MVP 之后必须在多核 + 流水线维度通过后立即打开"D = 用户上限"的精度回归,不允许"只测小 D 就交付大 D"。

**维度隔离调试纪律**:某维度失败时先回退 MVP 验证共享组件完好,再逐维度二分定位;具体方法见 `/ascendc-precision-debug`。

**产出**:维度打开顺序声明 + 各维度的精度回归覆盖。

---

## §7 流水线编排:必须回答的问题

> 微观实现层——把 §2.4 流水线深度、§3 多核切分、§4 内存划分的决策落到 stage 节拍、跨核同步、cache slot、barrier 设计契约上。

### §7.1 节拍数与同步事件规划

回答:**已选流水级数(§2.4 Q1)下,loop body 内各 stage 的执行顺序如何编排?各 stage 之间用哪个同步事件 ID?**

**与 I1 的关系**:同一 KV 分块内的 stage 依赖方向严格 C1→V1→C2→V2(I1 不变量,见 overview §3.1);**不同 KV 分块如何交错**(loop body 内 stage 顺序、是否反排、是否让某些 stage 早于其依赖的本轮 stage 执行)由设计决定,不在 I1 约束内。

**编排约束**:
- 在 loop body 内任何重排都不能让某个 stage 读到本轮尚未产生的状态(读旧轮的状态需要 slot 公式正确,见 §7.4)
- 反排某些 stage(让节拍编号大的先执行)可避免覆盖问题——典型场景:V2 需要读上一轮的 softmax state,若与 V1 在同一 loop body 中按节拍顺序执行,V1 会先覆写 state;故 V2 应**先于** V1 执行

**同步事件 ID 取值规则**:
- 不与平台保留 ID 冲突
- 各 ID 语义独立声明,**禁止**复用同一 ID 表达不同语义

**产出**:loop body 执行顺序声明(stage 顺序 + 各 stage 处理哪一轮的任务)+ 各 stage 同步事件 ID 列表。

> 一种已落地的实例化(三级流水排布,具体代码见参考实现):loop body 内 stage 顺序 `C1(本轮) → V2(上两轮) → V1(上一轮) + C2(上一轮)`,V2 提前执行确保 softmax state 不被 V1 覆盖。

### §7.2 跨核同步原语选择

回答:**AIC↔AIV 跨核同步用哪种 mode?PIPE 类型选什么?**

跨核同步原语:`CrossCoreSetFlag<mode, PIPE>` / `CrossCoreWaitFlag(id)`。

- **mode**:平台相关(A2/A3 用 mode=2;A5 推荐 mode=4 + AIV1 早返回;详见 `/npu-arch` 平台 API 版本表)
- **PIPE 类型**:AIC → AIV 用 `PIPE_FIX`(Fixpipe 完成后通知);AIV → AIC 用 `PIPE_MTE3`(GM 写完后通知)
- **PIPE_FIX 仅 AIC 有**:AIV 端无 PIPE_FIX,`HardEvent::FIX_*` 在 AIV 端无效,会死锁(用 cross-core sync 取代)

**自检清单 3(cross-core sync 时序)**:
- [ ] AIV → AIC 的 `CrossCoreSetFlag` 在数据写完之后
- [ ] AIC → AIV 的 `CrossCoreWaitFlag` 在数据读取之前
- [ ] 同核内的 PIPE 同步用 `SetFlag/WaitFlag`,跨核用 `CrossCoreSetFlag/WaitFlag`,不混用
- [ ] `mode=2 CrossCoreSetFlag` 是 fire-and-forget,不阻塞后续操作 → 不能依赖它做 back-pressure
- [ ] GM 自读自写段经输出 queue 包装写出(`AllocTensor → DataCopy → EnQue → DeQue → DataCopy → FreeTensor`)+ 配合 `MTE3_MTE2` fence,**禁止**直接 `DataCopy(GM 自读自写段, LocalTensor)` 不经 queue 包装(无 MTE3 同步事件 → 多 s2 时上一轮内容未落盘 → 周期性 chunk 错位)
- [ ] 跨阶段依赖(MTE2↔MTE1、M↔FIX 等)用对应 `HardEvent`,不用 `PipeBarrier` 替代
- [ ] PipeBarrier 冗余率(实际数量 / 设计契约数量 - 1)< 30%(见 §7.5)

**产出**:跨核同步 mode + 各方向 PIPE 类型 + 同核同步 HardEvent 清单。

### §7.3 cache slot 语义命名

回答:**loop body 中 cache slot 如何命名?**

**禁止**:数字下标(`info0` / `info1` / `info2` 这种)——下标与节拍号容易错位,改动时易写错。

**强制**:按"该 slot 在 loop body 内**哪个节拍执行 / 处理哪一轮的任务**"**语义化**命名。命名格式开放,但必须能从名字直接读出该 slot 对应的 stage 与轮次。

**产出**:cache slot 命名约定(具体名字开放,见参考实现)。

### §7.4 跨 loop buffer 与跨 task 状态 slot 公式分离原则

**核心原则**:不同语义的 slot 必须用**不同的计数器**取模**不同的常量**,**不能共用**。这是数学要求,不是命名约定。

| slot 语义 | 用途 | 取模计数器 | 取模常量 | 典型应用 |
|---|---|---|---|---|
| **跨 task 状态 slot** | 跨 task 持有 stateful 状态(默认 task 级模式下的 softmax max/sum/exp 等)| **task 计数器**(每完成一个 task 自增 1)| 跨 task 状态槽数(§2.4 Q2)| 跨 s2 在线累积的 softmax 状态段 |
| **跨 loop handshake slot** | 跨 stage 握手 buffer(上一 stage 写 → 下一 stage 读)| **loop 计数器**(每个 loop body 自增 1)| 跨 stage 握手槽数(§2.4 Q2)| 跨核握手段 |
| **跨 loop 自读自写 slot** | 同一段在不同 loop 迭代中被自身读写 | **loop 计数器** | 自读自写槽数(§2.4 Q2,可能 ≠ 握手槽数)| 默认 task 级模式下 V2 跨 s2 块的 O_acc 落地段 |

**公式骨架**(具体变量名见参考实现):

```
跨 task 状态 slot:taskSlot = taskCounter % preloadSlots
跨 loop handshake slot:handshakeSlot = loopCounter % handshakeSlots
跨 loop 自读自写 slot:selfRefSlot = loopCounter % selfRefSlots
```

**关键**:同一变量(loop 计数器)取模**不同常量**会得到不同 slot 序列 —— 这就是为什么 §2.4 Q2 要求各槽数独立声明。

**误用症状**:
- 用 task 计数器替代 loop 计数器(在 handshake / 自读自写 slot 上):multi-s2 块在 s2N ≥ 3 时精度漂移
- 用 loop 计数器替代 task 计数器(在 task 状态 slot 上):s2 累积状态错乱(max/sum 跨 task 串扰)
- 三类 slot 全部用同一常量取模:可能掩盖 race,但任一槽数的"实际并行度需求"变化时整体崩塌

**最小复现条件**:`single task + multi s2 块(s2N ≥ 3)`,或 `multi task per core + multi s2 块`。

**自检清单 2(slot 语义)**:
- [ ] 所有跨 stage handshake buffer 用 loop 计数器取模(handshakeSlots)
- [ ] 所有跨 loop 自读自写段用 loop 计数器取模(selfRefSlots,可能 ≠ handshakeSlots)
- [ ] 所有跨 task 状态段用 task 计数器取模(preloadSlots)
- [ ] 三类公式互不混用,亦不用单一公式 / 单一常量覆盖所有 buffer

**产出**:每个 buffer 的 slot 公式声明(明确取模计数器 + 取模常量)+ 公式分离的三类列表。

### §7.5 PipeBarrier 设计契约

回答:**算子内 `PipeBarrier<PIPE_ALL>` 应该出现在哪些位置?预计数量是多少?**

`PipeBarrier<PIPE_ALL>` 用于跨 pipe 整核 drain。设计阶段必须**显式列出**所有需要出现的位置(契约位置),禁止四处散布。

**FA 类的 PipeBarrier 来源**:
- 通用 Cube 同步约束(每次 Mmad 之后等)—— 属 Ascend C 通用 API 同步规则,见 `/ascendc-api-best-practices`
- 跨 pipe 边界(如 MTE2 → MTE1)无对应 HardEvent 时必须 PIPE_ALL 兜底 —— 通用规则
- 平台特定治本配方(如某些平台的多 K-pass Mmad 隐式状态需要在循环起始 drain)—— 见 `/npu-arch` 平台差异表

设计阶段应基于以上来源列出本算子的**契约位置清单**与预计数量。

**冗余率验收**:在审查前列出"设计契约位置数"与"实现实际位置数",冗余率 `(实际 / 设计 - 1) < 30%`,达到 50% 视为性能阻塞项。

**禁止**:
- 以"PIPE_ALL 是保险屏障,跟精度无关"为由四处散布。冗余 barrier 让 Cube 流水利用率系统性下降
- 同核内跨 pipe 依赖用 PipeBarrier 替代专门的 HardEvent(应使用对应 `HardEvent::X_Y`)
- 同一 pipe 内的顺序操作之间散布 barrier(不需要)

**产出**:PipeBarrier 设计契约位置清单 + 预计数量 + 各位置的依据(通用同步 / 跨 pipe 兜底 / 平台特定配方)。

---

## §8 开发起点

### §8.1 基线验证步骤

在写自定义 kernel 之前**必须**先编译运行 asc-devkit 中同 kernel 类型的至少一个参考示例,确认 `__mix__(N, M)` + TPipe 模式在当前 CANN 版本下可用。具体推荐示例:

- 标准 mix kernel 模板:`asc-devkit/examples/.../data_copy_ub2l1/data_copy_ub2l1.asc`
- 跨核同步原语验证:`asc-devkit/examples/.../cross_core_set_wait_flag/cross_core_set_wait_flag.asc`
- A5 平台 mode=4 同步基座(若目标平台为 A5):`asc-devkit/examples/01_simd_cpp_api/05_compatibility_guide/matmul_s4/matmul_s4.asc`

通用规则(基线验证强制 / 架构偏离管控)见 `workflows/development-guide.md` 对应章节。

---

## §9 参考资源

| 阶段 / 主题 | 入口 |
|---|---|
| 分析阶段产出 | [`overview.md`](./overview.md) |
| 子族扩展契约 | [`subfamilies.md`](./subfamilies.md) |
| 通用调试方法 | `/ascendc-precision-debug` |
| 运行时错误码 | `/ascendc-runtime-debug` |
| 精度门禁 | `/ops-precision-standard` |
| 平台差异 / 架构基础 | `/npu-arch` |
| Ascend C API 用法 | `/ascendc-api-best-practices` |
| 通用编码规范 / 工程配置 / 修复循环纪律 / 性能调优纪律 | `ops-direct-invoke` plugin `workflows/development-guide.md` |
