# 计算如何执行(节点 N4 + N8 + N10 + N11 + N12)

> **关注点**:softmax 算法 / 编译期特化 / 流水编排与同步 / 开发起点 / 优化产出。
> 节点序列:N4 softmax → N8 编译期特化 → N10 流水/同步 → N11 开发起点 → N12 优化产出。
> **依赖链核对**:N4 依赖 N3(消费 Roofline 的 V1 模式判定);N10 依赖 N1(§1.4 流水级数)+ N4 + N6 + N7。

---

## §1 V1/V2 Online Softmax 算法模式(节点 N4)

FA 类算子的 softmax 采用 FlashAttention-2 的 online 算法,分 V1/V2 两个阶段:

**V1 阶段**(计算 softmax 中间结果):
1. 对 QK scores 做 scale
2. ReduceMax → 行最大值 max
3. FusedExpSub / Exp(x - max) → 未归一化的指数值
4. ReduceSum → 行求和 sum

**V2 阶段**(rescale + 累加 + 归一化):
- 非末轮 s2:V_accum = V_accum × exp(old_max - new_max) + exp(x - max) × V_new
- 末轮 s2:V_out = (同上) / sum

**关键**:V1 输出的是 `exp(x-max)`,**不是** softmax 概率。归一化(÷sum)只在 V2 末轮执行。

> ⚠️ **online softmax 是有状态的,不是单次变换**。跨 s2 块的 online 算法必须在块间**接入并回读**三类状态:running max、running sum、rescale 因子 `exp(old_max − new_max)`。设计阶段必须确认这些状态量在 s2 循环间正确传递,**不能把 V1 当成一次性 `src → dst` 的无状态调用**。无论走 SoftmaxFlashV2 还是手动实现,这些状态的输入/输出都必须显式规划(对应 resources §1.1 的 softmax 状态 buffer)。具体接口的参数形态与个数以目标版本文档为准——本节只约束"状态必须接入"这一方法论要求,不锁定任何版本的确切签名。
>
> ⚠️ **首块初始化契约(易漏,漏则整行 O 错并沿 s2 传播)**:
> - **running max/sum**:首块用 `max = −∞`、`sum = 0` 初始化。
> - **O_acc**:首块累积的 O_acc 段(若落 GM workspace)**必须显式清零**——GM workspace 默认不清零,不清则首块 `rescale·O_acc + PV` 读到脏数据。
> - **rescale 因子**:首块**无前序状态,rescale 因子应等价于 1**(即首块直接 `O = P·V`,不做 rescale)。若用有"是否更新前序状态"开关的 softmax API,首块通常走"不更新"模式,此时该 API **不产出 rescale 因子**,设计须让 V2 首块走"跳过 rescale"分支,不要去读一个未定义的 rescale 值。
>
> ⚠️ **末块除零保护**:末块 `O = O_acc / sum` 前须对 `sum` 做下限保护(如加一个极小 eps 或 clamp)。padding 行或极端输入下 `sum` 可能为 0,直接相除会得到 inf/NaN 并写出。

**实现路径**(按公开可用性排序):

| 路径 | 定位 | 说明 |
|---|---|---|
| **SoftmaxFlashV2 API**(默认 / 推荐) | ✅ Ascend C 公开 API | asc-devkit `include/adv_api/activation/softmaxflashv2.h`,有文档与示例;`T` 仅支持 half/float,bf16 需 `SoftmaxFlashV2<float>` + 手动 Cast |
| **手动指令级实现** | ✅ 公开基础 API 组合 | 需绕开高阶 API 约束时,用 ReduceMax + Exp + ReduceSum 组合 |

> **默认路径选 `SoftmaxFlashV2`**(公开 API)。部分 FA 参考实现内部把 V1 融合成私有单函数以追求极致性能,但那类函数不是 Ascend C 公开 API、走公开 API 的开发路径无法直接调用——本 skill 不推荐、不依赖此类私有实现。

> ⚠️ **`isBasicBlock` 模板参数必须与 TilingFunc 一致**。`SoftmaxFlashV2<T, isUpdate, ..., isBasicBlock>` 的 `isBasicBlock`(布尔模板参数)与 `SoftMaxFlashV2TilingFunc(..., isBasicBlock)` 传入的值**必须相同**;不一致会导致 expMax 的 UB 布局与 kernel 预期错位 → 逐行 rescale 读到错误偏移 → 精度错(编译能过,静默数值错)。isBasicBlock 的生效条件(尾轴/行数对齐)及一致性用法以目标版本 SoftmaxFlashV2 文档与样例为准。

#### Padding 行 / causal mask 值处理

softmax 之前必须对 P 矩阵的 padding 行(`[mEff, mPad)`)以及 causal 屏蔽位写入屏蔽值。**无论选择哪条路径,padding / mask 都不可省略。**

> ⚠️ **mask 值必须用大有限负数(如 −1e30),禁用 `−∞`**(FA 类经典 NaN 陷阱)。若某行被**整行屏蔽**(causal 完全屏蔽 tile,或 padding 行),用 `−∞` 会让 `rowmax = −∞` → `exp(−∞ − (−∞)) = exp(NaN) = NaN`,sum/O 全变 NaN 并沿 s2 累积传播。用大有限负数则 `exp(mask − max) ≈ 0`,数值安全。
>
> ⚠️ **完全屏蔽的 tile:跳过 C1 后 V1 必须同步跳过**。causal 下若某 tile 的 KV 全在 mask 外(`qOff + mEff − 1 < kOff`),跳过 C1 GEMM 的同时,V1 也**不得对该 tile 跑 softmax / 更新 max·sum**(该 tile 贡献为 0);若跨核握手段需要有效数据供下游 GEMM,P 段应写全 0。**只跳 C1 不跳 V1** → 全 mask 行仍进 softmax → NaN。
>
> padding 行的输出应最终强制清零,softmax 状态不参与有效累积。

**产出**:V1/V2 softmax 实现路径选择 + 输出语义标注。

---

## §2 编译期特化(节点 N8)+ 自检清单 O13-O16

> 紧贴 N7 内存划分——编译期特化判定标准的核心依据是"该特性是否引入 / 删除 TBuf 成员或改变片上 buffer slot 数"。依赖:N7。

### §2.1 条件性功能识别

回答:**算子的可选特性中,哪些是条件性功能?**

特性维度来自 [`fundamentals.md` §4.3](../foundation/fundamentals.md) 枚举(dtype / causal / sparse / 量化 / ...)。其中**条件性功能**指:启用 / 关闭会改变算子内部结构,而不只是改变数值参数。

### §2.2 三维度分层判定(权威 taxonomy)

回答:**每个条件性功能走编译宏、constexpr 模板参数、还是运行时 if?**

**权威判定采三维度分层**(编译宏 / constexpr / 运行时 if)。早期方法学用的二分法(编译宏 vs 运行期 if)是其**粗粒度子集**——"运行期 if"桶按"是否在热路径分支"进一步细分出 constexpr 层:

| 判定条件 | 处理方式 | 理由 |
|---------|---------|------|
| 改动引入/删除 TBuf 成员或改变片上 buffer slot 数 | **编译宏分离独立 target** | TBuf 存在性影响编译器 UB 布局;运行期跳过 InitBuffer 不能避免布局漂移 |
| 改动影响热路径上的分支 | **constexpr 模板参数** | 编译期消除分支 |
| 改动只影响数值参数 | **运行时 if** | 不影响代码结构 |

**判定流程**:
```
改动是否引入/删除 TBuf 成员?
  ├─ 是 → 编译宏分离
  └─ 否 → 改动是否在热路径上?
            ├─ 是 → constexpr 模板参数
            └─ 否 → 运行时 if
```

**FA 类常见分层**:

| 维度 | 典型处理 | 理由 |
|------|---------|------|
| dtype(fp16/bf16)| **编译宏** | 影响 typedef 和 Mmad 模板参数 |
| causal on/off | **编译宏** | 影响 mask TBuf 是否声明 |
| sparse on/off | **编译宏**(若涉及 TBuf 变化)| 同上 |
| layout(BSND/BNSD)| constexpr 模板参数 | 影响 stride 计算 |
| hasRope / emptyTensor | constexpr false 消除 | 不需要时整个子树编译期消除 |
| 实际序列长度 | 运行时 if | 仅数值参数变化 |

按 fundamentals §4.3 列出的特性笛卡尔积组合,各自一个 target。

**产出**:每个条件性功能的分层归属表 + CMakeLists 中的 target 列表 + 各 target 的编译宏定义清单。

### §2.3 Dummy Block 模式

回答:**AIC/AIV 编译期分流是否需要 Dummy Block 模式?**

| 可选方案 | 说明 | 优势 | 代价 |
|---------|------|------|------|
| Dummy Block | Cube 核用 CubeBlock + VecBlockDummy | 零死代码 + 类型安全 | 需维护 Dummy 类 |
| 运行时 if 跳过 | `if ASCEND_IS_AIC { ... } else { ... }` | 无需 Dummy 类 | 未执行分支仍占编译空间 |

**模式**:Cube 核上 `CubeBlock` 真实、`VecBlock` 为 `VecBlockDummy`;Vector 核上相反。编译期通过 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 分流。

**推荐条件**:`__mix__(N, M)` kernel → Dummy Block。

**产出**:是否使用 Dummy Block + 理由。

### §2.4 GQA 合并策略

| 可选方案 | 说明 | 优势 | 代价 |
|---------|------|------|------|
| m 维合并 G 个 qHead | `mEff = curBq × curG` | KV 只搬运 1 次 | m 维增大 → L0A 压力 |
| 每个 qHead 独立 task | G 个 task 各自搬运同一份 KV | 实现简单 | KV 重复搬运 G 次 |

**推荐条件**:GQA → **必须** m 维合并。

> 这不是一个"优化选项"而是 **I2 不变量(正确性契约)的性能视角复述**——完整定义与 curG_max 求解见 [`base_design.md` §3.2](../subfamilies/base_design.md)。本表仅从性能角度说明"为何合并":避免 KV 重复搬运 G 次。禁止把它当可选优化跳过。

### §2.5 量化子循环分解

| 可选方案 | 说明 | 优势 | 代价 |
|---------|------|------|------|
| 单次 Mmad | 量化 K 维一次性计算 | 无子循环开销 | K 维超过单次限制 |
| K 维子循环 | 按固定粒度切分 K 维 | 满足限制 | 需子循环 + L1 双缓冲 |

**推荐条件**:量化 Mmad 的 K 维限制 < 实际 K 维 → **必须**子循环分解。

### §2.6 constexpr 级联派生

回答:**是否有从少量输入参数派生出大量编译期常量的场景?**

从 Host 传入的少量参数出发,通过 constexpr chain 派生出整个决策树。典型链路(角色名,实际变量名自定):
```
tilingKey → layout → 输入 dtype → 累加/中间 dtype → BMM2 输出位置(UB/GM) → 是否 split-D → ...
```
每一级由上一级 constexpr 推出,末级得到影响 TBuf 布局 / 分支的编译期常量。

链长度通常 3-5 层(太短无意义,太长难维护)。

**产出**:constexpr 级联链(如有)。

### §2.7 计算层自检清单 O13-O16

- [ ] **清单 O13(三维度分层完整)**:所有条件性功能已归类
- [ ] **清单 O14(Dummy Block)**:mix kernel 场景已确认是否使用 Dummy Block
- [ ] **清单 O15(GQA 合并)**:GQA 场景下 G 个 qHead 合并到 mEff
- [ ] **清单 O16(量化子循环)**:量化场景下 K 维子循环已实施(如适用)

---

## §3 流水编排与跨核同步(节点 N10)

> 微观实现层——把 N1 §1.4 流水线深度、N6 多核切分、N7 内存划分的决策落到 stage 节拍、跨核同步、cache slot、barrier 设计契约上。依赖:N1,N4,N6,N7。

### §3.1 节拍数、stage 顺序与 ring buffer 深度

回答:**已选流水级数(shape §1.4 Q1)下,loop body 内各 stage 的执行顺序如何编排?各 stage 之间用哪个同步事件 ID?ring buffer 交错多深?**

**与 I1 的关系**:同一 KV 分块内的 stage 依赖方向严格 C1→V1→C2→V2(I1 不变量,见 [`fundamentals.md` §3.1](../foundation/fundamentals.md));**不同 KV 分块如何交错**由设计决定。

**编排约束**:
- 在 loop body 内任何重排都不能让某个 stage 读到本轮尚未产生的状态
- 反排某些 stage 可避免覆盖问题——典型场景:V2 需要读上一轮的 softmax state,故 V2 应**先于** V1 执行

**同步事件 ID 取值规则**:
- 不与平台保留 ID 冲突
- 各 ID 语义独立声明,**禁止**复用同一 ID 表达不同语义

#### Stage 执行顺序(V2 vs V1 相对顺序)

| 可选方案 | 说明 | 优势 | 代价 |
|---------|------|------|------|
| 2 级正序(C1+V1 → C2+V2)| 同轮次 C1+V1 配对 | 逻辑清晰 | C2/V2 等待 |
| 3 级 V2 提前(C1→V2→V1+C2)| V2 在 V1 之前执行 | 保护 softmax state | 需要额外 slot 管理 |

**编排关键决策**:V2 是否必须先于 V1?
- **3 级流水**:V2 需要读上上轮的 softmax state,而 V1 会覆写该 state → **V2 必须在 V1 之前执行**
- **2 级流水**:V1 处理本轮 state,V2 处理 PRELOAD_N 轮前的 state,二者通过 ring buffer slot 物理隔离 → **V1 可以先于 V2 执行**

> 已落地的实例化:
> - **2 级流水**:loop body 内 `C1+V1(本轮)` 然后 `C2+V2(loop-PRELOAD_N 轮)`,V1 先于 V2 执行,通过 ring buffer slot 隔离 state
> - **3 级流水**:loop body 内 stage 顺序 `C1(本轮) → V2(上两轮) → V1(上一轮) + C2(上一轮)`,V2 提前执行确保 softmax state 不被 V1 覆盖

#### Ring Buffer 深度

回答:**跨 KV 分块的流水交错深度是多少?**

Ring buffer 深度(通常记为 `PRELOAD_N`)决定流水能"看到"多远的前/后轮次。

| 可选方案 | 说明 | 优势 | 代价 |
|---------|------|------|------|
| PRELOAD_N=1(2 slot)| 交错 1 轮 | buffer 最少 | 隐藏延迟能力有限 |
| PRELOAD_N=2(3 slot)| 交错 2 轮 | 平衡点 | 3 slot 取模有整数除法开销 |
| PRELOAD_N=3(4 slot)| 交错 3 轮 | 可用位掩码 | 多 1 slot buffer |

**决策依据**:AIC↔AIV 通信延迟、UB/GM 可用容量、跨 KV 分块数量。

> 📊 **数值标定项**:`PRELOAD_N` 的**精确取值**(1/2/3)是可后置的数值标定——设计期取默认 `PRELOAD_N=2` 并预留 buffer 余量即可,working baseline 跑起来后按 PipeUtilization profiling 在 1/2/3 间微调。但"是否启用 ring buffer 预取"这一**结构决策**必须设计期定死(**禁止**退化为逐 KV 块串行,见 [`base_design.md` §10.2 清单 10](../subfamilies/base_design.md))。

**产出**:loop body 执行顺序声明 + 各 stage 同步事件 ID 列表 + ring buffer 深度 + 顺序选择理由。

### §3.2 跨核同步粒度 + 原语

#### 同步粒度

回答:**AIC↔AIV 的同步是 per-stage、per-tile、还是 per-task?**

| 可选方案 | 含义 | 优势 | 代价 |
|---------|------|------|------|
| per-stage | 每个 stage 完成后立即通知 | 最细粒度,延迟最小 | 同步次数多 |
| per-tile | 一个 KV tile 的所有 stage 完成后通知 | 实现简单 | 对端等待整个 tile |
| per-task | 整个 task 完成后通知 | 最简单 | 完全无法流水 |

**说明**:per-stage 同步(每个 stage 完成即通知)配合流水可最小化对端等待,是 FA 4-stage 的常见选择。

#### 同步原语(架构模式 + modeId)

回答:**AIC↔AIV 跨核同步用哪种架构模式?平台 modeId 选什么?PIPE 类型选什么?**

本节涉及**两个正交决策**,设计阶段必须分别明确:

**决策 1 — 架构模式**(单/双 AIV):

| 架构模式 | 说明 |
|---------|------|
| 模式 B(双 AIV,推荐)| 双 AIV 全程参与计算,Vector 算力充分利用 |
| 模式 A(单 AIV,简化替代)| 仅 AIV0 参与,AIV1 early-return,代码更简单 |

**决策 2 — 平台 modeId**(由目标芯片决定):950PR/950DT(A5)支持 modeId 0/1/2/4,核内 AIC↔单 AIV 握手用 modeId=4;上一代(A2/A3)仅支持 0/1/2。**modeId 语义、双 AIV flagId 映射、ISASI flagId 冲突约束是实现层细节,统一见 [`implementation_ref.md` §1-§2](../implementation_ref.md)。**

**同步原语选择**:

| 可选方案 | 说明 | 优势 | 代价 |
|---------|------|------|------|
| `CrossCoreSetFlag<modeId,pipe>` / `CrossCoreWaitFlag<modeId,pipe>`(公开 ISASI API)| 精确跨核通知,fire-and-forget | 精确、开销小 | 需手动管理 flagId 与 modeId/pipe 模板参数 |
| PipeBarrier\<PIPE_ALL\> | 整核 drain | 简单粗暴 | drain 所有 pipe |

> ⚠️ **两端 modeId 必须一致**:`CrossCoreSetFlag<modeId,pipe>` 与配对的 `CrossCoreWaitFlag<modeId,pipe>` 若 modeId 不一致(如一侧 4、另一侧默认 0),flagId 空间独立 → 挂死。目标平台不支持的 modeId 不能用。省略模板参数会退回默认 `modeId=0`(多核同步),与 `modeId=4`(核内 AIC↔单 AIV)flagId 空间独立 → WaitFlag 永远等不到 → **挂死**(见 [`implementation_ref.md` §1](../implementation_ref.md))。

**判定标准**:
- `CrossCoreSetFlag/WaitFlag`:精确通知,fire-and-forget,FA 跨核握手的默认选择
- PipeBarrier<PIPE_ALL>:整核 drain,仅用于跨 pipe 无对应 HardEvent 时的兜底

**禁止**:以"PipeBarrier 是保险屏障"为由四处散布——冗余 barrier 让 Cube 流水利用率系统性下降。

**决策产出**:本节只需**声明**"架构模式 + 目标平台 modeId + 同步粒度 + 各同步点原语"。架构模式决策建议表、公开 API 实现骨架(签名、PIPE 映射表、AIV1 偏移)进入实现阶段查 [`implementation_ref.md` §3](../implementation_ref.md);实现自检清单见 [`implementation_ref.md` §5](../implementation_ref.md)。

### §3.3 cache slot 语义命名 + slot 公式分离(权威表)

> **本节是 slot 语义的唯一权威表**。shape §1.4(枚举)与 resources §1.5 Q2(workspace 段公式)均前指此处。

#### cache slot 语义命名

回答:**loop body 中 cache slot 如何命名?**

**禁止**:数字下标(`info0` / `info1` / `info2`)。

**强制**:按"该 slot 在 loop body 内**哪个节拍执行 / 处理哪一轮的任务**"**语义化**命名。

#### slot 分类标准 + 公式分离原则

**核心原则**:不同语义的 slot 必须用**不同的计数器**取模**不同的常量**,**不能共用**。

| slot 语义 | 取模计数器 | 取模常量 | 索引方式 | 典型应用 |
|---|---|---|---|---|
| **跨 stage 握手 slot** | **loop**(全局迭代器)| DB(如 2)| `loop & (DB-1)` | BMM1/BMM2 中间结果 DB / 跨核握手段 |
| **跨 task 状态 slot** | **mloop**(行计数器)| `PRELOAD_N+1`(如 3)| `mloop % (PRELOAD_N+1)` | 跨 s2 在线累积的 softmax max/sum 段 |
| **跨 loop 自读自写 slot** | **loop**(全局迭代器)| `PRELOAD_N+1`(如 3)| `loop % (PRELOAD_N+1)` | V2 跨 s2 块的 O_acc 落地段 / exp buffer |
| **task 环形缓冲区 slot** | **loop**(全局迭代器)| `CACHE_SIZE`(如 4)| `loop & (CACHE_SIZE-1)` | task 元信息 |

**公式骨架**(计数器名与上表一致:`mloop`=行计数器,`loop`=全局 task 迭代器):

```
跨 task 状态 slot:      stateSlot     = mloop % (PRELOAD_N+1)
跨 loop handshake slot: handshakeSlot = loop  & (DB-1)
跨 loop 自读自写 slot:  selfRefSlot   = loop  % (PRELOAD_N+1)
task 环形缓冲区 slot:   cacheSlot     = loop  & (CACHE_SIZE-1)
```

#### slot 寻址方式(取模 vs 位掩码)

| 可选方案 | 公式 | 优势 | 限制 |
|---------|------|------|------|
| 取模 | `slot = loop % N` | 通用 | 整数除法在热路径上有开销 |
| 位掩码 | `slot = loop & (N-1)` | 单条位运算 | 要求 N 为 2 的幂 |

> ⚠️ `mloop`(行计数器,每完成一行 M 递增)与 `loop`(全局 task 迭代器)**不可混用**。softmax max/sum 是行级状态,用 `mloop` 索引;exp buffer 是迭代级状态,用 `loop` 索引。
>
> ⚠️ `PRELOAD_N+1=3` 不是 2 的幂,**必须用 `%` 取模**,不可用 bitmask。task 环形缓冲区大小必须是 2 的幂(如 4),**用 `& MASK` 取模**以避免标量除法。当 slot 数为 2 的幂时 `loop & (N-1)` 比 `loop % N` 省一次标量除法(热路径收益);是否为凑 2 的幂多分配 1 个 slot,取决于该 buffer 的 UB 占用。

**误用症状**:
- 用 loop 计数器替代 mloop 计数器(跨 task 状态):multi-s2 块精度漂移
- 用 mloop 计数器替代 loop 计数器(跨 loop 段):s2 累积状态错乱
- 三类 slot 全部用同一常量取模:任一槽数变化时整体崩塌

**最小复现条件**:`single task + multi s2 块(s2N ≥ 3)`,或 `multi task per core + multi s2 块`。

> **禁止**:把不同语义的槽数写在同一个常量里——混用会导致 silent 数据污染或 race。

**产出**:每个 buffer 的 slot 公式声明 + 公式分离的三类列表 + slot 命名约定 + slot 寻址方式。

### §3.4 PipeBarrier 设计契约

回答:**算子内 `PipeBarrier<PIPE_ALL>` 应该出现在哪些位置?预计数量是多少?**

`PipeBarrier<PIPE_ALL>` 用于跨 pipe 整核 drain。设计阶段必须**显式列出**所有需要出现的位置(契约位置),禁止四处散布。

**FA 类的 PipeBarrier 来源**:
- 通用 Cube 同步约束
- 跨 pipe 边界无对应 HardEvent 时必须 PIPE_ALL 兜底
- 平台特定治本配方

**冗余率验收**:冗余率 `(实际 / 设计 - 1) < 30%`,达到 50% 视为性能阻塞项。**注:冗余率是实测验收指标(_governance §4.3(b)),归性能验收阶段,不在设计期判定**;设计期只产出契约位置清单 + 预计数量。

**禁止**:
- 以"PIPE_ALL 是保险屏障,跟精度无关"为由四处散布
- 同核内跨 pipe 依赖用 PipeBarrier 替代专门的 HardEvent
- 同一 pipe 内的顺序操作之间散布 barrier

**产出**:PipeBarrier 设计契约位置清单 + 预计数量。

### §3.5 Fixpipe 输出参数

回答:**Fixpipe(L0C→UB/GM)使用什么参数结构体?**

Fixpipe 参数结构体因平台而异(双目标分发 / 量化预处理 / 子块控制)。**设计阶段只需声明"是否需要双目标分发(dualDstCtl)"**;具体参数结构体(A5/DAV_3510 为 `FixpipeParamsArch3510`,见 asc-devkit `L0C到UB数据搬运（Fixpipe）.md`)与关键参数取值是实现层细节,见 [`implementation_ref.md` §4](../implementation_ref.md)。

> ⚠️ **中间段落地位置(片上 L0C→UB vs L0C→GM)是结构决策,默认优选片上 L0C→UB。** 950PR 的 Fixpipe **支持** L0C→UB 片上握手(见 [`implementation_ref.md` §3.3/§4](../implementation_ref.md) 与 [`roofline.md` §2.3](../foundation/roofline.md)),优先走片上通路以省一趟 GM 往返。调用 L0C→UB 时**必须显式构造 `isToUB=true` 的 FixpipeConfig**——默认 `CFG_ROW_MAJOR`(`isToUB=false`)指向 GM 物理地址,用它写 UB 目的地会**静默出错**(编译过、结果错),极易被误诊为"硬件不支持 L0C→UB"而错误退到全程 L0C→GM。

**产出**:是否需要 Fixpipe 双目标分发的声明。

### §3.6 Buffer 通信策略 + GM 自读自写段 + 自检清单 O1-O7

#### Buffer 通信策略

回答:**跨核传递中间矩阵时,用双缓冲还是单缓冲?**

| 可选方案 | 含义 | 适用条件 |
|---------|------|---------|
| 双缓冲(DB)| AIC 写 slot N,AIV 读 slot N-1 | 数据量大、buffer 充裕 |
| 三缓冲(3buff)| 3 slot 轮转,前向同步 | GM 段 |
| 单缓冲(SB)| AIC 写 → AIV 读同一 slot | 数据量小、buffer 紧张 |

**选择依据**:中间矩阵数据量大且 buffer 充裕时用多缓冲隐藏搬运延迟;数据量小(如量化后 dtype 减半)或 buffer 紧张时可退回单缓冲。缓冲深度与 §3.1 的 ring buffer 深度协同决定。

#### GM 自读自写段管理

回答:**V2 跨 s2 块的 O_acc 累积段如何包装?**

| 可选方案 | 说明 | 优势 | 代价 |
|---------|------|------|------|
| 封装管理类模式(推荐)| 自定义 GM 段管理类,内部维护 cross-core sync | 封装度高 | 需要初始化 |
| Queue 包装模式 | `AllocTensor → ... → FreeTensor` + MTE3_MTE2 fence | 有同步保证 | 代码量多 |
| 直接 DataCopy | 不经同步 | 代码量少 | 无同步保证 → 精度错误 |

**禁止**:直接 DataCopy 不经任何同步机制。

#### 流水层 + 同步层自检清单 O1-O7

- [ ] **清单 O1(流水编排完整性)**:流水级数、ring buffer 深度、stage 执行顺序、slot 寻址方式四项全部有选定方案和理由
- [ ] **清单 O2(V2-V1 顺序)**:3 级流水下 V2 在 V1 之前执行已确认;2 级流水下 V1/V2 的 slot 物理隔离已确认
- [ ] **清单 O3(slot 寻址)**:slot 寻址在热路径上无非必要整数除法
- [ ] **清单 O4(同步原语纪律)**:所有跨核同步点使用 `CrossCoreSetFlag/WaitFlag`(显式 modeId/pipe,两端一致),非 PipeBarrier 兜底;PipeBarrier 仅出现在契约位置
- [ ] **清单 O5(GM 自读自写管理)**:O_acc 累积段的 GM 管理方式已确定,有明确同步机制
- [ ] **清单 O6(PipeBarrier 契约位置已列举)**:设计期只需产出 PipeBarrier 契约位置清单 + 预计数量(见 §3.4);**冗余率 < 30% 是实测验收指标(_governance §4.3(b)),归性能验收阶段,不在设计期判定**
- [ ] **清单 O7(实现层校核)**:跨核同步实现细节已按 [`implementation_ref.md` §5](../implementation_ref.md) 清单校核(平台 mode / flagId / 双通知)

**产出**:每个跨核传递段的 buffer 策略 + GM 自读自写段的管理方式 + 同步机制。

---

## §4 平台基线 + 开发起点(节点 N11)

> 依赖:全部。

### §4.1 开发入口

进入实现前,确认设计方案中的关键硬件特化 API 已在架构承诺表中标记平台依赖状态(见 shape §4)。

**基线验证**(在写自定义 kernel 之前**必须**先执行):编译运行同 kernel 类型的至少一个参考示例,确认 `__mix__(N, M)` + TPipe 模式在当前 CANN 版本下可用。

> **基线示例来源**:asc-devkit **没有完整 FA 示例**,但有可用的 `__mix__(1,2)` Cube-Vector 融合示例与 softmax building block,可作为基线:
> - `examples/.../00_introduction/03_fusion_operation/matmul_leakyrelu_advanced_api/`(标准 `__mix__(1,2)` 融合)
> - `examples/.../05_best_practices/03_fusion_compute/matmul_gelu_high_performance/`(mix 融合最佳实践)
> - `examples/.../01_activation/softmaxflashv2/`(FA-2 softmax 变体 building block)
>
> 基线验证目标是确认 mix-mode 融合 + SoftmaxFlashV2 在当前环境可编译运行,不是提供完整 FA 参考。

**模块化构建与验证**:基线验证通过后,从空 Kernel 骨架开始,逐步添加 Tiling、搬运、计算等模块,每个模块编译通过并验证后再进入下一模块。

具体规则见 `workflows/development-guide.md` 对应章节。

---

## §5 优化策略产出(节点 N12)

> 依赖:全部。优化策略在 N1-N11 骨架确定后汇总。
>
> ⚠️ **"骨架确定后"指节点推导顺序,不是时间上的"以后再优化"**。设计离开本阶段时,优化策略必须已随骨架一并**锁定为设计目标**;本节记录的是"已承诺采纳的最优配置 + 逐项硬件约束下的合法退化",不是"留待实现后再补的 TODO"。任何标注为"deferred / 后续迭代"的性能项都视为 _governance §2 设计即最优门禁未通过。

结构性方案决策(流水 / 同步 / 负载 / 内存 / 计算五层)已在 _governance §4 定位、并在各节点(N6/N7/N8/N10)与正确性契约合一定死。本节汇总输出到 DESIGN.md。

### §5.1 变体特定优化策略索引(按变体选择性回答)

变体特定优化在对应 `subfamilies/` 文件的 §9 中展开:

| 变体 | `subfamilies/` §9 | 核心问题 |
|------|-----------|---------|
| FA / GQA | [`base_design.md` §9](../subfamilies/base_design.md) | gMaxPerTask 选择 |
| MLA(base 内变体分支)| [`base_design.md` §11 MLA 变体分支](../subfamilies/base_design.md) | latent absorption 流水 |
| 量化 FA | [`quantization_design.md` §9](../subfamilies/quantization_design.md) | 子循环分解 / buffer 缩减 / stride 访问 |
| 稀疏 FA | [`sparse_design.md` §9](../specialization/sparse_design.md) | sparse 剪枝 / KV chunk 跳过 |

### §5.2 DESIGN.md 优化策略产出模板

在 DESIGN.md 中按以下模板输出"优化策略"章节:

```markdown
## 优化策略

### 基本块概要(承接 shape §3 / roofline.md)
- mEff × Bk:...
- AI_HBM (= 2·mEff/sizeof(T)):...
- 性能区制:compute-bound / HBM-bound / cross-core-bound(三通道取 min)
- V1 模式:V1-fit / V1-chunk

### 流水编排(N10,execution §3.1)
- 流水级数:X 级
- Ring buffer 深度:PRELOAD_N = N
- Stage 执行顺序:
- Slot 寻址:取模 / 位掩码
- 决策理由:...

### 跨核同步(N10,execution §3.2-§3.6)
- 同步粒度:per-stage / per-tile / per-task
- 同步原语:CrossCoreSetFlag/WaitFlag(modeId/pipe)
- Buffer 策略:段 A 双缓冲 / 段 B 单缓冲
- GM 自读自写段:封装管理类 / queue 包装模式
- PipeBarrier 契约位置:...(预计 N 处)

### 负载均衡(N6,shape §5.5)
- 典型 shape 核数利用率:X%
- 均衡策略:zigzag / 动态 / sparse 剪枝
- Split-KV reduce:启用条件 / 不启用

### 编译期特化(N8,execution §2)
- 编译宏分离:...
- constexpr 固化:...
- 运行时 if:...
- Dummy Block:是 / 否
- constexpr 级联链:...

### 变体特定优化(§5.1)
- ...
```

**产出**:在 DESIGN.md 中按本模板输出"优化策略"章节。
