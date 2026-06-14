# FlashAttention 类子族扩展契约

> 本文档定义 FA 各子族在通用骨架(`overview.md` / `design.md`)之外的扩展点。新算子设计阶段确定子族后,本文档告诉你**只需要在哪些点上特化**,其余部分按通用骨架。
> **禁止**子族混用——一个算子只能属于一个子族,设计文档须显式声明。

---

## §1 子族总览与选型决策

| 子族 | 数学差异 | 适用场景 | 选型触发 |
|---|---|---|---|
| **FA / GQA**(默认)| `Hq = G × Hkv`,G ≥ 1 个 query head 共享一个 KV head | 主流推理 / 训练 | 默认路径;Sq ≥ 1 + Hq/Hkv = G ≥ 1 |
| **MHA** | `G = 1`(`Hq = Hkv`)| 经典 attention | GQA 的退化,通常作为 GQA 实现的特例 |
| **MLA** | KV 经 latent compression,`kvHeadNum = 1`,所有 KV 压缩为 latent vector | DeepSeek-V2 类模型 | KV 经 latent absorption |
| **稀疏 FA** | KV 维度有 sparse pattern(block-sparse / sliding window 等)| sparse attention 模型 | KV 维度非全连接 |
| **量化路径** | 输入 / 输出 / 累加器某项走 int8 / FP8 / FP4 | 量化推理 | post-quant / weight-quant |

**选型规则**:
- 数学公式与默认 FA-2 一致 + Hq 关 Hkv 整除 → **FA / GQA**
- 数学公式含 latent absorption 步骤 → **MLA**
- 否则按 sparse / 量化扩展点占位

> **注**:FlashDecoding(split-KV reduce)是 FA 算子的**并行实现技术**,不是子族。它可以叠加在任意子族之上,适用于 Sq=1 + 大 Sk + 核数富余场景。详见 [`design.md` §2.5 并行策略选择](./design.md)。

---

## §2 扩展点 capability-map

为了让上层在不同子族间复用同一份 Tiling 决策框架,本节定义 5 个扩展点。每个子族在这 5 个点上特化,通用骨架(`design.md` 各章)对所有子族一致。

### §2.1 ConstInfo / RunInfo 字段

| 字段类别 | 说明 |
|---|---|
| **ConstInfo 通用字段**(所有子族读)| `batchSize` / `qHeadNum` / `kvHeadNum` / `qSeqSize` / `kvSeqSize` / `headDim` / dtype / scale 等 |
| **ConstInfo 子族特定字段** | 仅对应子族读;不读的子族字段保持默认值。例如 GQA 子族需要 group 合并相关字段、MLA 子族需要 latent 维度相关字段(具体字段名见各子族参考实现)|
| **RunInfo 通用字段(per-task)**| 包含以下职责类别的字段(具体命名与字段数由实现决定):<br>· **任务标识**:batch / kvHead / Sq 块 / G 块等维度索引<br>· **块内偏移与实际尺寸**:`actS1Size` 类(实际 Sq 维处理量)/ `actS2Size` 类(实际 Sk 维处理量)/ m 维有效维 / m 维对齐维<br>· **边界标志**:跨 batch 重置、跨 s2 块首尾标志 等 |
| **子族特定 task 结构** | FA / GQA / MHA 复用 RunInfo 通用结构;MLA 复用 + 加 latent 维元数据字段 |

**设计原则**:
- ConstInfo **单一来源**——所有子族字段并集存于同一结构体,子族不消费的字段保默认即可,避免多结构维护
- 新增子族时优先扩 ConstInfo + 复用 RunInfo;仅当 task 维度本身不同时才另起 TaskInfo(典型见 design.md §2.5 描述的 split-KV reduce 并行模式)

### §2.2 Cube 端扩展点

| 扩展点 | 通用骨架 | 子族特化 |
|---|---|---|
| **是否走 Cube** | FA / GQA / MHA / MLA 走 Cube(Q·K^T 与 P·V 都是 GEMM)| 启用 split-KV reduce 并行模式(design.md §2.5)时 partial 计算与 combine 可能由 Vector 直接做,不走 Cube |
| **L1 rotation 深度** | task 级双缓冲(QP 与 KV 各 N 槽)| 因子族而异(见下表)|
| **L0 K-axis 分块** | 全子族通用 K-axis 分块(`design.md §2.3`)| 子族不变 |
| **L1 M / N 轴分块** | 通用骨架,触发条件按容量校验(`design.md §4.2 Q2`)| 子族不变 |

**L1 rotation 深度子族对照**:

| 子族 | L1 QP 槽数 | L1 KV 槽数 | 依据 |
|---|---|---|---|
| FA / GQA 推理(默认)| 标准 task 级双缓冲 | 标准 task 级双缓冲 | GQA 中 K/V 在同 kvHead 内复用,标准深度即够 |
| GQA prefill(高吞吐)| 同推理 | 推理深度 + 1 段预取(可选)| 长 KV 场景预取 1 个 chunk 提升 mte2 利用率 |
| MLA | 较深(QP 多槽)| 较深(KV 多槽)| Latent absorption 需保留更长 K_compressed 历史以做多轮 Mmad |

**禁止**:把跨核握手槽(Cube↔Vector 跨核固定深度,见 `design.md §2.4`)和 L1 rotation 槽(子族特定)写在同一常量。两者语义不同——前者是 Cube↔Vector 握手深度,后者是 task 间 KV 预取深度。

### §2.3 Vector 端扩展点

| 扩展点 | 通用骨架 | 子族特化 |
|---|---|---|
| **Softmax 变体** | `SoftmaxFlashV2` 类 API(支持是否更新状态 / 是否保留输出 / 广播 flag 等模板参数)| FA / GQA / MHA 标准 SoftmaxFlashV2;MLA 启用广播相关 flag 配合 latent。启用 split-KV reduce 并行模式(design.md §2.5)时,每核处理自己的 s2 切片,不走 stateful 累积,改为每切片独立计算后再 cross-core combine |
| **计算链插入点** | C1 → V1(softmax)→ C2 → V2(streaming 归一化)| MLA 在 V1 之后插入 latent absorption 两步计算链(具体函数名见参考实现)|

### §2.4 Workspace 扩展点

| 扩展点 | 通用骨架 | 子族特化 |
|---|---|---|
| **标准段** | 跨核握手段(每个 Cube↔Vector stage 之间 1 段)+ 自读自写段(若 V2 跨 KV 分块在线累积需要 GM 落地)+ 任务级状态段(若需要跨 task 持有 stateful 状态)| FA / GQA / MHA 默认 task 级并行下走以上标准段组合(具体段名见各子族参考实现)|
| **子族新增段** | — | MLA 增 int32 中间段(latent absorption 的 fp32 精度不足)。启用 split-KV reduce 并行模式(design.md §2.5)时另增 partial reduce 段(partial O / partial logsumexp max / partial logsumexp sum 各一段)|

按子族在 `design.md §4.4` 内列段时,直接按本表扩。

### §2.5 Service 类划分约定

**Service 类组织约定**:

- 按 Cube 计算 / Vector 计算两条线**独立成 service 头**(Cube 端服务 + Vector 端服务)
- 每个 service 头封装该端的 stage 实现(Cube 服务封装 ComputeMm1 / ComputeMm2 等 GEMM stage;Vector 服务封装 ComputeVec1 / ComputeVec2 等 softmax + accumulate stage)
- **子族独立命名**:MLA / 其他显著扩展的子族**必须独立命名 service 头**,**禁止**复用默认 FA / GQA service 头

| 子族 | service 类入口 | 关键计算链 |
|---|---|---|
| FA / GQA / MHA | Cube 服务 + Vector 服务(默认命名)| 标准 4 段 Q·K^T → Softmax → P·V → rescale |
| MLA | 独立 Cube 服务 + 独立 Vector 服务(与默认分离)| 标准 4 段 + latent absorption 两步链 |
| FlashDecoding 并行模式 | 仅 Vector 服务(无 Cube)| partial reduce + cross-block combine(无 Cube)|

具体文件命名、类名、函数名见各子族参考实现。

---

## §3 各子族扩展契约

### §3.1 FA / GQA(默认)

#### 数学差异

`Hq = G × Hkv`,同一 kvHead 下的 G 个 qHead 共享 KV。

#### 任务维度

`taskIdx → (batch, kvHead, gS1Block, gBlock)`。

#### 关键扩展契约

- **`gMaxPerTask`**:每个 task 合并的 qHead 数,`≤ G = Hq / Hkv`
- **`mEff = curBq × curG`**:task 内有效 m 维(Bq 是 Sq 维基本块,curG 是当 task 合并的 qHead 数)
- **mEff 行排布约定**(**Bq-major**):`row(g, i) = g × Bq + i`,即按 qHead 外、Sq 内排布

  **理由**:
  - Causal mask 模板可一份 + 整体平移 curG 次,不同 g 之间 mask 模式相同
  - Q 的 Nd2Nz BSND→NZ 自然连续
  - V2 输出按 (g, i) 二维循环逐行写回(见下条)

- **GQA 输出 BSND 写回必须逐行 DataCopy**:Q / Out BSND layout `[B, Sq, Hq, D]` 中同 qHead 内相邻 Sq 行间隔 `Hq × D`,**Bq 行不连续**。若用批量 `DataCopy(GM, UB, curBq × D)` 会破坏邻 qHead 输出。正确做法:按 (g, i) 二维循环,每行单独 `DataCopyPad(outGm[gOff + i × qRowStride], outUb[(g × Bq + i) × D_align], {1, D, ...})`

- **mPad > 16 + 多 s2 块组合**:GQA 维度首次打开(`mPad > 16` 即 `mEff > 16`)时,必须额外验证三类用例:
  - (a) 单 s2 块 + mPad > 16
  - (b) 多 s2 块 + mPad = 16
  - (c) 两者同时打开

  实测教训:(a)(b) PASS 不等价于 (c) PASS;(c) 路径在 P·V GEMM 上易出现精度漂移,需单独验证。

- **M 轴尾块 padding 行**:`mPad > mEff` 时 padding 行的 mask / softmax 必须明确处理(padding 行 mask 写 -INF 或在 Mul/Add 中跳过);`SoftMaxShapeInfo.srcM = mEff`(不是 mPad),确保 padding 不参与归约

#### 自检清单(进入审查前必跑)

- [ ] **清单 4(M 轴尾块 padding 行)**:padding 行 mask / softmax 处理已明确;`SoftMaxShapeInfo.srcM = mEff`;mEff 行排布 Bq-major
- [ ] **清单 5(causal mask 边界)**:`Duplicate` 起址 32B 对齐;尾块 padding 行 mask 处理;GQA group 复制(curG 份 mask 模式相同 + 整体平移)
- [ ] **清单 7(输出写回 BSND)**:按 (g, i) 二维循环逐行 `DataCopyPad`,不做批量 `DataCopy(GM, UB, curBq × D)`
- [ ] **清单 8(mPad > 16 + 多 s2 块)**:三类用例(a/b/c)已在精度回归中覆盖

### §3.2 MHA(G = 1)

#### 数学差异

`Hq = Hkv`(G = 1),即 GQA 的退化。

#### 实现契约

通常作为 **GQA 的特例**实现 —— 只需把 GQA 的 `gMaxPerTask = 1` 即可,无需独立 service 类。

#### 选型说明

新算子开发时:
- 仅支持 MHA 形态 → 仍走 GQA 骨架(GQA 实现已自然兼容 G=1),不另起 MHA 骨架
- 既支持 MHA 又支持 GQA → 同一份代码(GQA 路径),通过 Tiling 决策的 `gMaxPerTask` 区分

### §3.3 MLA(latent absorption)

#### 数学差异

KV 经 latent compression:K / V 压缩为 1 个 latent vector,`kvHeadNum = 1` 强制。

#### 扩展契约

- **ConstInfo 加字段**:latent 维度相关字段(具体字段名见参考实现)
- **强制 `kvHeadNum = 1`**:所有 KV 压缩为 latent
- **Cube 端 L1 rotation 加深**:见 §2.2 表(latent absorption 需保留更长 K_compressed 历史)
- **Vector 端计算链插入**:V1 之后加 latent absorption 两步计算链(具体函数名见参考实现)
- **Workspace 加 int32 段**:latent absorption 的 fp32 精度不足,需 int32 中间段
- **SoftmaxFlashV2 加广播相关 flag**(配合 latent 维度)
- **Service 类独立命名**:Cube / Vector 服务头**必须独立命名**,**禁止**复用默认 FA / GQA 的 service 头

#### 自检清单

继承 §3.1 GQA 全部清单 + MLA 特有的 latent absorption 数值校验(具体由实现层定义,本文档暂占位)。

### §3.4 稀疏 FA(扩展点占位)

#### 数学差异

KV 维度有 sparse pattern(block-sparse / sliding window / sink / 自定义 mask)。

#### 扩展契约(暂列扩展点,具体由首个落地实现细化)

- ConstInfo 加 sparse pattern 描述字段
- s2 loop 可能需要按 sparse mask 跳过部分 KV chunk
- Workspace 是否需要额外段取决于 pattern 是否需要预计算 sparse 索引
- 是否走 Cube 取决于 sparse pattern 粒度(block-sparse 可走 Cube,token-level sparse 可能需要 gather 后走 Vector)

#### 蒸馏纪律

本子族**仅占位**。落地第一个稀疏 FA 算子时再细化扩展契约——跨算子可复用前不固化具体规则。

### §3.5 量化 FA 子族(扩展契约骨架)

> 本子族的具体实现 HOW(字段公式 / API 数值参数 / cast 数值实现 / scale layout)由落地参考实现承接,本节只列**必须回答的问题**。
>
> 本骨架在首个量化 FA 算子(GQA × mxfp8 推理,A5 平台)落地过程中形成。具体 HOW(字段公式 / cast 数值 / scale layout / Mmad 模板)由各落地算子的 DESIGN 文档承接,本节不固化。

#### 数学差异

输入 / 输出 / 累加器某项走量化 dtype(int8 / FP8 / FP4 / MX 类格式 mxfp8 / mxfp4),计算流引入额外的 scale 加载、量化 cast、scale 生成等步骤,但 FA-2 数学骨架不变。

#### 选型触发

`dtype ∈ 量化集合` 即触发本子族;**与并行策略(默认 task 级 / split-KV reduce)和上层子族(GQA / MHA / MLA / 稀疏)正交叠加**,例如"量化 GQA"、"量化 MLA"是合法组合。

#### 受约束的不变量

- [`overview.md §3.5 I5`](./overview.md):MX 类 scale 必须沿 K-mmad 轴。设计阶段必须显式校核 Q/K/V/P 各 scale 张量的轴对齐。
- 其他 I1-I4 不变量正常继承。

#### 5 扩展点上必须回答的问题

##### ConstInfo / RunInfo 扩展(承接 §2.1)

设计阶段必须显式产出:

- 各 scale 张量(Q / K / V / P)沿哪个轴量化?layout 维度顺序是什么?(I5 校核)
- scale dtype 是什么?(典型 MX 类用 E8M0,其他量化方案可能不同)
- 是否新增 scale 相关 Tiling 派生字段供 kernel 引用?

##### Cube 端扩展(承接 §2.2)

设计阶段必须显式产出:

- Mmad 模板形态:走"同型 fm/filter dtype + 内置 ScaleA/ScaleB"路径还是其他?
- 是否需要 LoadData2DMx 系列 API 加载数据 + scale?(具体字段单位、对齐约束见 `/ascendc-api-best-practices`)
- K_BASE 与所选 LoadData API 的 yStep 单位整除性是否满足?

##### Vector 端扩展(承接 §2.3)

设计阶段必须显式产出:

- V1 末块的 P 量化路径如何设计?P 的 scale 沿哪个轴(应沿 PV Mmad 的 K-mmad = S_k)?
- V2 末块的"fp32 → 量化 dtype cast + scale 生成"两步链结构是什么?有几步、各步输入输出 dtype?
- amax 归约的精度路径?(为避免 cast 溢出量化 dtype max,数值公式如何保证;具体 cast 数值公式见 `/ascendc-api-best-practices` 量化精度章节)

##### Workspace 扩展(承接 §2.4)

设计阶段必须显式产出:

- 是否新增 scale staging segment?
- 容量公式是什么?(按运行时 Sk 动态分配,不静态预留 Sk_max)

##### Service 类扩展(承接 §2.5)

设计阶段必须显式产出:

- 量化子族是否独立命名 service 类?还是复用上层子族(GQA / MHA / MLA)的 service?
- (默认复用,除非量化 cast + scale 生成在 stage 流水中占比足以拆出独立服务)

#### 自检清单(进入审查前必跑)

- [ ] **清单 Q1(I5 落地校核)**:所有 scale 张量的内存连续轴是否对齐其消费 matmul 的 K-mmad 方向?V scale 沿 S_k 还是沿 D 已显式声明?
- [ ] **清单 Q2(数据载体 vs scale 载体公式分离)**:数据载体的 L1 head offset 公式与 scale 载体的 L1 head offset 公式是否分别独立推导?两者公式系数是否可能不同?
- [ ] **清单 Q3(V2 末块量化路径对齐链)**:V2 末块量化路径中各 API(DataCopy 类 / Reduce 类 / Cast 类)的对齐要求是否被显式校核?(具体对齐要求查 `/ascendc-api-best-practices`)
- [ ] **清单 Q4(cast 数值不溢出)**:V2 末块 cast 数值公式是否能保证 amax / scale 不会超量化 dtype max?
- [ ] **清单 Q5(LoadData 字段单位匹配 dtype)**:LoadData2DMx 系列 API 的 kStep / kStartPosition 等字段单位是否与所选量化 dtype 匹配?

#### 蒸馏纪律

本骨架基于首个量化 FA 算子落地形成。具体阈值、API 数值参数、cast 实现细节 → 各算子自己的 DESIGN.md。等第 2、第 3 个量化 FA 落地后,如果在 5 扩展点上出现稳定共性,可考虑把骨架升级为完整契约。

---

## §4 dtype 路径契约

dtype 是 FA 类的一个独立维度,各子族都要面对。本节定义 dtype 路径的扩展契约。

### §4.1 fp16

标准路径,所有子族默认支持。

- Mmad 模板:`Mmad<float, half, half>`(fp16 × fp16 → fp32 accumulator)
- V1 / V2 输出到 GM 的中间段与 KV_T 同型 = half
- 所有 Cast 仅在 UB 上执行

### §4.2 bf16(Cube 直通 bf16)

**关键契约**:bf16 路径**与 fp16 路径同型**——V 全程 bf16,**无任何 Cast / pre-cast / Reinterpret 中转**。

- **V 全程 bf16**:GM(bf16)→ Nd2Nz → L1(bf16)→ LoadData2D `ifTranspose=true` → L0B(bf16)→ `Mmad<float, bfloat16_t, bfloat16_t>`(bf16 × bf16 → fp32 accumulator)
- **V1 输出到 GM 的中间段 dtype = bf16**(与 KV_T 同型):V1 末尾 P 经 `Cast<bf16, float>` 写回 GM;**禁止**把 P Cast 到 half 让该段走 half(偏离官方实现,会引入精度退化)
- **Mmad 模板同型约束**:Mmad fm 与 filter 必须同 dtype(API 文档 dtype 矩阵硬约束);bf16 路径 fm / filter 均 bf16,fp32 accumulator
- **Fixpipe L0C(fp32)→ Cube↔Vector 中间段(fp32)**,V2 末块 `Cast<bf16, float>` → outGm

#### 禁止反模式

- **AIV pre-cast 到 half workspace**:增加 AIV Cast 开销 + 额外 GM workspace + cross-core sync 一次,无任何收益,偏离官方生产实现
- **L1 内 Cast<half, bfloat16_t>**:Cast 仅在 UB 上执行,不能在 L1 执行
- **Host CPU bf16 → half 预转换**:违反 `development-guide §1.4 API 使用规则`("Host 侧禁止计算型操作"小节)
- **Kernel 内逐 s2 块 bf16 → half Cast + 额外 workspace 轮转**:既触发 BSND flat DataCopy 不感知 stride 的 race,又因逐 s2 Cast 链经 rescale + accumulate 放大累积误差
- **V1 末尾把 P Cast 到 half 让中间段走 half**:偏离官方实现,V1 输出中间段必须与 KV_T 同型 = bf16

#### dtype 编译错诊断

`P_T = half` + `KV_T = bf16` 或 `P_T = bf16` + `KV_T = half` 同型约束未满足直接喂 Mmad → 编译失败 → 错误归因为"DAV-XXXX builtin 不接受 bf16 ptr"是反模式。**真根因**:Mmad 模板参数同型约束未满足(API 文档 dtype 矩阵硬约束),改 fm/filter 为同 dtype 即可。

#### 自检清单

- [ ] **清单 6(bf16 V Cube 直通)**:V 全程 bf16,无任何 Cast / pre-cast / Reinterpret / 额外 half workspace 中转;V1 输出中间段 dtype = bf16(与 KV_T 同型);Mmad 模板 fm/filter 同型 = bf16;Fixpipe L0C(fp32)→ Cube↔Vector 中间段(fp32);V2 末块 Cast<bf16, float> → outGm;**未引入** 额外 half workspace / AIV pre-cast / kernel 内逐 s2 bf16↔half Cast 等反模式

### §4.3 量化 dtype 路径(必须产出清单)

> 与 §4.1 fp16 / §4.2 bf16 同结构组织。具体 dtype 字节宽、cast 公式数值、layout 维度顺序等 HOW 由各算子自己的 DESIGN.md 承接,通用的 API 用法见 `/ascendc-api-best-practices`。

#### 必须产出的设计决策

- Mmad 模板形态(同型链:fm / filter / accumulator dtype 关系 + 是否内置 ScaleA/ScaleB)
- 中间段 dtype 选择(归约状态保 fp32 还是其他)
- V1 末块 P 量化路径(P_dtype / P_scale 沿什么轴 / cast 数值公式)
- V2 末块 cast 链结构(几步、各步输入输出 dtype)
- scale 路径(GM layout / L1 layout / 用什么 LoadData API)

#### 必须回答的禁止反模式问题

- scale 张量内存连续轴是否对齐 K-mmad?(I5 校核;**典型踩坑**:V scale 错放在 D 上,不兼容 PV Mmad)
- cast 数值公式是否能保证 amax / scale 不会超量化 dtype max?(典型踩坑:cast 溢出 → RINT 产出 NaN)
- LoadData 字段单位是否随 dtype 改变?(典型踩坑:照搬 fp16 路径的 CUBE_BLOCK 单位)
- 数据载体 vs scale 载体的 multi-head head offset 公式是否系数分离?

---

## §5 子族间禁止事项

### §5.1 子族混用禁止

同一算子内**禁止混用子族**:不能"主要是 GQA 但在某些路径上走 MLA latent absorption"。

设计阶段 DESIGN.md §架构选型必须显式声明子族(默认 FA/GQA、或 MHA、或 MLA、或稀疏、或量化),后续 Tiling 决策 / Workspace 公式 / Service 类划分按所选子族对应表实施。并行策略(默认 / split-KV reduce)是独立维度,与子族正交,详见 [`design.md` §2.5](./design.md)。

### §5.2 常量语义分离禁止

以下三类槽数**必须用不同常量定义**,**禁止**共用一个常量:

| 槽数 | 语义 | 决策位置 |
|---|---|---|
| 跨核握手槽 | Cube↔Vector 跨核握手 buffer 的轮转(典型固定值)| `design.md §2.4` |
| L1 rotation 槽 | 任务级 KV / Q 预取深度(子族特定)| 本文件 §2.2 |
| V2 自读自写槽 | V2 阶段在三级流水中同时活跃的 task 数(典型 > 跨核握手槽)| `design.md §2.4` + `§4.4 Q2` |

混用同一常量会导致 silent 数据污染或 race(O_acc 被覆写 / KV 预取阻断 / Cube↔Vector 握手错位)。

