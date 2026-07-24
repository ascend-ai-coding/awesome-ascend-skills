# FlashAttention FA / GQA / MHA / MLA 扩展契约

> 本文档定义 FA / GQA(默认)、MHA(G=1 退化)及 MLA(kvHeadNum=1 + latent absorption 变体)形态的扩展契约。
> **MLA 不是独立子族**——它是基础 FA 的一种 head / hidden-dim 变体(kvHeadNum=1 + latent 维压缩),与 GQA / MHA 共享同一 service 骨架,仅以变体分支承接 latent absorption 计算链(见 §11)。
> **通用设计流程** → [`design/`](../design/_governance.md)(节点 G→N12);**扩展点框架** → [`extension_points.md`](../foundation/extension_points.md);**选型入口** → [`patterns.md`](../patterns.md)。

---

## §1 宏观决策特化

### §1.1 Kernel 类型

FA / GQA 标准选择:`__mix__(1, 2)`(1 AIC + 2 AIV)。

**MHA 退化**:MHA 与 GQA 走同一 kernel,仅 `gMaxPerTask = 1`,无需独立 target。

### §1.2 UB 模式

GQA 默认走 streaming UB 模式(O_acc 不常驻 UB,见 design/shape.md §1.2)。

### §1.3 Cube 分块层次

GQA 的 Cube 分块层次与通用骨架一致(L0 K-axis / L1 M-axis / L1 N-axis),无特化。

### §1.4 流水级数

GQA 常见 **2 级流水**(C1+V1 配对 → C2+V2 延后)。参见 [`design/shape.md` §1.4](../design/shape.md)。

---

## §2 Roofline 特化

### §2.1 GQA 的 mEff 公式

FA 类 Mmad 的有效 m 维度不是 Bq 本身,而是:
```
mEff = Bq × curG
```

其中 curG 是同一 kvHead 下合并到一个 task 的 qHead 数(≤ G = Hq/Hkv)。

- **Prefill**(Sq 较大):Bq 本身足够大,mEff 主要受 L0A 容量约束,curG 可能 < G
- **Decode**(Sq=1):Bq=1,mEff = curG,应最大化 curG = G,此时 AI ∝ G

### §2.2 AI 公式修正

代入 GQA 的 mEff 后,主 Roofline 的 `AI_HBM = 2·mEff/sizeof(T) ∝ mEff`(见 [`roofline.md` §2.2](../foundation/roofline.md))保持不变,但 mEff 可行域受 curG ≤ G 约束:

- Prefill:mEff_max 受 L0A 容量约束
- Decode:mEff_max = G(受 Hq/Hkv 约束),故 `AI_HBM = 2G/sizeof(T)`

### §2.3 Decode 场景的优化杠杆

Decode(Sq=1)场景下 AI ∝ G,GQA group 合并是提升 AI 的**唯一杠杆**。若 G=1(MHA),AI 天花板较低,通常落在 HBM-bound 或 cross-core-bound,必须依赖流水隐藏延迟(见 [`roofline.md` §4](../foundation/roofline.md))。

---

## §3 多核特化

### §3.1 任务维度构造

GQA 默认 task 级并行下,任务空间沿 `(batch, kvHead, Sq 分块, G 分块)` 四个轴展开。**这四个轴是通用的;把它们映射到 taskIdx 有多种等价实现**——下面的闭式 decode 是一种常见构造,另一种是 metadata 驱动的 range-walker(为每个 axis 记 start/end,用 while 派发遍历),二者任务空间等价,选择取决于尾块处理与代码风格。

一种常见闭式构造:
```
taskIdx → (batch, kvHead, gS1Block, gBlock)
```

其中(角色名,实现命名自定):
- `batch`:batch 维索引
- `kvHead`:kvHead 维索引(同一 kvHead 下的 G 个 qHead 在同一 task 内合并)
- `gS1Block`:Sq 维分块索引(Bq 大小)
- `gBlock`:G 维分块索引(curG 大小)

该构造下任务总数:
```
totalTasks = B × Hkv × ceil(Sq/Bq) × ceil(G/curG)
```

无论用闭式还是 range-walker,I2(同 kvHead 同任务)与 I4(s2 不进任务维)都必须满足(见 §3.2)。

### §3.2 I2 不变量校核

GQA 的 task 维度构造必须让同一 kvHead 下的 G 个 qHead 在同一 task 内合并到 mEff。禁止把 G 个 qHead 拆到 G 个独立任务(违反 I2 → KV 重复搬运 G 次)。

> ⚠️ **UB/L0A 溢出时应缩小 curG,不是退到 gMaxPerTask=1**。合并整组(curG=G)可能使 mEff=Bq×G 溢出 UB 或 L0A。此时正确做法是**求解最大合法 curG**,而非把 curG 直接砍到 1(那等于放弃 I2,KV 被重搬 G 次)。联合可行域(两个约束同时满足):
> - **UB 约束**:并发存活 buffer 峰值(含双缓冲系数)≤ UB 容量。UB 峰值 ∝ mEff = curBq × curG。
> - **L0A 约束**:`mEff × K_BASE × sizeof(fm_dtype) ≤ L0A 容量`(见 [`design/resources.md` §1.4](../design/resources.md))。
>
> 解法:先按 §2.1 定 Bq(prefill 下 Bq 已较大),再取 `curG_max = min(G, ⌊UB 上限 / (Bq 相关项)⌋, ⌊L0A 上限 / (Bq×K_BASE×sizeof)⌋)`。若 curG_max ≥ 1 仍装不下,再考虑下调 Bq,**gMaxPerTask=1 仅当 curG_max=1 时才是被迫的合法退化**(等价 MHA,见 §1.1),不应作为回避求解的默认。

### §3.3 Split-KV reduce 启用条件

GQA 在 Sq=1 纯 decode 场景下,若 `totalTasks << usedCoreNum`,可启用 split-KV reduce(FlashDecoding 技术)。详见 design/shape.md §2。

---

## §4 内存特化

### §4.1 L1 Q/K/V/P 端口分配

| 张量 | L1 端口 | 理由 |
|------|---------|------|
| Q | A1 | 进 L0A 做 fm |
| P | A1 | 进 L0A 做 fm |
| K | B1 | 进 L0B 做 filter |
| V | B1 | 进 L0B 做 filter |

### §4.2 L1 rotation 深度

GQA 推理默认:
- QP:标准 task 级双缓冲(2 slot)
- KV:标准 task 级双缓冲(2 slot)

GQA prefill(高吞吐):KV 可选 +1 段预取(3 slot)。

详见 [`extension_points.md` §1.2](../foundation/extension_points.md)。

### §4.3 Workspace 段

GQA 默认 task 级并行下:
- **跨核握手段**:BMM1 中间矩阵(`[Bq, Bk]` S 矩阵)+ BMM1 中间矩阵(`[Bq, Bk]` P 矩阵)+ BMM2 中间矩阵(`[Bq, D]` PV 矩阵)
- **跨 loop 自读自写段**:V2 跨 s2 块的 O_acc 落地段(streaming UB 模式下存 GM)
- **跨 task 状态段**:softmax max / sum / exp(默认 task 级模式下 stateful 持有)

> ⚠️ **中间矩阵写 workspace 的行 stride 必须按下游读回 API 的 dtype 对齐粒度取整,不能按逻辑块宽紧打包**。上述握手段(S/P/PV)由 Cube 写、Vector 读回。写侧(Fixpipe/DataCopy)的行 stride 若按逻辑块宽(如 curBk)紧打包,当块宽非下游读回 dtype 的对齐倍数时,读侧 DataCopy(UB↔GM 要求 32B 对齐的行偏移与元素数,fp32 即 8 元素、bf16 即 16 元素)会**逐行静默错位/污染**(编译过、结果错)。规则:行 stride 取 `ceil(块宽 / 读回对齐粒度) × 对齐粒度`(如 fp32 读回时按 8 元素对齐、bf16 按 16),尾块同理。块宽恰为对齐倍数时侥幸正确,故须按最坏尾块规划,不能靠主 case 整除侥幸通过。

---

## §5 编译宏特化

### §5.1 GQA 常见编译宏分离

| 维度 | 是否分离 | 理由 |
|------|---------|------|
| `dtype`(fp16/bf16)| ✅ 必须 | 影响 typedef 与 Mmad 模板参数同型链 |
| `causal` on/off | ✅ 必须 | 影响 causal mask TBuf 是否声明 |
| `G` 值(GQA group 大小)| ❌ 通常不分离 | 通过 Tiling 决策的 `gMaxPerTask` 运行时适配 |
| `Sq` 值(prefill/decode)| ❌ 通常不分离 | 通过 chunkRows 运行时自适应 |

### §5.2 constexpr 级联链

GQA 常见 constexpr 级联(角色名,实际变量名自定;完整框架见 [`design/execution.md` §2](../design/execution.md)):
```
tilingKey → layout(BSND/BNSD)→ 输入 dtype → 累加/中间 dtype → BMM2 输出位置 → 是否 split-D
```

---

## §6 流水线特化

### §6.1 GQA 流水 stage 编排

**2 级流水(生产常见)**:
```
loop body:
  C1(本轮) + V1(本轮)      # 同轮次配对
  C2(loop-PRELOAD_N 轮) + V2(loop-PRELOAD_N 轮)  # 延后轮次
```

**3 级流水(可选)**:
```
loop body:
  C1(本轮)
  V2(上两轮)              # 提前执行保护 softmax state
  V1(上一轮) + C2(上一轮)
```

详见 [`design/execution.md` §3.1](../design/execution.md)。

### §6.2 GQA slot 分类

> ⚠️ **下表为某 2 级流水实例的槽数配置,非规范值**。槽数(DB / PRELOAD_N+1 / CACHE_SIZE)由各自的并行度需求推导,随流水级数、AIC↔AIV 延迟、UB/GM 余量变化——决策方法与可选值(PRELOAD_N=1/2/3 的权衡)见 [`design/execution.md` §3.1](../design/execution.md)。此处固定值仅示意**计数器与取模常量如何绑定到 slot 语义**(§7.4 的分离原则),不表示这些数字是唯一正确取值。常量命名由实现自定。

| 槽语义 | 槽数(示例)| 取模计数器 | 取模常量(示例)|
|---|---|---|---|
| 跨 stage 握手槽 | 2 | loop 计数器 | DB=2 |
| 跨 task 状态槽 | 3 | mloop | PRELOAD_N+1=3 |
| 跨 loop 自读自写槽 | 3 | loop | PRELOAD_N+1=3 |
| L1 rotation 槽 | 2 | — | 变体特定 |
| task 环形缓冲区槽 | 4 | loop | CACHE_SIZE=4 |

---

## §7 Host Tiling 特化

> **命名约定**:本节所有字段名(`gMaxPerTask` / `curG` / `gIdx` / `gOff` 等)为方法论**角色名**,承 [`fundamentals.md` §0](../foundation/fundamentals.md);它们承载"该字段决策什么",**实际变量名由实现自定**(经典实现可能作 `gSize` / `gS1O` 等)。字段的**职责类别**是通用契约,命名不是。

### §7.1 ConstInfo 字段扩展

GQA 形态特定的 ConstInfo 字段:
- `gMaxPerTask`:每个 task 合并的 qHead 数上限(≤ G)
- `curG`:实际合并的 qHead 数(运行时由 Tiling 函数决定)

### §7.2 RunInfo 字段扩展

GQA 的 RunInfo 字段(在通用结构基础上):
- `gIdx`:当前 task 在 G 维的索引
- `gOff`:当前 task 对应的 qHead 起始偏移
- `curBq`:当前 task 的实际 Sq 维处理量(尾块可能 < Bq)
- `curG`:当前 task 实际合并的 qHead 数(尾块可能 < gMaxPerTask)

### §7.3 TilingData 字段扩展

GQA 的 TilingData 字段(在通用结构基础上):
- `gMaxPerTask`:每个 task 合并的 qHead 数上限
- `s1Blocks`:Sq 维分块数 = ceil(Sq / Bq)
- `gBlocks`:G 维分块数 = ceil(G / gMaxPerTask)
- `oAccInUb`(**结构决策字段**):O_acc 是否常驻 UB(1)还是 GM streaming(0)。由 Host Tiling 按 [`design/resources.md` §1.2](../design/resources.md) 的推荐条件(`mPad × D_align × sizeof > UB 余量` → streaming)定死并写入;kernel **必须**读此字段选择 O_acc 布局,不得自行硬编码。设计期定死的**结构**决策(非 `chunkRows` 一类数值标定项)都遵此纪律——落成字段、被 kernel 读取。

---

## §8 dtype 路径

### §8.1 fp16(标准路径)

- Mmad 模板:`Mmad<float, half, half>`(fp16 × fp16 → fp32 accumulator)
- V1 / V2 输出到 GM 的中间段与 KV_T 同型 = half
- 所有 Cast 仅在 UB 上执行

### §8.2 bf16(Cube 直通 bf16)

**关键契约**:bf16 路径**与 fp16 路径同型**——V 全程 bf16,**无任何 Cast / pre-cast / Reinterpret 中转**。

- **V 全程 bf16**:GM(bf16)→ Nd2Nz → L1(bf16)→ `LoadDataWithTranspose`(L1→L0B,bf16 转置)→ L0B(bf16)→ `Mmad<float, bfloat16_t, bfloat16_t>`
- **V1 输出到 GM 的中间段 dtype = bf16**(与 KV_T 同型):V1 末尾 P 经 `Cast<bf16, float>` 写回 GM
- **Mmad 模板同型约束**:fm / filter 均 bf16,fp32 accumulator
- **Fixpipe L0C(fp32)→ Cube↔Vector 中间段(fp32)**,V2 末块 `Cast<bf16, float>` → outGm

#### 禁止反模式

- **AIV pre-cast 到 half workspace**:增加 AIV Cast 开销 + 额外 GM workspace + cross-core sync 一次,无任何收益
- **L1 内 Cast<half, bfloat16_t>**:Cast 仅在 UB 上执行
- **Host CPU bf16 → half 预转换**:违反 `development-guide §1.4`
- **Kernel 内逐 s2 块 bf16 → half Cast + 额外 workspace 轮转**:触发 BSND flat DataCopy 不感知 stride 的 race
- **V1 末尾把 P Cast 到 half 让中间段走 half**:V1 输出中间段必须与 KV_T 同型 = bf16

#### dtype 编译错诊断

`P_T = half` + `KV_T = bf16` 或 `P_T = bf16` + `KV_T = half` 同型约束未满足直接喂 Mmad → 编译失败。**真根因**:Mmad 模板参数同型约束未满足,改 fm/filter 为同 dtype 即可。

### §8.3 GQA BSND 输出写回特化

**GQA 输出 BSND 写回必须逐行 DataCopy**:

Q / Out BSND layout `[B, Sq, Hq, D]` 中同 qHead 内相邻 Sq 行间隔 `Hq × D`,**Bq 行不连续**。

**禁止**:批量 `DataCopy(GM, UB, curBq × D)` 会破坏邻 qHead 输出。

**正确做法**:按 (g, i) 二维循环,每行单独 `DataCopyPad(outGm[gOff + i × qRowStride], outUb[(g × Bq + i) × D_align], {1, D, ...})`。

### §8.4 GQA mEff 行排布约定(Bq-major)

`row(g, i) = g × Bq + i`,即按 qHead 外、Sq 内排布。

**理由**:
- Causal mask 模板可一份 + 整体平移 curG 次,不同 g 之间 mask 模式相同
- Q 的 Nd2Nz BSND→NZ 自然连续
- V2 输出按 (g, i) 二维循环逐行写回

---

## §9 优化策略特化

### §9.1 gMaxPerTask 选择

回答:**每个 task 合并的 qHead 数怎么选?**

**决策依据**:
- `mEff = curBq × curG`:mEff 越大 Mmad m 维利用率越高,但 L0A 压力增大
- Prefill(Sq 大):mEff 上限受 L0A 容量约束,curG 可能 < G
- Decode(Sq=1):mEff = curG,应最大化 curG = G

**产出**:gMaxPerTask 选择策略(prefill / decode 是否不同)。

### §9.2 mPad > 16 + 多 s2 块组合验证

GQA 维度首次打开(`mPad > 16` 即 `mEff > 16`)时,必须额外验证三类用例:
- (a) 单 s2 块 + mPad > 16
- (b) 多 s2 块 + mPad = 16
- (c) 两者同时打开

实测教训:(a)(b) PASS 不等价于 (c) PASS;(c) 路径在 P·V GEMM 上易出现精度漂移,需单独验证。

---

## §10 Self-Check 清单

### §10.1 继承通用清单(design/ 各节点 + implementation_ref.md)

- [ ] 清单 1(streaming UB 容量)
- [ ] 清单 2(slot 语义)
- [ ] 清单 3(cross-core sync 时序)([`implementation_ref.md` §5](../implementation_ref.md))
- [ ] 清单 9(长 Sk workspace 槽轮转)
- [ ] 清单 9b(workspace 外层乘子 = min(totalTasks, usedCoreNum))

### §10.2 GQA 特有清单

- [ ] **清单 4(M 轴尾块 padding 行)**:padding 行 mask / softmax 处理已明确;`SoftMaxShapeInfo.srcM = mEff`;mEff 行排布 Bq-major;padding 行输出最终清零
- [ ] **清单 5(causal mask 边界)**:`Duplicate` 起址 32B 对齐;尾块 padding 行 mask 处理;GQA group 复制(curG 份 mask 模式相同 + 整体平移);**mask 值用大有限负数(非 −∞)**;**完全屏蔽 tile 跳 C1 时 V1 同步跳过**(见 design/execution.md §1 mask 值陷阱)
- [ ] **清单 5b(online softmax 首块 / 除零)**:首块 O_acc 清零、rescale 因子等价 1(首块走跳过 rescale 分支)、末块 ÷sum 有下限保护(见 design/execution.md §1 首块初始化契约)
- [ ] **清单 6(bf16 V Cube 直通)**:V 全程 bf16,无任何 Cast / pre-cast / Reinterpret / 额外 half workspace 中转;V1 输出中间段 dtype = bf16;Mmad 模板 fm/filter 同型 = bf16;**未引入** 额外 half workspace / AIV pre-cast / kernel 内逐 s2 bf16↔half Cast 等反模式
- [ ] **清单 7(输出写回 BSND)**:按 (g, i) 二维循环逐行 `DataCopyPad`,不做批量 `DataCopy(GM, UB, curBq × D)`
- [ ] **清单 8(mPad > 16 + 多 s2 块)**:三类用例(a/b/c)已在精度回归中覆盖
- [ ] **清单 10(设计即最优,承接 design/_governance.md §2 门禁 / resources.md §2)**:`curG` 取 §2.1 / §9.1 求解的 `curG_max`(非 1,除非 `curG_max=1` 硬件被迫);流水启用 double buffer / 软流水重叠(非逐 KV 块 `PipeBarrier<PIPE_ALL>`);`mEff` / `K_BASE` 取 Roofline + L0A/L0B 容量最优点;每一处低于最优的取值都标注了绑定的硬件 / 容量约束;无脚手架残留、无 "S5 / 后续迭代" 式性能推迟

### §10.3 MHA 退化说明

MHA(G = 1)作为 GQA 的退化特例:
- 数学差异:`Hq = Hkv`,即 GQA 的 G=1
- 实现契约:走 GQA 骨架,仅 `gMaxPerTask = 1`,无需独立 service 类
- 选型说明:
  - 仅支持 MHA 形态 → 仍走 GQA 骨架(GQA 实现已自然兼容 G=1)
  - 既支持 MHA 又支持 GQA → 同一份代码(GQA 路径),通过 Tiling 决策的 `gMaxPerTask` 区分

---

## §11 MLA 变体分支(kvHeadNum=1 + latent absorption)

> MLA(Multi-head Latent Attention)在 GQA 骨架上引入 latent absorption 计算链。**MLA 属基础 FA 的 head / hidden-dim 变体,不单开子族**:它与 GQA / MHA 共享 §1-§10 的 service 骨架,仅在本节列出 kvHeadNum=1 + latent absorption 带来的变体分支特化。凡与本骨架一致处不再重复,只标注差异。
>
> ⚠️ **状态:未经算子验证**。截至本版本**尚无 MLA 算子落地**,本节的 latent absorption 计算链、int32 中间段、L1 rotation 加深等特化,是**从 GQA 骨架外推的前瞻性设计假设**,尚未经真实 MLA 算子回归验证。**设计 MLA 算子时应把本节当起点提示**;凡与实测(尤其 latent absorption 的数值精度、int32 段必要性、L1 深度)冲突,以实测为准并按蒸馏纪律回写(见 [`patterns.md` §6 架构约束](../patterns.md) 第 5 条"新变体落地后回写")。凡标注"由实现定义/由实现层定义"处,均为待落地时确定的开放项。

### §11.1 宏观决策差异

- **Kernel 类型**:`__mix__(1, 2)`,与 GQA 一致。
- **UB 模式**:默认 streaming UB 模式,与 GQA 一致。
- **Cube 分块层次**:与 GQA 一致,无特化。
- **流水级数**:与 GQA 一致(2 级或 3 级)。latent absorption 两步链插入在 V1 之后,不影响流水骨架(见 §11.6)。

### §11.2 Roofline 差异

**kvHeadNum = 1 强制**:MLA 强制 `kvHeadNum = 1`,所有 KV 压缩为 1 个 latent vector。AI 公式(见 [`roofline.md` §2.2](../foundation/roofline.md))中 Hkv=1,但 `mEff = Bq × curG` 不变(G 在 MLA 中通常 = Hq)。

**Latent absorption 对 AI 的影响**:latent absorption 两步链在 V1 之后执行,增加 Vector 端计算量。AI 公式不变(latent absorption 是逐元素操作,FLOPs 可忽略),但 Vector 端负载增加,可能需要调整 AIC:AIV 比例。

### §11.3 多核差异

**任务维度构造**:由于 `kvHeadNum = 1`,task 维度中 kvHead 维退化为常量 0:
```
taskIdx → (batch, kvHead=0, gS1Block, gBlock)
totalTasks = B × 1 × ceil(Sq/Bq) × ceil(Hq/curG)
```

**I2 不变量校核**:MLA 的 I2 继承 GQA——同一 kvHead 下(此处仅 1 个 kvHead)的 Hq 个 qHead 必须在同一 task 内合并处理(见 §3.2)。

### §11.4 内存差异

- **ConstInfo 加字段**:latent 维度相关字段(字段名由实现定义)、K_compressed 维度、V_compressed 维度。
- **Cube 端 L1 rotation 加深**:latent absorption 需保留更长 K_compressed 历史以做多轮 Mmad,L1 rotation 深度比 GQA 深(QP / KV 均较深)。依据见 [`extension_points.md` §1.2](../foundation/extension_points.md)。
- **Workspace 加 int32 段**:latent absorption fp32 精度不足,需 **int32 中间段**(段名由实现定义)。

### §11.5 编译宏差异

MLA 通常需要独立编译 target(latent absorption 链引入额外 TBuf 成员):

| 维度 | 是否分离 | 理由 |
|------|---------|------|
| MLA vs GQA | ✅ 必须 | latent absorption TBuf 不同 |
| MLA 内部 dtype | ✅ 同 GQA 规则 | fp16/bf16 影响 typedef |

### §11.6 流水线差异

**Stage 编排**:4-stage(C1/V1/C2/V2)骨架不变,在 V1 之后插入 latent absorption 两步链(函数名由实现定义):
```
C1 → V1 → latent_absorption_step1 → latent_absorption_step2 → C2 → V2
```

**Slot 分类**:与 GQA 类似(见 §6.2),但 L1 rotation 槽数更深。

### §11.7 Host Tiling 差异

- **ConstInfo**:在通用基础上加 latent 维度相关字段(字段名由实现定义)。
- **RunInfo**:复用 GQA 的 RunInfo 通用结构,加 latent 维元数据字段。
- **TilingData**:与 GQA 类似,但 kvHeadNum 固定为 1。

### §11.8 dtype 路径

MLA 的 dtype 路径与 GQA 一致(fp16 / bf16 同型规则)。详见 §8。

### §11.9 优化策略差异

- **Latent absorption 流水**:两步链在 V1 之后、C2 之前执行;Vector 端负载增加,可能需调整 AIC:AIV 比例;两步链与 V1 的 softmax 可融合(若平台 API 支持)。
- **K_compressed 历史保留**:MLA 的 L1 需保留更长 K_compressed 历史,可能影响 L1 rotation 深度的优化选择。

### §11.10 MLA 变体 Self-Check 清单

MLA 继承通用清单(清单 1/2/3/9/9b)与 GQA 基础清单(清单 4-8,见 §10.2),另加 MLA 变体特有清单:

- [ ] **清单 M1(kvHeadNum=1)**:kvHeadNum 强制为 1,所有 KV 压缩为 latent
- [ ] **清单 M2(latent absorption 数值校验)**:latent absorption 两步链的数值正确性已验证(具体由实现层定义)
- [ ] **清单 M3(int32 中间段)**:latent absorption 的 fp32 精度不足已通过 int32 中间段解决
- [ ] **清单 M4(latent absorption 变体分支承接)**:latent absorption 计算链在 **base FA service 内以变体分支承接**,不单开独立 service 类(MLA 属基础 FA 的 head / hidden-dim 变体,与 GQA / MHA 共享 service 骨架)
- [ ] **清单 M5(L1 rotation 加深)**:L1 QP / KV 槽数按 MLA 变体特定深度设置
- [ ] **清单 M6(softmax 广播 flag)**:SoftmaxFlashV2 已启用广播相关 flag 配合 latent 维度
