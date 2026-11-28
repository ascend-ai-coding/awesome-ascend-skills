# apace 算子优化手册（Optimization Playbook）

> 本文档沉淀 apace 算子从"能跑"到"跑好"的通用方法论：开发/优化阶段路径、现象→手法速查、失败案例教训、排查模式与实验纪律。内容来自生产算子的多轮优化演进复盘，已全部泛化——不绑定任何单一算子的实现细节与调参数值。
>
> 性能采集方法（msprof/L2 flush/多卡后处理）见 [`../shared/profiling_mc2.md`](../../../shared/profiling_mc2.md)；tileCnt 两阶段策略见 [`../shared/pipeline_tuning.md`](../../../shared/pipeline_tuning.md)。本文档不重复，只讲"什么时候用什么手法、怎么验证、怎么避坑"。

## 目录

1. [开发/优化阶段路径](#1-开发优化阶段路径)
2. [现象→手法→触发条件速查](#2-现象手法触发条件速查)
3. [失败案例教训（已证伪的方向）](#3-失败案例教训已证伪的方向)
4. [通用排查模式（现象→根因→修法）](#4-通用排查模式现象根因修法)
5. [实验纪律](#5-实验纪律)
6. [评审意见落地模式](#6-评审意见落地模式)

---

## 1. 开发/优化阶段路径

apace 算子的成熟过程遵循固定顺序，**不要跳级**（如在精度未达标时做性能优化，或在串行基线未建立时做流水重叠）：

```
① 可运行基线（精度优先，tileCnt=1 串行）
   → ② 精度修复（多 shape × 多 rank 全量 PASS）
   → ③ 验收归档基线性能（msprof round_001，作为后续所有对比的锚点）
   → ④ 架构级优化：通算流水重叠（收益最大，先做）
   → ⑤ 参数级调优：tiling / tile 粒度 / 通信粒度
   → ⑥ 热点子模块重写：op_summary 定位占比最高的子模块
   → ⑦ 微优化：双缓冲、去冗余指令、自适应参数
   → ⑧ 穷尽分析：确认无剩余空间，失败尝试同样归档
```

各阶段判据：

| 阶段 | 进入条件 | 退出条件 |
|:---|:---|:---|
| ① 基线 | 工程搭建完成 | 单 rank 冒烟输出非全 0 |
| ② 精度 | 基线可运行 | 全量用例 PASS（含 tail 非对齐、多 commTurn 场景） |
| ③ 基线性能 | 精度全 PASS | msprof 数据归档到 `docs/perf/round_001/` |
| ④ 流水重叠 | 基线性能已归档 | 通信耗时被计算掩盖（AIV time ≤ AIC time） |
| ⑤ 参数调优 | 流水已建立 | tileCnt/tile 粒度扫描到 Task Duration 最小值 |
| ⑥~⑦ 模块优化 | op_summary 有明确热点 | 单轮收益 ≥ 噪声水平（否则停止） |
| ⑧ 穷尽分析 | 连续多轮无收益 | 输出优化空间分析报告，成功与失败尝试均归档 |

**每轮优化必须绑定精度验证**：性能提交与"N/N case ALL PASS"证据同时落地，防止性能优化悄悄破坏精度。

---

## 2. 现象→手法→触发条件速查

按 op_summary 流水指标定位瓶颈，再选手法。**先判主导流水（CUBE/MTE2/V/FIXPIPE 谁最高），再动手**。

| 现象（判据） | 手法 | 触发条件与注意事项 |
|:---|:---|:---|
| AIC 计算与 AIV 通信串行，总时长 ≈ 两者之和 | per-tile 流水重叠：AIC 逐 tile `CrossCoreSetFlag` / AIV 逐 tile `CrossCoreWaitFlag` + 通信 | 计数式 flag 的 Set/Wait 必须严格配对；计数器 0-15 衡量未消费积压、不构成 T 上限（R9 口径——不设数值拒绝）；`tileCnt=1` 应自然退化为串行，便于回退对比 |
| tiles/core ≈ 1，cube_utilization 低，CUBE bound | 减小 baseM（如对 SWAT tiling 结果做减半后处理 256→128），增加 tile 数提升核间负载均衡 | 触发条件：SWAT 在 `blockNum ≥ aicNum` 时不触发 `AdjustBasicBlock()`、输出 `baseM==256` 且减半后满足 M 轴整除；⚠️ 副作用：DMA 次数随之增多（MTE2 上升），**仅 CUBE bound 时值得**；baseM 必须保持 CUBE_BLOCK(16) 整数倍；减半后须重置 M 尾块参数（`mTailTile=1, mBaseTailSplitCnt=1, mTailMain=0`）并验证 L1 buffer 安全（`stepK`/`scaleKL1`/`nBufferNum` 不变仍有效） |
| 仅个别核干活，其余核空转（如后处理只跑 block 0） | 多核并行：按 blockIdx 均分任务 + 余数前摊 | 后处理类 AIV 工作默认应全核参与；先确认任务可分行/分块且无跨核依赖 |
| 自研基础模块（如归约/拷贝）性能差 | **优先找同生态已验证的参考实现整体替换**，只适配语义差异，而不是继续微调自研版 | 这是历史上单轮收益最大的手法（2 倍以上）；替换后用同一基线量化验证 |
| 同步次数随某维度（rank 数/行数/tile 数）线性增长，该维度大时性能退化 | 批量摊薄：利用 MTE2/V/MTE3 流水 FIFO 性，一批数据只做 1 次 SetFlag/WaitFlag；批大小按 UB 容量自适应 | "重构后某维度退化"几乎必然指向同步开销；先数同步次数再改代码 |
| MTE2 加载与 V 计算串行 | pingpong 双缓冲：输入 buffer 双份，按计数奇偶交替 | 累加器等有语义依赖的 buffer **保持单份**，不能盲目复制 |
| V 指令链路过长（如 低精度→Cast→Add→Cast回） | 去冗余 Cast：用硬件原生低精度运算直接累加 | ⚠️ 触发条件苛刻：**仅当目标平台硬件级低精度累加的精度已实测达标**才能去掉高精度中间累加；必须先用高精度版本保底（见 §4 精度排查模式）。另需先确认该模块未被流水掩盖——被掩盖模块降精度实测收益趋近于零，而精度风险随 rank/配置漂移，生产上曾因此回退 |
| 固定 tile 粒度在小分片场景（如多 rank）效率低 | 自适应 tile 粒度：host 按 rank 数/shape 分档选择首 tile 大小 | 本质是"同步次数 vs 流水粒度"的权衡：分片小→减小 tile 让通信尽早启动；分片大→增大 tile 减少同步。**分档值只是起点**——异常 case（tileCnt 非最优、SyncAll 占比偏高、GM 带宽争抢特征）需 per-case 搜索首 tile 大小，判据用通信覆盖率与收益实测；具体分档阈值需按平台实测标定，不跨平台/case 复用单点数值 |
| 通信核与计算核互等 | 角色错峰：通信核发通信时，其余核算上一 tile 的残留工作 | 需要跨核同步点（SyncAll）配合；同步次数不可裁剪（见 §4）。compute-first 下进阶形态：fragment 重排 [remote..., local] + 双 flag（remote done 提前启动通信），见 [`fusion.md`](../fundamentals/fusion.md) §6.2.3 双 flag 计数式 |
| 通信与归约串行排队（时分复用），AIV 侧成为关键路径 | **通信/归约核严格分离（专职化）**：后 rankSize 核专职通信（通信对象 totalJobs=rankSize，每核 1 个 target 并行 PUT）、前 (核数-R) 核专职归约，`AllToAll(t) ∥ Reduce(t-1)` 错位流水，无 Phase 2 | 收益稳定（R 越大越显著）：消除 Phase 2 串行的收益大于归约算力减少 R 核的代价。⚠️ 反面教训：通信对象退化为单核（臆造"多核写同一 UBMEM flag 竞态"约束）会让 R 个 target 串行 PUT，通信时间放大 R 倍——该竞态约束不存在；TeamBarrier totalJobs 按分核映射取值（官方前 R 核映射 = rankSize；compute-first 后 R 核映射 = 1 + 显式 CrossDevice，见 [`communication.md`](../fundamentals/communication.md) §2.2） |
| 通信等待含框架内建跨设备 rendezvous，与分核守卫耦合 | **BARRIER_NONE + 手动 CrossDevice**：`Wait<BARRIER_NONE>` 仅 Drain 本 block channel，SyncAll 后由 block 0 显式 `teamBarrier_.CrossDevice()`，再一次 SyncAll 放行 | 多核并行通信 + 严格分离编排的配套手段；同步点显式可控（见 [`fusion.md`](../fundamentals/fusion.md) §6.2.3） |
| 归约（reduceSum）模块本身耗时高，逐行处理 flag 次数多 | **手动 UB + 多行批量归约**：LocalTensor 手动偏移布局（FP32 累加器单份 + src 双缓冲 pingpong + 输出 buffer），一次处理多行摊薄 flag 次数 | 显著摊薄同步开销；per-tile 高频调用场景避免 TPipe 重复 InitBuffer 耗尽 UB（归约 buffer 循环外一次性分配） |
| 归约 VEC 占比异常低但 flag/同步次数频繁 | **逐行归约（blockCount=1）**：flag/同步次数 = 行数 × N段数 × R | 改批量：2D DataCopyPad blockCount=本批行数（多行一次搬运），flag 次数除以 batchRows 摊薄——量化判据：逐行归约 = 性能 FAIL（见 [`fusion.md`](../fundamentals/fusion.md) §6.2.6） |
| AIC SCALAR 占比异常高、CUBE 反而很低 | **R×T 子 mm 调用的 Params 重建开销**：检查是否每轮循环重建完整 Params | 改 FragmentTensor 一次调用消 R 循环，或减少 T（增大 tileM），或 Params 增量更新只改地址字段；小/中 shape 下 SCALAR 易成为主 bound（见 [`fusion.md`](../fundamentals/fusion.md) §6.2.2）。⚠️ 反向认知：小 R×T 时两者性能持平，FragmentTensor 的价值在代码统一与 per-fragment L1 隔离——SCALAR 红线针对大 R×T 才成立 |
| 重复构建地址/描述符 | 增量复用：按轮次索引递推偏移，而非每轮重建 | 适用于 FragmentTensor 地址表、GM 指针族 |
| mac_ratio 高但 cube_utilization 低 | 瓶颈在**数据供给（MTE2 带宽）而非计算**：MAC 在等数据。优化搬运链路（双缓冲、L2 footprint、布局），不要再压计算侧 | 经典误判点：mac_ratio 高 ≠ 计算 bound；结合 cube_util 与 mte2_ratio 构建 GM→L1→L0→L0C→GM 搬运链路图定位 |
| L2 footprint 超容量、cache 命中低（大 shape 明显） | **problem 维度拆分子调用**：tile 大小不变、只缩 problem M（如 R×T 子调用），L2 footprint 缩小 → cube 利用率提升，大 shape 实测显著收益 | ⚠️ 与"拆小 tile"严格区分：后者 MTE2 次数翻倍，已证伪（§3）。拆分不得增加单位数据的搬运次数 |

---

## 3. 失败案例教训（已证伪的方向）

以下方向在生产算子上实测**无收益或负收益**，遇到相同现象时不要再试：

| 尝试 | 结果 | 通用教训 |
|:---|:---|:---|
| 增大 L0C 双缓冲深度（如 dbL0c=2） | fixpipe 时间反而上升 | L0 是稀缺资源，加深缓冲会与对侧 buffer 竞争空间 |
| 增大 stepK（K 方向步进） | 无收益 | **当两条 pipe 已高度重叠（≥90%）时，减少任何一侧的次数都无收益**——先看重叠率再决定优化对象 |
| 减小 baseN 提升 cube_util | 整体恶化 | fixpipe/DMA 的**固定开销随 tile 数线性放大**，cube_util 收益被抵消。减小 tile 前必须评估固定开销占比 |
| 把框架通信原语的多核 work partition 直接套到语义不同的场景 | 大面积数据错误后回退 | 复用框架原语前必须核对**地址偏移语义**（work partition 按 targetRank 还是 sourceRank 切分）；语义不匹配就换优化维度，不要死磕框架 |
| 通信退化为单核（仅 blockIdx==0 执行 Commit/Wait），理由"避免多核写同一 UBMEM flag 竞态" | R 个 target 串行 PUT，通信时间放大 R 倍；R≥4 且真同步时暴露死锁竞态 | **该竞态约束是臆造的**：TeamBarrier 按 job 分槽写入，无共享 flag 竞态；TeamBarrier totalJobs 按分核映射取值（见 [`communication.md`](../fundamentals/communication.md) §2.2）。设计阶段引入"约束"前必须在 skill/官网代码中找到实证来源 |
| 用 `SyncAll` 替代跨设备 fence | 接收方在远端 PUT 到达前读未初始化数据（大面积元素错） | **SyncAll 是本地核间同步，无跨设备能力**——跨设备可见性只能走框架 CrossDevice（TeamBarrier），本地同步加多少次都无效 |
| 手动轮询远端 GM flag 做跨设备同步 | 死循环超时 | **MTE 跨设备 GM 写可见性无框架 fence 保证**（本端 MTE2 读看不到远端 MTE3 写）——手动跨设备轮询不可行，须用框架 CrossDevice |
| PUT/GET 数据覆盖 Win 区内元数据/barrier 区 | 精度"假通过"（碰巧正确），同步机制已被破坏，大 shape/多轮时紊乱 | 数据区与元数据区分离：官网布局 barrier flag 在独立 BARRIER_BUF（Win 数据区从 0 可用）；共享布局按约定偏移跳过头部，host 预留与 kernel 读写偏移同源——此类"假通过"最危险：精度验证无法发现，必须靠设计红线拦截 |
| 单轮 PUT 数据量过大 | bring-up 期过大单轮曾见间歇失败（边界未定，非普适上限；同平台亦有更大单轮稳定运行的工程实例） | 无官方上限，勿以经验阈值做 host 强制拒绝；放大单轮后按 case 复核（连续多轮精度 + 重复 launch 一致性）；可靠性边界与带宽高效区间是两回事，不要混用 |
| 为增加通算重叠把 tile/K-window 数翻倍（拆小 tile） | MTE2 次数翻倍至饱和，负收益回退 | 重叠收益要靠**拆 problem 维度**（tile 大小不变）获得，不能靠增加单位数据的搬运次数；改之前先看 MTE2 余量 |
| local 段前置/后置重排以期通信提前启动 | 通信被计算完全掩盖的算子收益 ≈ 0 | **重排类优化先 profiling 确认瓶颈归属**（AIV WaitFlag 占比高 ⇒ 跳过重排）；仅通信暴露场景有收益。注意：compute-first 的 localLast 编排因 flag 配对契约依赖仍须保留，此处针对"额外重排"类提议 |

---

## 4. 通用排查模式（现象→根因→修法）

| 现象特征 | 根因方向 | 修法 |
|:---|:---|:---|
| **仅非对齐 shape 失败**，其余全部 PASS | tail 路径：尾块不被 tile 粒度整除，matmul 产出垃圾数据 | tail 单独 tiling + padding 到对齐边界 + 用 real size 字段限制实际读取范围；**host 侧内存分配必须与 kernel 实际读取范围一致**（详见 [`fusion.md`](../fundamentals/fusion.md) §6.2.7） |
| **精度误差随 K（或累加维度）放大**（小 K PASS、大 K FAIL） | 累加位宽不足，低精度累加误差随累加长度线性放大 | 先升高精度中间累加保底（如 BF16→F32→Add→Cast回）；降回低精度须同时满足：全量配置（含大 rank 数）实测达标 + 收益显著（未被流水掩盖），并保留回退开关 |
| **重构后某维度性能退化**（其余维度正常） | 同步/flag 次数随该维度线性增长 | 数清每单位数据的同步次数，改批量处理摊薄（见 §2） |
| **复用框架通信原语后大面积元素错误**（非边界、非尾块） | work partition 地址偏移语义与算子语义不匹配 | 读框架实现的偏移公式（按 targetRank 还是 sourceRank）；不匹配则回退单核通信基线，改从其他维度（后处理多核化等）要性能 |
| **删减跨核同步后出现 NaN / 偶发错误** | SyncAll 保障跨核数据可见性，次数不可裁剪 | 恢复同步点。同步是正确性保障不是性能开销，优化同步前先证明数据依赖已消除 |
| **少量元素错位/超差，分布无明显 tile 规律** | 参数误传：行步长/计数值等参数语义与签名不符（如行步长误传为段高、tile 计数误传 0） | **先逐参数核对调用点与签名的语义**（单位、轴、起算点），再怀疑算法；此类 bug 占比高且排查成本低 |
| **越界/踩数据，地址看似正确** | 同一地址偏移在多处独立计算，两处公式漂移 | 地址偏移**单一来源**：派生量在一处计算后传递，不在多个文件各算一遍 |
| **wall-clock 测得 kernel 极快（微秒级）但与 msprof 矛盾** | `<<<>>>` 是异步 launch，wall-clock 只测到下发开销 | 性能结论只以 msprof device 侧 Task Duration 为准（见 [`../shared/profiling_mc2.md`](../../../shared/profiling_mc2.md)） |
| **统一分支路径后出现全零/未初始化数据** | 原 if/else 掩盖了退化路径的初始化遗漏 | 统一路径是好事——它会暴露退化路径的初始化 bug；补齐退化路径的 tiling/参数初始化 |

---

## 5. 实验纪律

性能优化是实验科学，每条优化假设都按"可证伪的实验"执行：

1. **单变量**：每轮只改一个变量（一个手法、一个参数），否则无法归因
2. **同一锚点**：所有对比都相对同一个归档基线（`docs/perf/round_001/`），不相对"上一轮"链式对比（误差会累积）
3. **稳态取值**：丢弃 warmup 轮次取稳态均值；多卡场景跨 rank 取 max（木桶效应）——采集细则见 [`../shared/profiling_mc2.md`](../../../shared/profiling_mc2.md)
4. **量化回退**：不达标就回退，**回退提交同样记录量化数据**（R5/R6/R7 式编号留档），证明"试过、为什么不采用"
5. **失败也归档**：负收益尝试与成功尝试同等篇幅写进优化报告，防止后人重复踩坑（§3 即来源于此）
6. **精度绑定**：每轮 perf 提交附带全量精度 PASS 证据
7. **穷尽收尾**：确认无剩余空间时输出分析报告（含理论耗时对比、各路流水占比、已证伪方向清单），作为算子的性能档案
8. **实证复核**：静态分析得出的"不可行/无收益"结论，在有争议或收益量级不明时应设计最小实验复核（典型：拆 problem 维度的收益主因常是 L2 footprint 效应，静态分析易遗漏）。被推翻的假设是有价值的档案，不隐瞒
9. **选择性路由**：一个手法只对部分 case 有效、对其他 case 退化时，不做统一开关，按 case 特征（rank 数/tile 粒度/shape 规模/bound 类型）路由到不同方案，保留各方案的局部收益

---

## 6. 评审意见落地模式

评审意见（尤其是重构类）的落地方式是**"语义等价的高性能实现 + 量化回退机制"**，不是字面执行：

1. 逐条实现评审意见，每条用同一基线量化对比
2. 字面实现性能退化时，**诚实回退并记录数据**，然后寻找满足评审意图的替代实现（典型：为消除循环的"一次全覆盖"实现若实测回退，可改用 FragmentTensor 打包分段地址实现同等意图）
3. 阻塞项（正确性/红线）修完即 PASS；建议项按收益排期
4. 重构多发期主动做一次**设计文档 vs 代码的一致性核对**，防止文档腐化（文档与实现漂移会误导后续所有优化决策）

---

## 后续阅读

- [`../shared/profiling_mc2.md`](../../../shared/profiling_mc2.md) — 性能采集与多卡后处理细则
- [`../shared/pipeline_tuning.md`](../../../shared/pipeline_tuning.md) — tileCnt 两阶段策略
- [`fusion.md`](../fundamentals/fusion.md) — 通算融合编排模式与硬限制
- [`development-guide.md`](../operator-design/development-guide.md) — 工程改造食谱与验收清单
