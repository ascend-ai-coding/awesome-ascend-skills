# apace 路线代码审查验收条件

> Reviewer 在 Step 4 逐项检查。违反任意**硬红线**项 = FAIL。
> 本文件是**全局红线**的唯一规范源（规则全文、理由与"详见"链接）；**场景约束**的完整定义以对应场景文档为规范源，本表只做索引与一句话判定。编号 R1-R21 保持稳定，跨文档引用不断链。
>
> **条目分级（Reviewer 必读）**：本表条目分两级——
> - **硬红线**：违反 = FAIL（正确性/死锁/共享层完整性问题，如 R1-R8/R11-R13/R20 的主体）
> - **风险提示**：原文含"生产实测经验值/官方无显式约束/边界未定"限定词的数值约束（如 512KB 单轮 PUT、flag 计数深度），违反**不直接 FAIL**——按 case 复核（连续多轮精度 + 重复 launch 一致性 + profiling 证据）；仅当复核证据缺失或实测失败才 FAIL。**镜像/转述这些条目时禁止删去限定词升格为硬门禁**（历史教训：经验阈值写成 host 强制拒绝曾误拒合法 shape）。
>
> **镜像同步纪律（修订者必读）**：R 条目镜像处——本表（唯一规范源）/ SKILL.md 红线索引 / fusion.md §6.2.10 / architecture.md §10 / operator-anatomy.md §5.8/§7.6 / 场景 design.md / 场景 development.md。修改任意一处 R 条目时，**其余镜像处必须同步检查**；本文件为唯一规范源，镜像处只做索引与一句话判定，禁止在镜像处展开规则细节（512KB 事故即镜像措辞漂移逐级扩散所致）。

## 全局红线定义（所有 apace 算子必须满足）

| # | 约束 | 说明 | 详见 |
|---|------|------|------|
| R1 | 禁止 `__schedmode__(1)` 和 `[[bisheng::core_ratio(1,1)]]` | 会导致 AIC/AIV 串行调度→死锁（`aclError:507015`）；核配比唯一由 `KERNEL_TYPE_MIX_AIC_1_1` 保证为 1:1 | [`architecture.md`](fundamentals/architecture.md) §10 ① |
| R2 | 有 `KERNEL_TYPE_MIX_AIC_1_1` | 每个入口函数都含核配比声明 | [`architecture.md`](fundamentals/architecture.md) §10 ① |
| R3 | 入口变体覆盖 dtype 合同全部组合 + host 运行期 dispatch | 变体数由 dtype 合同决定（FP8 E4M3/E5M2 双组合 = 4 变体；单一组合可单入口）；硬编码单入口 → 异 dtype 字节流被错误模板解释，精度系统性失败 | [`operator-anatomy.md`](operator-design/operator-anatomy.md) §5.3/§7.2 |
| R4 | `block/` `tiling/` 未修改 | 与官网仓原始文件完全一致 | [`architecture.md`](fundamentals/architecture.md) §10 ③ |
| R5 | CrossCore flag idx 配对 | AIV `WaitFlag` idx == AIC `SetFlag` idx（含计数式配对） | [`fusion.md`](fundamentals/fusion.md) §3 |
| R6 | CommContext 与引擎匹配 | UDMA 模式有 `__gm__ CommContext*`；HCCL windows 无 | [`communication.md`](fundamentals/communication.md) |
| R7 | 禁止 `AscendC::Matmul` 高阶 API | 高阶 API 不支持 AIV+URMA 直调场景（与 blaze-shmem 共享此约束） | [`compute.md`](fundamentals/compute.md) |
| R8 | 禁止 HCCL 高阶 API（`Hccl::*`） | HCCL 集合通信库依赖框架注入上下文，AIV+URMA 直调场景拿不到（与 blaze-shmem 共享此约束） | [`comm_shmem.md`](../blaze-shmem/comm_shmem.md) §5 |
| R11 | host 前置校验在 fork/建链前拒绝非法输入 | 整除/对齐/核数下限/Win 容量等**正确性类**校验，报错可操作，main.cpp 与 gen_data.py 双侧；compute-first 场景的完整清单（含归约核数/对齐/Win 容量）见场景文档。⚠️ 风险提示类数值（512KB 单轮 PUT、flag 计数深度）**不进强制拒绝清单**（见头部条目分级） | [`development-guide.md`](operator-design/development-guide.md) §3.5 |
| R12 | UB 静态通信区隔离 | commBuf/barrierBuf 与 TPipe 管理 buffer 物理隔离（静态偏移或 guard TBuf），混用重叠 → 通信数据被踩踏 → 死锁 | [`communication.md`](fundamentals/communication.md) 陷阱 #9 |
| R13 | 通信默认多核并行 | 通信对象 `totalJobs=rankSize`（每核负责 1 个 target 并行 PUT）；TeamBarrier `totalJobs` 官方为 `rankSize`（`teamBarrier_.Init(buf, ctx, rankSize, GetBlockIdx())`——前 R 核映射下 CrossDevice 由 `Wait<BARRIER_DEVICE>` 内建触发）。compute-first 后 R 核映射的自研编排可采用 TeamBarrier totalJobs=1 + 显式 CrossDevice 替代，两种映射不得混用。"多核写同一 UBMEM flag 竞态 → 退化为 totalJobs=1"是已证伪的臆造约束：退化为单核串行 PUT 通信时间放大 R 倍 | [`communication.md`](fundamentals/communication.md) §2.2；[`optimization-playbook.md`](troubleshooting/optimization-playbook.md) §3 |
| R14 | Win 区数据/元数据分离（硬红线）；单轮 PUT 大小（风险提示） | ① **硬红线**：PUT/GET 数据不得覆盖 Win 区内元数据/barrier 区（偏移按 host 建链布局确定，host/kernel 同源）：官网布局 barrier flag 在独立 BARRIER_BUF、Win 数据区从 0 可用；共享布局须按约定偏移跳过头部（布局细则与已验证实现见 [`communication.md`](fundamentals/communication.md) 陷阱 #12；缺失 → 覆盖 flag → "假通过"）。② **风险提示**：单轮 PUT 大小无官方硬上限（bring-up 期过大单轮曾见间歇失败，边界未定）——大单轮 PUT 按 case 复核精度+重复 launch 一致性，不作 host 强制拒绝 | [`communication.md`](fundamentals/communication.md) 陷阱 #12/#13；[`fusion.md`](fundamentals/fusion.md) §6.2.4 |
| R15 | 投产级性能验证门槛 | 精度 PASS 不算投产：性能验证必须覆盖真实大 shape 矩阵（非 toy shape）× R=2/4 双档，并与参考路径对标归档；仅基线采集、无对标 = 未达投产门槛。基线候选按算子族枚举：mc2 融合算子（RS 族含 `aclnnMatmulReduceScatterV2`，CANN 9.1.0/9.2.0 opp 内置 ascend950/ops_transformer 下已验证存在、fp8_e4m3fn 输入实跑可用；注意其无 MX scale 输入为 per-tensor 量化语义，对标须声明差异）/ hccl 分步（mm + HcclReduceScatter）。**aclnn 对标算子调用契约坑（CCU 引擎直调不可行、scale 形状校验、group 机制等生产试错记录）见 [`scenarios/compute-first-reduce-scatter/development.md`](scenarios/compute-first-reduce-scatter/development.md) §4.1**。候选检索方法：按 op 名模式搜 `$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/kernel/`，勿只查单一算子名即下 N/A 结论 | [`host-and-testing.md`](operator-design/host-and-testing.md)；[`optimization-playbook.md`](troubleshooting/optimization-playbook.md) §1 |
| R20 | perf 模式 L2 flush 实接线 | 性能采集每轮主 kernel 前必须实际调用 L2 flush kernel（msprof 记录数 == 轮数）；只分配 cacheFlush buffer 不调 kernel = 死代码 = MTE2 带宽虚高 | [`host-and-testing.md`](operator-design/host-and-testing.md) §4 |

## 场景约束（按算子语义特征自动适用，完整定义以场景文档为规范源）

> **适用判据（不依赖场景注册表命中）**：代码含 compute-first 编排（先算后通信、staging 即通信源）即适用 R9/R10/R16；含归约模块（reduceSum）即适用 R17-R19。Reviewer 只要看到对应代码形态就必须检查，不得以"场景未正式命中"为由跳过。

| # | 适用场景 | 约束（一句话判定） | 规范源 |
|---|---------|---------------------|--------|
| R9 | compute-first | CrossCore flag **Set/Wait 严格配对**（硬红线：不配对 = 未定义行为/挂死）+ flagId 数量 16（0-15）为硬件事实；**每 flagId 计数器 0-15 衡量未消费积压**（官方有据；紧邻配对下 ≈1-2 不触顶，**不构成 T 上限**；计数式固定 flagId 配对工程验证 T=16）——T 上限按 case 实测确定，不设 host 强制数值拒绝 | [`fusion.md`](fundamentals/fusion.md) §6.2.3 |
| R10 | compute-first | T 派生优先 `T \| mSeg` 无尾块（实现最简）；**单尾块且尾块 ≤ 头块（16 对齐）时 PUT 直传合法**（偏移精确，工程验证）；多尾块/尾块>头块走策略 A（tail padding 32 对齐 + realFragmentSize 限读 + 多套 tiling）——三条路径均合法，禁止为凑无尾块而牺牲流水粒度 | [`fusion.md`](fundamentals/fusion.md) §6.2.7 |
| R16 | compute-first | A 各 rank 段 GM 连续时 mm 默认 FragmentTensor 消 R 循环（约束 `R ≤ 32` = 单次构建 fragment 数上限，与 T 无关）；vendor R×T 子调用路径须论证 SCALAR 占比可接受（R×T 小 + 大 shape） | [`fusion.md`](fundamentals/fusion.md) §6.2.2 |
| R21 | compute-first | **localLast 编排禁止移除**：frag kernel 必须含 fragment 重排 `[remote..., local]` + 边界提前 SetFlag(flagA)；以"每轮 Set 双 flag"替代会导致 flagId 用量翻倍 + 丧失通信提前启动。`cFragAddrs_` 顺序写错才是 A/C 错位根因，非 localLast 本身 | [`fusion.md`](fundamentals/fusion.md) §6.2.2 |
| R17 | compute-first（含归约模块） | 归约搬入/输出用 2D DataCopyPad 且 blockCount=本批行数（多行批量 + 手动 UB）为**性能推荐形态**；**例外**：MTE2-bound 算子中逐行 1D 路径可为参考形态（通信被 mm 掩盖时性能不劣化——须 profiling 证据支撑，性能不达标再改批量）；strided 场景隐式上限防御见下层 API 文档。**归约必须多核分治**：归约核集合内按连续行块均分本轮行（余数前摊，每核只处理自己的行区间），禁止单核归约（如 `GetBlockIdx()==0` 独立承担全量行 = 写竞争/归约独占 = FAIL） | [`fusion.md`](fundamentals/fusion.md) §6.2.6；`ascendc-api-best-practices` skill `references/api-pipeline.md` / `api-datacopy.md` |
| R18 | compute-first（含 BF16→FP32 归约） | 禁止 in-place BF16→FP32 Cast（FP32 占 2× 空间覆盖未读 BF16 → 精度系统性错误），必须独立 srcFP32 双缓冲 | [`fusion.md`](fundamentals/fusion.md) §6.2.6 纪律 2；`ascendc-api-best-practices` skill `references/api-precision.md` |
| R19 | compute-first（含归约模块） | MTE2_V/V_MTE2/V_MTE3/MTE3_V 四类 HardEvent 同迭代 Set/Wait 配对（含循环结束消费残留事件）；Set 无配对 Wait = 挂死（507014） | 事件模板：[`scenarios/compute-first-reduce-scatter/development.md`](scenarios/compute-first-reduce-scatter/development.md) §5.3；纪律：[`fusion.md`](fundamentals/fusion.md) §6.2.6 |

## 红线项（操作化检查方法）

> 场景约束（R9/R10/R16/R21/R17-R19）仅在对应场景命中时检查；其余为全局检查项。

| # | 检查方法 |
|---|---------|
| R1 | `grep -rn "schedmode\|core_ratio"` 应为空 |
| R2 | 每个 `__global__` 入口函数含 `KERNEL_TYPE_MIX_AIC_1_1` |
| R3 | 入口变体数 == dtype 合同组合数（FP8 双组合 = 4 变体）；main.cpp 有运行期 dtype dispatch，无硬编码单一入口 |
| R4 | `block/` `tiling/` 与官网仓原始文件 diff 一致；算子目录无共享层副本 |
| R5 | AIV `WaitFlag` idx == AIC `SetFlag` idx（含计数式配对） |
| R6 | UDMA 模式入口有 `__gm__ CommContext*`；HCCL windows 无 |
| R7 | `grep -rn "AscendC::Matmul"` 应为空 |
| R8 | `grep -rn "Hccl::"` 应为空 |
| R9 | AIV `WaitFlag` idx == AIC `SetFlag` idx 且 Set/Wait 次数严格配对（含计数式固定 flagId）；flagId ∈ [0, FLAG_ID_MAX)；**无计数深度类数值断言要求**（计数器 0-15 衡量未消费积压、不构成 T 上限，T 上限按 case 实测）；**若以"每轮 Set 双 flag"替代 localLast 双 flag = 性能 FAIL**（[`fusion.md`](fundamentals/fusion.md) §6.2.2/§6.2.3） |
| R10 | T 派生：`T \| mSeg` 无尾块 / 单尾块 ≤ 头块直传 / 策略 A padding 三形态任一合法；单尾块时 tailM 16 对齐且 `commTilingData` 如实填 tail 字段、tail tiling 入参 m=R×tailMSize；多尾块或尾块>头块必须走策略 A |
| R11 | host 前置校验在 fork/建链前执行，main.cpp 与 gen_data.py 双侧；非法输入报错可操作；compute-first 场景 9 项齐全（场景 development.md §3.1） |
| R12 | commBuf/barrierBuf 与 TPipe 管理 buffer 物理隔离（静态偏移或 guard TBuf）；**`TPipe` 与 `MakeMemPtr` 必须二选一，禁止混用**——`grep -n "TPipe\|InitBuffer" kernel/` 与 `grep -n "MakeMemPtr" kernel/` 若同时出现于同一 kernel 目录且未隔离 = FAIL（[`operator-anatomy.md`](operator-design/operator-anatomy.md) §4.3） |
| R13 | 通信对象 `totalJobs=rankSize` 多核并行；TeamBarrier `totalJobs` 按分核映射取值（官方前 R 核映射 = rankSize；compute-first 后 R 核映射 = 1 + 显式 CrossDevice），两种映射不得混用；出现"totalJobs=1 避免 UBMEM flag 竞态"类设计（指**通信对象**退化单核）= FAIL |
| R14 | Win 数据/元数据偏移 host/kernel 同源（硬红线）；单轮 PUT 大小无强制 512KB 校验要求（风险提示级：大单轮 PUT 须有 case 复核证据——连续多轮精度 + 重复 launch 一致性） |
| R15 | profiling/ 含真实大 shape × R=2/4 双档 × 三路径对标归档；仅 toy shape 基线 = FAIL |
| R16 | mm 内核为 FragmentTensor 自研；vendor 路径 DESIGN.md 有 SCALAR 占比论证；"vendor kernel + FragmentTensor C 输出"设计 = 阻塞级错误 |
| R21 | frag kernel 含 localLast fragment 重排（`[remote..., local]` + 边界提前 SetFlag(flagA)）；`grep -n "localLast\|remote.*local\|flagA" kernel/` 无 localLast 相关代码 = FAIL；每轮 Set 双 flag（flagA+flagB）替代 localLast 双 flag = 性能 FAIL（峰值 2T） |
| R17 | 归约搬入/输出 2D DataCopyPad 且 blockCount=本批行数；strided 场景（N>单次列宽）有 redUbM≤32 或 1D 退化（例外条件见场景约束 R17） |
| R18 | 归约有独立 srcFP32 双缓冲；`grep -n "Cast<float" kernel/` 确认无 in-place 加宽 Cast |
| R19 | MTE2_V/V_MTE2/V_MTE3/MTE3_V 同迭代 Set/Wait 配对；循环结束有残留事件消费（含次数守卫） |
| R20 | perf 循环每轮调用 L2 flush kernel；msprof 结果中 flush kernel 记录数 == 轮数 |

## 常见 FAIL 原因

| 现象 | 根因 | 修复方向 |
|:---|:---|:---|
| 代码中含 `__schedmode__(1)` | 误加调度属性 | 删除，核配比由 `KERNEL_TYPE_MIX_AIC_1_1` 保证 |
| 代码中含 `Hccl::AllReduce` | 误用 HCCL 高阶 API | 改用 `CollectiveComm` 四段式 API |
| 代码中含 `AscendC::Matmul` | 误用 asc-devkit 接口 | 替换为 `Blaze::Gemm::Block::BlockMmad` |
| `block/` 或 `tiling/` 有改动 | 误改共享层 | 恢复共享层文件，只在 `kernel/<op>/` 下改 |
| 精度对不上但无报错 | flag idx 不配对 / splitKNum 配置错 | 核对 flag 编排和 splitKNum 规则 |
| 死锁（aclError:507015） | schedmode 或 flag 不配对；**或通信 UB 静态区被 TPipe 分配覆盖** | 检查无 schedmode；检查 CrossCore flag idx 配对；检查 guard TBuf/静态偏移隔离（[`communication.md`](fundamentals/communication.md) 陷阱 #9） |
| golden 全错且误差不收敛 | golden 切分轴/每卡语义写错 | 回设计阶段核对 golden 语义小节（[`workflow_integration.md`](workflow_integration.md) Step 2 §golden 语义），先修 gen_data 再怀疑 kernel |
| T=1 全 PASS、T>1 精度错 | 多 tile 布局或流水路径 bug（T=1 退化路径会掩盖） | 按 tile/轮次定位超差分布，核对多 tile 下的输出布局与 flag 计数配对 |
| 性能差：通信时间 ≈ R × 单 target 时间 | 通信对象被退化为单核 totalJobs=1 串行 PUT（臆造"UBMEM flag 竞态"约束） | 通信对象改 totalJobs=rankSize 多核并行（R13）；TeamBarrier 按分核映射取值（官方前 R 核映射 = rankSize；compute-first 后 R 核映射 = 1 + 显式 CrossDevice） |
| 性能差：AIC CUBE/MTE2 流水占比接近饱和但 cube_utilization 极低 | R×T 子 mm 调用每轮重建 Params，SCALAR 主 bound（特征信号） | 改 FragmentTensor 一次调用消 R 循环（R16）/ 减少 T / Params 增量更新（[`fusion.md`](fundamentals/fusion.md) §6.2.2） |
| 性能差：归约模块耗时高、归约 VEC 占比低但 flag/同步频繁 | 逐行归约（blockCount=1），flag 次数 = 行数×N段数×R | 手动 UB + 多行批量归约 + 2D DataCopyPad blockCount=多行（R17） |
| 精度"假通过"但大 shape/多轮紊乱 | PUT/GET 数据覆盖了 Win 区内元数据/barrier 区 | 数据区与元数据区分离，host/kernel 偏移同源（R14）——精度验证发现不了，靠红线拦截 |
| compute-first 归约读 staging 得 0/旧值 | 多核归约写竞争（多核写同一 yGm）、flag 配对缺失，或归约地址与 mm 写入不同源——**非 cache 问题** | 按序排查：flag idx 配对 → 归约多核分治（R17）→ staging 写/读地址同源；**禁止直接对 staging 加 dcci**（staging 可见性由 CrossCore flag 配对保证，参考实现不依赖 dcci，加 dcci 属误诊掩盖真根因，见 [`communication.md`](fundamentals/communication.md) 陷阱 #14） |
| 死锁（aclError:507014，Ascend 950 归约路径） | 归约 SetFlag\<V_MTE2\> 无配对 WaitFlag（跨迭代 Set-Set 无中间 Wait） | 补齐 pingpong slot 事件配对 + 循环结束消费残留事件（R19，事件模板见 [`scenarios/compute-first-reduce-scatter/development.md`](scenarios/compute-first-reduce-scatter/development.md) §5.3） |
| 精度 FAIL：E5M2 变体全元素不通过且误差量级稳定（不收敛） | main.cpp kernel dispatch 硬编码为 E4M3E4M3 入口，E5M2 字节流被 E4M3 模板错误解释 | 实现全部 dtype 变体入口 + host 运行期 dtype dispatch 宏（R3，[`development-guide.md`](operator-design/development-guide.md) §3.5） |
| 精度 FAIL：N>redUbN 时部分输出为零，呈周期性分布（period=redUbM） | 2D DataCopyPad srcStride>0 时 blockCount 超 DAV_3510 隐式上限（约 29-32 行），超出行静默丢零 | host 限制 redUbM ≤ 32（方案 A）+ strided 场景 1D 逐行（方案 B），（R17 例外，[`fusion.md`](fundamentals/fusion.md) §6.2.6 纪律 3） |
| 精度 FAIL：归约结果系统性错误（误差随来源序累积） | in-place BF16→FP32 Cast，FP32 输出覆盖同 buffer 未读 BF16 数据 | 独立 srcFP32 双缓冲（R18） |
| perf 模式 MTE2 带宽虚高 / 性能数据不可复现 | L2 flush 未接线（cacheFlush buffer 分配但未调用 kernel，死代码） | 接入 heavy_add_kernel 每轮 flush + 同步（R20，[`host-and-testing.md`](operator-design/host-and-testing.md) §4 模板） |
| 大 shape 无法运行 / 运行期间歇 FAIL | 正确性类校验缺失（m%R、scale 偶数、tailM 16 对齐、Win 容量、usedCoreNum≥R+1）；或单轮 PUT 超大**且无 case 复核证据**（风险提示级，非必然缺陷——更大单轮亦有稳定运行实例） | 补正确性类 host 校验（R11）；单轮 PUT 大时先跑通小单轮再放大并复核精度+重复 launch 一致性（R14 风险提示） |
| DESIGN.md 与代码不一致 | localMatmul 等参数变更后未同步文档 | 同步 DESIGN.md |
