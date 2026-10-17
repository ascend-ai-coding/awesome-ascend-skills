# CANNBot Step 1→7 在 apace MC2 场景下的工作流映射

本文档把父级 `plugins-official/ops-direct-invoke/AGENTS.md` 的 7 步流程具体化到 apace 通算融合算子场景。CANNBot 主控和 Architect/Developer/Reviewer 三类 Subagent 在每个 Step 进入前都应先读对应小节，明确"本阶段在 apace 场景下要做什么、门禁是什么"。

> 父级流程定义以 `plugins-official/ops-direct-invoke/AGENTS.md` 为准；本文件只补充 apace 场景的差异化要点，不重写父级规则。
>
> **与 skill 四步模型的对应**：plugin Step 1 ≈ skill Step 1；plugin Step 2/2.5 ≈ skill Step 2+3；plugin Step 3-7 ≈ skill Step 4。**plugin 场景下以本文档为主要消费文档**；通用开发方法论（五阶段流程 + D1-D6 决策点 + 跨阶段纪律）见 [`operator-design/dev-methodology.md`](operator-design/dev-methodology.md)，路线模型、R1-R21 红线（[`review-checklist.md`](review-checklist.md)）与场景注册表（[`scenarios/`](scenarios/)）为两边共用的合同。

## Step 1：环境检查（门禁）

| 校验项 | 达标条件 | 失败处理 |
|--------|---------|---------|
| NPU 架构 | dav-3510 | 非 3510 直接终止 |
| CANN 版本 | 已验证 9.1.0/9.2.0（内置 apace 路径形态不同，以 `test -d` 实测为准）；其他版本未验证 | 提示风险，用户决策 |
| 多卡环境 | ≥ rankNum 张卡 | 单卡无法运行 |
| HCCL 可用 | `hccl.h` 存在 | 检查 CANN 安装 |
| CANN 内置 apace | 目录 `test -d` 实测存在（路径形态见 [`workflow/step1-project-setup.md`](workflow/step1-project-setup.md) §1） | 检查 CANN 安装版本 |
| 上游仓可拉取（可选） | 网络可访问 | 仅跟踪 master 契约时需要，缺失不阻塞 |

> **路径实测纪律**：environment.md 中每条路径（bisheng/头文件/.so/apace 目录）必须 `test -x`/`test -f`/`test -d` 实测通过才标 ✅——凭记忆填写的路径会在编译阶段才暴露。

## Step 2：设计（Architect）

### DESIGN.md 必须额外包含的小节

#### §约束显式确认

| 约束 | 验收条件 |
|:---|:---|
| ① 禁止 `__schedmode__(1)` | 核配比由 `KERNEL_TYPE_MIX_AIC_1_1` 保证 |
| ② Matmul 走 Blaze 模板 | `BlockMmad` + `BlockScheduler` + `MatmulWithScaleMx`，无 `AscendC::Matmul` |
| ③ 禁止修改 `block/` 和 `tiling/` | 只在 `kernel/<op>/` 下创建文件 |
| ④ 直调仅 UDMA | HCCL windows 不支持直调 |

#### §D1-D6 决策点定值

按 [`dev-methodology.md`](operator-design/dev-methodology.md) §2 对六个决策点逐项定值（数据流方向/切分轴与 T/数据通路/同步合同/累加合同/入口合同），每项附依据。技术细节锚点：

- 切分轴按数据分布语义确定——通信原语只决定搬运方向，不决定切分轴（[`compute.md`](fundamentals/compute.md) §4.7）
- 两阶段 tileCnt 策略：精度期 T=1 串行基线 → 性能期扫描（[`../shared/pipeline_tuning.md`](../../shared/pipeline_tuning.md)）
- Win 区预算：data 段 `rankSize × rankDataBytes` + scale 段 `rankSize × scaleKaSize × axisM`（[`operator-anatomy.md`](operator-design/operator-anatomy.md) §3.5）

#### §golden 语义（每卡输入/输出契约）

明确每卡输入（本地数据 + 远端来源）与输出语义、切分轴、聚合方式——golden 切分轴写错则精度验证整体失效。

#### §API 验证清单

选用的每个 API 标注「验证来源（官方文件:行号）+ 当前 CANN 版本可用性状态」。未验证 API 禁止入设计。

### Architect 加载顺序

[`architecture.md`](fundamentals/architecture.md)（心智模型）→ 按需 [`communication.md`](fundamentals/communication.md) / [`compute.md`](fundamentals/compute.md) / [`operator-anatomy.md`](operator-design/operator-anatomy.md) §3（tiling）/ [`fusion.md`](fundamentals/fusion.md) §5（localMatmul）。

### 门禁

双文件齐全（DESIGN.md + PLAN.md；`unsupported` 仅 DESIGN.md）；约束确认 4 项勾选；D1-D6 定值 + golden 语义 + API 验证清单齐备；路线决策已记录（`apace_custom` 须含 `selected_scenario`）。

## Step 2.5：设计串讲

关注：`[REUSE]`/`[MODIFY]` 标记合理；AIC 的 rank 遍历正确；Win 区预算够；D1-D6 取值可解释（判据见 [`dev-methodology.md`](operator-design/dev-methodology.md)）。严格 1 轮串讲，分歧写入 WALKTHROUGH.md 仲裁小节。

## Step 3：开发（Developer）

基础工程验收（工程结构/共享层零修改/编译/冒烟/精度/文档同步）按 [`development-guide.md`](operator-design/development-guide.md) §5 执行；apace 增量项：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | host 前置校验 | 整除/对齐/核数下限/Win 容量等正确性类校验在 fork/建链前（[`development-guide.md`](operator-design/development-guide.md) §3.5） |
| 2 | UB 静态区隔离 | commBuf/barrierBuf 与 TPipe buffer 物理隔离（[`communication.md`](fundamentals/communication.md) 陷阱 #9） |
| 3 | 分阶段 bring-up | 新链路（bias/新通信对象/新归约）先退化对照（zero 值、T=1、rank=2）再铺开 |
| 4 | 精度标准 | 以算子 ST `verify_result.py` 为准（容差分型见 [`host-and-testing.md`](operator-design/host-and-testing.md) §5） |

**开发期红线**：禁 `__schedmode__(1)`/`core_ratio`；禁 `Hccl::*` 高阶 API；禁 `AscendC::Matmul`；禁改 `block/`/`tiling/`；关键参数（如 localMatmul）改动必须同步 DESIGN.md；改完每个 `[MODIFY]` 文件立即跑精度冒烟。

## Step 4：审查（Reviewer）

按 [`review-checklist.md`](review-checklist.md) 逐项检查（R1-R21，含操作化方法与常见 FAIL）。违反任意红线 = FAIL → Step 5；PASS/PASS WITH NOTES → Step 6。

## Step 5：修复循环

最多 3 轮（Developer 修复 → Reviewer 复审），仍未通过暂停上报。常见"修不动"问题速查（根因分析详见 failure-navigation）：

| 问题 | 修复路径 |
|:---|:---|
| CrossCore flag idx 不配对 | 核对 flag 编排（[`fusion.md`](fundamentals/fusion.md) §3） |
| CommContext 填充错误 | 必须由 `CommChannelBuilder::CreateDeviceContext` 填充 |
| tiling 不匹配 | 切换 tileCnt 后必须重调 `GetTilingData` |
| splitKNum 配置错 | localMatmul=1 → rankSize-1；0/2 → rankSize（[`fusion.md`](fundamentals/fusion.md) §5） |
| localMatmul=1 死锁 | RunLocalMatmul 与 RunMatmul 之间补 `PipeBarrier<PIPE_ALL>()`（开销远小于通算并行收益，勿直接回退 mode 2） |

## Step 6：精度与性能验收

**6a 精度**（Reviewer）：独立运行，报告归档 `docs/precision/summary.txt`；不达标回 Step 5（计数器重置，另允 3 轮）。

**6b 性能**（Developer）：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 采集模式 | msprof task-based 采集 |
| 2 | L2 cache flush | 每轮实际调用 flush kernel（记录数 == 轮数；模板见 [`host-and-testing.md`](operator-design/host-and-testing.md) §4） |
| 3 | 多卡后处理 | 官方口径（parse_prof.py：跳 3 轮 warmup + 1.2×min 去离群 → 卡均值 → 总体 = 卡均值平均）；skill 投产另报跨 rank max，两层分开归档 |
| 4 | tileCnt 扫描 | 扫 `tileCnt ∈ {1,2,4,8,16}` 选最优（上限约束见 R9 口径） |
| 5 | 数据归档 | `docs/perf/round_NNN/` 含多个 `PROF_*` 子目录 |
| 6 | 性能达标 | R15 门槛：真实大 shape × R=2/4 双档 × 参考路径对标归档（[`host-and-testing.md`](operator-design/host-and-testing.md) §6） |

> 官网 `tests/st/{op}/run.sh --perf` 提供 msprof 采集 + `parse_prof.py` 解析链路，可复用；详细流程见 [`../shared/profiling_mc2.md`](../../shared/profiling_mc2.md)。

## Step 7：完成汇报

主控汇总：最终判定（PASS/PASS WITH NOTES）、总分、代码路径、各 dtype 精度概要、性能概要（Task Duration/主导流水/通信隐藏率）、遗留 NOTE。失败案例按"现象→根因→修复方向"回流 [`review-checklist.md`](review-checklist.md)。

## 后续阅读

| 文档 | 何时读 |
|:---|:---|
| [`dev-methodology.md`](operator-design/dev-methodology.md) | 通用方法论：五阶段流程 + D1-D6 决策点 + 跨阶段纪律 |
| [`architecture.md`](fundamentals/architecture.md) | 第一次了解 apace 三层架构 |
| [`development-guide.md`](operator-design/development-guide.md) | 工程搭建与改造验收 |
| [`communication.md`](fundamentals/communication.md) / [`compute.md`](fundamentals/compute.md) / [`fusion.md`](fundamentals/fusion.md) | 通信/计算/融合组合机制 |
| [`operator-anatomy.md`](operator-design/operator-anatomy.md) / [`host-and-testing.md`](operator-design/host-and-testing.md) | 算子骨架 / host 序列与 ST 工程 |
| [`../shared/pipeline_tuning.md`](../../shared/pipeline_tuning.md) / [`../shared/profiling_mc2.md`](../../shared/profiling_mc2.md) | 通算并行调优 / 性能采集流程 |
