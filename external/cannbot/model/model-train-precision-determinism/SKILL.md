---
name: external-cannbot-model-model-train-precision-determinism
description: 诊断 PyTorch on Ascend NPU 训练中固定输入、权重、Seed、确定性计算和配置后重复运行仍不一致的 E08 症状。适用于两次运行
  Loss、GradNorm、梯度或中间张量不一致；不用于单次曲线波动、Checkpoint 续训对齐、NaN/Inf 或稳定的标杆有限值偏差。
original-name: model-train-precision-determinism
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-26'
synced-commit: edcd4b94390c2a6b664f0935fbdd8a65e3644e9f
license: UNKNOWN
---

# 训练确定性复现偏差诊断

把同一受控环境的重复运行作为天然标杆，验证随机性契约或 Preflight 差异；未闭环时定位第一个无法固定的 Module/API/Tensor。

## 进入条件

- 数据集及顺序、初始化或加载权重、训练参数和运行环境受控一致。
- 已固定 Seed，并开启现场框架支持的计算与通信确定性。
- 至少两次独立运行的 Loss/GradNorm 或中间结果仍不一致，且没有先发 NaN/Inf。

Checkpoint 断点续训轨迹不一致和普通训练噪声不属于本技能。

## 工作流程

### Step 1：建立完整随机性契约

- 记录进程启动方式、Python/NumPy/PyTorch/NPU Seed、各 Rank 派生规则、sampler epoch、shuffle、worker seed、样本身份和顺序。
- 同时记录 Dropout、随机 API、MoE routing 噪声、初始化/加载权重、optimizer/scaler 初态、计算与通信确定性、编译模式和并行拓扑。
- `seed_all` 只能作为辅助；先探测现场签名和覆盖范围，不能用“已固定一个 Seed”代替完整契约。
- 在获批的短受控复跑中读取 [实验与结果分析](references/experiment-and-analysis.md)，确认实际加载代码里的所有 Seed/确定性设置入口，
  并在初始化后、首次 forward/backward 前、首个 Step 结束后记录当时实际生效且可验证的随机性与确定性状态；检查业务代码、
  框架封装和配置重载是否重复覆盖设置。
- 上述运行时状态在首差异前发生变化时，先做单变量修正、回退和 A/B 复跑；未闭环前不进入算子 dump。
- 任一运行先出现 NaN/Inf/Overflow 时停止 E08 并转 E04。

### Step 2：执行并对齐独立重复运行

- 在同一契约下执行至少两次独立运行，以 run A/run B 互为天然标杆；启动重复训练前先取得用户批准。
- 按 Step、Rank、样本 ID 和调用实例对齐，不按日志行号盲配；模型状态和 optimizer/scaler 初态也必须一致。
- 记录 Loss、GradNorm 或项目指定信号的比较口径、有效 Step 和首差异 Step 分布。每次都复现但首差异 Step 不固定，仍记为稳定复现。

### Step 3：判断确定性控制是否已经消除症状

- 若两次运行在项目口径下已一致，记录“确定性控制后症状消失”并停止大规模 dump。
- 若只改变一个已确认因素后多组 A/B 均恢复一致，可按该随机性契约、配置、环境或并行粒度形成因果闭环。
- 若一次补齐多个 Seed、数据顺序或确定性控制后才恢复一致，只能判定与随机性/非确定性相关；不能命名其中某一项为根因。

### Step 4：逐项验证 Preflight 与随机性差异

- 对已确认的 Seed 派生、sampler/worker、Dropout、随机 API、确定性开关、环境、编译或并行差异逐项建立假设。
- 每轮只改变一个因素，保持其余契约不变并执行多组 A/B；修改配置、代码或拓扑前先取得用户批准。
- 单变量实验未闭环时只保留候选，不因某项“看起来不一致”直接归因。

### Step 5：使用校验值定位首差异

- 外部差异未闭环且 A/B 仍不一致时，读取 [实验与结果分析](references/experiment-and-analysis.md)，按
  [msProbe 配置模板](references/msprobe-configs.md) 选择最小 Step、Rank 和采集范围。调用前先确认实际训练解释器中的 msProbe CLI、
  import 和所需 API 可用；任一不可用时停止 msProbe 工具分支并记录原始错误，本 Skill 不安装依赖或猜测包名。
- 经授权分别采集 `statistics` + `summary_mode: "md5"`；当前 PyTorch 字段实际记录 CRC-32 校验值，不是密码学 MD5，也不表示误差幅度。
- 采集前声明不依赖 compare 结果的 A/B 不一致信号、原复现窗口和等价判据。采集后分别核对退出状态、`stop()`/`step()`、
  预期 Step×Rank、`dump.json`、错误日志和范围，记录两侧 `dump_integrity`。
- 在插桩后的 A/B 比较对上按 Step 2 的相同契约和口径判定 `symptom_reproduction`；两侧 `dump_integrity=valid` 且比较对为
  `reproduced` 才能 compare。A/B 已一致或无法对齐时，阻断分析并按共享门禁处理。
- 使用现场 compare schema 和首差异产物定位不相等节点；`bench_path` 等能力仅在现场版本明确支持时使用。
- 按实验 Reference 定位首个可观测不相等边界；边界之后的差异扩大属于传播或影响，不作为 E08 起源定位轴线。边界不自动等于
  起源，只有输入、参数、状态、调用和通信参与者均对齐而输出首次不同时，当前节点才是非确定行为引入候选。

### Step 6：区分随机输入与内部非确定行为

- 若 compare 报告的首个问题 API 输入已经不一致，不把该 API 判为确定性根因。把最后一个已对齐节点记为 `node_a`、
  首个问题 API 记为 `node_b`，将 `node_a → node_b` 标为可能漏采或无法对齐的可疑区间。
- 对可疑区间先检查采集范围、调用实例和疑似遗漏算子，确认 custom API 是否已按现场签名注册；必要时经授权扩大到 mix 或
  targeted tensor。不要为了补齐证据直接修改模型计算逻辑。
- 补采后输入仍先不一致时，引用 Preflight 已核对的模型状态、数据和随机性证据，继续追查数据顺序、随机 API、Rank Seed、
  控制流、通信或更早的上游节点。
- 相同输入下输出首先不同：把候选收敛到非确定算子、融合/编译路径或通信归约。
- 仅某个 Rank 不一致：保留 rank-local 证据，检查该 Rank 的数据分片、Seed、通信配对、进程角色和拓扑；不得直接推广为全局根因。
- 同名 Module/API 调用次数或 shape 不同：先处理样本、控制流和调用实例对齐，不能强行比较。
- 校验值差异只证明不相等；需要量化误差或验证真实输入输出时再进入 targeted tensor。

### Step 7：按需采集 Tensor 并构造最小复现

- 只有构造算子或通信最小复现需要真实输入时，才对已锁定的不相等边界或引入候选采集 targeted tensor；不得默认全量采集。
- 固定 dump 输入、调用参数、Rank/通信参与者和执行次数，分别重复候选计算，判断相同输入能否稳定产生相同输出。
- 具体 API/Module 已定位后，可按实验 Reference 设计 CPU 单变量对照；CPU 会改变算法和同步，症状消失不能直接证明 NPU 算子不确定。
- 全量 Dump 改变时序、轻量采集已锁定候选而常规单算子脚本不复现时，按实验 Reference 经批准分别选择：用 `mssanitizer
  --tool=racecheck` 拉起实际调用候选 Kernel/API 的单算子 application，检查算子内竞争；或依据整网最后一致到首个不相等边界的
  真实子图和流/事件关系构造最小并发图，检查算子间多流竞争。工具报告或并发复现只形成候选，仍需修复/回退和整网 R0 验证。
- 上述检查不可用、仍不复现或未闭环时，只记录反证及缺失的整网条件；由 Plugin 编排返回 Scope Reducer，在整网 R0 上继续单变量
  实验，再回到同一 E08 路径。
- tensor dump、源码插桩、通信或算子最小复现实验均需提前授权，并记录 instrumentation 造成的差异消失或迁移。

### Step 8：验证并形成结论

- 修正候选随机性契约、替换/配置确定性算法或回退环境后，用 Step 2 的同一口径完成多组 A/B 复跑。
- 单变量变化使重复运行稳定一致，或内部首差异经最小复现和修复验证后，才能写 `verified root cause`。
- 仅发现校验值差异、首差异节点或一次不复现时，降为 `strong candidate` 或 `insufficient evidence`，并列出未控制因素。

## 现场版本要求

先检查 `msprobe --help`、`compare --help`、Python API 签名和实际配置解析结果。
不要假定 `bench_path`、`diff_nums`、`is_enhanced` 或 custom API 注册在所有版本可用。

## 输出要求

- 列出两次运行的环境、代码、权重、样本/Step/Rank 对齐证据。
- 给出单变量因果证据，或首差异 Module/API/Tensor 及输入/输出判定；不适用字段写 `not-applicable` 并说明原因。
- 将数据顺序、随机 API、非确定算子、通信归约和未控制环境分开归类。
- 记录 instrumentation 导致差异消失或迁移的观察者效应。

## 边界

- 不把“固定了一个 Seed”当作完整确定性证明。
- 不把校验值不同直接等价为数值误差大小；需要真实 Tensor 才能量化差异。
- `dump_integrity` 为 `incomplete/unknown`，或 `symptom_reproduction` 为 `not-reproduced/unknown` 时不得 compare、声明原始运行一致或认证根因。
- 文件完整但 A/B 差异未复现时只保留为观察者效应/反证；回退 instrumentation 复核 R0 后，再经批准尝试更轻量采集。
- 不在未经用户确认时启动重复训练或大规模 dump。
- 在 Plugin 编排下执行多节点新实验时，服从 Primary 已确认的共享存储、现场启动方式、独立 run/attempt 和全局 Rank 汇集门禁；
  Skill 不自行生成调度器、SSH/pssh、host 清单、文件搬运或 rendezvous 命令。
