---
name: external-cannbot-model-model-train-precision-numerical-mismatch
description: 诊断 PyTorch on Ascend NPU 训练中有标杆可对照的有限值数值偏差。适用于同输入同 Step 下 Loss、Logits、激活、梯度或
  GradNorm 与标杆不一致的 E01/E02 症状；不用于 NaN/Inf/Overflow、重复运行不确定性、OOM 或推理精度问题。
original-name: model-train-precision-numerical-mismatch
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-26'
synced-commit: edcd4b94390c2a6b664f0935fbdd8a65e3644e9f
license: UNKNOWN
---

# 训练有限值精度偏差诊断

定位 E01 正向数值偏差和 E02 反向信号偏差。可通过单变量实验验证 Preflight 差异，或继续定位首个
Node/Rank/Step/Module/API/Tensor 差异；不要仅凭最终 Loss 或单个误差指标归因。

## 进入条件

- 已确认问题发生在训练路径，观测值均为有限值。
- 有可比较标杆，且标杆与异常侧的输入、权重、数据顺序、超参和确定性设置可对齐。
- 若受控重复运行自身不一致，先转 `model-train-precision-determinism`。
- 若出现 NaN/Inf/Overflow，转 `model-train-precision-nonfinite`。

没有标杆时，只记录缺口并协助建立标杆；不得运行 `msprobe compare`，也不得套用本技能的对齐阈值。

## 默认对齐门槛

用户或项目已有口径优先；否则使用：

| 指标 | 达标标准 |
| --- | --- |
| 首个 Loss 相对误差 | `< 0.5%` |
| 平均 Loss 相对误差 | `< 1%` |
| Global Norm 平均相对误差 | `<= 10%` |

标杆值为零时不得直接计算相对误差，改用项目约定的绝对误差口径并记录公式。

训练级门槛复用 `model-train-log-visualization` 对齐后的 `loss rel error` 和 `grad_norm rel error` 序列，但验收时取绝对值：
首个 Loss 使用第一个有效对齐 Step 的 `abs(loss rel error)`；平均 Loss 和 Global Norm 分别使用所有有效对齐 Step 的
`mean(abs(rel error))`。不得使用有符号均值抵消正负偏差；缺失、NaN 或未对齐 Step 不计为零，并记录有效样本数。

## 工作流程

### Step 1：建立可比较契约并复现

- 记录 target/golden 的实际训练入口、代码和工作区、数据与样本顺序、权重、训练参数、并行拓扑、版本和环境。
- 固定 Seed、计算与通信确定性、Dropout 和数据顺序；按相同 Node/Rank/Step/样本身份复现，不能按日志行号盲配。
- 先验证 target 自身可重复：若两次受控运行不自洽，停止 E01/E02 并转 E08；若出现 NaN/Inf/Overflow，转 E04。
- 无外部标杆或关键对齐项仍为 `unknown` 时，不执行 compare；先建立标杆，无法建立则以证据不足结束。

### Step 2：确认训练级 mismatch 与方向

- 使用对齐后的 Loss、Logits、激活、梯度和 GradNorm 确认首个异常 Step，并按项目口径或默认门槛判断症状是否成立。
- Logits、激活或 Loss 在正向首先出现有限值偏差时进入 E01；正向仍在门槛内、梯度或 GradNorm 首先偏差时进入 E02；证据不足时保留 `unknown`。
- Loss 在 Step N 首次偏差时，同时检查 Step N 的正向及 Step N-1 的反向和参数更新，避免把历史更新偏差误判为当前正向根因。
- 单独记录训练级验收结果；普通 compare 的 `Result` 或首差异文件不能替代这一步。

### Step 3：优先验证 Preflight 差异

- 审查已确认的数据、Label/Mask、权重加载、代码、训练语义、精度策略、环境、编译和并行差异，并为每项差异建立可证伪假设。
- 每轮只改变一个因素，在原复现窗口下执行对齐、修复或回退；需要启动训练、修改配置或源码时先取得用户批准。
- 若单一因素变化使偏差稳定消失，且同口径复跑通过，即可按数据、权重、配置、环境或并行粒度形成根因闭环；不强制继续 dump。
- 一次改变多个因素或仅发现相关差异时，最多保留为强候选。

### Step 4：采集 Statistics 并比较

- Preflight 未闭环时，读取 [实验与结果分析](references/experiment-and-analysis.md) 选择异常 Step、前一 Step、方向和最小采集范围；按
  [msProbe 配置模板](references/msprobe-configs.md) 编写配置。调用前先确认实际训练解释器中的 msProbe CLI、import 和所需 API 可用；
  任一不可用时停止 msProbe 工具分支并记录原始错误，本 Skill 不安装依赖或猜测包名。
- 从 L0/mix 定模块，再按需缩到 L1 `list`/`scope`；采集应覆盖待分析的 forward、backward 和必要的 optimizer 边界。
- 采集前获得授权，并声明不依赖 compare 结果的训练级症状信号、原复现窗口和等价判据。采集后立即核对退出状态、
  `stop()`/`step()`、预期 Step×Rank、`dump.json`、错误日志和范围，分别记录两侧 `dump_integrity`。
- 在插桩后的 target/golden 比较对上按 Step 2 的相同 Step、Rank、样本和训练级门槛判定 `symptom_reproduction`；两侧
  `dump_integrity=valid` 且比较对为 `reproduced` 才能 compare。比较已回到口径内或无法对齐时，阻断分析并按共享门禁处理。
- 普通 compare 使用 `Result`/`Err_message` 筛选候选；`compare -da` 按现场 `is_same`、`op_items` 和阈值文件定位首差异。
  这些节点级规则不得套用训练级 Loss/Global Norm 门槛。

### Step 5：追踪首差异

- 从首个训练异常 Step 向前追踪同 Rank、同调用实例的 Module/API 输入输出摘要，先解决名称、调用次数、shape 或控制流未对齐。
- 按 [实验与结果分析](references/experiment-and-analysis.md) 同时检查有限值的量级和误差趋势：E01 按 forward 顺序，E02 按
  backward 逆序，区分首个可观测差异边界、有限放大起点和后续放大器；任一位置都须经单变量验证，不能仅凭先后顺序认定根因。
- 首个问题 API 的输入已偏时，不把当前 API 定为误差引入点。记录最后一致节点 `node_a` 到首问题节点 `node_b` 的可疑区间，
  排查 target/golden 结构与映射、采集范围、漏采算子、融合/通信/控制流和 custom API 可见性，再继续追溯上游。
- 仅某个 Rank 不一致时保留 rank-local 证据，核对该 Rank 的数据、映射和通信参与者；通信候选可按结果分析规则选择性汇总多 Rank，
  不得直接推广为全局根因。
- 输入、参数和状态对齐而输出首次不一致时，普通 statistics compare 已提供 `Result`/`Err_message` 则直接消费工具结论：由
  NormRelativeErr 规则标为 `error`、元数据对齐且执行顺序最早的节点，记为“工具标记的误差引入候选”，不重复实现判定逻辑。
- 只有 `Result`/`Err_message` 缺失、冲突或现场版本不明时，才按 [实验与结果分析](references/experiment-and-analysis.md) 核验
  NormRelativeErr 单位和输入/输出阈值；其他统计量超标或 10%～50% 灰区只作待验证证据。
- statistics 摘要通常足以判断差异来自上游还是当前节点；首差异或单个红项仍只是定位证据，不自动等于根因。

### Step 6：按需采集 Tensor 并构造最小复现

- 只有构造最小算子复现确实需要真实输入时，才对已锁定候选采集 targeted tensor；不得默认执行整网 tensor dump。
- 使用落盘输入保留 dtype、shape、device、关键非 Tensor 参数和调用方向，构造最小前向；E02 或反向候选再加入最小反向。
- 具体 API/Module 已定位后，可按实验 Reference 设计 CPU 单变量对照；保持输入/权重/dtype/shape/非 Tensor 参数，并记录搬运、
  cast、隐式同步和性能变化。CPU 结果不能替代 NPU 修复验证。
- 最小单算子/通信脚本不复现时，只记录反证及缺失的整网并发、通信、内存和上下文条件；由 Plugin 编排时返回 Scope Reducer，
  在最后可复现的整网 R0 上继续单变量缩圈，不得直接排除候选。
- tensor dump、源码插桩和复现实验均需提前授权，并记录观察者效应。

### Step 7：验证并形成结论

- 对候选修复或回退后，在 Step 1 的同一契约和 Step 2 的同一门槛下复跑，同时检查是否引入新回归。
- 单变量因果闭环或内部首差异经最小复现/修复验证后，才能写 `verified root cause`；缺少验证时降为
  `strong candidate` 或 `insufficient evidence`。
- 输出 E01/E02、标杆与对齐口径、Node/Rank/Step、证据支持的最高定位粒度、回退状态和未闭环项。

## 输出要求

- 明确 E01 或 E02；无法判定时保留 `unknown`。
- 报告标杆对象、对齐口径、因果实验或首差异位置、误差指标和复现实验；不适用的 Module/API/Tensor 写 `not-applicable` 并说明原因。
- 区分 `verified root cause`、`strong candidate`、`insufficient evidence`。
- 记录 instrumentation 导致症状消失、Step 漂移或数值变化的观察者效应。

## 边界

- 不把相关性、最终 Loss 偏差或单次 compare 红项直接写成根因。
- 不用普通 compare 的 `Result`、`compare -da` 的 `is_same` 或 `diff_analyze` 代替 Loss/Global GradNorm 训练级验收。
- `dump_integrity` 为 `incomplete/unknown`，或 `symptom_reproduction` 为 `not-reproduced/unknown` 时不得 compare 或认证原症状根因。
- 文件完整但 mismatch 未复现时只保留为观察者效应/反证；回退 instrumentation 复核 R0 后，再经批准尝试更轻量采集。
- 不把 `summary_mode: "md5"` 称为密码学 MD5；PyTorch 当前实现记录 CRC-32，现场版本优先。
- 不在未经用户确认时启动训练、插桩源码或执行大规模 tensor dump。
- 在 Plugin 编排下执行多节点新实验时，服从 Primary 已确认的共享存储、现场启动方式、独立 run/attempt 和全局 Rank 汇集门禁；Skill 不自行生成调度器、SSH/pssh、host 清单、文件搬运或 rendezvous 命令。
