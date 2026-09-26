---
name: external-cannbot-model-model-train-precision-nonfinite
description: 诊断 PyTorch on Ascend NPU 训练中的 NaN、Inf 与 Overflow（E04）。适用于 Loss、Logits、激活、梯度、参数或
  Optimizer/Scaler 状态首次出现非有限值；不用于仅有有限值的标杆偏差、重复运行不确定性、OOM 或推理问题。
original-name: model-train-precision-nonfinite
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-26'
synced-commit: edcd4b94390c2a6b664f0935fbdd8a65e3644e9f
license: UNKNOWN
---

# 训练非有限值诊断

验证导致 NaN/Inf/Overflow 的外部差异，或定位第一个产生非有限值的计算节点及传播路径，区分前向、反向和 Optimizer/Scaler 三类边界。

## 进入条件

- 训练观测或 dump 中出现 NaN、`+Inf`、`-Inf` 或明确 Overflow 信号。
- 若只有有限值偏差，转 `model-train-precision-numerical-mismatch`。
- 外部标杆不是必需条件，但相同输入的正常运行可提高归因可信度。

## 工作流程

### Step 1：冻结复现条件并记录首发现象

- 固定随机性、确定性、数据顺序和训练配置，在原拓扑或已确认的最小可复现拓扑中复现；新增训练作业需先取得用户批准。
- 记录最先出现 NaN、`+Inf`、`-Inf` 或 Overflow 的 Node/Rank/Step、日志字段和首次观测边界，不把后续 Loss NaN 当作首发位置。
- 若实际只有有限值偏差，停止 E04 并转 E01/E02；外部标杆不是进入 E04 的必要条件。

### Step 2：先用 Logits 划分阶段

- 在实际训练步骤的 `logits` 观测点检查有限性：`logits` 首先非有限，判为 forward；`logits` 有限而反向或梯度首先非有限，判为 backward。
- `logits` 和梯度均有限，但参数、Optimizer/Scaler 状态或下一 Step 的已更新权重首先异常时，判为 optimizer-scaler。
- 现有证据无法覆盖首发边界时保留 `unknown`，再用最小侵入观测补齐；显式同步造成症状消失或迁移时记录观察者效应。

### Step 3：优先验证 Preflight 差异

- 审查权重加载、代码、数据、混合精度、loss scale、优化器状态、环境、编译和并行差异，并为已确认的单一差异建立假设。
- 每轮只对齐、修复或回退一个因素，在原 Node/Rank/Step 窗口复跑；一次改变多个因素不能形成因果闭环。
- 单变量变化使非有限值稳定消失并通过同口径复跑时，可按对应配置、权重、环境或并行粒度认证根因，不强制继续内部定位。

### Step 4：按需使用 detect_anomaly

- Preflight 未闭环且症状为反向报错或反向 NaN 时，先确认运行时实际加载的训练文件，再让
  `torch.autograd.detect_anomaly(check_nan=True)` 覆盖相关 forward 和 backward。
- 只把失败反向函数对应的前向 traceback 和反向 NaN 抛错作为候选证据；不得宣称它覆盖前向 NaN、任意阶段 Inf 或完整非有限值检查。
- 若出现已知 `_is_any_true` 分片策略兼容错误，记录现场版本和原始报错，按下文兼容说明处理；若症状不复现，记录观察者效应并转轻量 dump。

### Step 5：采集 Statistics 并运行 overflow_check

- 仍未闭环时读取 [实验与结果分析](references/experiment-and-analysis.md)，按
  [msProbe 配置模板](references/msprobe-configs.md) 选择异常 Step、前一 Step 和最小范围。调用前先确认实际训练解释器中的 msProbe
  CLI、import 和所需 API 可用；任一不可用时停止 msProbe 工具分支并记录原始错误，本 Skill 不安装依赖或猜测包名。
- 经授权采集 statistics；采集前声明不依赖 overflow_check 结果的非有限值信号、原 Node/Rank/Step 窗口和等价判据。
  采集后先核对退出状态、`stop()`/`step()`、预期 Step×Rank、`dump.json`、错误日志和范围，记录 `dump_integrity`。
- 再用同次插桩运行的日志、原始报错、显式有限性信号或完整 `dump.json` 中的直接非有限统计判定 `symptom_reproduction`。只有
  `dump_integrity=valid` 且 `symptom_reproduction=reproduced` 才能运行 `overflow_check`；不得用分析文件是否生成反推症状复现。
- 有异常时读取现场实际生成的 `anomaly_analyze_*.json`；工具明确成功且报告无异常时可能没有分析文件。
  不得寻找或使用 compare/diff 产物代替 overflow_check 结果。

### Step 6：复核非有限转换边界、有限放大与传播链

- 优先检查有限输入首次产生非有限输出的转换点；输入已异常时继续向生产者追溯，搬运/复制节点通常只保留为传播证据。
- 从首个非有限转换点向前检查有限前驱的量级和趋势，区分更早的有限放大起点与后续放大器。转换点只界定 NaN/Inf 出现边界，
  放大起点、后续放大器和转换点均不能未经单变量验证直接写成根因。
- 使用现场实际生效的 `ignore_rules.yaml`；初始化/占位、通信野值和 inplace 输入即使命中过滤规则，也要检查实际写入者或消费者。
- `where`、`triu`、`tril` 等产生的合法 `-Inf` 必须同时满足 mask/位置/消费者语义；合法 `-Inf` 不豁免下游首次 NaN。
- 每次过滤记录算子、Tensor、Rank、理由及下游检查结果，不能靠白名单直接认证根因。

### Step 7：按需采集 Tensor 并最小复现

- 只有构造最小算子复现需要真实输入时，才对已锁定的有限放大或非有限转换候选采集 targeted tensor；不得默认进行整网全量 tensor dump。
- 使用固定 dump 输入复现前向，反向候选增加最小反向；保留有限性、dtype、shape、参数和输入输出关系。
- 具体 API/Module 已定位后，可按实验 Reference 设计 CPU 单变量对照。CPU 上不再出现非有限值只形成强候选，仍需 R0′ 和 NPU
  路径正式修复验证。
- 最小脚本不复现时，不排除整网并发、通信、内存或上下文影响；由 Plugin 编排时返回整网 Scope Reduction，再进入同一 E04 路径。
- tensor dump、源码插桩和最小复现实验均需提前授权。

### Step 8：修复验证与证据分级

- 对候选修复或回退后，在 Step 1 的同一复现窗口复跑，确认非有限值消失且未迁移到其他 Node/Rank/Step。
- 单变量因果闭环，或已锁定候选的输入输出关系经最小复现及修复验证后，才能写“已验证根因”。
- 只有 traceback、传播节点、最终 Loss NaN 或不完整 dump 时，降为“强候选”或“证据不足”，并列出最小下一步。

## detect_anomaly 语义边界

`detect_anomaly(check_nan=True)`：

- 让失败的反向计算打印创建该 backward function 的前向 traceback；
- 在反向计算生成 NaN 时抛错；
- 不是通用的前向/反向 NaN/Inf 检查器，不保证捕获前向 NaN 或反向 Inf。

因此必须保留显式 `torch.isfinite` 或 dump 证据。启用前用 `inspect.getsourcefile` 确认运行时实际导入的训练文件，并记录性能下降或症状消失等观察者效应。

若出现 `NotImplementedError: Operator aten._is_any_true.default does not have a sharding strategy registered`，按现场 PyTorch 版本核对并参考
[PyTorch PR #170951](https://github.com/pytorch/pytorch/pull/170951) 中的临时规避方法；记录原始报错、版本和处理方式。

## overflow_check 产物边界

- 当前官方语义下，有异常节点时主分析产物是 `anomaly_analyze_{timestamp}.json`。
- 工具明确成功并报告没有异常节点时，可能不生成分析文件。
- `compare_result*` 和 `diff_analyze*` 属于 `msprobe compare -da`，不得当作 overflow_check 产物。
- 没有分析文件不等于未采集范围外不存在异常。
- `dump_integrity` 为 `incomplete/unknown`，或 `symptom_reproduction` 为 `not-reproduced/unknown` 时不得运行 overflow_check 或认证原症状根因。
- 文件完整但非有限值未复现时只保留为观察者效应/反证；回退 instrumentation 复核 R0 后，再经批准尝试更轻量采集。
- 应用过滤规则前必须定位并核对现场安装版本的 `ignore_rules.yaml`；不得假定其路径、内容或规则集合与当前文档一致。
- 内置过滤命中只表示该节点需按规则排除或继续追溯，不能替代对实际写入者、生产者和消费者的人工复核。

## 输出要求

- 分类为 forward / backward / optimizer-scaler / unknown。
- 给出单变量因果证据，或有限放大起点、首个非有限转换点及其 Node/Rank/Step/Module/API/Tensor、输入输出有限性和传播链；不适用字段写 `not-applicable` 并说明原因。
- 每个白名单过滤记录算子、Tensor、Rank、理由及下游检查结果。
- 结论区分已验证根因、强候选和证据不足。

在 Plugin 编排下执行多节点新实验时，服从 Primary 已确认的共享存储、现场启动方式、独立 run/attempt 和全局 Rank 汇集门禁；
本 Skill 不自行生成调度器、SSH/pssh、host 清单、文件搬运或 rendezvous 命令。

## 官方资料

- [msProbe overflow_check](https://gitcode.com/Ascend/msprobe/blob/master/docs/en/user_guide/overflow_check/overflow_check_instruct.md)
- [PyTorch detect_anomaly](https://docs.pytorch.org/docs/2.8/autograd.html#torch.autograd.detect_anomaly)
