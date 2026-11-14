# E01/E02 实验与结果分析

## 1. 冻结比较契约

两侧至少记录：代码 commit/工作区差异、训练入口和命令、数据样本及顺序、Label/Mask、预训练权重、
模型结构、优化器和混合精度参数、TP/DP/PP/CP/EP、torch/torch_npu/CANN/msProbe 版本。
任何未确认项写 `unknown`，不得假定一致。

同一次定位中保持固定 Seed 和确定性设置。`seed_all` 的参数随版本变化，先检查现场签名；数据加载顺序仍需单独固定。

## 2. 优先验证 Preflight 差异

对已确认的数据、权重、代码、配置、环境或并行差异逐个建立假设。每次只改变一个因素，在原复现口径下执行对齐、修复或回退；
若偏差稳定消失并完成同口径复跑，可在对应粒度形成根因闭环，不强制执行 msProbe dump。一次同时改变多个因素只能说明相关性。

## 3. 选择 Step 与方向

- E01：在同输入同 Step 下先检查输入、Label、Mask、Logits、Loss reduction，再向正向上游追溯。
- E02：先确认正向仍在门槛内，再检查 loss scaling、反向 API、参数梯度、GradNorm、梯度同步和 optimizer 前边界。
- Loss 在 Step N 首次偏差时，同时检查 Step N 正向以及 Step N-1 的反向和参数更新。
- 多卡场景先按 Rank 对齐，再比较同名 Module/API；命名无法直接匹配时使用现场版本支持的映射能力。

## 4. 训练级 mismatch 判定

优先使用 `model-train-log-visualization` 对齐 target 与 golden 的 Step，并取得逐 Step 的 `loss rel error` 和
`grad_norm rel error`。该 Skill 的曲线保留误差方向；本诊断的验收值必须取绝对值，不能直接使用可能正负抵消的有符号 `mean`。

设异常侧为 `T`、标杆侧为 `G`，有效 Step 的有符号相对误差为 `(T-G)/abs(G)`，训练级验收固定为：

| 指标 | 计算值 | 默认门槛 |
| --- | --- | --- |
| 首个 Loss 相对误差 | 第一个有效对齐 Step 的 `abs(loss rel error)` | `< 0.5%` |
| 平均 Loss 相对误差 | `mean(abs(loss rel error))` | `< 1%` |
| Global Norm 平均相对误差 | `mean(abs(grad_norm rel error))` | `<= 10%` |

项目已有口径优先。缺失、NaN 或未对齐 Step 不得填零；记录总对齐 Step 数、有效 Step 数和被排除项。标杆值为零时，
该 Step 不计算相对误差，改用项目声明的绝对误差公式；没有项目绝对误差口径时把该项写为 `unknown`。

上述指标负责确认训练症状是否达到 mismatch 标准。statistics compare 的任意 API/Module 行、`Result`、`is_same` 和
`diff_analyze_*.json` 只负责内部节点定位，不能覆盖或替代训练级验收结论。

## 5. Statistics 与可选 Tensor

### Statistics

先采集异常 Step 及其前一 Step。大模型优先 L0/mix 定模块，再用 `list`/`scope` 缩到 L1 API。
`start` 应覆盖待分析的 forward、backward 和必要的 optimizer 边界；L0/mix 是否需要传 `model` 以现场签名和官方文档为准。

执行前检查：

```bash
msprobe --help
msprobe compare --help
```

采集前声明不依赖 compare 结果的训练级症状信号、可比窗口和等价判据。采集结束后立即核对：进程正常退出，预期 Step 已执行
现场要求的 `stop()`/`step()`，每个预期 `Step × Rank` 的 `dump.json` 存在、非空、可解析且采集范围匹配，并检查
`dump_error_info.log`。target 和 golden 分别记录 `dump_integrity=valid / incomplete / unknown`；再按对齐日志和训练级门槛对
该比较对记录 `symptom_reproduction=reproduced / not-reproduced / unknown`。只有两侧完整且插桩后 mismatch 仍复现才能 compare。
API 数量或 key 集不同先核对控制流、调用次数、映射和采集范围，不直接判成采集不完整，也不盲目配对。

同卡或多卡路径格式必须按现场 `compare --help` 和实际 dump 目录确定。启用 `-da` 时，当前官方语义会生成
`compare_result_rank{rank_id}_{timestamp}.json` 与 `diff_analyze_{timestamp}.json`；普通 compare 还可能生成 CSV/XLSX。
只分析实际存在且命令明确报告成功的文件。

### 有限值趋势、首个可观测差异与放大位置

在首差异附近分开记录“首个可观测差异边界”“有限量级异常从哪里开始”和“差异在哪里继续放大”：

- E01 按实际 forward 执行顺序遍历；E02 先确认正向仍在门槛内，再按 backward 的实际逆向传播顺序分析。
- 对同 Step、Rank、调用实例和可比 Tensor，优先读取本次有效 PyTorch msProbe statistics dump 的 `dump.json` 原始 `max`、`min`、
  `mean`、`norm`，以及普通 compare 实际输出的对应 diff/relative error。字段缺失、为 `null` 或现场版本语义不明时写
  `unknown`，不得补造。近零分母时保留原始统计量和项目口径，不补造相对误差。
- statistics 摘要相等不证明真实 Tensor 逐元素相等。按现有证据最早由一致变为不一致的位置记为“首个可观测差异边界”；只有输入、
  参数、状态和调用均对齐而输出首次偏离时，当前节点才是“误差引入候选”，仍不是已验证根因。
- 在首个可观测差异之前，若 target 相对 golden 已出现可重复的异常量级增长，但摘要精度或采集范围尚不足以判为差异，记为“有限放大
  起点候选”并补证；若两侧始终同幅增长，只记为数值风险。输入已经偏差而输出误差或绝对量继续增长的节点记为“后续放大候选”。
- 不设置跨模型通用倍数或阈值。有限放大起点、误差引入节点和后续放大器可以不同；必须在同一契约下重复复现并通过单变量修复/回退，
  才能提升为根因或已确认贡献因素。

### Statistics 模式 API 级定位 mismatch 判定

statistics compare 比较两侧张量摘要，不是逐元素真实 Tensor。必须先确认训练级 mismatch，再用 API/Module 规则定位；两层阈值
相互独立：

| 层级/指标 | 默认判定口径 | 用途与限制 |
| --- | --- | --- |
| 训练级首个/平均 Loss 相对误差 | 首个 `< 0.5%`、`mean(abs(loss rel error)) < 1%` | E01 整体验收；用户或项目明确标准优先 |
| 训练级 Global GradNorm 平均相对误差 | `mean(abs(grad_norm rel error)) <= 10%` | E02 整体验收；不能反推任一 API 的局部阈值 |
| statistics API/Module 的 `NormRelativeErr` | 非空的可比 input/parameters 集合全部 `< 10%`，且某个 output `> 50%` | 当前普通 compare 的 `error` 语义；常规路径读取 `Result`/`Err_message`，该阈值只用于解释或异常时复核 |
| output 相对 input/parameters 的 `NormRelativeErr` 放大 | 官方文档表述为 10 倍；当前本地源码为 `output/input > 10` | 普通 compare 的 `warning` 语义；常规路径读取工具标记，不能替代更早的引入点 |
| `Max/Min/Mean/L2 Norm diff`、`MaxRelativeErr`、`MinRelativeErr`、`MeanRelativeErr` | 无跨模型通用硬阈值 | 受数值尺度、近零分母和聚合方式影响，只作定位证据；阈值必须来自用户、项目或实验 |

正常情况下不要在 Agent 中复制 msProbe 的 Result 判定器。按以下方式消费普通 compare 结果：

1. 先确认 Step、Rank、forward/backward、调用实例和执行顺序对齐，API/Module 映射有效。
2. 读取每行 `Result` 和 `Err_message`。若 `Result=error` 且原因明确为 NormRelativeErr 输入/输出规则，把按已验证执行顺序最早的
   一项记为“工具标记的误差引入候选”；不再手工重算 `< 10%`/`> 50%` 以制造第二套判定结果。
3. `Result=warning` 且原因为 NormRelativeErr 放大时，只记为“工具标记的放大候选”，继续检查是否存在更早的 error 或输入偏差。
4. error 由 shape、dtype、requires_grad 或非 Tensor 参数触发时，按对应结构/参数差异分类，不把它改称为 NormRelativeErr
   误差引入；NPU 独有 NaN/Inf 应停止 E01/E02 并重新路由 E04。

只有 `Result`/`Err_message` 缺失、两者与原始指标冲突，或现场版本规则无法确认时，才人工复核：

1. 记录结果 schema、原始单元格、`Err_message`、msProbe 版本及实际分析源码；不要静默覆盖工具结果。
2. 核验 NormRelativeErr 规则时，确认至少存在一个有效的可比较 input/parameter，所有可比较 input/parameters `< 10%`，且当前
   某个 output `> 50%`。shape、dtype、requires_grad、非 Tensor 参数或映射仍有缺陷时，不能套用该数值规则。
3. 当前官方公式按百分比展示，本地源码把带 `%` 的 CSV/XLSX 字符串除以 100 后再与内部比例阈值比较：`8%`/`55%` 对应
   `0.08`/`0.55`。不带 `%` 时先确认它是百分数 `8`/`55` 还是比例 `0.08`/`0.55`，不得猜测。
4. 10 倍规则使用归一化后同单位的 `output/input`；当前本地源码使用严格 `> 10`，输入最大误差为零时另以输出 `> 10%`
   触发 warning。边界以现场版本为准。
5. 只有 Max/Min/Mean 等指标、缺少可用 `NormRelativeErr`，或 output 落在 10%～50% 时，不能补造上述 error；满足现场放大规则
   时最多保留 warning 候选，否则只作为待验证证据。部分通信 API 的输入为占位值，涉及输入的 Result 规则按现场特殊场景列表
   排除，不能机械套用“输入 `< 10%`”。

普通 compare 的 `Result=pass` 不代表训练级 Loss/Global GradNorm 门槛通过；`warning/error` 也不能直接认证训练级 mismatch、
首差异或根因。

`compare -da` 的逐 Rank JSON 不保证包含普通 compare 的 `Result`。已核对的当前源码主要使用 `is_same`、`op_items` 以及
安装包内 `diff_analyze_threshold.yaml` 指定的 `MaxRelativeErr`、`MinRelativeErr`、`MeanRelativeErr`、
`NormRelativeErr` 阈值；当前源码默认均为 `0.5`（50%）。执行前必须从实际导入的 msProbe 包目录定位并记录该文件的
路径、哈希和实际阈值，不得硬编码，也不得把它换成项目的 0.5%/1%/10% 门槛。`diff_analyze_*.json` 只定位首差异。

若 Loss 和 Global GradNorm 被明确采集为标量输出，可作为日志验收的交叉验证：Loss 查看对应输出行的
`MeanRelativeErr`，Global GradNorm 查看对应输出行的 `NormRelativeErr`。使用前必须确认 Step、Rank、调用实例、
`state=output` 和标量语义；不得把其他 API 行按这两个阈值判定。

规则来源：[msProbe PyTorch 精度比对指标说明](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)。
目标版本的规则、单位或特殊场景发生变化时，以现场安装包、实际结果和 `Err_message` 为准并记录差异。

compare 完成后还要检查返回码、成功日志和每个预期 Rank 的非空可解析结果。缺 Rank 或结果损坏时，本次 compare 为
`incomplete`；不得仅凭 `diff_analyze_*.json` 是否存在判断 compare 完整。

### 首差异分支

#### 首个问题 API 的输入已经不一致

无论来自普通 compare 还是 `compare -da`，该结果都只证明偏差在当前 API 之前已经存在，不能把当前 API 定为误差引入点或根因。
先确认 Step、Rank、forward/backward、调用序号和 shape 已正确对齐，再把最后一个输入输出均一致的可见上游节点记为
`node_a`，首个问题 API 记为 `node_b`。将 `node_a → node_b` 记录为“可能漏采、无法对齐或两侧结构不同”的可疑区间，不能直接
断言 msProbe 漏采。

按以下顺序检查：

1. 核对 target/golden 的 statistics `level`、`scope`/`list`、`data_mode`、方向、调用次数和 Step/Rank 范围完全对应。
2. 核对两侧模型结构、代码路径、框架/版本和 API 命名。结构确有差异时按现场 `fuzzy_match`、`data_mapping`、`cell_mapping` 等
   能力判断是否可比；无法建立一一映射时记录结构差异，不强行把区间配成同一计算链。
3. 检查框架包装、融合、通信、控制流和原地操作之间是否存在未显示的算子/API，保留 `node_a` 与 `node_b` 输入的原始证据。
4. 检查项目 custom API 是否被两侧一致采集。确认运行时实际导入模块，以 `inspect.signature` 核对 `register_custom_api`、
   `restore_custom_api` 或现场等价接口；target/golden 使用相同注册集合、顺序和生命周期。
5. 仍无法闭合时，经 Primary 批准把区间提升到现场支持的 mix，或仅对区间候选做 targeted tensor。先补证据，不直接修改模型逻辑。
6. 补采后若 `node_b` 输入仍先偏，结合 Preflight 已确认的数据/样本、权重和 buffer、optimizer 状态、精度策略、代码、环境、并行
   与通信证据继续追查更早上游；区间仍不可见时保持 `insufficient evidence`。

#### 仅某个 Rank 不一致

保留该 Rank 的原始 compare、Step、样本 ID、进程角色、target/golden Rank 映射和拓扑证据，并与其他 Rank 分开判断。检查该 Rank
的数据分片、参数/optimizer 分片、pipeline stage、collective 参与者与调用顺序、点对点通信配对和两侧拓扑是否等价。单 Rank
差异不能直接推广为全局算法根因；首差异在 collective 输出时，还要检查所有参与 Rank 的对应输入、group 和调用序号。已有满足
输入约束的普通 statistics/tensor XLSX 时，可进入下文 `merge_result` 可选分支横向观察，但汇总结果不能替代逐 Rank 原始证据。

#### 相同输入下输出首先不一致

只有输入、调用实例、参数/buffer/optimizer 状态、非 Tensor 参数和通信参与者都对齐后，才能把当前 API/Module、精度策略、
融合/编译路径、运行栈或通信归约列为内部数值强候选。普通 compare 按上文 `Result`/`Err_message` 记录“工具标记的误差引入候选”；
`compare -da` 按现场 `is_same`、`op_items` 和阈值文件解释，不能混用两套 schema。候选仍需最小复现或修复/回退复跑才能认证根因。

### 通信或 rank-local 可疑时：可选 merge_result

仅在 E01/E02 的普通 compare 已把候选收敛到通信 API/Module，或显示只有部分 Rank 偏差时，使用 `merge_result` 横向查看指定
通信节点在各 Rank 的数值指标。它是结果分析的可选分支，不用于 E04/E08，也不因为作业是多卡就默认执行。

执行顺序：

1. 先完成训练级 mismatch 判定、双侧 dump 完整性检查、插桩后症状复现判定和普通 compare；只有
   `dump_integrity=valid + symptom_reproduction=reproduced` 才能执行 compare。计划汇总时，普通 compare 必须使用 `--xlsx`。
2. 对照预期 Rank 检查每份 `compare_result_rank*.xlsx` 存在、非空且可读取；核对无缺失、重复或冲突 Rank，并确认所有文件的
   target/golden、Step、调用实例和 dump 模式一致。任一项无法确认时停止汇总，不能用已存在的部分 Rank 代替集群结论。
3. 运行 `msprobe merge_result --help` 探测现场参数。当前官方命令格式为：

   ```bash
   msprobe merge_result -i <compare_xlsx_dir> -o <output_dir> -config <config.yaml>
   ```

   配置只列入待验证的通信 API/Module 和所需指标；`compare_index` 必须属于 statistics 或 tensor 模式的现场支持集合。
4. 当前版本不接受 CSV、MD5 或 `compare -da` JSON。已有结果不满足输入条件时，不做格式猜测或伪转换；是否重新 compare/采集
   由 Primary 按授权和证据增益决定。
5. 命令成功后确认 `multi_ranks_compare_merge_*.xlsx` 存在、非空且可读取，并核对预期 API、指标 sheet 和 Rank 列。单元格空白
   表示该 Rank 未找到相应 API/Module，不得解释为 `pass`；先回查调用实例、采集范围和逐 Rank 原始 compare。
6. 使用汇总表识别 rank-local 离群或通信前后误差分布，再回到逐 Rank 证据检查该 collective 的所有参与 Rank 输入、调用序号、
   group、拓扑和上游生产者。汇总结果不能单独认证通信算子或某个 Rank 为根因，也不能替代 Loss/Global GradNorm 门槛。

### Tensor（可选）

statistics 的输入/输出摘要通常足以判断差异来自上游还是当前节点。需要使用真实输入构造最小单算子复现时，才对锁定的候选采集
真实 Tensor，避免整网全量 dump。新的 tensor 采集同样必须重新通过完整性与症状复现联合门禁；采集后还可核对：

- 输入是否已经偏差；
- 正常输入下输出是否首次越过阈值；
- dtype、shape、requires_grad 和非 Tensor 标量是否一致；
- 多次调用实例是否正确对齐。

### 定位后的 CPU 单变量对照（可选）

只有候选已收敛到具体 API/Module，且实验已获批时才使用 CPU 对照；它不是 Scope Reduction：

1. 保持相同输入、权重、dtype、shape、调用方向和非 Tensor 参数，只改变候选计算设备；记录 device 搬运、显式/隐式 cast、同步和性能变化。
2. 目标侧 CPU 替代用于测试 target 的 NPU 路径；必要时对 golden 侧执行同构 CPU 负对照，以区分平台路径与模型/实现敏感性。
3. CPU 输出恢复一致只形成 NPU 路径强候选。CPU/NPU 算法或精度路径不同、搬运造成同步、参数语义未对齐时，实验不能形成因果证据。
4. 实验后回退到 R0′，并在 NPU 正式修复后按原训练级门槛复跑；CPU 结果不能代替该闭环。

## 6. 最小复现与归因

最小脚本固定使用候选 API 的 dump 输入，保存 dtype/shape/device、前向和必要的反向，输出 max/min/mean/norm、
项目规定的绝对/相对误差或 cosine。必须记录运行命令和版本。

若最小单算子或通信脚本不复现，记录脚本相对整网缺少的通信/计算并发、内存竞争、拓扑和上下文状态。该结果是反证而非排除证据；
由 Plugin 编排时回到最后稳定复现的整网 R0，按 Scope Reduction 一次只改变一个因素并验证 R0′，再把范围缩减结果交回 E01/E02。

以下任一路径完成后才可写为已验证根因：

- 单变量因果闭环：Preflight 差异已确认，仅改变该因素后偏差消失，并完成同口径复跑；
- 内部数值定位闭环：输入在口径内、候选输出首次异常、同一输入可重复，修复或回退后问题消失且无新回归。

前一路径不强制提供 Module/API/Tensor；不适用字段记录 `not-applicable` 及原因。

## 官方资料

- [PyTorch 场景数据采集](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/pytorch_data_dump_instruct.md)
- [PyTorch 精度比对](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)
- [配置 JSON 说明](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/config_json_introduct.md)
