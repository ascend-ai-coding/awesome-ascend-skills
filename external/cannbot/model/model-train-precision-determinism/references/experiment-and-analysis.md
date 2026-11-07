# E08 实验与结果分析

## 1. 随机性契约

至少确认：

- Python/NumPy/PyTorch/NPU Seed 及每 Rank 派生规则；
- sampler epoch、shuffle、worker 数、worker seed、样本 ID 和顺序；
- Dropout、stochastic depth、随机增强、MoE routing 噪声；
- 计算确定性和 HCCL 通信确定性；
- 初始化权重、预加载权重和 optimizer/scaler 初态；
- 代码、环境、编译模式和并行拓扑。

`msprobe.pytorch.seed_all` 可以辅助固定多项随机性，但参数和覆盖范围随版本变化，数据读取顺序仍需单独确认。

### 1.1 运行时随机性与确定性状态检查

“状态”指训练进程在具体检查点**实际生效且能通过 API、日志或运行时源码验证**的随机性与确定性设置，而不是配置文件中的
声明，也不是某个 setter 曾经执行过。进入算子 dump 前，用获批的短受控复跑确认这些设置是否持续生效：

1. 先确认实际加载的训练模块、业务封装和框架入口；在这些路径中定位 Python/NumPy/PyTorch/NPU Seed、
   `torch.use_deterministic_algorithms`、`seed_all` 或现场等价封装的所有调用，以及配置重载/恢复逻辑。
2. 对 `seed_all` 等封装使用 `inspect.signature`、`inspect.getsourcefile`，必要时读取实际加载源码；不假设缺省 `mode` 或同名函数在
   不同版本具有相同覆盖范围。
3. 至少在初始化完成后、首次 forward/backward 前、首个 Step 结束后，记录当时实际生效且可验证的状态，包括各级 Seed/RNG、
   Rank Seed 派生、sampler/worker 状态，以及随机模块和确定性开关。PyTorch 可记录
   `torch.are_deterministic_algorithms_enabled()`；NPU、HCCL 和框架状态只记录现场 API/日志能验证的值，无法读取时写 `unknown`。
4. 每个检查点同时记录调用源、Rank/进程角色、Seed 派生、sampler/worker 状态和配置哈希，判断业务代码、框架封装或配置重载是否
   在初始化后重复设置随机种子或关闭确定性。
5. 若上述运行时状态在首差异之前改变，先只修正一个覆盖点，完成多组 A/B，再撤销修正复跑。单变量闭环可定位到该随机性契约；一次删除多个
   setter 只能形成相关性。未闭环前不进入 msProbe 算子 dump。

## 2. 两次运行对齐

用 run A 作为天然标杆、run B 作为 target。比较前检查 Step 对应相同样本 ID；多卡场景检查相同 Rank 的数据分片。
每次非固定 Step 发生、但每次运行都发生，仍可标为稳定复现，同时记录首次差异 Step 的分布。

## 3. 优先验证随机性契约或 Preflight 差异

对已确认的单一 Seed 派生、sampler/worker、Dropout、随机 API、确定性开关、环境或并行差异，只改变该因素后重复运行。
若多次运行恢复一致并完成同口径复跑，可认证对应粒度的根因，不强制 dump。若一次补齐多个控制项，只能判为相关或强候选。

## 4. 校验值定位

`statistics` 下使用 `summary_mode: "md5"`。当前 PyTorch msProbe 语义中字段包含 CRC-32 校验值和统计量；
它用于快速判等，不代表密码学 MD5，也不表示误差幅度。

采集前声明不依赖 compare 结果的 A/B 不一致信号、可比窗口和等价判据。run A/run B 采集结束后分别确认正常退出、预期 Step
已完成现场要求的 `stop()`/`step()`，逐项核对预期 `Step × Rank` 的 `dump.json` 存在、非空、可解析且范围匹配，并检查
`dump_error_info.log`，分别记录 `dump_integrity`。再按相同契约和项目口径对 A/B 比较对记录 `symptom_reproduction`。
只有两侧 `dump_integrity=valid` 且 A/B 仍为 `reproduced` 才能 compare；其他组合按共享门禁回退、复核 R0 或以证据不足结束。

启用 `msprobe compare ... -da` 后，当前版本可输出 `compare_result_rank*.json` 和 `diff_analyze_*.json`；
输出名和格式以现场帮助及目录为准。

## 5. 首差异分支

### 首个问题 API 的输入已经不一致

该结果只证明异常在当前 API 之前发生，不能把当前 API 定为确定性根因。先确认 Step、Rank、方向、调用序号和 shape 已正确对齐，
再把最后一个输入输出均一致的上游节点记为 `node_a`，把 compare 报告的首个问题 API 记为 `node_b`。将
`node_a → node_b` 记录为“可能漏采或无法对齐”的可疑区间，而不是直接声明 msProbe 一定漏采。

按以下顺序检查该区间：

1. 核对 statistics 的 level、scope/list、data_mode、forward/backward 方向和调用次数，排除采集范围或配对错误。
2. 检查框架包装、融合、通信、控制流和原地操作之间是否存在未显示的算子/API；保留最后一致节点和首问题输入的原始证据。
3. 检查项目 custom API 是否已被 msProbe 采集。先确认运行时实际导入模块，再用 `inspect.signature` 核对
   `register_custom_api` 或现场版本提供的注册方式；run A/run B 使用相同注册集合和调用边界。
4. 仍无法闭合时，经 Primary 批准把该区间提升到现场支持的 mix，或对候选做 targeted tensor。先补采证据，不直接修改模型逻辑。
5. 补采后若 `node_b` 输入仍先不一致，结合 Preflight 已确认的权重/optimizer 状态、样本身份、数据分片和随机状态，继续追查
   通信或更早上游；若区间仍不可见，结论保持 `insufficient evidence`。

### 仅某个 Rank 不一致

保留该 Rank 的原始 compare、Step、样本 ID、Seed、进程角色和拓扑证据，并与其他 Rank 分开判断。优先检查该 Rank 的数据分片、
worker seed、collective 参与者/顺序、点对点通信配对、pipeline stage 和 Rank 映射。单 Rank 差异不能直接推广为全局算法根因；
若首差异发生在 collective 输出，还要检查所有参与 Rank 的对应输入和调用序号。

### 相同输入下输出首先不一致

只有在输入、调用实例、参数/状态和通信参与者均对齐后，才能把当前算子、融合/编译路径或通信归约列为内部非确定性强候选；
仍需最小复现或修复/回退复跑才能认证根因。

## 6. 首个可观测不相等边界

按 Step、Rank、方向和调用实例找到第一个校验值或 Tensor 不一致的位置，记为“首个可观测不相等边界”。校验值只能界定当前采集
范围内的判等边界；CRC-32 不表示误差幅度。只有输入、参数、状态、调用和通信参与者均对齐而输出首次不同时，当前节点才是非确定
行为引入候选，仍不是已验证根因。

E08 不把边界之后的差异扩大作为另一条起源定位轴线。边界之前 A/B 仍相等，共同出现的大有限值不构成非确定性证据；边界之后的
差异扩大属于传播或业务影响，不能把定位目标从更早的首次不相等位置移到下游节点。若 statistics 校验值不足以界定更早位置，先核对
采集范围、调用实例和对齐关系，必要时补充 targeted tensor，而不是根据下游误差幅度改写边界或根因。

非确定行为引入候选仍需重复性和单变量修复/回退验证。需要量化差异对业务指标的影响时可另行分析真实 Tensor，但该结果不进入 E08
起源定位证据链。

## 7. Targeted Tensor 与定位后 CPU 对照（可选）

需要构造算子或通信最小复现时，才采集已锁定的不相等边界或引入候选的真实 Tensor。新的 tensor 采集同样必须重新通过完整性与症状复现联合门禁；
采集后检查：

- 输入校验值先不同：继续追上游随机状态、数据与通信；
- 输入一致而输出不同：候选为非确定算子或通信；
- Module 相同但调用次数不同：先解决控制流或数据对齐，不强行配对。

现场支持 `bench_path` 时，可把 run A 的校验值 dump 作为基准，在 run B 只保存差异节点真实 Tensor。

候选已收敛到具体 API/Module 后，可经批准执行 CPU 单变量对照。它用于判断问题是否与设备执行路径相关并缩小怀疑范围，
不属于 Scope Reduction，也不能单独认证根因。保持输入、权重、dtype、shape、非 Tensor 参数和调用次数，并记录 device 搬运、
cast、隐式同步和算法变化。实验序列定义为：

- R0：CPU 对照前、能够稳定复现 A/B 不一致的原始 NPU 基线；
- R1：只把已定位的候选 API/Module 替换到 CPU，其余契约保持不变；
- R0′：撤销 R1、恢复原始 NPU 路径后的回退复跑。

若 R0 不一致、R1 恢复一致且 R0′ 再次不一致，可降低“输入或模型状态本身不一致”的可能性，并把怀疑收敛到 NPU 特有算法、
异步时序、并发或候选算子路径。由于 CPU 可能天然串行或选择不同实现，这条证据仍不能单独证明候选 NPU 算子本身不确定。
要认证该根因，还需在 NPU 上以相同输入重复候选计算，并通过确定性配置或实现的修复及回退完成验证。

## 8. 全量 Dump 改变时序且单算子不复现时的竞争检查（可选）

仅当以下事实同时成立时进入本分支：原始整网 R0 的 A/B 不一致可重复；全量 Dump 使症状消失或迁移，已记为观察者效应；更轻量且
仍能复现症状的采集已把范围锁定到具体候选；使用相同输入和参数的常规单算子脚本多次运行仍一致。单算子负结果只说明串行、孤立上下文
未复现，不能排除算子内竞争或算子间多流竞争。以下两类实验必须分别获批并一次只选择一类：

### 8.1 用 mssanitizer 拉起单算子 application 检查算子内竞争

这里不是再次要求单算子脚本复现 E08 数值不一致，而是借该脚本实际触发候选 Kernel，供 `racecheck` 观察算子执行期间的内存访问
竞争。即使第 7 节的单算子脚本在常规运行下始终一致，仍可将它作为 msSanitizer 的 user program/application；但必须确认它加载的
算子实现、Kernel 分支、shape/dtype、非 Tensor 参数、tiling 和 block/core 配置与整网候选一致。检测对象不是候选名称或 dump 文件，
而是以下一种能够真正执行候选 Kernel/API 的 application：

- Kernel 直调：调用 `<<<>>>` Kernel 的可执行程序，例如 `./op_test`；
- 单算子 API：调用已确认 `aclnn*` API 的可执行程序或 `bash run.sh`；
- PyTorch 接入：现场版本明确支持时，使用只调用候选算子的 `python op_case.py`。PyTorch 图模式、编译选项和检测范围必须按现场官方
  文档核对，不能默认整网训练脚本或任意框架模式支持竞争检测。

按以下顺序执行：

1. 运行 `mssanitizer --version`、`mssanitizer --help`，并按官方“适用场景”选择 Kernel 直调、单算子 API 或 PyTorch 接入方式。
   核对候选 Kernel 的实际名称、application 加载的算子包/动态库和输入契约。`--kernel-name` 只在名称已验证时用于缩小检测范围；
   不指定时会检测 application 运行期间调度的所有算子。
2. 核对编译支持。官方指南的竞争全量检测需要相应的 sanitizer 编译支持；需要增加检测编译选项、调试信息，重新编译或替换算子包时，
   先取得独立授权。现场二进制、产品或框架模式不支持时写 `unknown`，不得用受限模式的“无报告”排除竞争。
3. 官方建议先运行内存检测，确认 application 能正常执行且没有先发内存异常，再单独运行竞争检测。命令骨架为：

   ```bash
   mssanitizer --tool=memcheck [--kernel-name="<verified-kernel>"] -- <application> [application-args]
   mssanitizer --tool=racecheck [--kernel-name="<verified-kernel>"] [--log-file=result.log] -- <application> [application-args]
   ```

   例如 Kernel 直调的 `<application>` 是 `./op_test`，aclnn 单算子脚本可以是 `bash run.sh`，PyTorch 单算子脚本可以是
   `python op_case.py`。模板中的可选参数必须先由现场 `--help` 验证。只有假设明确指向未配对的 `SetFlag/WaitFlag` 时，才另建实验使用
   `mssanitizer --tool=synccheck -- <application>`；`synccheck` 不能替代 WAW/WAR/RAW 竞争检查。
4. 保存控制台输出和 `mindstudio_sanitizer_log` 中的实际日志，记录 hazard 类型（WAW/WAR/RAW）、Kernel、device/block、PIPE 或线程、
   内存空间、PC/serialNo 和可用调用栈。工具退出状态正常但没有竞争报告，只表示本次 application、输入和检测范围内未报告，不能外推整网。
5. 报告只形成“算子内竞争候选”。只有报告指向的位置经单变量同步/实现修正后不再报告，原 E08 症状同时消失，撤销后恢复，且正式修复
   在整网 R0 的多组 A/B 中通过，才能认证根因。

### 8.2 基于整网真实子图构造最小并发 application

最小并发图不是从单个候选名称凭空拼装，也不是 msSanitizer 的算子间竞争报告。它依据仍能复现 E08 的轻量采集、实际运行栈/图结构，
以及“最后一致节点 → 首个可观测不相等边界”的真实数据依赖构造：

1. 保留候选及与其共享数据或 buffer 生命周期的必要生产者、消费者、异步拷贝/通信节点；只有现场证据表明相关时才保留更多节点。
2. 用原框架和执行模式构造可直接运行的普通 NPU application，复用已采集输入，并保持 dtype、shape、非 Tensor 参数、调用次数、
   stream 创建与绑定、event 依赖、launch 顺序、buffer alias/复用及必要通信参与者。不得为了简化把待验证的多流依赖串行化。
3. 不开启会使原症状消失的全量 Dump；按 E08 原判据重复运行并记录输出校验值/真实 Tensor、不一致次数和首次不相等边界。
4. 建立同构串行控制组，每轮只改变一个因素：把相关节点放到同一 stream，或只在候选边界增加一次显式同步，二者不能同轮混用。
   并发版本不一致、串行控制恢复一致且回退并发后再次不一致，只能把范围收敛到算子间多流时序、依赖或 buffer 生命周期，不能直接指定
   某个算子为根因。
5. 若最小并发 application 仍不复现，记录未保留的拓扑、通信参与者、内存压力和整网上下文。该结果仍是反证，不是候选排除。

两类检查均不能用全量 Dump 下已经消失症状的产物认证原问题。检查不可用、未复现或未闭环时，由 Plugin 编排返回最后稳定复现的整网
R0，按单变量 Scope Reduction 和 R0′ 回退复现缩圈，再回到 E08。

## 9. 根因门禁

根因必须能通过最小改动验证，例如固定 worker seed、替换或配置确定性算法、修正 Rank seed 派生、开启通信确定性后，
在原口径下重复运行恢复一致。该单变量因果闭环不要求 Module/API/Tensor 证据；未找到可闭环的外部差异时，才继续定位内部首差异。
只发现 Preflight 差异或首差异而未验证时写“强候选”。

## 官方资料

- [msProbe 训练调试指南](https://gitcode.com/Ascend/msprobe/blob/master/docs/en/wiki/train_debug_guide.md)
- [PyTorch 场景数据采集](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/pytorch_data_dump_instruct.md)
- [PyTorch 精度比对](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)
- [MindStudio Sanitizer 使用指南](https://gitcode.com/Ascend/mssanitizer/blob/master/docs/zh/user_guide/mssanitizer_user_guide.md)
