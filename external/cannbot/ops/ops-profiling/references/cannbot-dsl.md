# CANNBotDSL 公开入口测量

用于精度已通过的算子性能摸底，不修改实现或用例。输入为调用方选定的核心 case、公开模块/入口、测试目录、性能目标及输出目录；case 必须来自已确认场景的 active blackbox 行，不自造输入。

## 采集

1. 编写单进程采集脚本，按 `CANNBOTDSL_OP_MODULE` 导入公开入口，非 `run` 入口按 `CANNBOTDSL_OP_ENTRY` 定位。复用 testcase.csv 中该 case 的 inputs_json 与 Golden 的 make_inputs，确保与精度验证输入一致。
2. 固定设备、实现版本、驱动、测试框架和输入。将 `CANNBOTDSL_CACHE_DIR` 指向本次新建的空缓存目录并记录运行标识；预热 10 次、正式采样 10 次。编译在预热阶段完成，正式采样全部命中缓存；保留编译日志或缓存证据。
3. 采集环境不设置 `CANNBOTDSL_DUMP_*`、`CANNBOTDSL_PRINT_*`、`CANNBOTDSL_REUSE_ASCENDC`、`CANNBOTDSL_PIPE_STAGE`。Kernel 不含设备打印或 dump 操作；Host 普通输出允许，其开销计入公开入口测量。
4. 分别记录首次编译耗时、预热后公开入口同步完成耗时和每个设备 Kernel 耗时。无法单独测量编译时报告首次调用总耗时及限制，不能称为纯编译耗时。设备同步由测试脚本界定调用完成边界；输入构造、断言、快照及采集工具开销单独说明。
5. 使用 msprof 采集完整公开调用区间，固定 `ASCEND_RT_VISIBLE_DEVICES`，不与其他进程共用设备。按环境中已确认的 msprof 接口采集 PipeUtilization、Task Duration 与 AI Core 数据，例如 `msprof --aic-metrics=PipeUtilization --task-time=on --ai-core=on --application=<脚本及参数> --output=<输出目录>`；执行前核对安装版本支持的参数。
6. 列出正式调用区间内全部设备任务与源码启动点、次数及耗时的对应关系，不能只过滤目标 Kernel。按算子生产执行约束核查额外 torch 设备小算子、诊断操作和检查 Kernel；需求规定的 metadata 前置接口与主计算分别记录。速度提升比例不能替代此检查。

## 报告与验收

保存采集脚本、命令、退出码、环境与版本、原始 msprof 输出及全部样本。每个目标 Kernel 的 10 次 Task Duration 取最小值作为摸底统计，注明 Kernel 名称与源码对应关系；同时报告公开入口耗时及首次编译或首次调用耗时。对已确认性能目标只作事实对照，不以单个 Kernel 统计代替完整算子耗时。

验收者核对输入、版本、预热/采样和原始数据对应关系，确认正式采样没有重新编译，完整设备任务满足生产执行约束；对已有完整证据不重复运行。缺少当前实现的设备证据时注明“算子性能尚未实测”，不能保证性能不受影响；违规或证据缺失不予通过。

生成采集脚本时按 `cannbot-dsl-test-generator` 的脚本执行契约处理失败、结果与路径；生产路径判定按 `cannbot-dsl-code-reviewer`，具体工具用法参见 [msprof 指南](msprof-guide.md)。

## DSL 编译/prime 启动与配对确认（kernel 级标准计时适配）

DSL 实现接入本 skill 的标准可比较计时（`collect_perf_application.py`）时：

- **preparation launch**：DSL 的编译/prime 不属于被测耗时。runner adapter 实现 `prepare_target(inputs)`，在每 case 正式 pair 循环前调用；返回的 kernel 名写入 `preparation_kernel_sequence`，采集器映射为 `role=preparation, sample_index=-1`，计入会话容量但不计入 target 样本耗时。协议细节见 [application_runner_contract.md](application_runner_contract.md)；torch 模板见 `assets/app_runner/torch/torch_app_runner.py`。
- **配对确认（shared candidate confirmation）**：小改动（<3%）走 confidence 判定时，用事前冻结的 AB/BA 配对与共享输入 SHA 恢复，禁止采集器看结果后补配置；确认元数据随 collection 归档。
- 上述证据与 [DSL 公开入口测量](#采集) 的编译/缓存证据互补：公开入口口径证明端到端，标准计时口径证明 kernel 级收益。
