---
name: external-cannbot-tools-tool-npu-compute
description: 使用 npu-compute 采集和分析 NPU 性能指标，帮助用户生成或执行采集命令、查询工具支持的指标，并依据带宽、流水线、存储访问和资源冲突等采集结果定位性能问题。支持按指标集合（Set，由多个采集项组成）或单类采集项（Section，生成一类独立结果）选择采集范围，解包
  .npu-rep，分析 Section CSV、summary.jsonl、HardwareInfo.jsonl 和 PipeTrace.json。适用于明确要求使用
  npu-compute、采集 NPU 性能指标或解释上述报告的请求；仅要求一般性能分析或算子优化时不触发。
license: CANN-2.0
original-name: tool-npu-compute
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-26'
synced-commit: edcd4b94390c2a6b664f0935fbdd8a65e3644e9f
---

# npu-compute 使用与数据分析

根据用户意图生成或执行采集命令、解包报告并解释结果，帮助用户确认采集是否成功，以及从带宽、流水线、存储访问、缓存和资源冲突等数据中定位性能问题。依据实际命令输出和采集文件给出结论，不用目标程序自身的成功信息代替 npu-compute 的采集结果。

`Section` 表示一类可单独采集和分析的性能指标及其结果文件，例如 `Memory` 或 `L2Cache`；`Set` 表示按常见分析目标预先组合的一组 Section，例如 `basic` 和 `full`。用户不熟悉这些概念时，先用采集目标说明推荐项，再给出对应命令。

## 何时加载

根据当前请求和已经明确的任务上下文判断：

- 请求使用 npu-compute、解包或分析 `.npu-rep`、解释 `PipeTrace.json`，或查询、采集、分析 `PipeUtilization`、`ArithmeticUtilization`、`ResourceConflictRatio` 等 NPU 指标时加载。
- 明确要求采集、选择或解释 NPU 性能指标时加载。仅要求分析性能、定位访存等性能瓶颈或优化算子时不加载，即使提供 Ascend 用例目录。
- `Memory`、`Pipeline`、`Set`、`Section`、CSV、报告、性能、带宽等通用词需要 NPU 性能采集或结果分析语境。仅修复编译错误、整理文本或偶然引用工具名，不构成加载条件。
- 结合相关前文理解省略表达：前文明确为 npu-compute 采集结果时，“分析这个 Memory.csv”适用；前文仅在讨论 Ascend 算子性能时，“看看访存瓶颈”不足以触发。代码块、日志和历史记录中的词语需结合当前意图判断。
- 按语义识别 `NPU-Compute` 等工具名大小写变体；生成命令时仍遵守 CLI 名称和参数值的大小写规则。

例如，“用 npu-compute 分析 examples/add_custom”应加载本技能，并进入下面的目录输入流程。仅提供该相对路径并要求分析性能瓶颈时不加载。

## 判断用户意图

- 用户询问“怎么采集”“命令怎么写”时，只生成命令并解释必要参数，不执行采集。
- 满足上述加载条件后，只有用户明确要求“实际运行”“执行采集”“帮我实际执行”时，才按请求执行 npu-compute；输入为目录且需要新增采集时，按下面的流程核对请求和既有授权。
- 用户要求解包 `.npu-rep` 时，执行导入命令并读取终端给出的实际解包目录。
- 用户要求查看完整报告内容时，先检查实际解包目录并区分根级结果和各子结果集合。
- 用户提供 CSV、`summary.jsonl`、`PipeTrace.json` 或 `HardwareInfo.jsonl` 时，先检查实际文件，再解释数据。
- 目标程序、工作目录或必要参数缺失且无法可靠推断时，先说明缺少的信息。

命令语法、参数规则和路径行为见 [CLI 使用说明](references/cli-usage.md)。从命令生成到采集、报告确认和解包的完整步骤见[采集与报告流程](references/collection-workflow.md)。

### 命令生成快速路径

- 用户明确要求“只生成命令”“只告诉我命令”或“不要执行”时，直接根据已知 CLI 契约生成命令并简要说明。此时不检查目标程序是否存在，不扫描工作目录，不探测 npu-compute 是否安装，也不运行 lookup、list 或检查脚本。
- 已知 Set、Section 和结果文件的命令生成直接使用本技能已给出的规则。只有名称未知、用户明确要求核验，或进入实际执行流程时，才查询 catalog、运行 lookup 脚本或调用 `--list-sets`、`--list-sections`。
- 用户在请求中提供 CSV、JSON 或 JSONL 内联片段时，直接分析该内联片段，并且只读取结论所需的最少 reference 或 catalog；不创建临时文件、不扫描目录，也不运行要求完整文件路径的检查脚本。
- “给出查询命令”属于命令生成；只有用户明确要求查询当前安装环境的实际结果时，才执行查询命令。

## 加载后的用例目录处理

此流程用于已明确需要本技能能力的目录输入；目录或泛化性能分析请求本身不作为加载依据。

1. 只读检查用户目录中的 README、运行脚本、可执行文件和已有报告，确认分析目标、工作目录及参数来源。
2. 已有采集结果适用于本次目标时，按现有结果检查和指标分析流程处理。
3. 需要新增采集时，根据当前请求和既有授权确定执行范围；已授权步骤直接继续。必要目标、参数或执行权限缺失时，说明具体缺项，不猜测。
4. 只有源码而没有可执行程序时，说明实际构建入口和所需准备；构建按用户任务授权或适用的开发技能处理，不凭关键词修改源码。
5. 其他性能技能负责总体分析时，仅在任务明确需要本工具的采集、解包或指标解释能力时参与。瓶颈结论需由实际指标、定义或比较依据支持。

## 生成或执行采集命令

1. 从用户请求中识别目标程序、Set、Section、目标程序参数、工作目录和报告输出要求。
2. 用户只要求生成集合查询命令时，给出 `npu-compute --list-sets`，不附加目标程序且不执行；用户明确要求查询当前安装环境的实际支持集合时才运行该命令。
3. 按采集意图选择参数：
   - 明确要求一个或少量指标时，使用一个或多个 `--section`。
   - 要求常用或基础指标时，使用 `--set basic`。
   - 要求完整或全部指标时，使用 `--set full`。
   - 未指定指标时，不添加 `--set` 或 `--section`；npu-compute 默认使用 `basic`。
   - 同时要求 Set 和额外 Section 时，按用户表达顺序生成选项；执行时按命令行出现顺序展开，并按 Section 首次出现的位置去重。
4. 实际执行、名称未知或用户明确要求核验时，使用 [Set/Section 查询脚本](scripts/lookup_section.py)核对 Set、Section、结果文件和解析入口；名称区分大小写。命令生成快速路径中的已知名称不运行该脚本。
5. 实际执行前运行 `npu-compute --list-sets` 或 `npu-compute --list-sections`，确认当前安装版本支持所需名称。
6. 保持每个目标程序参数的独立边界；需要明确结束工具选项时使用 `--`。`--` 后第一项是目标程序，其余内容均是目标程序参数。
7. 目标是无执行权限的脚本时，使用对应解释器，例如 `bash run.sh`。
8. 不安装软件包，不修改目标程序，不猜测会改变程序行为的参数。

命令说明要求：当用户在 `basic` 基础上追加 Section 时，除完整命令外，说明该组合是否等价于 `full`、按命令行出现顺序展开，并按首次出现位置对重复 Section 去重。仅生成 `--list-sets` 查询命令时，说明它只查询支持的 Set/Section，不启动目标程序，也不生成采集报告。

典型命令：

```bash
npu-compute ./add_custom
npu-compute --set basic ./add_custom
npu-compute --set full ./add_custom
npu-compute --set basic --section ResourceConflictRatio ./add_custom
npu-compute --set basic -- ./add_custom --help
```

## 工作流输入与输出契约

按以下契约执行，避免根据缺失信息猜测路径、参数或成功状态：

| 场景 | 必要输入 | 执行动作 | 成功证据 | 失败输出 |
|---|---|---|---|---|
| 查询 Set/Section | 查询类型，可选名称、是否实际执行 | 仅需命令时生成 `--list-sets` 或 `--list-sections`；明确要求当前环境结果时才执行命令或对应 lookup 脚本 | 命令生成返回正确命令；实际查询返回支持名称、顺序和结果文件映射 | 实际查询时返回命令错误和实际合法候选值 |
| 采集 | 目标程序、工作目录、指标范围、程序参数、报告输出要求 | 先确认工具能力，再按命令边界执行 npu-compute | 退出码为 `0`、终端出现 `report`、报告文件实际存在 | 返回失败阶段、退出码、终端错误；不声称报告生成成功 |
| 解包 | 已存在且可读的 `.npu-rep`、可写的导出父目录（可选） | 执行 `--import`，记录终端给出的实际 `unpacked` 路径，再运行解包检查脚本 | 退出码为 `0`、目录存在、包含可识别结果文件 | 返回导入/解包错误和输入路径；不猜测输出目录 |
| 结果分析 | CSV、`summary.jsonl`、`HardwareInfo.jsonl`、`PipeTrace.json` 或完整解包目录 | 先运行对应只读检查脚本，再读取匹配的 reference 和 catalog | 获得 JSON/Markdown 检查结果，并能对应输入文件 | 返回文件路径、行号、字段或结构错误；未知文件保留相对路径 |

查询名称区分大小写；Set 与 Section 的展开顺序和去重规则以 [CLI 使用说明](references/cli-usage.md) 为准。没有必要输入时先列出缺失项，不执行可能改变目标程序行为的命令。

## 判断采集和解包结果

采集成功需要同时满足：

1. npu-compute 退出状态为 `0`。
2. 终端输出 `npu-compute: report= <路径>`。
3. 对应 `.npu-rep` 文件实际存在。

解包成功需要同时满足：

1. npu-compute 退出状态为 `0`。
2. 终端输出 `npu-compute: unpacked= <目录>`。
3. 对应目录实际存在并包含可识别的采集文件。

指定 `--export` 作为解包父目录时，文件写入该目录下新建的唯一子目录；以终端输出的 `unpacked` 路径为准。
使用[解包结果检查脚本](scripts/inspect_collection.py)检查该路径，不将不同子结果集合中的同名文件合并。

## 解释采集文件

- 输入是用户消息中的内联片段时，直接按片段分析，不运行下面要求文件路径的检查脚本；仅在需要确认字段语义时读取最少的 reference 或 catalog。
- 分析 CSV 时，先运行 [CSV 检查脚本](scripts/inspect_csv.py)识别 Section、表头、行数和字段可用性。
- 分析完整报告时，先运行[解包结果检查脚本](scripts/inspect_collection.py)识别根级结果、子结果集合和未知项目。
- 分析 `summary.jsonl` 时，先运行 [summary 检查脚本](scripts/inspect_summary.py)，再读取 [summary 字段](references/summary.md)。
- 分析 `HardwareInfo.jsonl` 时，先运行 [HardwareInfo 检查脚本](scripts/inspect_hardware_info.py)，再读取[HardwareInfo 字段](references/hardware-info.md)。
- 分析 `PipeTrace.json` 时，先运行 [PipeTrace 检查脚本](scripts/inspect_pipe_trace.py)，再读取 [Pipeline 时间线](references/pipeline.md)。
- 查询 Section 及结果文件时，使用 [Section 目录](assets/section-catalog.json)或 [Section 查询脚本](scripts/lookup_section.py)。
- 查询 CSV 字段时，使用 [CSV 指标目录](assets/metric-catalog.json)或 [CSV 字段查询脚本](scripts/lookup_metric.py)。
- 需要确认 summary 记录结构时，使用 [summary 字段目录](assets/summary-catalog.json)。
- 需要确认 HardwareInfo 或 PipeTrace 结构时，使用 [HardwareInfo 字段目录](assets/hardware-info-catalog.json)或 [PipeTrace 格式目录](assets/pipe-trace-catalog.json)。

## 指标分析流程

1. 检查实际输入，不假定所有预期文件或指标都存在。
2. 读取[指标通用规则](references/metric-foundations.md)，确认单位、公式、时长选择和缺失值规则。
3. 根据 Section 读取对应字段说明：
   - [ArithmeticUtilization 字段](references/arithmetic-utilization.md)
   - [PipeUtilization 字段](references/pipe-utilization.md)
   - [ResourceConflictRatio 字段](references/resource-conflict-ratio.md)
   - [Memory 字段](references/memory.md)
   - [MemoryL0 字段](references/memory-l0.md)
   - [MemoryUB 字段](references/memory-ub.md)
   - [L2Cache 字段](references/l2-cache.md)
4. 需要区分 `NA` 原因时，读取[数据可用性说明](references/data-availability.md)。
5. 需要确认字段或格式依据时，读取[技术事实与验证依据](references/verification-basis.md)。
6. 需要结构化输出时，使用[分析报告模板](assets/analysis-report-template.md)。

## 解释规则

- `NA` 表示数据不可用或字段不适用，不等于数值零。
- 名称含 `ratio` 的字段表示原始比值；名称含 `(%)` 的字段表示百分比。
- 使用数据方向和计数含义解释字段依赖，不使用固定 Event 编号代替指标语义。
- 只有指标定义或已提供的比较标准能够支持结论时，才能根据数值高低判断性能瓶颈。
- 不按行号直接比较数据，应先匹配 `block_id`、`sub_block_id` 和适用 Core。
- `PipeTrace.json` 的 `ts` 和 `dur` 按微秒解释；不同轨道的事件可以重叠。

## 失败处理

- npu-compute 返回非零状态时，报告终端中的具体错误及失败阶段，不声称报告生成成功。
- 输入文件不存在或不可读时，报告路径和读取错误，不推断文件内容。
- 解包结果中存在未知文件或目录时，报告相对路径，不将其解释为已知采集结果。
- CSV 表头不匹配时，报告未知、缺失、重复或顺序错误的字段，不猜测 Section。
- summary、HardwareInfo 或 PipeTrace 结构错误时，报告检查脚本给出的记录、字段或事件位置。
- Set、Section 或字段不存在时，返回查询脚本列出的可选值，不生成目录中没有的定义。
- 硬件环境不一致时，先报告差异，再说明受影响的比较结论。

## 安全边界

- 启动目标程序或采集须在当前请求或既有授权范围内；加载技能本身不构成执行授权。用户只要求方法说明或已有结果分析时，不额外启动采集。
- 只读取用户明确指定的输入目录、报告文件和本 Skill 自带的 `references/`、`assets/`、`scripts/`、`tests/` 资源；不扫描无关目录。
- 不访问未知网络服务，不上传采集报告，不安装软件包或下载运行时依赖。
- 不读取、传播或写入 token、密钥、密码、凭据及其他敏感配置；命令输出中发现敏感信息时先停止并提示用户脱敏。
- 没有 NPU、npu-compute 不可用或版本能力无法确认时，只执行静态检查、fixture 验证或 dry-run，不声称真实采集成功。
- 不修改、删除或覆盖用户已有采集数据和报告。
- 已有报告文件不会被采集命令覆盖；解包写入唯一目录。
- 目标报告、解包目录或导出路径冲突时先报告冲突并停止，不自行清理或覆盖。
- 不执行嵌套 npu-compute 采集。
- 分析脚本只读取输入文件和 Skill 自带资源。

## 验证与维护

- 维护目录、reference 或真实样本后，运行[目录校验脚本](scripts/validate_catalog.py)。
- [语义评测](evals/evals.json)用于验证字段解释和能力边界。
- 在技能目录运行仓库测试时，执行 `python3 -m pytest -p no:cacheprovider --confcutdir=tests tests -q`。
