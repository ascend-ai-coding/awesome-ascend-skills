# 报告与可交换结果

先把执行记录整理成 `bundle.json`。随报告保存计划、环境与标准版本、输入和输出哈希、执行命令、原始 ATK 报告和复现包。按所用标准判定后填写结果。渲染脚本按输入记录生成表格和统计。

## 规范化结果

使用 Python 3.9 或更高版本。可用合成用例检查输出格式。[报告示例](../assets/report-example.json)

```bash
python3 scripts/render_report.py run/bundle.json --out run/report
```

输出目录须尚不存在。保持整个 run 目录的相对结构，HTML 才能定位证据。输入 `schema_version` 是字符串 `"1.0"`。

| 字段 | 含义 |
| --- | --- |
| `run.id`, `run.standard_id` | 本轮标识、正式引用的标准或诊断规则标识 |
| `run.evidence_class` | `hardware`、`synthetic` 或 `unverified`；标识证据来源，不代表通过 |
| `run.metadata` | 标准来源与摘要、设备、软件、计划数、完成数、缺失/多余 ID 及重复数；JSON 对象 |
| `results[].case_id, operator, phase` | 实例、公开调用面、阶段。多个模式使用不同 ID 或在同一行记录重复的汇总 |
| `suite, backend, reference` | 五类测试之一：`function`、`accuracy`、`memory`、`performance`、`determinism`。设备和标杆标签自定；无标杆填空字符串 |
| `status, detail` | `PASS`、`FAIL`、`ERROR`、`TIMEOUT`、`UNSUPPORTED`、`NOT_RUN`、`NOT_APPLICABLE`、`BLOCKED` 或 `MEASURED`；附解释 |
| `metrics` | 数组；每项 `name, value, unit`，可选数值 `limit`。输出叶子或 rank 可放入指标名，并在元数据解释；未取得数值不要填零 |
| `evidence, reproduction` | 相对 `bundle.json` 所在目录的文件路径；前者是列表，后者是复现清单路径 |
| `metadata` | JSON 对象。记录 dtype、shape、layout、属性、用例来源、适用条件、失败签名、阈值比较符，以及 ATK 执行与精度的原始状态 |

行键是 `(case_id, phase, suite, backend, reference)`。ATK 数字 ID 仅在对应批次内解释。若以多种判据分别输出，应在 case ID 或 reference 中明确区分其身份，并保存规则标识。显示无穷比值时在详情中标明原因，不将非有限数写入 JSON 数值字段。

记录 2.1 浮点双标杆结果时，在 `reference` 写同精度对照，在 `metadata` 写 Golden、等级、节点角色和比较符号。用 `metrics` 保存两端相对 Golden 的 MARE、MERE、RMSE 及各项比值，以 `evidence` 关联覆盖核对和原始报告。在离线重放中，ATK worker 的设备名可能与标杆实际执行设备不同，按小样验证的角色映射读取结果。

HTML 分别解释设备角色、三组直接比较和浮点 2.1 的误差比。以 NPU、H20、CPU Golden 为例：NPU–CPU 和 H20–CPU 产生两端相对同一 Golden 的误差；NPU–H20 显示两端直接差异。浮点验收使用前两组的正常值域指标比及小值域错误数，不以直接差异替代。非计算类、整数类及量化类仍按各自规定的直接比较分支验收。

三组诊断使用实际执行端名称，并在宽表前给出紧凑展示，使候选–对照数据无需横向滚动到末尾才能找到。标明相对误差的分母、统计范围与实际 dtype；全有限元素诊断不能直接冒充 2.1 正常值域指标。逐叶 dtype 表仅展示已提供的执行记录，Golden 是否按映射升精或保持原 dtype，还须检查[计算精度证据](precision.md#golden-计算精度核验)。FP32、complex64 不升精是规定方案；不得把 FP64、complex128 的结果换标签后替代。旧方案结果保留溯源，受影响条目补算前标为待更新。

## 2.1 判定记录

默认 `run.standard_id` 为 `ascend-operator-precision-2.1`。记录所采用原文的版本和 SHA256；用户已指定并提供 2.1 原文时，以该文档为依据，不要求另找一个未指定的修订号。公开参考用于交叉核查，不覆盖任务指定原文。公式、表格继承或特殊 dtype 映射无法从原文确认时，在相应条目下记录待确认项。

按算子和阶段分别保存数值初判、原输入复现、换种子复检、覆盖、执行要求及正式结论。建议字段如下；它们放在通用 `metadata` 中，不改变 CSV 行键。

| 记录位置 | 字段与内容 |
| --- | --- |
| `run.metadata.standard` | `version`、`source_sha256`、适用范围、原文未明确项和采用的解释 |
| `run.metadata.standard_acceptance` | 各算子/阶段的 `numerical_status`、`coverage_status`、`execution_status`、正式状态和未满足项；只有全部适用要求满足才为正式 PASS |
| 结果 `metadata` | `criterion_id`、`assessment_scope`（单例数值或算子正式验收）、`operator_category`、`accuracy_level`、三方实现及真实计算/输出 dtype |
| 正常值域 | `relative_error_epsilon=1e-7`；`ratio_denominator_err`、`ratio_err_source`、`ratio_err_interpretation`；候选/对照相对同一 Golden 的 MARE/MERE/RMSE、三项比值与 `<=` 上限 |
| 小值域 | `small_value_threshold`、`small_value_error`、实际掩码元素数、两端 ErrorCount、计数比；分别保留 `<`、`>`、`<=` 的比较关系 |
| 非有限值 | 芯片与实际 `INF_NAN_MODE_ENABLE`、三方 NaN/正负 Inf 的数量及位置、排除条件、分支结论与待查项 |
| 覆盖与执行 | 按 dtype/格式/维度/Attr/分布/来源统计计划与实测；普通用例去重数、特殊用例数、输出元素数、L2泛化与模型比例、平台、逐例压测次数、NaN预填证据和确定性结果 |
| 复检 | 初判失败指标、原输入复现状态、换种子样本数、三项 Ratio 样本、2000 次 Bootstrap、随机种子、分位数方法、95% CI及复检结论；不足200样本的复检记 FAIL |
| ATK | 版本、节点角色校准、原始执行/精度状态、计划ID与完成ID、报告路径和失败复现包；基础设施错误与数值失败分开 |

普通用例数量不能按种子、重复次数、设备数、输出叶子数或结果行数累加。初判失败不从报告删除，复检结果另列并关联原用例。单例数值满足阈值时可记录其数值结果，但覆盖不足仍不能把算子正式状态改为 PASS。缺必需标杆的正式结果记 `BLOCKED`；只取得数值尚未评定时记 `MEASURED`。`NOT_EVALUATED` 可用于元数据中的正式状态，不是结果行的 `status` 枚举。

精度验收仅采用 2.1。三方逐位、ULP 和直接绝对/相对误差作为诊断记录，不设置额外通过门槛。CPU 在双标杆中承担规定精度 Golden 时，其误差参与 2.1 计算；原始 Golden 不因诊断所需的舍入或转换而被覆盖。

报告示例包含正常值域三项比值和边界、小值域计数、三方身份及未执行的复检字段，全部为合成数据。渲染器只校验报告结构，不重新执行算子、计算精度或证明标准覆盖达标。

两个 CSV 都使用 UTF-8。`results.csv` 每个结果一行，`metrics.csv` 每个指标一行，两表用行键关联。上述结构化字段保留在 `metadata_json`，指标以长表保存，适用于不同项目和设备。可能被表格软件解释为公式的文本和原本以单引号开头的文本会多一个前导单引号。程序读取 CSV 时可去掉这一前缀，或直接读取保留原值的 JSON。导入 XLSX 时将 ID 列设为文本。

## 逐输出三方精度宽表

需要横向比较时，在 `bundle.json` 顶层增加 `precision_comparisons` 数组，每项对应某 API 调用的一个前向输出或反向梯度。非空数组会新增 `precision_comparison.csv`，并在 HTML 中显示同一张宽表；未提供或为空时保留原有四个输出文件。CSV 使用稳定英文列，HTML 使用中文表头；角色名称来自数据，不固定为某个硬件品牌。

| 字段组 | 必备内容 |
| --- | --- |
| 调用与输出 | `case_id`、`operator`、`phase`、`api_type`、`api_family`、`api_layer`、`output_name`、`shape`、`dtype`、`candidate_name`、`peer_name`、`golden_name` |
| 可选调用绑定 | `torch_api`、`backend_operator`。只有后端名称而没有公开 Torch 调用证据时，前者留空，不根据 ACLNN 或 kernel 名猜绑定 |
| `standard21` | `level`、`numerical_status`、`formal_status`、`detail`；三项误差比及分母下限、小值域阈值/error、两端 ErrorCount 及比值、非有限值和复检状态 |
| `diagnostics` | `candidate_golden`、`peer_golden`、`candidate_peer` 三组，每组记录 `bitwise_equal`、`bitwise_mismatch_count`、`max_ulp`、`max_abs`、`mean_abs`、`max_rel`、`mean_rel`、`rmse`、`excluded_count`；最后一组可附 `row_max_mean_rel` |
| 计算定义 | `diagnostic_definition` 保存原 dtype 位比较、ULP 计算 dtype/舍入/符号零及非有限值规则、相对误差分母与掩码、剔除原因、行最大位置的轴/参考端/并列处理 |
| 可选上下文 | `context` 保存模型、模块、片段、源表 T、输出位置、来源行等。它们不作为各项目都必须填写的固定列 |

`api_type` 使用 `non_computational`、`integer`、`quantized`、`floating` 或 `random`，对应 2.1 类别；`api_family` 表示矩阵乘、归一化、激活、通信等功能。`phase` 明确前向/反向或更细阶段。RMSNorm 的 y、rstd 以及 input/weight 梯度分别成行，各自保留真实 shape 和 dtype。多个输出共用原调用的 case ID，输出行数不计为独立用例数。

三方诊断不产生单独的通过/失败结论。`bitwise_mismatch_count` 统计原 dtype 位模式不同的**元素数**，不是不同 bit 总数。跨 dtype 的原始位比较不适用；如另做转换后的 ULP，必须在方法中记录目标 dtype 与舍入。未知指标使用 `null` 或省略，不能用 0 或 false 代替。零剔除数也不能代替小值域 ErrorCount。

`standard21.issue_fields` 将标准字段名映射到具体问题说明。表中这些单元格以及 `FAIL`、`ERROR`、`TIMEOUT`、`BLOCKED` 的标准状态标红，分别表明已超标、执行错误或必需证据缺失。诊断列不按大小自动标红；不一致的位模式或较大的 ULP/直接相对差不会触发浮点 2.1 失败。缺级别、RMSE、小值域或计算依据时注明缺项并保留待判定状态，不能仅凭最大/平均相对误差宣布通过。

渲染器只验证结构并展示输入结论，不重新计算 2.1，也不把未知指标改为失败数值。`issue_fields` 只能引用 `standard21` 中的字段，不能把诊断字段列为独立验收问题。已有原始输入/输出、标准版本、执行与覆盖证据仍需随报告交付。

## 核对报告

- 按行键核对计划和实测，检查多余项和重复项。每个适用阶段关联输入、输出、标杆和执行次数。多输出用例按返回结构和顺序逐项核对。只有完成执行和所需比较并满足判据的结果才记为 `PASS`。必需标杆缺失记为 `BLOCKED`，未运行记为 `NOT_RUN`。
- 在交付目录离线打开 HTML，检查用例明细、原始报告和复现链接。受限输入注明获取方式。
- `summary.json` 汇总结果行数和状态。精度记录保留 ATK 原始状态、2.1 数值及正式结论；三方诊断宽表的输出叶子数与独立用例数分开统计。

数据量大时按算子或阶段拆分 HTML。汇总表按行键显示结果，重复执行的明细放在原始数据文件。
