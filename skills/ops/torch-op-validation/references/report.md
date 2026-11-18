# 报告与可交换结果

以真实执行生成的记录为输入。随结果保留计划清单、环境指纹、标准版本与哈希、输入和输出文件哈希、真实命令、原始报告、失败复现入口。解析 ATK 版本特定字段的工作由执行方完成；本技能的渲染脚本不解释原始 ATK 报告或替代精度评估。

## 规范化结果

[合成报告示例](../assets/report-example.json) 无硬件依赖；它展示格式，不声称已执行设备测试。Python 3.9+ 运行：

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

CSV 使用 UTF-8 和标准引号规则。 `results.csv` 为长表，便于筛选和归档；`metrics.csv` 用同一行键关联任意数量的指标，不为每个框架新增一组列。工具为可能被电子表格解释为公式的文本及原本以单引号开头的文本加一个单引号；程序使用 CSV 原文时可移除这一处前缀还原，标准 JSON 始终保留原值。无需在 CSV 中填固定 NPU、H20、CPU 三套列。若需 XLSX，可按两个 CSV 分工作表导入；保留 ID 为文本，避免自动识别成日期或截断大整数。

## 核对报告

- 将计划键集合与执行键集合做差。缺项作为 `NOT_RUN` 或 `BLOCKED` 逐行输出并写明原因，多余项和重复项另查，不能仅核对总数。每个适用阶段都要对应实际输入、输出、标杆和执行次数。
- HTML 可离线打开，查看各项计数、用例明细、数值、证据与复现链接。放到同一交付目录中再核对链接；未发布的敏感输入只附访问说明，不附私有路径或凭据。
- `summary.json` 是行数与状态统计，不是认证证明。正式等级结论由经校验的判据、平台覆盖、独立用例和执行次数共同支持。项目额外回归、ATK 原生结论及正式标准结论均要可区分。

数据量大时按算子或阶段拆分 HTML；重复执行的原始数据保留在独立文件，汇总表每用例、阶段和专项一行，不将每次重复展开成 HTML 行。
