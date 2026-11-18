# 报告与可交换结果

先把执行记录整理成 `bundle.json`。随报告保存计划、环境与标准版本、输入和输出哈希、执行命令、原始 ATK 报告和复现包。

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

两个 CSV 都使用 UTF-8。`results.csv` 每个结果一行，`metrics.csv` 每个指标一行，两表用行键关联。可能被表格软件解释为公式的文本和原本以单引号开头的文本会多一个前导单引号。程序读取 CSV 时可去掉这一前缀，或直接读取保留原值的 JSON。导入 XLSX 时将 ID 列设为文本。

## 核对报告

- 按行键核对计划和实测，检查多余项和重复项。每个适用阶段关联输入、输出、标杆和执行次数。多输出用例按返回结构和顺序逐项核对。只有完成执行和所需比较并满足判据的结果才记为 `PASS`。必需标杆缺失记为 `BLOCKED`，未运行记为 `NOT_RUN`。
- 在交付目录离线打开 HTML，检查用例明细、原始报告和复现链接。受限输入注明获取方式。
- `summary.json` 汇总行数和状态。精度记录保留所用规则及其结论，包括 ATK 原始结果、项目回归或正式标准验收。

数据量大时按算子或阶段拆分 HTML。汇总表按行键显示结果，重复执行的明细放在原始数据文件。
