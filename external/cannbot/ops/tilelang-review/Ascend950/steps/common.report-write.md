# 撰写报告（文件检视 + PR 检视）

## 输入

- YAML结果目录：`{yaml_dir}`
- 报告输出路径：`{report_output_path}`
- 产物目录：`{review_output_dir}`（阶段0按 `instructions.md` 的共用规则固定；用于校验报告路径）
- 检查对象：`{file_input}`；PR检视时为变更文件列表
- 文件类型：`{file_types}`
- 代码侧别：`{side}`
- 匹配规则文件：`{matched_references}`
- 检视时间：`{timestamp}`

## 执行步骤

### 1. 组装报告正文

执行：

```bash
python3 {skill_base}/scripts/workflow.assemble_report.py \
    --dir {yaml_dir} \
    --output {report_output_path}
```

脚本读取 `{yaml_dir}` 下所有可用 YAML，将一份汇总 `type: format` 结果和普通 `type: clause` 结果组装为同一份 Markdown 报告。可解析的 `_dupN` 文件全部独立进入报告和统计；无法解析的 YAML 告警后跳过；普通条例的证据总分与 `confidence_value` 不一致时，脚本按 `core/methodology.md` 重新计算，并原地回写。

命令失败或未生成非空报告时，停止本步骤并返回实际错误，不得伪造报告。

### 2. 补全检视概览

报告的 `## 1. 检视概览` 含以下占位符。主 Agent使用实际输入逐项替换：

| 占位符 | 替换值来源 |
|---|---|
| `{{CODE_FILE}}` | `file_input`；PR检视时使用变更文件列表 |
| `{{FILE_TYPES}}` | code-summarize 返回的文件类型 |
| `{{SIDE}}` | code-summarize 返回的代码侧别 |
| `{{DOC_LIST}}` | plan-design 返回的匹配规则文件列表 |
| `{{TIMESTAMP}}` | 本次检视时间戳 |

使用 `apply_patch` 替换占位符。替换后搜索 `{{`，确认报告中没有未处理的占位符。

### 3. 报告路径

文件检视与PR检视均检查 `report_output_path` 是阶段0固定的 `review_output_dir` 下的绝对路径，然后向 assembler 的 `--output` 传入该路径。不得在此阶段从当前 shell 目录、临时检视仓库或输入文件重新推导产物目录。报告命名规则：

- 文件检视：`{review_output_dir}/{source_file}_review_summary.md`
- PR检视：`{review_output_dir}/{review_id}_review_summary.md`

本步骤不额外定义多文件或目录输入时 `{source_file}` 的推导规则。

### 4. 检查超长报告

补完头部元信息后检查报告行数：

```bash
wc -l {report_output_path}
```

- 不超过5000行：报告完成。
- 超过5000行：读取并执行 `steps/common.report-filter.md`，压缩格式详情、按严重程度删除不严重条例发现并更新统计；过滤完成后报告才视为最终成品。

## 报告内容

`workflow.assemble_report.py` 统一生成：

1. 检视概览与检查摘要；
2. Ruff lint、Python格式和C/C++格式结果；
3. 格式检查工具、未检查文件和工具执行错误；
4. 普通条例的PASS、FAIL和SUSPICIOUS统计；
5. HIGH、MED和LOW发现详情；
6. PR范围外备注（存在 `out_of_range` 结果时）；
7. 问题统计摘要和下一步操作。

## 约束

- 报告正文只能由 assembler 根据 YAML 结果生成；主 Agent只补头部元信息和触发超长报告过滤。
- 不在本步骤重新检视源码、运行格式检查、创建或修改问题结论。
- 不生成design、S1-S7、D8、style、API预研或行号校对章节。
- 格式问题只能在全部检查完成且用户明确同意后修复；本步骤不得修改待检视文件。
- 保留 PR workflow 所需的 `out_of_range` 展示和PR报告路径。
- 报告生成后不删除 `{yaml_dir}`。
