# 撰写报告（文件检视 + PR 检视）

## 执行步骤

### 1. 调用报告拼接脚本

```bash
python3 {skill_base}/scripts/workflow.assemble_report.py \
    --dir {yaml_output_dir} \
    --output {报告输出路径}
```

脚本读取 `{yaml_output_dir}` 下所有 clause 类 yaml（含 category: clause / style / sunset），组装「检视概览 + 检视意见单一大表」两章节报告正文，写入 `{报告输出路径}`。

**统计口径**：问题数 = FAIL + SUSPICIOUS + style/sunset 违规 + 范围外（`out_of_range`）；PASS 不渲染、不计数。严重级别取 yaml 的 `severity` 字段（条款声明的高/中/低），脚本据此分档计数与排序；yaml 缺 severity 时按置信度兜底（HIGH→高 / MED→中 / LOW→低）。

### 2. 主 Agent 补头部元信息

脚本输出的报告 `## 检视概览` 章节含 2 个占位符，主 Agent 用实际值替换：

| 占位符 | 替换值来源 |
|--------|----------|
| `{{FILE_COUNT}}` | 检视文件数：文件检视 = 阶段0 统计的 file_input 文件计数（目录输入按 find 枚举计数）；PR 检视 = code-fetch 返回的变更文件数 |
| `{{TOTAL_LINES}}` | 代码行数：文件检视 = 阶段0 按 `workflow.review-thresholds.yaml`「统计口径（行数）」统计的代码行数；PR 检视 = code-fetch 返回的变更行数（新增+删除） |

**占位符值必须是纯数字**——禁止附加单位、括号解释或差额说明。替换方法：Read 报告文件，用 Edit 工具逐个替换；或用 `sed -i` 批量替换。

### 3. 报告路径

- 文件检视：`./operators/{operator_name}/{source_file}_review_summary.md`（单文件输入）；多文件/目录输入时 `./operators/{operator_name}/{operator_name}_review_summary.md`
- PR 检视：`./operators/pr-{pr_number}/{pr_number}_review_summary.md`

### 4. 超长报告过滤

补完头部元信息后，检查报告行数：

```bash
wc -l {报告输出路径}
```

- 若 ≤2000 行：跳过过滤，报告完成
- 若 >2000 行：Read + 执行 `steps/common.report-filter.md`，派发过滤子 Agent。子 Agent 读取报告、按严重程度过滤「检视意见」大表中的不严重条目、更新概览计数、原地覆盖报告文件。过滤完成后报告才视为最终成品

## 报告章节结构（脚本生成）

1. `# 代码检视报告` + `## 检视概览`：检视文件数（`{{FILE_COUNT}}`）/ 代码行数（`{{TOTAL_LINES}}`）2 个占位符由主 Agent 替换；发现问题总数与高/中/低计数由脚本统计
2. `## 检视意见`：单一大表——`条例编号 | 严重级别 | 问题描述 | 代码位置 | 修改建议`；按严重级别（高→中→低）→ 文件路径 → 行号排序，同文件意见连续；SUSPICIOUS 在问题描述尾部标「（疑似，建议人工确认）」、范围外（out_of_range，仅 PR 检视）在代码位置尾部标「（PR检视场景，非PR提交的相关代码）」、行号未校对确认标「（行号待确认）」；无问题时显示「本次检视未发现问题」

## 约束

- 报告正文由脚本一次性组装，主 Agent 仅负责补头部元信息（替换 2 个占位符）和超长过滤触发（>2000 行时派发过滤子 Agent）
- 每个 FAIL/SUSPICIOUS 发现的代码位置行号使用阶段2 行号校对后的结果（yaml 已被 workflow.line_verify.py 更新）
- PR 检视的代码位置标注完整文件路径（相对 repo_path），由子 Agent 写入 yaml 时保证
- 代码片段内容与假设检验证据不进报告——yaml 保留 `code`（行号定位依赖）与 `evidence`（置信度校验依赖），报告仅呈现行号
- 结果缺失条款（三级补派后仍未落盘）不写入报告：波次完整性校验与补派阶梯照常执行（静默保证覆盖完整）
- yaml 中 `type: design` 为已停用类型（设计一致性检查不再执行），collector 拒收、报告跳过
