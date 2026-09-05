【最高优先级指令 — 必须严格遵守，覆盖 skill workflow 中的任何冲突指令】

1. 如果 prompt 中已包含【确定性路由结果】，则直接使用，禁止调用 clause_routing.py、plan-design 或任何 routing/plan 脚本重新运行路由。如果 prompt 中未包含路由结果，则按 skill 原流程执行路由。
2. 如果 prompt 中已包含【确定性路由结果】，禁止裁剪条例清单。必须逐条检视其中列出的所有条例，不得以"无关""噪声""不适用"为由跳过任何条例或分组。如果认为某条条例不适用，在报告中标注 PASS 并说明原因，但必须检视。
3. 禁止使用 generate_report.py 或任何报告生成脚本，除非 skill workflow 已内置 assemble_report.py 脚本负责报告组装（此时按 skill 原流程使用该脚本）。如果 skill 未内置报告组装脚本，你必须自己用 Write 工具写 Markdown 报告到指定路径，并在报告末尾手动追加 ## 结构化发现 YAML 块。
4. 报告必须包含 ## 结构化发现 章节。如果没有任何 FAIL/SUSPICIOUS 发现，也要输出空的 YAML 块（findings: []）。缺少此章节视为检视失败。
5. 检视子 Agent 必须先 Read 被检视代码文件的实际内容再做检视。禁止根据条例文档示例或 API 文档示例推断代码中存在的问题。每条 FAIL/SUSPICIOUS 发现必须引用实际代码的具体行号和代码内容，不得描述代码中不存在的问题。
6. 检视范围：你可以 Read 算子目录下所有关联文件获取上下文（如变量来源、上游校验、调用链），但只对 diff 变更文件中的代码报告 FAIL/SUSPICIOUS。对非 diff 文件中发现的历史问题，仅在报告附录中标注为"上下文观察"，不计入结构化发现 YAML。

使用 ascendc-code-review skill 检视以下 PR 代码。PR 检视模式。

【重要】diff 和完整源码已就绪，跳过 code-fetch 阶段（阶段0步骤3-4），直接从阶段0步骤5开始执行，使用以下路径：
- diff 文件路径：{diff_file}
- 完整源码路径（repo_path）：{repo_dir}

检视完成后，必须用 Write 工具将报告写入指定文件（不要用脚本生成）。

【结构化输出要求】在 Markdown 报告末尾追加 `## 结构化发现` 章节，用 YAML 格式列出所有 FAIL/SUSPICIOUS 发现，每条字段对齐 LlmComment 结构：
```yaml
findings:
  - id: "条例ID"
    path: "文件路径"
    start_line: 起始行号
    end_line: 结束行号
    severity: critical|high|medium|low
    category: bug|security|performance|maintainability|test|documentation|style|other
    confidence: 0-100整数
    content: "问题描述一句话"
    suggestion: "修复建议一句话"
    suggestion_code: "修复代码片段（如有）"
    existing_code: "问题代码片段（如有）"
```
severity 使用 critical/high/medium/low 四级（critical=崩溃/越界/数据损坏, high=逻辑错误/安全漏洞, medium=规范违反/魔数, low=风格/命名）。
