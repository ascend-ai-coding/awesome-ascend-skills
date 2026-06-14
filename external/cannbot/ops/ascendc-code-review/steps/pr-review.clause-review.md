# 逐条检视 prompt 模板（PR 检视）

workflow 按 clause-routing 产出的波次规划，逐波派发子 Agent。每组使用以下 prompt 模板。

## prompt 模板

```
【已由上游完成】
- 代码侧别识别：{Kernel侧/Tiling侧}
- 条款过滤：已按侧别过滤，保留以下条款
- 代码概要：{code_summary_path}

检视 PR diff：{diff_file_path}
检视代码范围：{仅 op_kernel/ | 仅 op_host/ | 全部变更文件}
完整源码路径：{repo_path}

检视条款：{条例ID-1} {条例标题}、{条例ID-2} {条例标题}

【执行要求】
- 第一步加载 ascendc-code-review skill，然后 Read skill 目录下的 `core/methodology.md` 掌握假设检验方法、置信度标准、红线问题和 PR 交叉验证规则
- 若提供了代码概要，Read 获取全局视角（重点关注「API 调用索引」和「跨文件防御摘要」）
- 对每条分配的条例，Grep `^{条例ID}` 在 references/ 中定位起始行号，再 Grep 下一个 `^####` 标题定位结束行号，Read offset={start} limit={end-start}。**只读该条例章节，禁止 Read 整个规则文档。**若条例包含专属检视方法，必须严格按该指引执行
- 若检视条款来自 ascendc-api / ascendc-perf / simt-api-analysis / mc2-specific，必须先使用 `/ascendc-docs-search` skill 查阅对应 API 的最新官方文档
- 先 Read diff 了解变更范围，再 Read 完整源码追溯变量来源
- 严格按假设检验驱动流程执行（H0/H1、证据收集、自信值计算）
- 所有条款检视完成后直接输出逐条结果，禁止生成报告文件
```

## 输出格式

PASS 条例仅列出 ID，无需展开分析：
```
[条例ID] PASS
```

FAIL/SUSPICIOUS 展开完整分析：
```
[条例ID] FAIL 置信度:HIGH
- 问题描述：{描述}
- 代码片段（{完整文件路径} 行 N-M）：
  ```cpp
  {至少 10 行代码，含上下文}
  ```
- 假设检验证据：正向证据 + 负向证据 + 自信值 = {累计}% ≥ 70% → 判定违规
- 修复建议：{建议}
```

**文件路径硬约束**：代码片段必须标注完整文件路径（相对于 repo_path），格式为 `{完整文件路径} 行 N-M`。禁止只写短文件名。
- 假设检验证据：正向证据 + 负向证据 + 自信值 = {累计}% ≥ 70% → 判定违规
- 修复建议：{建议}
```

禁止为 PASS 条例输出置信度或分析过程。禁止生成报告文件。
