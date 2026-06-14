# 逐条检视 prompt 模板（文件检视）

workflow 按 clause-routing 产出的波次规划，逐波派发子 Agent。每组使用以下 prompt 模板。

## prompt 模板

```
【已由上游完成】
- 代码侧别识别：{Kernel侧/Tiling侧}
- 条款过滤：已按侧别过滤，保留以下条款
- 代码概要：{code_summary_path}

检视文件：{file_input}

检视条款：{条例ID-1} {条例标题}、{条例ID-2} {条例标题}

【执行要求】
- 第一步加载 ascendc-code-review skill，然后 Read skill 目录下的 `core/methodology.md` 掌握假设检验方法、置信度标准和红线问题
- 若提供了代码概要，Read 获取全局视角（重点关注「API 调用索引」、「跨文件防御摘要」和「跨文件关系」）
- 对每条分配的条例，Grep `^{条例ID}` 在 references/ 中定位起始行号，再 Grep 下一个 `^####` 标题定位结束行号，Read offset={start} limit={end-start}。**只读该条例章节，禁止 Read 整个规则文档。**若条例包含专属检视方法，必须严格按该指引执行
- 若 file_input 含多个文件，对每条条例在所有文件中检查，结果标注文件路径
- 若检视条款来自 ascendc-api / ascendc-perf / simt-api-analysis / mc2-specific，必须先使用 `/ascendc-docs-search` skill 查阅对应 API 的最新官方文档
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
- 代码片段（行 N-M）：
  ```cpp
  {至少 10 行代码，含上下文}
  ```
- 假设检验证据：
  正向证据：{类型 +X% 描述}
  负向证据：{类型 -X% 描述}
  自信值 = Σ正向+Σ负向 = {累计}% ≥ 70% → 判定违规
- 修复建议：{建议}
```

禁止为 PASS 条例输出置信度或分析过程。禁止生成报告文件。
