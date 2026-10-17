# 逐条检视 message 模板（文件检视）

workflow 按 `plan-design` 产出的检视计划逐波派发子 Agent。每个分组使用以下 message 模板。

## message 模板

```text
【已由上游完成】
- 检视组：{group_id}
- 文件类型：{Python/TileLang DSL/Markdown/混合}
- 代码侧别：{Kernel/Host/混合/N/A}
- 条例过滤：已完成，只执行下方分配的条例
- 代码概要：{code_summary_path}
- YAML 提交端点：http://127.0.0.1:{collector_port}/submit

正式检视文件：{file_input}

分配条例：
- {reference_file_1}/{原文条例ID_1} {条例标题_1}
- {reference_file_2}/{原文条例ID_2} {条例标题_2}

【执行要求】
1. 首先加载 `tilelang-op-review` skill，再读取该 skill 的 `core/methodology.md`，掌握假设检验、证据、置信度和红线问题的判定方法。
2. 读取代码概要，获得正式检视范围、API索引、参数来源、跨文件防御和跨文件关系。概要中列出的关联文件只用于核实调用链或防御，不得加入正式检视范围。
3. 对每条分配条例，使用 reference 文件名和原文条例ID精确定位条例标题，只读取该条例到下一条条例标题之前的内容。必须保留条例内部的“问题描述”“检视方法”“证据要求”“排除规则”和示例等小节；禁止读取整份规则文件。条例存在专属检视方法时必须执行。
4. 对每条条例分别建立假设、收集正向与负向证据并作出判定，不得用整个分组的综合判断代替任何一条条例。若正式检视输入包含多个文件，每条条例都要覆盖其计划指定的全部正式文件，并在问题结果中标明实际文件路径。
5. 只有当前条例的判定依赖 TileLang API真实语义时，才按需查阅当前实际安装的 TileLang/PTO源码、lowering实现或仓库内有效用例。核对当前可疑代码涉及的具体API即可，禁止扫描或预研本次代码使用的全部API，也不得凭记忆推断版本相关行为。
6. 所有条例检视完成后，每条条例分别按下方 YAML schema 提交一次。提交命令：

   curl -sS --noproxy 127.0.0.1,localhost -X POST "http://127.0.0.1:{collector_port}/submit?group={group_id}&rule={reference_file}&clause={原文条例ID}" --data-binary @- <<'YAML_EOF'
   <YAML 内容>
   YAML_EOF

7. 检查每次 curl 的退出状态和 collector 响应。访问本地collector必须保留 `--noproxy 127.0.0.1,localhost`，不得受环境中的HTTP或SOCKS代理影响。收到 HTTP 400 时，根据返回的 schema 或 identity 错误修正后重新提交；无法成功提交时如实返回失败条例和错误，不得声称全部完成。
8. 禁止直接写入或读取 YAML 输出目录，禁止生成最终检视报告，也禁止在文本回复中重复输出检视结果。完成后只返回提交数量和未成功提交的条例（如有）。

【提交前自检】
- 每个分配条例均已读取自己的完整章节并单独检视。
- 每个分配条例均覆盖计划指定的全部正式检视文件。
- 每个分配条例均有且只有一次成功提交；因HTTP 400修正后的失败请求不计入成功次数。
- 每个FAIL/SUSPICIOUS均能对应当前条例的具体问题模式，并具有可复核的源码或文档证据。
- 没有直接操作 YAML 输出目录，也没有生成综合报告。
```

## YAML 输出格式

每条条例通过带 `group/rule/clause` 完整身份的 URL 提交。collector 负责补充：

```yaml
group_name: {group_id}
rule_file: {reference_file规范化为.md文件名}
canonical_id: {reference文件名去掉.md}/{原文条例ID}
submission_key: {group_id}::{canonical_id}
```

子 Agent不得在请求正文中自行指定这些字段。

### PASS

```yaml
type: clause
clause_id: PERF-1
clause_title: 避免热点循环逐元素 GM 操作
status: PASS
```

PASS 不得包含 `confidence` 或 `evidence`。

### FAIL/SUSPICIOUS

```yaml
type: clause
clause_id: PERF-1
clause_title: 避免热点循环逐元素 GM 操作
status: FAIL              # FAIL 或 SUSPICIOUS
confidence: HIGH          # 按 core/methodology.md 填写 HIGH、MED 或 LOW
problem_desc: {问题描述}
code_snippet:
  file_path: {正式检视文件的路径}
  start_line: {问题片段起始行号}
  end_line: {问题片段结束行号}
  code: |
    {问题原文及足以理解问题的必要上下文}
evidence:
  positive:
    - type: {证据类型}
      score: {正向分值}
      desc: {可复核证据}
  negative:
    - type: {证据类型}
      score: {负向分值}
      desc: {可复核证据}
  confidence_value: {累计置信度}
fix_suggestion: {与当前问题直接对应的修复建议}
```

Markdown 条例也使用同一普通条例格式。`code_snippet.code` 填写存在问题的文档原文和必要上下文，`file_path`、`start_line`、`end_line` 对应 Markdown 文件位置。

## 字段约束

collector 会解析并校验 YAML。提交前确保：

1. `clause_id` 必须与 URL 的 `clause` 参数完全一致，保留原始大小写、数字、连字符和小数点。
2. `file_path` 只填写文件路径，不拼接行号、注释或多个位置。
3. `start_line` 和 `end_line` 使用一基实际行号；片段必须包含问题行。上下文以足够理解问题为准，不强制固定行数。
4. `code` 只填写源码或 Markdown 原文，不添加行号前缀和文件路径注释。
5. `code_snippet` 和 `evidence` 必须是 mapping；`evidence.positive`、`evidence.negative` 必须是 list，没有负向证据时填写 `negative: []`。
6. 正向和负向证据必须来自实际代码、文档、调用链、安装源码、lowering、有效用例或可复现验证。找不到证据时不得虚构。
7. `status`、`confidence` 和 `confidence_value` 必须符合 `core/methodology.md` 的映射关系。

每条条例正常情况下只产生一个成功 YAML。collector 对意外重复提交保留原文件并使用 `_dup1`、`_dup2` 后缀，但不得主动利用该机制为同一条例制造重复结果。

禁止生成报告文件。
