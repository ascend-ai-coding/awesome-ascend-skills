# 检视计划设计（主 Agent 直接执行）

file-review 与 pr-review 共用,主 Agent 在 code-fetch 返回（PR 模式）或输入解析（文件模式）后直接执行,不派发子 Agent。职责：**代码侧别识别** + **规则文档级检视分组**。条款级侧别过滤与适用判断由各组检视子 Agent 执行时自行完成。

执行前，若任务提供了领域关联依据，保留来源备用；没有则忽略。关联依据仅供声明了 `关联适用` 的规则使用。

## 输入

- （PR 模式）diff 路径：{diff_file_path}、repo_path
- （文件模式）file_input（或源码根目录）
- 关联依据（若有）：{本次任务提供的领域关联依据}
- scope_hint：{scope_hint}（条例类别,空则全量）
- 检视类型：{review_type}（file 或 pr）
- 检视标识：{review_id}（PR 检视时为 PR 号；文件检视时留空）

## Step 1 创建 yaml 输出目录

调用 `{skill_base}/scripts/workflow.create_review_dir.py` 创建本次检视的结构化 yaml 输出目录：
- 命令：`python3 {skill_base}/scripts/workflow.create_review_dir.py --type {review_type}`，仅当检视类型为 pr 时追加 ` --id {review_id}`（file 检视的 review_id 为空，**必须省略 --id 参数**——空值会导致 argparse 报错退出）
- 捕获 stdout 输出作为 `yaml_dir`（绝对路径，如 `/tmp/pr1234_a3b7x9`）。yaml_dir 保留用于启动 collector 服务和阶段2/3 脚本调用,**不传给 clause-review 子 Agent**（子 Agent 通过 collector HTTP 端点提交 yaml，不接触目录路径）
- 目录创建失败（exit code 非 0）则终止，报错返回

## Step 2 侧别识别

按文件路径判定,不读代码内容：含 `op_kernel/` → Kernel；含 `op_host/`/`op_graph/`/`op_api/` → Host；两类并存 → 混合。PR 模式从 diff 变更文件列表取路径,文件模式从 file_input 取。输出：Kernel / Host / 混合。

## Step 3 声明式匹配（文件级）

执行 sed -n '/<适用>/,/<\/适用>/p' {skill_base}/references/*.md,逐文件判定:语言匹配（规则语言=代码语言或不限）;侧别匹配（规则侧别=All 或含代码侧别）;默认启用=false 跳过;领域规则（领域=true）:触发:必须触发→无条件通过,否则触发关键词在 diff（PR 限定变更行）/代码 grep 命中→通过；未命中时，仅对声明了 `关联适用` 的规则，读取传入的关联依据,满足声明条件→通过，否则跳过；未声明该字段→按原逻辑跳过。文件 <适用> 含 排除场景: 字段则按其自定义逻辑排除。
scope_hint 非空时:类别名（如\"数值安全\"）→ 只保留该类条例所在文件;\"secure\" → 只保留 cpp-secure+ascendc-topk;空 → 全保留。
禁止 Read 源码仓;reference 只 sed 取声明头与快速索引段。

## Step 4 分组（文档级）

每个匹配的规则文件 = 一个组,负责该文件全部条款,不拆分、不跨文件合并。cpp-style 专项组照旧（组名 style,标签 [全部],全文 Read,结果以 yaml `category: style` 字段输出）。

## Step 5 输出检视计划

```
yaml_dir: /tmp/{目录名}
检视计划 [标准]
检视范围: {全量/scope_hint} | 代码语言: {C++/Python/Build/混合} | 侧别: {Kernel/Host/混合}
匹配规则文件: {文件列表}（共 N 个文件）
关联依据（仅供声明关联适用的已选条款使用，无则省略）: {对应文件/条款、关系、来源、证据位置及确认状态}
跳过文件: {文件}:{原因}

分组（单波全部派发）:
  {组1（来源,标签）}: 条例ID {标题},...
  style（cpp-style,[全部]）: 全部 16 条 | 读取:全文 Read cpp-style.md | 标记:yaml category: style

共 G 组（G = 匹配文件数,含 style 专项组）。
```

禁止生成报告文件。
