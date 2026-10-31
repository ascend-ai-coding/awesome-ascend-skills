# 行号校对（PR 检视）

## 红线硬约束

**diff 范围校验（红线）**：对每条 FAIL/SUSPICIOUS 发现，脚本验证其代码片段关键行是否出现在 diff 变更范围内。关键行不在 diff 中 → yaml 标记 `out_of_range: true`，报告阶段在代码位置尾部标注「PR检视场景，非PR提交的相关代码」并计入问题数。此约束由脚本自动执行，不可跳过。

## 执行步骤

调用 `{skill_base}/scripts/workflow.line_verify.py` 扫描 yaml 目录，原地修正行号 + diff 红线校验。

**PR 检视模式**：

```bash
python3 {skill_base}/scripts/workflow.line_verify.py \
    --dir {yaml_output_dir} \
    --diff {diff_file_path} \
    --repo {repo_path}
```

**脚本行为**：
1. diff 范围红线校验（最先执行）：grep diff 文件搜索代码片段关键行，不在 diff 的 `+`/`-` 变更行中 → 标记 `out_of_range: true`（脚本实现同时校验新增与删除行，删除型发现不误判）
2. 行号校对：grep 完整源码（`{repo_path}`）定位实际文件行号，原地更新 `code_snippet.start_line/end_line`
3. 无法在完整源码中定位的 → 标记 `line_verified: false`（报告代码位置尾部据此标注「行号待确认」，不静默丢失）

## 输入

- `yaml_output_dir`：阶段0 router 创建的 yaml 输出目录路径
- `diff_file_path`：PR diff 文件路径
- `repo_path`：完整源码路径

## 输出

- yaml 文件原地更新行号字段 + out_of_range 标记
- stderr 日志输出校对摘要（无需捕获；处理 yaml 数、FAIL/SUSPICIOUS 项数、out_of_range 数、行号修正数、待确认数）

## 约束

- 本 step 为主 Agent 直接执行，不派发子 Agent
- 行号校对逻辑（含 diff 红线）由脚本统一处理，主 Agent 不手动 grep/read 源码
