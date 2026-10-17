# PR 代码获取

派发为 `general` 子 Agent 执行。

## 派发

```
Agent({
  subagent_type: "general",
  description: "PR 代码获取",
  prompt: "PR 代码获取

【输入】
- PR URL: {pr_url}

【执行步骤】

0. 清理旧缓存（同一 PR 重复检视时，防止读取陈旧代码）：
   - 从 PR URL 中提取 PR 编号
   - 若 `./operators/.pr_diff/{pr_number}.diff` 存在 → 删除该文件
   - 若 `./operators/.pr_repo/{pr_number}/` 存在 → 删除整个目录（`rm -rf`）
   - 若 `./operators/pr-{pr_number}/code_summary.md` 存在 → 删除该文件
   - 若 `./operators/pr-{pr_number}/api_prestudy.md` 存在 → 删除该文件
   - 若 `./operators/pr-{pr_number}/{pr_number}_review_summary.md` 存在 → 删除该文件
   确保 diff、源码、概要、预研全部来自本次获取，不混用不同次检视的数据。
1. 识别 PR 托管平台（URL 含 `gitcode.com` → GitCode；非 GitCode PR → 输出「当前仅支持 GitCode 平台」并终止）
2. 获取 diff：
   - 脚本路径 = `{skill_base}/scripts/get_gitcode_pr_diff.py`
   - `mkdir -p ./operators/.pr_diff`
   - 执行脚本：`python3 {skill_base}/scripts/get_gitcode_pr_diff.py --repo {repo_url} --pr {pr_number} --output ./operators/.pr_diff/{pr_number}.diff`（repo_url 从 PR URL 推断）
3. 克隆完整源码：
   - `mkdir -p ./operators/.pr_repo`
    - 执行 `python3 {skill_base}/scripts/clone_pr_source.py --repo {repo_url} --pr {pr_number} --clone-dir ./operators/.pr_repo/{pr_number}/`
    - 克隆失败则终止
4. 统计 diff 规模并判定检视档位：
   - 变更文件数 = diff 中所有变更文件路径数
   - 变更行数 = diff 新增+删除行数，按 `{skill_base}/scripts/workflow.review-thresholds.yaml` 顶部「统计口径（行数）」段统计（唯一定义处，勿自行实现）
   - 执行 `python3 {skill_base}/scripts/workflow.review_mode.py --lines {变更行数} --files {变更文件数}`，解析返回的 JSON 取 mode/routing/guidance

【输出】

返回以下结构化信息：
- `operator_name`: {提取的算子名}
- `diff_path`: `./operators/.pr_diff/{pr_number}.diff`
- `repo_path`: `./operators/.pr_repo/{pr_number}/`
- `mode`: {minimal|compact|standard|oversized}
- `routing`: {pr-review}（唯一值——超大 PR 不再切换 workflow）
- `guidance`: {对应档位的 plan agent 工作指导}
- `变更文件数`: {步骤4 统计的 diff 变更文件数——供阶段3 报告头部元信息}
- `变更行数`: {步骤4 统计的 diff 变更行数——供阶段3 报告头部元信息}

禁止生成报告文件。"
})
```
