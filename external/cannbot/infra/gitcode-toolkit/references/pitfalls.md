# Git 操作易错点

常见 Git 操作错误及正确做法对照。

---

## 易错点对照表

| 问题 | 正确做法 | 错误做法 |
|------|----------|----------|
| refs 路径 | `refs/merge-requests/{n}/head`（GitCode） | `refs/pull/{n}/head`（GitHub） |
| 比较基准 | `git diff $MERGE_BASE pr_xxx` | `git diff base_branch pr_xxx` |
| merge-base 失败 | `git fetch --deepen=500` 后重试 | 直接使用 base_branch |
| 克隆深度 | `--depth=500`（标准）/ `--depth=200`（轻量） | 不指定 depth（全量克隆太慢） |
| PR 文件列表 | 优先用 `git diff --name-status` | 仅依赖 API（可能认证失败） |
| triple-dot vs double-dot | `git diff A...B`（A 和 B 的共同祖先到 B） | `git diff A..B`（A 到 B 的差异） |

---

## 平台行为易错点（GitCode API 实测）

评论、关闭、机器人收编等平台行为与直觉不符的实测坑：

| 问题 | 正确做法 | 错误做法 |
|------|----------|----------|
| 评论正文 `@login` 被平台重写为 Markdown 链接 | 回读/去重锚点选不含 `@` 的稳定子串（数字开头登录名不被改写） | 用 `@login` 子串做回读匹配（实测高频失配） |
| 评论 POST 后列表回读有秒级延迟 | 延时数秒重查，或按正文稳定锚句匹配 | 秒级回读 MISS 即判失败并重发（造成重复评论） |
| 关闭类操作的评论时序 | 先评论、回读确认后再关闭 | 先关闭再评论（已关闭/已合入的 PR 拒绝评论，意见带不上） |
| PR 可被删除，删除后 API 404 | 按「不存在」收口：核对编号后不重试，与关闭态区分 | 把 404 当瞬时错误反复重试，或当关闭态处理 |
| 机器人合入判据 | 票齐后确认全部行内讨论（含机器人自动分析评论）已解决，再等收编 | 票齐即视为可合入（未解决的行内讨论会拦合入） |
| closed PR 列表 `merged_at` 置空 | 按 `updated_at` 截断拉取，候选逐笔直查终态 | 按 `merged_at` 过滤 closed 列表（已合入项全漏） |
