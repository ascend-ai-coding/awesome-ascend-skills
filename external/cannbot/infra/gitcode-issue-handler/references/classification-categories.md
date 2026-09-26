# 分类结果字段速查

仅在需要解释 category 时读取；运行分类以脚本输出为准。

常见分类：

| category | bucket | 含义 |
| --- | --- | --- |
| `planning_record` | no_attention | 纯路线图、开发规划或工作汇总，不属于本次问题响应对象 |
| `assigned_requirement` | no_attention | 历史兼容类别；新分类统一按账号关系处理，不再仅凭需求类型和任意负责人豁免 |
| `self_assigned` | no_attention | 负责人与 Issue 同作者，或有效同作者 PR 且已有负责人；两条件任一即可，即使另有他人 PR 也免首响；不计已解决 |
| `needs_first_response_with_pr` | need_attention | 非自提且已有有效关联 PR，尚无他人实质回复；概述 PR 方案并首响，无负责人时接着分配 PR 作者 |
| `needs_pr_owner_handoff` | need_attention | 已有首响或自提免首响，但没有负责人；在 response 阶段分配 PR 作者，分类器只输出动作 |
| `auto_assign_via_pr` | no_attention | 历史结果：已根据关联 PR 自动指派（新分类器不写入） |
| `auto_assign_failed` | need_attention | 自动指派失败 |
| `association_scan_incomplete` | need_attention | PR 列表扫描失败，等待自动重试且不执行外部动作 |
| `comment_scan_incomplete` | need_attention | 评论获取失败，等待从缓存断点续跑 |
| `reporter_followup` | need_attention | 提出者在维护者回复后追加评论，需优先跟进并恢复`进行中` |
| `reopened_followup` | need_attention | 保留既有关闭/终态追加评论分类；正常入口先过滤核心 closed，仅核心 open 的终态项按 follow-up 恢复`进行中` |
| `assignee_followup` | need_attention | 有效转交后责任人新增且尚未获回应的实质进展；不以挂起/watch 为前提，按需恢复`进行中`并判断下一步 |
| `awaiting_reporter_setup` | need_attention | 等待提出者的 watch 存在但状态偏离`挂起`，需修复状态 |
| `awaiting_assignee_setup` | need_attention | 历史分类兼容项；日常批量不再因缺少挂起/watch 自动生成，历史补录须由用户明确要求 |
| `awaiting_reporter` | no_attention | 已明确请求补充，当前保持`挂起`且由 watchlist 定点刷新 |
| `awaiting_assignee` | no_attention | 已回复并转交责任人，无新跟进；沿用已有状态/watch，不为历史补录进入本轮，不参与静默关闭 |
| `needs_manual_no_pr_author` | need_attention | 有 PR 但无法确定作者 |
| `needs_first_look` | need_attention | 无负责人、PR 和有效评论 |
| `needs_only_assign_cmd` | need_attention | 只有 `/assign` 评论 |
| `replied_no_owner` | no_attention | 无负责人但已有实质回复 |
| `operator_routing_required` | need_attention | 已回复但无 assignee，且明确涉及具体算子，需解析责任人 |
| `our_team_done_with_pr` | no_attention | 已有负责人、有效 PR 和实质回复；无需首响不代表问题已解决 |
| `our_team_only_assign_cmd` | need_attention | 团队负责但只有指派命令 |
| `our_team_replied` | no_attention | 团队已实质回复 |
| `our_team_needs_work` | need_attention | 团队负责但尚无响应 |
