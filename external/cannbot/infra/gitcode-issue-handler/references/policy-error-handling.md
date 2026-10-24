# 跨阶段错误与卡点处置

仅在出现相应故障/等待时读取该条目；正常路径无需加载整个错误目录。业务授权以 [authorization-contract.md](authorization-contract.md) 为准，外部评论只包含提出者可行动的信息。
内部环境、权限、CI 和重试细节写入使用者报告；单项失败不阻塞无依赖的其他 Issue。

## 交互与恢复

- `single` 和 `batch` 共用回复与交付两种检查点；初始“处理/自动处理/auto apply”请求不是代码交付批准；首响与指派按 `automation.md` 的开关执行。
  当前会话已有准确操作授权则复用，不能按内部阶段重复询问。“是否继续”不替代缺失的具体输入。
- 每个适用能力检查点最多请求一次输入，缺失算子责任人请求最多 1 次；owner 候选表随 summary 交付，线下确认即可，不强制当轮收齐。候选调查不延迟回复，本身不授权 assign 或 direct；临时指派须另满足 `automation.md` 的开关或会话授权。
- 仅在首个依赖操作前收集 Token、无法推导的 remote、或将 commit 列入统一执行预览前缺失的 git author。已有检查成功结果复用；目标必需接口 401/403 才触发凭据恢复，替换 Token 不扩大授权。
- 缺工具、不可写目录、NPU/依赖缺失、可选配置缺失，按下表降级或记录 blocker，不用用户回答“继续”掩盖技术问题。平台权限审批与业务授权不同，不让同一动作重复接受业务确认。
- 输入未到或确认被拒绝时保存恢复点与已完成工作，不执行依赖动作；输入到达从该点恢复，不重做已通过预检、获取、诊断或其他 Issue。direct push 的 commit 后独立确认始终保留。

## 按故障选择动作

| 条目 | 动作与结果 |
| --- | --- |
| C08 已关闭 Issue | 正常 single/batch 均 `skipped_closed`；按 intake 入口过滤，不因追加回复或显式目标跟进/自动 reopen |
| C09/C12 缺信息或环境匹配但无法复现 | 只索要真正缺失的最小字段，记录 `waiting_context`；不得改源码。获授权后按评论回查→挂起回查→watch 落盘，见 `issue-followup.md` |
| C10/C15 目标环境不匹配或外部资源缺失 | `waiting_environment`，逐项记录目标与本地 CANN、源码 commit/分支、SoC，未知填 unknown；不用非目标环境证明复现，不自动安装系统依赖。只在问题原文确实缺定位必需信息时向提出者索要 |
| C11 偶发复现 | 同输入/环境最多 3 次有界尝试，记录次数与触发条件；不稳定则 `intermittent_waiting`，准备证据回复并继续其他项，不改代码 |
| C13 未知 owner | `pending_operator_owner`，逐算子核查核心贡献候选；获授权可先临时指派，否则等待用户确认具体账号或当前 Issue 的 `direct`，不能静默回退到 Agent 正常处理。详细转交协议见 `operator-handoff.md` |
| C13 指派失败 | `assignment_failed`，保存 owner 和失败证据，不转入自行修复；未验证 assignee 不能挂起或写 assignee watch |
| C14 无 NPU | 能在 CPU/仿真执行的 ophost 等测试照常运行；opapi/opkernel 执行相关编译/`--noexec`，标记 `degraded_validation`，交付说明“未执行真实 NPU UT”，不虚报上板通过 |
| C16/C17 交付确认或 author 缺失 | 保留未提交 worktree 与操作预览，按 `delivery-confirmation.md` 恢复；只有将创建 commit 才检查 author，绑定邮箱不一致只告警 |
| C20 CI 失败 | 获取日志区分代码/基础设施/偶发；偶发或基础设施原样重试最多 1 次，有明确代码根因最多 2 轮修改→本地门禁→推送→重触发。每次写入仍须被准确授权覆盖；耗尽后 `ci_blocked`，保留 PR 继续其他组，只在报告给下一步建议，不追加“下一步怎么做”的询问，不向外部 Issue 披露内部 CI 故障或自动合并 |
| C21/C22 必需回复失败 | `comment_failed`，保留同一评论结果文件，停止该 Issue 的指派、状态、代码/PR 依赖动作，其他独立项继续；结果回评失败不回滚已有修复或 PR |

## 评论与网络错误

使用 toolkit `post_issue_comment.py` 的去重、POST 和 GET 回查；POST 结果未知时只回查，不得把普通 GET 的重试策略套到非幂等 POST，也不得清空/换结果文件绕过未知状态。
幂等 GET 复用共享客户端：总计最多 3 次 attempt（首次 + 2 次重试）；网络错误/5xx 退避 2、4 秒，429 按服务端 `Retry-After`/reset 与共享 cooldown 等待，不叠加 Agent 重试层。
422 是明确拒绝，先核对原因；若确需修改正文，由 Agent 在保持必要信息后重新核对授权，只允许一次修正后的新 operation，仍失败则保留 blocker；不由脚本自动清理或缩短正文。
401/403/404 不盲目重试。

`/user` 对评论执行器的作者核验是必需接口，失败不能跳过身份检查发送；仅在当前动作不依赖某账号接口时，才可将该接口失败视为可选降级。目标必需接口的 Token 无效按 `runtime-capability-checks.md` 恢复；替换后仍被拒绝才记录权限 blocker。

## 状态与报告

等待提出者、等待责任人和任一方追加回复的状态/恢复顺序，以 `issue-followup.md` 为准。
只有有效回复、指派与状态都回查成功后才能记 `delegated` 或已进入等待；等待责任人没有静默自动关闭期限。候选不是 owner，不公开候选 @，不替使用者确认责任。

Token/author 缺失是待输入，不伪装成发布失败。权限、保护分支拒绝，或必须依赖 rebase/reset/force push 等破坏性恢复时，停止该发布路径并报告，不能借降级绕过。
CI 失败报告至少保留 PR、阶段、分类、已用轮次与下一步；其他字段按状态契约按需填写。
