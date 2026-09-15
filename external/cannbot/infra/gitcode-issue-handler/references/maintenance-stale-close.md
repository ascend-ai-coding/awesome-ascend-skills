# 维护路径：已答复咨询 Issue 自动闭环

## 目标与边界

`scripts/auto_close_stale_issues.py` 对已答复、超过静默期且没有关联 PR 的咨询类 Issue
执行固定闭环。默认静默期为 48 小时，默认只预览。交互运行中必须先展示 dry-run 选出的
每个 Issue、完整固定评论和关闭操作，用户明确确认该精确清单后才传 `--apply`；初始“自动
处理 / auto apply”请求不是这次批准。只有已单独审定候选范围、评论模板和运行策略的定时
任务，才能依据其部署授权直接使用 `--apply`。

脚本必须同时确认：

1. Issue 仍为打开状态，并由标签或标题明确标识为咨询类。
2. Issue 提交者可识别；不要求 Issue 已设置 assignee。
3. 至少一名作者可识别的非 Issue 提交者发表了非空、非 `/assign` 的实质答复。索要
   日志/版本/复现等补充信息通常不算已答复；但若该评论已按主流程成功切到`挂起`并写入
   `awaiting_reporter` watch，可按明确等待提出者的路径参与静默闭环。
4. 从最后一条非提交者实质答复起已满静默期；watch 路径从 `waiting_since` 起算。多名
   处理人连续补充时，以最后一条为准。
5. Issue 提交者在最后一条处理答复后没有新回复。
6. Issue 正文、评论和 PR 关联扫描均未发现关联 PR。
7. PR 列表与原生关联补查完整；达到页数或 API 预算上限时保守跳过。

`awaiting_assignee` 表示维护侧仍欠处理结论，即使超过静默期也不得自动关闭；只有
`awaiting_reporter` watch 可以按上述信息请求路径参与静默闭环。

评论时间、作者或 PR 关联证据不完整时不得自动关闭。

## 使用

先预览：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/auto_close_stale_issues.py" \
  --config .cannbot/gitcode-issue-handler/config/classify_config.yaml
```

交互运行在用户确认上述精确预览后执行；定时任务则必须已有单独记录的部署授权：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/auto_close_stale_issues.py" \
  --config .cannbot/gitcode-issue-handler/config/classify_config.yaml --apply
```

脚本从 `GITCODE_TOKEN` 读取 Token。静默期、咨询标签、标题标识、固定评论和 PR 扫描
预算在统一配置目录的 `classify_config.yaml` 的 `auto_close` 下配置；
`--hours`、`--comment`、
`--pr-fetch-pages` 和 `--pr-linkage-api-budget` 可覆盖单次运行参数。

## 写入顺序

对每个候选 Issue：

1. 重新 GET Issue 和评论，防止使用过期快照。
2. 再次执行全部资格判断。
3. 检查固定评论是否已存在，避免重复评论。
4. 必要时 POST 固定评论，并 GET 回查正文。
5. 再检查评论快照，发现并发新回复时不关闭。
6. PATCH `state=closed`，随后 GET 验证状态。
7. 只有关闭回查成功后才移除对应 follow-up watch；失败时保留供下轮刷新。

评论未写入或未通过回查时不得关闭。单个 Issue 的网络或权限失败记录为失败，继续处理
其他候选项。

## 输出

脚本向标准输出写 JSON，包括模式、静默期、PR 扫描诊断和每个 Issue 的判定理由。
常见动作包括：

- `would_close`：dry-run 中满足全部条件。
- `closed`：评论已回查且关闭状态已验证。
- `skipped` / `skipped_*`：不满足条件或刷新后条件变化。
- `comment_failed` / `close_failed` / `action_failed`：外部操作失败。

存在失败动作时退出码为 1；全部成功或仅跳过时退出码为 0。
