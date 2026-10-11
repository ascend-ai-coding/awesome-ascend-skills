# 运行时：知识维护

只有用户明确要求知识维护或独立维护任务已获授权时读取本文件。
只有刷新动作确实访问 GitCode API 时，才在该动作前确认 `api` 能力已就绪；使用受审卡片或已有本地快照时不为此额外预检。

## 刷新命令

仅在用户明确要求知识维护或独立维护任务已获授权时调用刷新器；full/incremental 的选择、TTL、重叠窗口与周期保持既有规则：

```bash
export ISSUE_HANDLER_REPOSITORY_ROOT="<步骤 0 记录的 repository_root>"
(
  cd "$ISSUE_HANDLER_REPOSITORY_ROOT"
  python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/refresh_issue_knowledge.py" \
    --url "https://gitcode.com/<owner>/<repo>"
)
```

刷新器按以下状态机自行选择模式：

1. corpus/状态缺失、摘要校验失败、schema 变化、仓库或过滤策略变化时执行全量 bootstrap；显式 `--force-full` 也走全量。
2. 上次成功刷新仍在 TTL 内时返回 `fresh`，不访问远端。默认 TTL 为 15 分钟，避免同一会话的多次查询重复抓取。
3. 其他正常运行使用持久化 `cursor_at` 加 5 分钟重叠窗口，按 `updated_at` 倒序获取变更 Issue 和 PR；只有整个刷新成功后才推进游标。
4. 默认每 7 天强制全量校准。增量列表无法可靠发现已删除或失去访问权限的对象；周期全量会删除本地已不存在的 Issue/PR，并无条件重抓评论，从而校准评论编辑或删除。
5. 若远端没有按 `updated_at` 倒序返回，增量提前停止不再安全，当前运行自动退回全量。

GitCode Issue 的 `updated_at`、评论数以及 PR 的 `updated_at` 只用于降低请求量，不作为永久真相：Issue 状态、负责人、正文或评论发生变化时刷新对应记录；评论缓存同时校验 Issue `updated_at` 和评论数；PR 记录更新标题、正文引用、状态和 head SHA。任何服务端未反映到水位线的评论编辑/删除，以及增量期间不可见的 Issue/PR 删除，都由周期全量校准。

## 一致性与失败语义

- `.cannbot/gitcode-issue-handler/cache/issue-knowledge-refresh.lock` 使用进程文件锁；并发刷新不能同时写同一快照。
- corpus、报告和状态先写同目录临时文件、`fsync`，再原子替换；状态最后写入并保存 corpus SHA-256。查询器只读取 schema、仓库和摘要一致的已提交快照。
- 网络、限流、权限或单项评论抓取失败时不推进游标，也不生成部分成功快照。已有快照返回 `stale_fallback` 并继续主 Issue 流程；没有快照时返回 `unavailable`，主流程仍使用静态受审知识卡并记录知识证据不可用边界。
- 默认失败退出码为 0，避免知识增强阻塞 Issue 主流程；维护或 CI 检查可追加 `--strict`，令刷新失败返回非零。
- Token 只从当前参数或会话环境读取，不写入 corpus、状态、报告或错误字段。

## 维护输出

仅实际执行刷新时，将刷新器返回的 `status`、`mode`、`snapshot_usable` 和 `corpus` 写入既有运行状态：

```yaml
knowledge_refresh_status: fresh | refreshed | stale_fallback | unavailable
knowledge_refresh_mode: none | skip | full | incremental
knowledge_snapshot_usable: true | false
knowledge_corpus_path:
```

- `fresh` / `refreshed`：使用已校验快照；
- `stale_fallback`：刷新失败但旧快照仍通过摘要校验，继续使用并记录陈旧边界；
- `unavailable`：没有可信快照，只使用随 Skill 发布的受审卡片并继续。

## 运维参数

- `--ttl-seconds N`：调整轻量刷新 TTL。
- `--overlap-seconds N`：调整增量重叠窗口；不得设为负数。
- `--full-interval-seconds N`：调整全量校准周期。
- `--force-full`：立即全量重建并校准删除/编辑。
- `--skip-prs`、`--include-self-assigned`：改变语料策略；与上次策略不同时自动全量重建。

不要把全量刷新设置为每次查询前必做：大仓库会重复拉取全部 Issue、评论和 PR，既慢又更容易触发限流。也不要永远只做增量：删除、权限变化和服务端时间戳边界需要周期全量校准。
