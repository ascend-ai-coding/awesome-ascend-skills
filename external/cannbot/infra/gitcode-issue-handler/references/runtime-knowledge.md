# 运行时：知识查询边界

## 读取时机

主 Issue 流程在第一次知识检索前读取本文件。使用受审卡片或已有本地快照时不为此额外预检。
本文件只管理运行时历史证据，不修改随 Skill 发布的受审知识卡。

知识维护的刷新命令、完整刷新状态机、一致性与失败语义、维护输出状态字段及运维参数见 [`knowledge-maintenance.md`](knowledge-maintenance.md)；只有用户明确要求知识维护或独立维护任务已获授权时才读取该文件。

## 两层知识与信任边界

| 层 | 位置 | 更新方式 | 诊断权重 |
| --- | --- | --- | --- |
| 受审知识卡 | `$ISSUE_HANDLER_SKILL_ROOT/knowledge/reference/`、`runbooks/` | 代码评审后随 Skill 版本发布 | 优先读取，可作为规则或调查方法依据 |
| 运行时历史证据 corpus | 目标仓库 `.cannbot/gitcode-issue-handler/data/issue-history.json` | 独立维护调用 `refresh_issue_knowledge.py` | `provisional/low`，只能提出候选调查方向 |

历史卡中的分类/授权口径不覆盖当前 workflow references；不为读取相关链接遍历全部知识卡。
自动刷新不得创建、覆盖或提升受审知识卡。Issue 已关闭、评论声称已修复或存在关联 PR 都不是当前 Issue 根因或修复有效性的充分证据。把运行时案例提升为稳定知识时，仍须按 `knowledge/SPEC-Issue.md` 人工复核、更新逐层索引并走代码评审。

## 查询与维护分离

正常首查直接使用下述查询协议，复用校验通过的已有快照及受审知识卡，并按当前 Issue 定向读取源码、评论或 PR。快照缺失、过期或无命中均不阻塞首响；不为单项查询同步构建全仓 corpus。

## 查询协议

`knowledge_query.py search/preflight` 保持 `results` 和 `read_first` 为受审知识卡，另外返回：

- `runtime_corpus`：快照是否 `usable`、`missing`、`invalid` 或 `disabled`；
- `runtime_candidates`：最多 5 个低置信度历史证据候选，包含公开来源 URL 和不可直接归因警告；
- `route: runtime_evidence_only`：没有受审卡命中、但存在历史候选。此时不能把候选放进 `read_first` 冒充规则卡。

调查时先读 `read_first`，再按需打开 `runtime_candidates[].resource`，最后用当前 Issue、当前源码、版本历史或稳定复现独立验证。

查询器默认从当前目录读取目标仓库的 `.cannbot`。进入外部受管 worktree 后，必须显式保留原目标仓库根，不能误读 worktree 内同名路径：

```bash
export ISSUE_HANDLER_REPOSITORY_ROOT="<步骤 0 记录的 repository_root>"
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/knowledge_query.py" preflight \
  --task "<当前问题>"
```

也可在子命令前传 `--repository-root "<repository_root>"`。显式 `--runtime-corpus/--runtime-state` 优先级最高，主要用于维护和测试。

## 普通查询输出

普通查询记录查询器返回的 `runtime_corpus` 可用性与证据路径，不把“读了快照”记为本轮刷新成功。

保持 `ISSUE_HANDLER_REPOSITORY_ROOT` 指向原目标仓库；后续进入组 worktree 再次查询时仍依靠它读取同一快照。完成后进入步骤 1。
