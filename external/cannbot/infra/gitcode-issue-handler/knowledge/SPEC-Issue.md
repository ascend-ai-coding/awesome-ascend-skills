# Issue 知识卡规范

## 内容边界

- `reference/` 记录稳定规则：模式判定、证据要求、GitCode 协作约束。
- `runbooks/` 记录可复用流程：分类、复现、根因分析、交付和回评。
- `runbooks/field_notes/` 记录有公开来源的代表案例，不收录原始 Issue 全文。
- 原始全量语料只存在 `.cannbot/gitcode-issue-handler/data/`，由脚本重建，不作为知识正文提交。

## frontmatter

卡片使用 `okf.v1`，至少包含：

`schema_version`、`kind`、`type`、`source_family`、`title`、`description`、`tags`、`resource`、`sources`、`status`、`confidence`、`created_at`、`updated_at`。

每卡至少一个 `sources`，且有且仅有一个 `role: primary`；`resource` 必须等于 primary URL。Git 仓来源固定到 commit，历史案例使用公开 Issue/PR URL。

## 历史案例准入

案例必须同时具备：

1. 可观察现象或诉求；
2. 公开评论、关联 PR、代码历史或复现记录之一；
3. 明确区分“事实”“处理动作”“可复用启示”；
4. 适用边界与不可推断项。

“Issue 已关闭”不能单独证明“问题已修复”。只有规则统计、没有人工复核的条目保持 `provisional`，不得作为当前 Issue 根因的唯一证据。

## 更新规则

- 同一概念增强已有卡，不按每个 Issue 重复造卡。
- 新增或修改卡片时同步逐层 `index.md`，并通过人工复核和代码评审保留变更证据。
- `reference/` 和 `runbooks/` 是受审知识卡，只能通过人工复核和代码评审更新；运行时刷新
  不得写入这些目录。
- 历史证据由 `scripts/refresh_issue_knowledge.py` 首次全量、日常增量并周期全量校准；原始
  corpus 只存放在目标仓库 `.cannbot/gitcode-issue-handler/`。卡片只吸收复核后的稳定模式
  和代表案例。
- 不在卡片中保存 Token、私有路径、不可公开日志或大段原文。
