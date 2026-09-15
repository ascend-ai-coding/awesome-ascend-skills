# Issue 处理知识库

面向 `gitcode-issue-handler` 的内嵌内容根。目录沿用 CANNBot 知识库的渐进式结构，skill 与知识正文分离。

这里的 `reference/` 和 `runbooks/` 是随 Skill 发布的受审知识，不由运行时脚本自动改写。
自动刷新的历史 Issue 证据只写入目标仓库 `.cannbot/gitcode-issue-handler/`，查询时以
`provisional/low` 候选补充，不能替代当前 Issue 的根因验证。

- [reference/](reference/index.md) — 从 CANNBot 和项目规则迁移的稳定处置知识。
- [runbooks/](runbooks/index.md) — 跨 Issue 可复用的调查流程与历史 field notes。
- [SPEC-Issue.md](SPEC-Issue.md) — Issue 知识卡准入与证据规范。
