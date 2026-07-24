# Pipeline · 大版本升级（ref 切换，如 9.0.0→9.1.0）

> **适用**：asc-devkit 仓库的大版本升级（ref 切换 + 架构性变更可能）。作为 `ops-knowledge-ingest` 编排器的第三条路由。
> **不适用**：同 ref 的 commit 级增量 → 走 [`incremental-sync.md`](incremental-sync.md)；首次接入 → 走 [`new-source-onboarding.md`](new-source-onboarding.md)。
> **权威 SPEC**：目标知识库根目录的 `SPEC-Version-update.md`（本文件是执行流程摘要，指回权威 SPEC）。
> **适用范围**：仅 asc-devkit（git 源、有 commit diff）。文档站 bundle 的版本切换另立流程。

## 范式

```
0 prep        : 目标 ref 前置验证 + 改脚本常量 + 双 clone + pin 新 sha
1 diff        : 结构性 diff (确定性 version-diff) + 架构性变更识别 (agent) → impact_inventory
2 map         : 影响面映射到现有卡 (knowledge_query 检索) → ledger
3 curate      : 派 subagent 复用 ops-knowledge-reference-ingest §4，按 5 类变更分流著作
4 review      : 派独立 subagent 复用 ops-knowledge-reference-ingest §5 + 版本升级专项
5 修订        : 按 findings 改至 clean
6 migrate-note: 架构性变更 → runbooks/version-migration/<from>_to_<to>.md
7 finalize    : index --no-pin → 检索 → 图谱 → 覆盖账(两本) → 迁移态 verify + knowledge-lint → finalize-version-bump(原子 advance-pin+viz+verify+log)
```

## 变更类型分流（5 类）

| 类型 | 触发 | 卡动作 | `sources[].url`+`resource` 指向 | 图谱 |
|---|---|---|---|---|
| `new` | 新版本有、旧无 | 新建卡 | `new_sha` | 新节点 |
| `update` | 同路径内容变 | 增量融合；旧 url **重写**为新路径@new_sha | `new_sha` | 重判该卡 |
| `deprecate` | 旧有、新无 | 标弃用保留 | **`deprecated_commit`**（本卡自承） | 边保留、viz 标灰 |
| `move_rename` | 路径变、内容同 | 改 `sources[].url` 为新路径@new_sha | `new_sha` | **重判该卡**（sources 变致 card_fp 变） |
| `structural` | 架构性变更 | 触发 migrate-note + 波及卡 update | 波及卡按 `update` | migrate-note 新节点 |

> **SHA 策略 = 严格单版本 + 弃用卡自承**：active 卡（new/update/move_rename）所有 sources 指向 new_sha；deprecate 卡（`deprecated: true`）全部指向本卡 `deprecated_commit`（= 升级前 watermark old_sha，跨升级累积各自自承）。

## 关键命令（从 ops-knowledge-ingest skill 目录执行，目标知识库用 `--knowledge-root` 指定）

- **prep**：`git -C .build/asc-devkit fetch --tags --prune` + `git rev-parse --verify 9.1.0^{commit}`（不存在则停，不改常量）
- **diff**：`python3 ../ops-knowledge-reference-ingest/scripts/asc_devkit_extract.py --knowledge-root <知识库根> version-diff --old-tree .build/asc-devkit@<oldsha> --new-tree .build/asc-devkit`
- **finalize**：
  1. `python3 ../ops-knowledge-reference-ingest/scripts/asc_devkit_extract.py --knowledge-root <知识库根> tags`
  2. `python3 ../ops-knowledge-reference-ingest/scripts/asc_devkit_extract.py --knowledge-root <知识库根> index --no-pin`（保留旧 pin）
  3. `python3 ../knowledge-query/scripts/knowledge_query.py --knowledge-root <知识库根> build`
  4. `python3 scripts/okf_graph.py --knowledge-root <知识库根> candidates → judge → inject → viz → verify`
  5. 覆盖账两本：新版本账 `union(active 卡 sources) ∪ skipped == new version docs`；移除账 `deprecated ∪ move_rename.old_path == (old - new) docs`
  6. `python3 ../ops-knowledge-reference-ingest/scripts/asc_devkit_extract.py --knowledge-root <知识库根> verify --old-sha <oldsha> --new-sha <newsha>`（迁移态专用，按 `deprecated` 分流校验 sources 全量 sha + URL 存在性，不读 index pin）+ `python3 ../knowledge-lint/scripts/knowledge_lint.py --knowledge-root <知识库根>`（全库通用原则，不读迁移态 SHA）
  7. **`python3 ../ops-knowledge-reference-ingest/scripts/asc_devkit_extract.py --knowledge-root <知识库根> finalize-version-bump`**（原子化落地开关：内部 advance-pin → `okf_graph.py viz` → `okf_graph.py verify` → 写日志，任一步失败回滚——执行前备份 `index.md`+`graph/viz*.html`+`log/<date>.md`，均用 `.tmp` → 原子 `os.replace`，失败从备份恢复。**最后一步，之后无其他落地步骤。**）

## 红线速查
- [ ] **`finalize-version-bump` 是唯一「已落地」开关，原子化 pin+viz+verify+log，且是最后一步**。
- [ ] **SHA 严格单版本 + 弃用卡自承**：active 卡 == new_sha；deprecate 卡 == 该卡 `deprecated_commit`。SHA 分流校验只在 `asc_devkit_extract.py verify --old-sha --new-sha`（不读 index pin）；knowledge-lint 不读迁移态 SHA。
- [ ] **目标 ref 前置验证**：`git rev-parse --verify <ref>^{commit}` 通过后才改脚本常量；不存在则停。
- [ ] **deprecate 不物理删除**：frontmatter `deprecated` 五元组（含 `deprecated_commit`）+ 正文提示。
- [ ] **move_rename 重判图谱**（sources 变致 card_fp 变，不能跳过）。
- [ ] **URL 存在性校验**（git 对象库多源 fallback，不 checkout 目录）：`.build/asc-devkit` → `.build/asc-devkit@<oldsha>` → `git fetch --depth=1 origin <sha>` 后重试；fetch by SHA 失败只报「对象缺失」绝不报 404。
- [ ] **架构性变更有专门落点**：`runbooks/version-migration/`，不散落在各卡。
- [ ] **watermark 缺失则停下问基线，不猜**。

> 完整流程、findings 修正历史、回滚实现细节见目标知识库根目录的 `SPEC-Version-update.md` §4–§8。
