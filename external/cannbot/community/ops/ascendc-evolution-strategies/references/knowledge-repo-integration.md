# 知识仓绑定与本地知识库（统一说明）

本文件是 `ascendc-evolution-strategies` 与 cannbot-knowledge 三仓分离后的**唯一说明入口**。
SKILL.md 只保留协议摘要，脚本用法、目录契约、验收方法都以本文为准。

---

## 1. 三个仓分别是什么

| 仓 | 本地路径示例 | 职责 | 本 skill 如何使用 |
|---|---|---|---|
| **cannbot**（插件仓） | `…/cannbot/plugins-community/ascendc-ops-perf-evolution` | 编排：agents / hooks / `init.sh` | 安装 skill、绑定知识仓、驱动进化循环 |
| **cannbot-skills**（技能仓） | `…/cannbot-skills/community/ops/ascendc-evolution-strategies` | 能力：策略索引 / 场景路由 / 脚本 | **本目录** |
| **cannbot-knowledge**（知识仓） | `…/cannbot-knowledge` | 知识正文：runbooks / cards / discovered | 运行时按需读取；本地库审核后经 PR 回写 |

知识仓内与本 skill 相关的路径：

```
<root>/knowledge/index.md
<root>/governance/schemas/registries.yaml
<root>/knowledge/contrib/ascendc/index.md
<root>/knowledge/contrib/ascendc/runbooks/performance/            # Playbook（142）
<root>/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/       # 策略卡（161）
<root>/knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/  # X 发现（89）
```

---

## 2. 知识根解析规则

唯一实现：`scripts/knowledge_root.py`。

优先级（高 → 低）：

1. `--knowledge-root <path>`（CLI）
2. `CANNBOT_KNOWLEDGE_ROOT` 环境变量
3. 自 cwd 向上最近的 `.cannbot/knowledge.env`
4. 兄弟检出 `<repo>/../cannbot-knowledge`（**仅兜底**，打印 WARN）

**fail-loud 规则**：

- 1/2/3 任一被配置但路径无效 → 直接 `KnowledgeRootError`，退出码 3，**不再静默换仓**；
- 只有完全未配置时才允许第 4 级兜底；
- 校验项见 `validate_root()`：`knowledge/index.md`、`governance/schemas/registries.yaml`、
  `contrib/ascendc/index.md`、`runbooks/performance/`、`cards/`、`discovered/`。

`.cannbot/knowledge.env` 由 `init.sh` 或 `knowledge_root.py --write-env` 生成：

```bash
export CANNBOT_KNOWLEDGE_ROOT='/path/to/cannbot-knowledge'
CANNBOT_KNOWLEDGE_MODE='consumer'
CANNBOT_KNOWLEDGE_COMMIT='<git HEAD>'
```

---

## 3. 初始化

### 3.1 只用 skill 仓

```bash
bash scripts/../init.sh \
  --project <项目根> \
  --knowledge-root /path/to/cannbot-knowledge
```

执行内容：校验知识仓 → 写 `<项目>/.cannbot/knowledge.env` → 初始化
`<项目>/.cannbot/knowledge-local/{staging,promoted,rejected}` → 跑 smoke。

只检查不写配置：`bash init.sh --project <项目根> --check`。

### 3.2 通过插件安装（推荐）

```bash
bash <plugin>/init.sh project claude \
  --knowledge-root /path/to/cannbot-knowledge \
  --skills-repo /path/to/cannbot-skills \
  --project <项目根>
```

插件 `init.sh` 新增参数：

| 参数 | 作用 |
|---|---|
| `--knowledge-root <dir>` | 显式绑定知识仓；无效 → exit 2 |
| `--skills-repo <dir>` | 三仓分离布局下定位 skills 仓（否则依次找 `vendor/cannbot-skills`、`../../../cannbot-skills`） |
| `--project <dir>` | 写 `.cannbot/knowledge.env` 的项目根（默认当前目录） |

---

## 4. 脚本清单

| 脚本 | 用途 | 常用命令 | 读/写 | 退出码 |
|---|---|---|---|---|
| `scripts/knowledge_root.py` | 知识根解析/校验/写配置 | `--print`、`--validate`、`--write-env --project DIR` | 写 `.cannbot/knowledge.env` | 0 / 2（解析失败） |
| `scripts/load_playbook.py` | 按策略 ID 加载 Playbook 正文 | `--strategy-ids P1 --list-only`、`--output -` | 读知识仓 | 0 / 3（解析失败） |
| `scripts/generate_index_json.py` | 重建 `references/INDEX.json` | `--knowledge-root DIR` | 读本地 preconditions + 知识仓 cards/discovered；写 INDEX.json | 0 / 3 |
| `scripts/query_strategies.py` | 按瓶颈/算子族筛策略 | `--bottleneck ... --limit 5` | 读本地 preconditions | 0 / 1 |
| `scripts/knowledge_local.py` | 本地知识库 | `init/add/list/promote/reject/pr` | 写 `.cannbot/knowledge-local/` | 0 / 1 |
| `scripts/knowledge_smoke.py` | 知识加载 smoke | `--json report.json` | 只读 | 0 / 1 |
| `scripts/check_three_repo_contract.py` | 三仓适配契约 | `--skills A --plugin B --knowledge C` | 只读 | 0 / 1 |
| `init.sh` | 绑定 + 初始化 + smoke | `--project DIR --knowledge-root DIR` | 写配置与本地库 | 0 / 2 |

### 4.1 `knowledge_smoke.py` 输出

```json
{
  "passed": true,
  "source": "cli|env|project_env|sibling",
  "knowledge_root": "...",
  "git_commit": "...",
  "counts": {"cards": 161, "playbooks": 142, "discovered": 89, "preconditions": 166},
  "index_stats": {"...": "..."},
  "negative_test": {"returncode": 3, "passed": true},
  "checks": {
    "layout": {"passed": true},
    "counts_nonzero": {"passed": true},
    "index_consistency": {"passed": true},
    "fingerprints": {"passed": true},
    "fail_loud_on_invalid_env": {"passed": true}
  }
}
```

### 4.2 `check_three_repo_contract.py` 检查矩阵

| 仓 | 检查项 |
|---|---|
| skills | skill 目录、必需文件、`cards/` 只保留 `SCHEMA.md` + `perf_shape_spec_01.md`、preconditions 非空 |
| plugin | `plugin-sources.json` 的 skill 可解析、`init.sh` 支持 `--knowledge-root/--skills-repo`、`agents/*.md` 中 `.claude/skills/...` 引用全部可解析、无三类旧路径 |
| knowledge | 目录、6 项布局、cards/playbooks/discovered 计数（`--run-check` 时跑 `check.sh`） |

---

## 5. 本地知识库（local-first）

```
<project>/.cannbot/knowledge-local/
  staging/       # 进化运行期写入，未审核
  promoted/      # 审核通过，待提 PR
  rejected/      # 未通过，留档
  manifest.jsonl # id/kind/title/status/op/run_id/metrics/created_at/path
```

### 5.1 写入（运行期）

```bash
python3 scripts/knowledge_local.py add \
  --project <项目根> --id X90 --file draft.md \
  --op <算子名> --run-id <运行ID> --metric speedup=1.23
```

`add` 会校验 OKF frontmatter 九个字段（type/title/description/tags/status/sources/
created_at/updated_at/platforms），失败直接拒绝。

**运行期不写知识仓、不改 `references/strategy-index.md`。**

### 5.2 审核与流转

```bash
python3 scripts/knowledge_local.py list   --project <项目根> [--status staging]
python3 scripts/knowledge_local.py promote --project <项目根> --id X90
python3 scripts/knowledge_local.py reject  --project <项目根> --id X90 --reason "..."
```

### 5.3 提 PR

```bash
# 只打印步骤（默认）
python3 scripts/knowledge_local.py pr --project <项目根> --id X90

# 执行到“复制 + 重建索引 + check.sh”，commit/push 仍由人工确认
python3 scripts/knowledge_local.py pr --project <项目根> --id X90 --execute
```

生成的步骤：`git checkout -b knowledge/evolution-local` → `cp` 到知识仓 `discovered/`
→ `knowledge_index.py build` → `check.sh` → `git add/commit/push`。

---

## 6. 验收：怎么证明知识库被加载 / 三仓适配完整

```bash
# 1) 知识加载 smoke（来源 / 计数 / 指纹 / 负向测试）
python3 scripts/knowledge_smoke.py --knowledge-root <root> --json /tmp/smoke.json

# 2) 三仓契约
python3 scripts/check_three_repo_contract.py \
  --skills  <cannbot-skills> \
  --plugin  <cannbot>/plugins-community/ascendc-ops-perf-evolution \
  --knowledge <cannbot-knowledge> \
  --json /tmp/contract.json
```

判读：

- `SMOKE: PASS` 才说明知识仓被正确解析、正文可读、显式错误会 fail-loud；
- `CONTRACT: PASS` 才说明 skill / plugin / knowledge 三方的路径与清单互相匹配；
- 任一 `FAIL` 行都带具体缺失路径，可直接定位到仓/文件。

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `知识仓布局不完整（source=env）` | `CANNBOT_KNOWLEDGE_ROOT` 指错 | 改成完整 checkout，或 unset 后用 `--knowledge-root` |
| `知识仓布局不完整（source=project_env）` | `.cannbot/knowledge.env` 指错 | 重新跑 `init.sh --knowledge-root` 覆盖 |
| `[WARN] 回退到兄弟检出` | 未显式配置 | 生产环境请用 `init.sh` 绑定 |
| `INDEX.json 缺失或不可解析` | 未重建索引 | `python3 scripts/generate_index_json.py --knowledge-root <root>` |
| `agent 路径引用不可解析` | 旧路径漂移 | 按 `check_three_repo_contract.py` 的 FAIL 明细修正 |

---

## 8. 旧布局（迁移前）说明

旧布局 checkout（迁移前的分支）的 skill 仍在
`plugins-community/ops-perf-evolution/skills/evolution-strategies/`，且
`references/{cards,playbooks,discovered}/` 仍为本地副本。该布局下的行为：

| 场景 | 行为 |
|---|---|
| 未显式配置知识仓 | `load_playbook.py` / `generate_index_json.py` **回退本地正文**（禁用 sibling 自动命中，避免误用知识仓） |
| 显式 `--knowledge-root` / env / `.cannbot` | 优先知识仓；显式来源无效 → fail-loud |
| smoke | 用 `--corpus local`（`init.sh` 会自动检测 `references/playbooks` 并加上该参数） |
| 契约检查 | `--legacy-layout --skill-dir <本 skill 目录>` |

具体命令：

```bash
SK=plugins-community/ops-perf-evolution/skills/evolution-strategies

python3 $SK/scripts/knowledge_smoke.py --corpus local --json /tmp/smoke.json

python3 $SK/scripts/check_three_repo_contract.py \
  --skills  <skills-repo> \
  --plugin  <skills-repo>/plugins-community/ops-perf-evolution \
  --knowledge <cannbot-knowledge> \
  --skill-dir <...>/skills/evolution-strategies \
  --legacy-layout
```

迁移到新布局（正文全部进知识仓）后，去掉 `--corpus local` / `--legacy-layout`，
改用默认 `--corpus knowledge` 与 `community/ops/ascendc-evolution-strategies` 路径。


---

## 9. 知识使用 Trace（审计）

`scripts/knowledge_trace.py` 生成 `knowledge_trace.md`，记录：
知识仓绑定（root/mode/commit）、本地库清单（staging/promoted/rejected + manifest）、
transcripts 中的检索证据（load_playbook/卡片读取/本地库命中/策略 ID）、EVO_DIR 策略落地与评估结果，
并判定 `retrieval`（知识是否被真实利用）与 `repo_writes`（local-first 纪律，应为 0）。

```bash
python3 scripts/knowledge_trace.py --evo-dir "$EVO_DIR" --since 2026-10-22T16:30 \
  --projects-filter ascendc-ops-perf-evolution --out "$EVO_DIR/knowledge_trace.md"
```

调用时机：每个 round checkpoint 与会话完成时（见插件 AGENTS.md「知识使用 Trace（强制步骤）」）。
