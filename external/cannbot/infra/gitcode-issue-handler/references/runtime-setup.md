# 运行时：初始化、路径与基线同步

## 目录

1. [读取时机](#读取时机)
2. [前置条件](#前置条件)
3. [旧路径兼容](#旧路径兼容)
4. [Skill 路径](#skill-路径)
5. [目标仓库解析](#目标仓库解析)
6. [默认配置](#默认配置)
7. [步骤 -1：请求分流与运行能力检查](#步骤--1请求分流与运行能力检查)
8. [步骤 0：安全获取基线](#步骤-0安全获取基线)
9. [发布目标](#发布目标)
10. [输出](#输出)

## 读取时机

仅真实执行请求在步骤 -1 与 `runtime-state.md`、`runtime-capability-checks.md` 一起完整
读取本文件；`policy_query` 不得读取这些执行 reference。完成对应能力检查前不要执行
依赖该能力的 API、Git、临时落盘或提交操作。

## 前置条件

- `gitcode-issue-handler` 与 `gitcode-toolkit` 已通过 CANNBot Marketplace、
  `install-helper` 或 `npx skills` 安装到同一 `skills/` 根目录；不要另建 toolkit 副本。
- 进入仓库操作后，始终把已解析的目标仓库根目录作为命令工作目录。
- Python、Git、临时目录和 git author 都是 handler 按操作触发的能力，不是 Skill
  加载条件。只按 `runtime-capability-checks.md` 在首次相关操作紧前检查。首次运行 Python handler 脚本时，
  还要按 `requirements.txt` 验证可导入 `requests` 和 `yaml`（PyYAML）。
- 仓库地址来自 Issue URL、`--url` 或 `GITCODE_URL`。Token 来自 `--token`
  或 `GITCODE_TOKEN`，但只在即将访问 GitCode API 时检查；Token 规则以
  `gitcode-toolkit` Skill 的 `references/token-config.md` 为准。
- `.cannbot/gitcode-issue-handler/config/operator_owners.yaml` 是运行期责任人映射；
  缺失、为空或仍是模板时不阻塞运行初始化，
  但步骤 2c 识别出明确算子后必须按 `issue-routing.md` 请求责任人或 `direct`
  决定，禁止静默回退到 Agent 自行处理。
- 配置模板位于本 Skill 的 `assets/`；运行配置属于目标仓库，不写回 Skill 安装目录。
  单 Issue 不需要配置文件；批量模式缺少配置时可使用当前仓库 remote / `--repo`，运行
  目录由流程按需创建。

只有真实执行请求才准备运行树。解析目标仓库后，先完成 Git 能力检查；写入运行产物前
再验证目标父目录可写，然后非覆盖地准备统一运行树。`policy_query` 不执行以下命令：

```bash
ISSUE_HANDLER_RUNTIME_ROOT=".cannbot/gitcode-issue-handler"
mkdir -p "$ISSUE_HANDLER_RUNTIME_ROOT"/{config,data,reports,logs,cache,images,repro,tmp,worktrees}
issue_handler_exclude="$(git rev-parse --git-path info/exclude)"
mkdir -p "$(dirname "$issue_handler_exclude")"
if ! grep -Fqx '/.cannbot/gitcode-issue-handler/' "$issue_handler_exclude" 2>/dev/null; then
  printf '/.cannbot/gitcode-issue-handler/\n' >> "$issue_handler_exclude"
fi
```

排除规则只在当前仓库的 `.git/info/exclude` 中精确查重并追加，不创建或覆盖工作树里的
`.gitignore`。统一目录树为：

```text
.cannbot/gitcode-issue-handler/
├── config/      # classify_config.yaml、operator_owners.yaml
├── data/        # issues.json、groups.json、last_check.json
├── reports/     # 分类报告、历史处理报告和 latest.md
├── logs/        # 需要保留的运行日志
├── cache/       # Issue/PR 和知识构建缓存
├── images/      # Issue 图片
├── repro/       # 复现证据
├── worktrees/   # 受管 worktree manifest
└── tmp/         # 可删除临时文件
```

这只准备本地配置、缓存和报告目录，不安装 Skill、不修改仓库级 `AGENTS.md` /
`CLAUDE.md`，也不写入 Token。

## 旧路径兼容

- 新路径和仓根同名配置同时存在时，始终优先读取
  `.cannbot/gitcode-issue-handler/config/`，不合并、不覆盖任何一份。
- 新配置缺失时，分类器和责任人工具可只读回退到仓根的
  `classify_config.yaml` / `operator_owners.yaml`。责任人工具更新映射时会把完整内容写入
  新路径，不改写旧文件。
- 旧分类配置中恰好等于原默认值的 `last_check_file` / `report_file` /
  `cache_dir` 会在内存中转换到新目录；其他用户自定义路径仍保持权威。
- 旧 `issue_analysis_data/` 和仓根 YAML 不会自动移动或删除。完成新目录验证后，
  再由用户手工归档；所有新默认产物只写入统一目录。

## Skill 路径

根据当前已加载的 `SKILL.md` 绝对路径确定根目录，不要假定 Skill 被复制到目标仓库根：

```bash
ISSUE_HANDLER_SKILL_ROOT="<gitcode-issue-handler 的绝对目录>"
GITCODE_TOOLKIT_ROOT="$(dirname "$ISSUE_HANDLER_SKILL_ROOT")/gitcode-toolkit"
```

通过 `$ISSUE_HANDLER_SKILL_ROOT/scripts/` 调用主 Skill 脚本和内嵌知识检索；
`$GITCODE_TOOLKIT_ROOT` 不存在时立即报告安装不完整，不猜测其他路径。

## 目标仓库解析

- `batch`：启动目录必须是 Git 仓库；仓库标识从匹配的 remote 推导。
- `single`：先从 Issue URL 解析 `owner/repo/iid`。当前目录的任一 remote 匹配
  `owner/repo` 时可作为目标仓库；否则先执行临时目录能力检查，再在选定根中 clone canonical
  仓库。仅回评模式可使用该只读 clone；需要修改时再配置可推送 fork remote。
- 单 Issue 不需要 `.cannbot/gitcode-issue-handler/config/classify_config.yaml`；
  获取器把 `repository` 写入输入元数据。
  批量模式可从 `assets/classify_config.yaml.example` 复制配置到目标仓库，
  或使用 `classify_issues.py --repo owner/repo`。

## 默认配置

- 交付模式：`pr`
- 本地基线分支：`master`
- 目标分支名：`master`
- 目标 remote：优先匹配 Issue URL 或有效 `classify_config.yaml.repo` 指向的 canonical 仓库；
  `origin` 只在它确实指向 canonical 仓库或仓库只有一个 remote 时作为默认值。
- 仅当用户在当前任务中明确指定其他值时覆盖上述值。

## 步骤 -1：请求分流与运行能力检查

1. 先仅根据用户文本区分 `policy_query` 和真实执行。`policy_query` 在主 Skill
   直接回答并结束；不建运行树、不读取执行阶段 reference、不运行任何环境检查。
2. 真实执行只初始化状态和解析 Issue URL/当前仓库；解析纯文本 URL 不要求
   Token、Git、临时目录或 git author。
3. 每项能力检查按 `runtime-capability-checks.md`，只在它保护的第一个真实操作紧前执行；
   统一调用本 Skill 的脚本，不把业务路由参数传给 toolkit：

```bash
bash "$ISSUE_HANDLER_SKILL_ROOT/scripts/preflight.sh" --checks api
bash "$ISSUE_HANDLER_SKILL_ROOT/scripts/preflight.sh" --checks git --work-dir "$PWD"
bash "$ISSUE_HANDLER_SKILL_ROOT/scripts/preflight.sh" --checks tmp --work-dir "$PWD"
bash "$ISSUE_HANDLER_SKILL_ROOT/scripts/preflight.sh" \
  --checks author --work-dir "<group-worktree>"
```

Token 已由用户保存在当前会话而未放入环境变量时，`api` 调用追加
`--token-available`。临时目录的 handler 私有 fallback、git author 的 local → global
顺序、`needs_user` / `blockers` 路由和失败恢复都以 `runtime-capability-checks.md` 为准。
能力就绪不构成 GitCode 写入或交付授权。

## 步骤 0：安全获取基线

本次运行只执行一次：

```bash
git fetch --all --prune
git branch -r
git rev-parse --verify <canonical-remote>/master^{commit}
```

按以下规则执行：

1. 只更新远程跟踪引用，不切换分支、不执行 pull，不修改用户当前工作区和本地
   `master`。
2. 按以下顺序确定 canonical remote，不发起选择问题：用户显式指定；remote URL 精确
   匹配 Issue URL / `classify_config.yaml.repo` 的 owner/repo；唯一的 `<remote>/master`；
   只有一个 remote 时使用它。fork remote 则优先选择 URL owner 与 canonical owner
   不同、repo 名相同且当前凭据可推送的 remote，通常为 `origin`。
3. 若仍有多个完全等价候选，只在即将执行依赖 remote 的 Git 操作前请求一次选择；
   不得在启动时预问，也不要拖到步骤 8 才询问。
4. 记录基线为不可变的 `base_ref` 和 `base_commit`。PR 模式下本次运行的全部 worktree
   从该 commit 创建，避免并行组读取不同基线。direct-push 多组模式按
   `code-worktree.md` 串行刷新目标分支。
5. fetch 失败或基线引用不存在时先做有界诊断并重试一次；仍失败则报告明确 blocker。
6. 不得使用 stash、merge、rebase、reset、强制切换或其他方式规避失败。

## 发布目标

列出远程分支后确定 `delivery_mode` 和 `target_remote_branch`：

- `pr`：目标值是 PR 的目标远程分支。
- `direct-push`：目标值是确切推送目标。

推导值或用户指定值必须以 `<remote>/<branch>` 表示，并且能在远程分支列表中找到；
否则停止并报告证据。不要使用仓库名不匹配的 `origin` 作为静默兜底。

步骤 0 不创建功能分支。步骤 2e 分组完成后按
`code-worktree.md` 从 `base_commit` 创建组分支和 worktree。随后按
`runtime-knowledge.md` 完成步骤 0a，再进入步骤 1。

## 输出

更新运行状态：

```yaml
capability_checks:
  api: not_started | ready | blocked
  git: ready
  tmp: not_started | ready | blocked
  author: not_started | ready | blocked
sync_completed: true
base_branch: master
base_ref: <remote>/master
base_commit: <full-sha>
delivery_mode: pr | direct-push
target_remote_branch: <remote>/<branch>
remote_branches: []
```

完成后进入步骤 0a。
