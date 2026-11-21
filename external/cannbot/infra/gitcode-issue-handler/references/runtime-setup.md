# 运行时：初始化、路径与基线同步

## 读取时机

已配置 batch 的新一轮处理先读 [pipeline.md](pipeline.md)，配置和所需能力就绪后运行 `python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/issue_pipeline.py" resume --new-run --repository-root .` 并按 `next_action` 续跑；不默认加载本文及全部旧 references。初始化、目标或路径需要处理时，再与 [runtime-state.md](runtime-state.md) 一起读取本文相应部分；single 保留原入口；仅配置时读 [configuration-setup.md](configuration-setup.md)，需要推导目标时再读本文相应部分。能力文档仅在对应操作前读取；`policy_query` 不读取执行 reference。能力检查完成前不得执行依赖该能力的 API、Git、临时落盘或提交操作。

## 安装与运行目录

- `gitcode-issue-handler` 与 `gitcode-toolkit` 必须安装到同一 `skills/` 根目录；不要另建 toolkit 副本。按当前 Skill 绝对路径得到 `ISSUE_HANDLER_SKILL_ROOT`，并令 `GITCODE_TOOLKIT_ROOT="$(dirname "$ISSUE_HANDLER_SKILL_ROOT")/gitcode-toolkit"`；后者不存在即报告安装不完整。
- Python、Git、临时目录和 git author 都是按操作触发的能力，不是 Skill 加载条件；仅运行本地配置脚本时需要 Python 3.10+ 与 `yaml`（PyYAML），首次运行依赖 API 的 Python handler 脚本时再按 `requirements.txt` 验证可导入 `requests`。
- 配置模板来自本 Skill 的 `assets/`，运行配置只写目标仓库；不改 Skill 安装目录、仓库级 `AGENTS.md`/`CLAUDE.md`，不写 Token。配置和运行树按需创建，空 `repo` 在目标确认后保存。
- 进入仓库操作后始终以已解析仓库根为命令工作目录。依赖 Git 的操作先完成 Git 检查，写产物前确认父目录可写，再非覆盖地创建 `.cannbot/gitcode-issue-handler/{config,data,reports,logs,cache,images,repro,worktrees,tmp}`。仅在 `.git/info/exclude` 精确查重并追加 `/.cannbot/gitcode-issue-handler/`，不改 `.gitignore`。

## 旧路径兼容

新路径与仓根同名配置并存时优先读取 `.cannbot/gitcode-issue-handler/config/`，不合并、不覆盖。新配置缺失时，分类器和责任人工具可只读回退仓根 `classify_config.yaml`/`operator_owners.yaml`；责任人更新只把完整内容写新路径。旧分类配置中恰为原默认值的 `last_check_file`、`report_file`、`cache_dir` 在内存转换到新目录，其他自定义路径保持权威；旧 `issue_analysis_data/` 与仓根 YAML 不自动移动或删除。

## 目标仓库解析

以下是运行时解析规则；进入初始化问卷时，推导结果仅作为仓库题候选，仍按 [configuration-setup.md](configuration-setup.md) 获取明确回答。

仓库地址来自 Issue URL、`--url` 或 `GITCODE_URL`；Token 来自 `--token` 或 `GITCODE_TOKEN`，只在即将访问 API 时按能力门禁检查。 `repo` 非空直接使用，不因 remote 名称或分支变化改选。

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/resolve_repository.py" --repository-root .
# 显式目标再附加：--target <Issue-URL 或 owner/repo>
```

返回 `resolved` 后使用脚本返回的 `repo` 和 `config_path`；只更新顶层 `repo`，保留其他配置与注释。

缺失/为空时，有显式目标使用显式目标；否则解析 GitCode remote 的 `owner/repo`，归一化 HTTPS/SSH 和 `.git` 后缀并去重。只有一个候选自动保存；多个返回 `needs_selection`，Agent 优先用可用问卷工具（如 `request_user_input_async`），不可用才直接询问，发一次仓库问卷（列出候选并允许自由输入；无候选直接询问 `owner/repo`），收到选择后用 `--select` 重跑。未收到答复不得选定，也不得访问 Issue API。

配置与显式目标不一致同样问卷确认，不静默覆盖。若选择配置仓库而原 Issue URL 属于其他仓库，要求用户改为批量处理或提供对应 Issue，不能移植 IID。`batch` 根为启动 Git 仓库；`single` 从 URL 得到 `owner/repo/iid`。目标与当前 remote 不匹配时，经临时目录检查后 clone，在目标工作目录保存配置；不未经确认修改原仓库配置。解析歧义由脚本以 `needs_selection` 和退出码 2 返回。

## 配置合并与默认值

进入已确定的目标仓库后，按 [configuration-setup.md](configuration-setup.md) 调用 `setup_config.py --repository-root .`。它复用现有初始化逻辑补齐缺失文件，再区分模板、已有用户配置和已完成引导；Marketplace/install-helper 已创建配置时也执行该检查。只在首次需要引导或用户主动要求时询问常用参数，已明确选择保留设置的不重复询问；原缺 Token 等待点优先于配置问卷。旧仓根配置优先迁移且原文件保留。

除上节 `repo` 冲突须确认外，默认值、仓库配置、命令行参数按此顺序覆盖：字典逐项合并，列表和显式空容器整体替换；默认文件缺失用模板，显式指定文件缺失或格式错误报错。责任人映射缺失/模板不阻塞初始化，但识别出算子后须按 `issue-routing.md` 请求责任人或由用户决定 `direct`，禁止静默自修。

默认交付 `pr`，基线/目标分支 `master`。目标 remote 优先匹配 Issue URL 或 canonical `repo`；`origin` 仅在匹配或仅有一个 remote 时使用。

首响和临时指派的两开关及依赖校验见 [automation.md](automation.md)。关闭自动分配不关闭候选调查；单次会话覆盖不修改配置。

## 步骤 -1：分流与基线前置

batch 配置就绪后由 pipeline 复用持久状态、范围缓存和未完成队列，不重建已完成工作；single、配置、知识维护和咨询闭环仍使用各自原路径。新入口不扩展发布权限，closed、责任范围及授权门禁不变。能力检查严格按 [runtime-capability-checks.md](runtime-capability-checks.md) 在首个相关操作紧前调用。未确定认证 API 前不索取 Token。

## 步骤 0：安全获取基线

只有诊断需要源码基线或进入修复时执行，本轮只同步一次；API 答疑可跳过：

```bash
git fetch --all --prune
git branch -r
git rev-parse --verify <canonical-remote>/master^{commit}
```

只更新远程跟踪引用，不切换分支、pull 或修改工作区/本地 `master`。canonical remote 依次取用户指定、匹配 Issue/config 的 URL、唯一 `<remote>/master`、唯一 remote；同名 fork 优先 URL owner 不同且凭据可推送的 remote。多个完全等价候选只在依赖 remote 的 Git 操作前询问一次。记录不可变 `base_ref`/`base_commit`，PR worktree 全从该 commit 创建；direct-push 多组按 `code-worktree.md` 串行刷新目标分支。fetch 或基线缺失有界诊断并重试一次，仍失败报告 blocker；不得 stash、merge、rebase、reset 或强制切换。

`target_remote_branch` 在 `pr` 模式表示 PR base，在 `direct-push` 模式表示确切推送目标。
列出远程分支后确定 `target_remote_branch`，必须用 `<remote>/<branch>` 且存在于列表；不存在停止，不静默使用不匹配的 `origin`。步骤 0 不建功能分支，步骤 2e 后按 `code-worktree.md` 创建。

## 输出

仅实际同步成功后记录 `sync_completed: true`、`base_branch`、`base_ref`、`base_commit`、`delivery_mode`、`target_remote_branch` 与 `remote_branches`；完整字段见 [runtime-state-schema.md](runtime-state-schema.md)。

首次知识检索按 [runtime-knowledge.md](runtime-knowledge.md) 复用受审卡和已有历史快照；知识维护独立显式执行，不作为首响前置步骤。

首次落盘可使用以下幂等命令（先检查父目录可写）；单 Issue 不需要预先建全套目录：
```bash
ISSUE_HANDLER_RUNTIME_ROOT=".cannbot/gitcode-issue-handler"
mkdir -p "$ISSUE_HANDLER_RUNTIME_ROOT"/{config,data,reports,logs,cache,images,repro,tmp,worktrees}
ISSUE_HANDLER_EXCLUDE="$(git rev-parse --git-path info/exclude)"
mkdir -p "$(dirname "$ISSUE_HANDLER_EXCLUDE")"
grep -Fqx '/.cannbot/gitcode-issue-handler/' "$ISSUE_HANDLER_EXCLUDE" 2>/dev/null || printf '/.cannbot/gitcode-issue-handler/\n' >> "$ISSUE_HANDLER_EXCLUDE"
```
Marketplace/install-helper 与 `npx skills` 安装方式见 [安装指南](../docs/installation-guide.md)。
