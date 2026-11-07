# pre-commit 使用与钩子失败处置（agent 操作规程）

仓级完整说明（安装/排查/FAQ）见仓内 `docs/pre-commit_guide.md`；本文是 agent 在 GitCode 协作流程（clone→改→commit→push）中的操作要点。

## 使用前提（新克隆环境必做）

每次新克隆的仓**没有安装钩子**——`git commit` 不会触发任何检查。开工前先装：

```bash
pip3 install pre-commit && pre-commit install   # 仓库根目录执行；取消：pre-commit uninstall
git config --global alias.pc '!f() { pre-commit run --files $(git diff --name-only "$@"); }; f'  # 每环境一次
```

日常手动检查：`pre-commit run`（暂存区）/ `pre-commit run --files <清单>` / `git pc HEAD~x`（最近 x 笔改动）。

## 铁律

1. **禁止 `--no-verify` 绕过**，除非用户明确要求。用户说"先不要触发/先跳过"时，通常是推迟而非永久禁止——拿不准就问。
2. **只处理本次触碰的文件**；全仓存量债不动（发现大批量存量问题时记录并建议专项，不顺手修）。
3. 格式化修复与内容改动**分两笔提交**（见下「两笔提交结构」）。

## 失败分类决策表

| 钩子形态 | 类别 | 处置 |
|---------|------|------|
| clang-format / ruff-format（"files were modified"） | 自动重排 | 直接 `git add` 重排结果 → 重跑到收敛 → 单独成笔 |
| ruff-check（`--fix` 已修） | lint 自动修 | 复核 diff 是否合理（如 f-string 降级、import 拆分）→ 留下修复 |
| ruff-check（不可自动修，如 F821 未定义名） | lint 真错 | 修代码（这类多为真 bug），非格式化 |
| codespell | 拼写 | 真 typo 就改；领域术语/标识符误报 → 加入配置 `-L` 白名单（先入白名单再提交，不绕过）。命中清单在 `pre-commit_reports/codespell.log` |
| check-json / check-yaml | 语法 | 修语法；JSONC 文件（tsconfig/.claude/settings.json）应已在豁免清单，命中说明豁免失效需修配置 |
| check-added-large-files | 大文件 | 确认合法性（eval 数据集走 jsonl 豁免）；非法大文件移出提交 |
| detect-private-key | 密钥 | 真密钥立即移出并告知用户轮换；误报再议豁免 |
| oat-check（版权头/License） | 合规 | 按仓内既有文件样式补 License 版权头；marker 跳过问题见 oat_check.sh 的 done-marker 语义 |

## 两笔提交结构

```bash
# 第 1 笔：纯内容改动（--no-verify 跳过钩子，保证 diff 干净、可校验）
git commit --no-verify -m "<内容改动>"

# 第 2 笔：对上一笔的变更文件补跑钩子，自动修复单独成笔
git pc HEAD~1        # 仓约定别名：pre-commit run --files $(git diff --name-only <commit>)
                     # 无该别名时手写等效命令；多轮跑到收敛
```

**收敛判定以 `git status` 为准**——钩子输出措辞会变/漏报，不要 grep 输出关键词判收敛；
收敛后再用显式 `pre-commit run --files <清单>` 复核一遍。
若 `git pc` 后工作区零改动（收敛轮为 0），则无需第 2 笔。

## 提交前自检清单

1. `git show --stat HEAD` 复查提交里只有预期文件（防 `git add -A` 带入临时文件）
2. 临时文件一律写 `/tmp`，不落仓根目录
3. 推送后 `git ls-remote origin refs/heads/<分支>` 核对远端指针（push 输出有误导性）

## 本仓钩子特异性（skills-dev）

12 项：通用 7（trailing-whitespace 保留 md 硬换行 / end-of-file / check-yaml 允许多文档 /
大文件豁免 *.jsonl / merge-conflict / detect-private-key / check-json 豁免 JSONC）+
clang-format（Google 风，豁免 kb/ 提取快照）+ ruff-check/format（豁免仅 E501,RUF001-003,E402，
F 系列全放——未定义名/未用 import 这类是真 bug，必修）+ codespell（只查 md 与 yaml，
领域词白名单维护在配置 args 里）+ OAT（增量扫描，分叉分支自动 PR-range 全量）。
配置以仓内 `.pre-commit-config.yaml` 为准；本文不重复参数细节，只给处置规程。
