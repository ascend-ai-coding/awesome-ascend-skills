# 依赖自检

检视依赖下列 skill。**全部已装则立即进入下一步，不执行任何联网操作。**

## 依赖清单

安装名 = 目录名，`ops/`、`infra/` 仅源路径前缀：

| 安装名 | 源路径 |
|--------|--------|
| ascendc-code-review | ops/ascendc-code-review |
| ascendc-docs-search | ops/ascendc-docs-search |
| ascendc-api-best-practices | ops/ascendc-api-best-practices |
| npu-arch | ops/npu-arch |
| cann-error-code-standard | ops/cann-error-code-standard |
| gitcode-toolkit | infra/gitcode-toolkit |

## 执行

目标根 `{skills_root}` = `{skill_base}` 的父目录。

1. **检查**：`{skills_root}/{安装名}/SKILL.md` 逐一存在？
   - 全部存在 → 直接进入下一步
   - `{skills_root}` 名不是 `skills`（开发态）：兄弟 skill 已随仓库就位，同样直接进入下一步
2. **补齐缺失**（仅在缺失时）：
   - `TMP=$(mktemp -d)`，`git clone --depth 1 --filter=blob:none --sparse https://gitcode.com/cann/cannbot-skills.git $TMP/repo`
   - clone 失败先重试，最多 3 次；每次失败清掉 `$TMP/repo` 后重来
   - `git -C $TMP/repo sparse-checkout set <缺失的源路径...>`
   - 逐个装到临时同级名再原子替换：`cp -r $TMP/repo/<源路径> {skills_root}/.{安装名}.tmp && rm -rf {skills_root}/{安装名} && mv {skills_root}/.{安装名}.tmp {skills_root}/{安装名}`
   - `rm -rf $TMP`
3. **失败即中止**：clone 重试 3 次仍失败（判定网络不通）、取源或拷贝任一步失败，清掉残留临时目录，中止检视并向用户报告 git 错误。不跳过、不简化检视。

## 约束

- 主 Agent 直接执行，不派发子 Agent
- 无交互、不向用户申请权限
- 已装即走，不做版本比对、不重复安装
