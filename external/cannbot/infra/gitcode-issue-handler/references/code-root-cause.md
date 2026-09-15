# 代码处理：环境门禁、稳定复现与根因确认

## 读取时机

仅对 `code_change` 分组执行步骤 3–4 前完整读取本文件。修改源代码前必须完成本阶段。
每组命令的工作目录必须是 manifest 记录的 `worktree_path`，不得在原始目标仓库工作区
运行会生成构建产物的复现命令。

## 步骤 3：复现

历史案例只转化成待验证检查项；当前版本的环境、复现和代码证据优先。

### 环境一致性门禁

运行任何构建、测试、样例或复现命令前：

1. 从 `required_environment` 读取目标 CANN/框架版本、源码 revision、SoC、CPU 架构
   和实际工具。编译、运行时、精度和性能 Issue 缺少 CANN 版本时，准备最小信息请求并
   标记 `waiting_context`；把完整正文加入统一执行预览，确认前禁止发布。
2. 采集 `ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME`、`ASCEND_OPP_PATH`、
   `ASCEND_AICPU_PATH`，`asc_opc`/`ccec`/`atc` 的实际路径、CANN version 和
   `git rev-parse HEAD`。
3. 执行确定性检查：

   ```bash
   mkdir -p .cannbot/gitcode-issue-handler/repro/<IID>
   python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/check_repro_env.py" \
     --issue-json .cannbot/gitcode-issue-handler/data/issues.json --issue <IID> \
     > .cannbot/gitcode-issue-handler/repro/<IID>/environment.json
   ```

   没有持久化 Issue JSON 时显式使用 `--expected-cann <版本>`。

按状态分流：

| status | 动作 |
| --- | --- |
| `match` | 继续定位和复现 |
| `mismatch` / `mixed_environment` / `environment_unavailable` | 不运行复现，仅内部报告差异，标记 `waiting_environment` |
| `expected_unknown` | 环境敏感问题索要版本；静态或文档问题注明 `not_applicable` |

源码、SoC、架构和关键依赖不一致同样阻断该 Issue 的复现，但不阻断批次。不得把不同
环境中的失败描述为复现证据，也不得向外部 Issue 披露维护侧环境。

### 最小复现

阅读正文、评论和图片，定位入口、模块和触发条件。按以下顺序尝试，命中即停：

1. 已有单元测试或端到端测试。
2. 最小临时脚本或测试。
3. 必要时启动 dev server 手工复现。

记录命令、预期与实际、关键日志、稳定性、环境指纹和证据路径。临时产物统一放入
`.cannbot/gitcode-issue-handler/repro/<IID>/`。

### 复现分流

| 结果 | 动作 |
| --- | --- |
| 稳定复现 | 进入步骤 4 |
| 偶发 | 相同输入和环境最多重试 3 次；仍不稳定则回评并标记 `intermittent_waiting` |
| 无法复现 | 排除非问题后索要最小上下文，标记 `waiting_context` |
| 环境不匹配/缺失 | 标记 `waiting_environment`，仅内部报告 |
| 属于预期行为 | 解释并跳过代码修改 |

禁止在稳定复现前修改源代码，禁止吞异常或调高阈值来掩盖问题。

## 步骤 4：确认根因与方案

只有稳定复现的 Issue 才进入本步骤。再次用更精确的错误、函数和边界条件检索
本 Skill 的 `scripts/knowledge_query.py`；即使当前目录已进入组 worktree，也必须传
`--repository-root "$ISSUE_HANDLER_REPOSITORY_ROOT"` 读取步骤 0a 的目标仓库快照。受审卡片
优先，`runtime_candidates` 仍只是低置信度候选证据。

根据问题选择：

- 从堆栈定位首个仓内帧。
- 二分路径或添加临时日志。
- 使用 `git blame` 和对照实验确认回归。
- 先核对文档、类型和接口约定，排除理解偏差。
- 用户指出具体标识符或需要追溯引入历史时，完整读取
  `code-git-history.md`。

最终根因必须由当前代码、稳定复现或已验证的 Git 历史支持。确认后更新
`root_cause_hypothesis` 为最终根因。需要回评精简结论时只准备目标和完整正文，待本地
实现/验证完成后纳入统一执行预览；确认前禁止发布。

修改方案至少包含：

```yaml
final_root_cause:
change_locations: []
change_strategy:
compatibility_risks: []
verification_plan: []
```

方案未成型前不得进入步骤 5。

### 统一预览输入

`single` 和 `batch` 都把最终根因及证据、预计修改文件/函数、变更策略、兼容风险和验证
计划写入运行状态，不在此处发起方案确认。下一阶段只允许在本次 manifest 管理的独立
worktree 中形成可丢弃、未暂存的最小修改并验证；`batch/interactive` 同样不得 commit 或
产生远端写入。

实际 changed files、diff 摘要和测试结果形成后，连同精确评论、commit、分支、PR 和首次
CI 一起进入 `delivery-confirmation.md` 的统一预览。这样正常路径只需一次执行确认，不会
因为验证结果和 PR 正文尚未确定而必然再次询问。

### 合并组复核

多 Issue 组中，若最终方案仍修改同一位置且无冲突，保持合并；否则拆成独立组并更新
`group_id`、`members`、`proposed_branch`。已确认的根因和方案继续保留。

把最终 `change_locations` 规范化为仓库相对 `planned_paths`，按
`code-worktree.md` 重新运行冲突计划。新发现路径重叠或独占资源竞争时，相关组不得
同时进入步骤 5；让已获执行许可的组先完成，其他组进入后续波次。该调度变化不是人工
审批点。

完成后进入 `code-validation.md`。
