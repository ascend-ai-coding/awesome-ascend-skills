# Issue Handler 聚合授权契约

## 读取时机

初始化运行状态时读取本文件；生成统一执行预览、复用批准证据或执行自动闭环前再次核对。
GitCode 写入的通用安全边界仍以同级 `gitcode-toolkit` 的
[通用授权边界](../../gitcode-toolkit/references/authorization-contract.md) 为准，本文只定义
Issue Handler 的业务授权模型。

## 授权模式

| 模式 | 适用范围 | 默认行为 |
| --- | --- | --- |
| `interactive` | 单 Issue，以及尚未取得分析后统一批准的批处理 | 只分析和生成预览；写前必须有覆盖当前 operation 的明确确认 |
| `approved_batch` | 用户明确批准过精确 Issue 清单和完整操作预览的批量处理 | 仅执行批准清单内的 operation，不逐项重复确认 |

`single` 和 `batch` 都从 `interactive` 开始。初始“处理 Issue”“自动处理”或配置文件声明
不能产生 `approved_batch`。只有用户基于实际 Issue、最终 diff、验证结果和完整操作预览明确
批准，`batch` 才能切换为 `approved_batch`；`single` 保持 `interactive`，复用该次确认的
operation 证据。

模式未知、批准证据缺失、目标或操作超出作用域时，一律停止尚未执行写入并回到统一预览，
不得静默扩大批准范围。`approved_batch` 不适用于单 Issue。

## 必需上下文

```yaml
authorization:
  mode: interactive | approved_batch
  source: default | explicit_user_approval
  scope:
    repository: owner/repo
    issue_iids: []
    operation_ids: []
    delivery_mode: pr | direct-push | none
  approved_at:
  evidence:
    checkpoint:
    preview_digest:
```

- `approved_batch` 必须记录当前会话中的明确批准、仓库、精确 Issue 清单、operation IDs、
  交付模式、批准时间和完整预览摘要。
- Issue 集合、changed files、评论正文、状态目标、commit、分支、PR 内容或交付模式变化时，
  未执行 operation 的原批准失效。
- Token/环境能力检查、算子责任人方案选择和运行平台工具审批都不是业务写入授权。
- 用户要求“不评论”或缩小范围时，从清单删除对应 operation，并同步缩小批准作用域。

## 聚合检查点

分析后统一执行检查点可以覆盖预览中明确列出的多项 operation：

- Issue 评论：精确 Issue 和完整正文；
- 指派：目标 Issue、目标 login 和用于指派的具体操作；
- Issue 状态迁移：当前/目标状态名称，以及是否 reopen 核心 state；
- 交付：精确 changed files、commit message、功能分支 push、PR 标题与完整正文、首次 CI；
- 批次：上述每项 operation 所属的 Issue 或修复组及其执行顺序。

评论获批不隐含指派或状态迁移获批；它们可以放在同一预览中，但必须是独立 operation。
子流程只执行检查点覆盖的准确 operation，已有合法证据时不重复询问。正文、目标或依赖关系
实质变化时更新预览摘要并重新确认尚未执行的部分。

## 直接推送

direct push 不属于统一执行确认或 `approved_batch`。commit 形成后必须独立展示 remote
名称与 URL、目标分支、commit SHA、共享/保护分支提示和非快进检查结果，再取得确认。
没有确认时保留本地 commit，记录 `delivery_waiting_confirmation`，不得自动改走 PR。

## 自动闭环

`auto-close-stale` 是独立维护路径，不继承 `approved_batch`。交互运行时逐 Issue 展示完整
固定评论和关闭操作，确认后才传 `--apply`；部署模式只接受单独记录、精确约束仓库和策略的
部署授权。无 `--apply` 时必须保持 dry-run。

## 写后证据

- 评论、指派和 Issue 状态变更后执行 GET 回查；
- push 后执行 `git ls-remote` 回查；
- PR 创建后回查源/目标分支和 opened 状态；
- 回查失败时不得标记 operation 完成，结果未知的非幂等写入先查后重试。

每项外部写入在运行状态中记录 operation ID、授权证据、执行结果和回查证据。
