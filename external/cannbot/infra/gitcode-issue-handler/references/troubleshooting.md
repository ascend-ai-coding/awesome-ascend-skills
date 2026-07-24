# 异常处理 / 反模式 / 使用示例

## 异常处理速查

| 阶段 | 现象 | 处理 |
|------|------|------|
| Step 1 | Issue 已 `closed` | AskUserQuestion 确认是否仍要处理（Comment 路径仍可发评论） |
| Step 1 | Issue body 几乎为空 | AskUserQuestion 让用户补充复现/期望/问题描述 |
| Step 1.5 | 用户消息同时含"修复"和"只评论" | 取"只评论"为准——"只"是明确缩小意图 |
| Step 1.5 | 内容判定为 PR 但执行 Step 3 时发现并非代码问题 | 回 Step 1.5 重选 Comment，**不要硬把"无需改动"塞进 PR**|
| Step 1.5 | 内容判定为 Comment 但 Step C-3 发现需要改代码 | 回 Step 1.5 重选 PR，**不要在评论里塞代码 patch** |
| Step 2 | fork_url 不是该 Issue 仓库的 fork（也不是它的 renamed fork）| 报错要求更正；renamed fork 的判定看 `parent_full_name` |
| Step 2 | clone 卡住 | 检查 gitcode.com 网络可达性，按 env-check 重试 |
| Step 3 | 复现不出 Issue 现象 | 不要硬猜根因，回到用户问环境 |
| Step 4 | 测试失败但用户认为是预期的 | AskUserQuestion 让用户明确是否调整测试基线，不擅自改测试 |
| Step 5 | push 被拒（同名分支已存在） | 让用户选 rebase 强推 / 换分支名 / 取消 |
| Step 6 | "Another open merge request already exists" | 取已有 PR 链接给用户，问是覆盖更新还是终止 |
| Step C-5 | 评论 POST 返回 200 但 GET 回查 body 为空 / 被截断 | 改走 PATCH 套路写入；不要看到 200 就以为成功 |
| Step C-5 | 用户在确认弹窗里要求"加点改动建议" | 礼貌拒绝，提示这是 Comment 路径，要改动请回 Step 1.5 切到 PR |

## 反模式（不要这么干）

**通用**：

- ❌ 不看 Issue 内容就默认要改代码——很多 Issue 只是问问题
- ❌ 跳过 Step 1.5 模式判定 / 跳过 Step 3 根因分析直接动手——很容易跑偏
- ❌ 把多行 commit message / PR body / 评论 body 塞进 AskUserQuestion 的 `preview` 字段——preview 仅在聚焦时出现在侧栏，用户多数时候根本看不到；要审查的内容必须先在对话主流以代码块形式打印出来
- ❌ 不经用户确认就 push / 建 PR / 发评论

**PR 路径专属**：

- ❌ 在调用点 `try/except: pass` 把异常吞了当"修好"
- ❌ 顺手 reformat 整个文件 / 重命名变量——破坏最小修改
- ❌ 改测试断言让测试过——这是骗自己
- ❌ commit message 里加 `Co-Authored-By`（违反项目约定）
- ❌ 用中文写代码注释或 PR 描述正文
- ❌ 自创 PR 模板章节名（如把 `## 描述` 改成 `## Description`、把 `## 关联的Issue` 改成 `## Related Issue`）——这会让仓库维护者第一眼觉得是不熟悉规范的提交；必须沿用 `git show upstream/${base_branch}:.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` 的原章节，仅替换 `<!-- ... -->` 占位

**Comment 路径专属**：

- ❌ Comment 路径下还去 fork 代码 / 创建分支 / 跑测试套件——这都是 PR 路径的事
- ❌ 在评论里塞代码改动方案 / patch / "建议把第 N 行改成 M"——答疑就是答疑，改代码请回 Step 1.5 切 PR
- ❌ 用英文回复一条全中文 Issue（反之亦然）——评论的读者是提问者本人，可读性优先
- ❌ 不引用具体 `file:line` / 不贴关键源码片段，光打嘴炮泛泛而谈——维护者无法据此快速验证你的结论
- ❌ 评论里写"不知道"就交差——找不到根据时应明确写"现有代码 / 文档未规约，建议向 xxx 进一步确认"，给提问者一个下一步指引

## 使用示例

### 示例 1：PR 路径（典型 bug-report）

```
gitcode-issue-handler \
  --issue_url https://gitcode.com/cann/ops-math/issues/123 \
  --fork_url  https://gitcode.com/your-name/ops-math.git
```

期望交互节奏：

```
[Step 0]   环境预检 ✅（git author 待 1.5 判定后再决定是否查）
[Step 1]   解析 Issue #123: "tiling crash on empty shape"，labels=bug-report
[Step 1.5] 模式判定 → 推荐 PR（bug-report + 有复现栈）→ 用户确认
[Step 1.6] fork_url 已提供，跳过
[Step 2]   克隆 fork → /tmp/gitcode-issue-handler_ops-math_123_20260511_103045
           切出 fix/issue-123-tiling-empty-shape
[Step 3]   根因初判 + 修复策略 → 等待用户确认
[Step 4]   改 2 个文件 → pytest tests/test_tiling.py 通过
[Step 5]   commit message 预览 → 确认 → push 确认
[Step 6]   PR 标题/描述预览 → 确认 → 创建
[Step 7]   ✅ PR: https://gitcode.com/cann/ops-math/merge_requests/1600
```

### 示例 2：Comment 路径（典型答疑）

用户消息：

```
帮我看看这个 issue 想问什么、给个答复
https://gitcode.com/cann/ops-math/issues/456
```

期望交互节奏：

```
[Step 0]   环境预检 ✅（git author 未查，因为可能不需要 commit）
[Step 1]   解析 Issue #456: "请问 PadV2 在 ascend910b 上是否支持 INT8?"
           labels=question
[Step 1.5] 模式判定 → 推荐 Comment（question 标签 + "请问 / 是否支持"）→ 用户确认
[Step C-2] 克隆 cann/ops-math → /tmp/gitcode-issue-handler_ops-math_456_..._readonly
[Step C-3] 读 conversion/pad_v2/op_host/pad_v2_def.cpp 与
           conversion/pad_v2/op_host/config/ascend910b/pad_v2_binary.json
           → 确认当前 dtype 白名单：FLOAT16 / FLOAT / BF16，不含 INT8
[Step C-4] 起草中文答复（含 file:line 引用）
[Step C-5] 主流打印评论 body → 用户确认 → POST → GET 回查 ✅
[Step C-6] ✅ 评论已发布：https://gitcode.com/cann/ops-math/issues/456#note_xxxx
```

### 示例 3：用户显式要求只回复

用户消息：

```
issue_url=https://gitcode.com/cann/ops-math/issues/789
先别改代码，只在 issue 下评论回复一下我的疑问就好
```

→ Step 1.5 检到"只评论 / 不改代码"关键词，直接锁定 Comment 路径，跳过内容分析（但仍会在主流摘要 Issue 内容供用户复核）。

### 示例 4：选错路径中途切换

Step 1.5 推荐 PR、用户确认，但 Step 3 根因分析后发现 Issue 描述的"问题"其实只是误用：

```
我跑去 Step 3 读了代码，发现 Issue 报告的"参数顺序错误"实际上和文档一致，
是提问者用错了。是否切换到 Comment 路径，回复一条澄清评论？
[切到 Comment] [继续 PR（强行写澄清 commit / 文档改动）] [取消]
```

切到 Comment 后，**抛弃 PR 路径已切出的工作分支**（不 push、不 commit），重新进入 Step C-2 用一个新的只读目录起步。
