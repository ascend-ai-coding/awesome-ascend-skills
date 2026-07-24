# Comment 路径详细步骤

从 Step 1.5 判定为 Comment 时进入。**全程只读**，不改任何代码、不创建分支、不 commit、不 push、不开 PR；做完直接发评论。

## Step C-2: 克隆上游主仓（只读）

```bash
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="/tmp/gitcode-issue-handler_${upstream_repo}_${issue_number}_${timestamp}_readonly"
git clone --depth=200 \
    "https://gitcode.com/${upstream_owner}/${upstream_repo}.git" "$WORK_DIR"
cd "$WORK_DIR"
```

`--depth=200` 而不是 1：Issue 经常涉及最近若干提交的来龙去脉（"为什么 PR #xxx 这么改"/"这个限制是什么时候加上的"），过浅会看不到上下文。

后缀 `_readonly` 是给自己看的提示：这个工作目录不应该出现 `git add` / `git commit` / `git push`。

## Step C-3: 代码与 Issue 联合分析

原则与 PR 路径 Step 3 类似，但目标是"答清楚问题"而非"找根因"：

1. **先读代码再下结论**：Issue 提到的文件 / 函数 / 接口，用 Read / Grep 实地确认行号与上下文，不要凭 Issue 描述脑补。
2. **答到对应抽象层**：
   - 用户问"为什么这么实现"→ 给出**实现历史 + 当时取舍**（可结合 `git log` / `git blame` / `git show`），别复述代码"做了什么"。
   - 用户问"是否支持 X" → 先核实"支持"的边界（API 入参范围、配置项、特殊 dtype / shape / soc / 平台），别用一句话粗糙打发。
   - 用户问"如何使用 Y" → 给最小可运行示例 + 必要前置；不要把 README 整段抄过来。
3. **承认未知**：找不到根据时，明确写"现有代码未提供该能力 / 此处行为未在代码或文档中规约，建议向 PR #xxx 作者 / 维护者进一步确认"。
4. **不要顺手提改动建议**：Comment 路径的产物是答复；如果分析过程中发现 Issue 其实需要改代码，回到 Step 1.5 让用户改选 PR，**不要在评论里塞改动方案 / 代码 patch**——那是 PR 才该有的内容。

## Step C-4: 起草答复评论

**评论语言**：跟随 Issue 正文主语言。中文 Issue → 中文评论；英文 Issue → 英文评论；中英混合时按正文主体判断（通常看复现步骤 / 期望行为段落的语言）。

通用结构（不强模板，按 Issue 提问形状裁剪）：

```
【一句话结论】

【分析】
- 关键代码定位：${file}:${line_start}-${line_end}（一两句解释这里做了什么）
- 行为依据：<引用源码片段，5~10 行；或引用文档段落>
- 边界 / 约束：<如有；例如仅支持 INT32/INT64、仅 master 分支、仅 ascend910b>

【参考】
- <related file paths / 相关 PR / 相关 Issue / 文档链接>

【建议 / 后续】（可选）
- <仅当问题确有改进空间时给出；不要主动展开成改动方案，那是 PR 路径的事>
```

引用源码用 GitHub-flavored markdown 围栏标语言，例如 ` ```cpp `、` ```python `；并保留 `path/to/file.cpp:42` 这种行号引用方便维护者跳转。

## Step C-5: 用户确认 + 提交评论

沿用主流可见约定：

1. 先在对话主流以代码块完整打印评论 body：

       即将向 Issue #${issue_number} 提交以下评论：
       ```
       <comment body 全文>
       ```

2. 再发 AskUserQuestion 三选一：「确认提交」/「我来修改」/「取消」；preview 字段留空或只放一行摘要，不要重复 body 全文。

确认后调 GitCode 评论 API：

```bash
curl -X POST "https://api.gitcode.com/api/v5/repos/${upstream_owner}/${upstream_repo}/issues/${issue_number}/comments?access_token=${token}" \
  -H 'Content-Type: application/json' \
  --data-urlencode "body=${comment_body}" \
  --connect-timeout 60 --max-time 180
```

> 部分 GitCode 端点对超长 body 直接 POST 会 400 或截断。若评论较长（>3KB），保险起见走 PATCH 套路：先 POST 占位（如 `"占位，即将更新"`）拿到 `comment_id` → 再 PATCH `/repos/{owner}/{repo}/issues/comments/{comment_id}` 写入完整 body → GET 回查长度非 0 与预期一致。详见 [gitcode-api.md](../../gitcode-toolkit/references/gitcode-api.md)。

API 返回里取 `html_url`（或 `id`，拼成 `${issue_url}#note_{id}`）作为评论链接。
