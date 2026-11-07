# GitCode PR 行内评论（diff_comment）工作流

PR 行内评论（`diff_comment`）锚定 diff 的具体代码行，区别于落在讨论页的普通评论
（`pr_comment`）；两者共用 `POST /pulls/{number}/comments` 端点，唯一区别是是否携带有效
`position`，参数名错误会静默落成普通评论。本文只定义平台机制：行号定位、提交、回查、
删除。评论正文的业务内容由调用方负责。

> API 端点与错误码见 [gitcode-api.md](gitcode-api.md)，写前确认与回查边界见
> [authorization-contract.md](authorization-contract.md)，Token 获取见
> [token-config.md](token-config.md)。

## 1. 参数契约

```text
POST /repos/{owner}/{repo}/pulls/{number}/comments
{"commit_id": "<head_sha>", "path": "<文件相对路径>", "position": <行号>, "body": "<正文>"}
```

`position` 是 head 版本文件中的绝对行号（1-based），不是 diff 相对行号。只有 `position`
能创建行内评论。错误写法均被服务端静默忽略、落成普通 `pr_comment`，无报错：

```json
{"path": "...", "line": 92, "body": "..."}                                           // ❌ line → pr_comment
{"path": "...", "line": 92, "side": "RIGHT", "body": "..."}                          // ❌ side 无效 → pr_comment
{"path": "...", "start_line": 92, "end_line": 93, "body": "..."}                     // ❌ 行范围 → pr_comment
{"path": "...", "diff_position": {"start_new_line": 92, "end_new_line": 93}, "body": "..."}  // ❌ 对象 → pr_comment
{"commit_id": "<head_sha>", "path": "...", "position": 92, "body": "..."}            // ✅ position → diff_comment
```

## 2. 行号定位

`position` 必须与 head commit 的文件实际行号一致，两种方法互证：

1. diff 换算：hunk 头 `@@ -a,b +c,d @@` 中 `c` 为新侧起始行，context 行与 `+` 行逐行计数。
2. contents API 核对（长 hunk 推荐）：

```bash
curl "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "${path}")?ref=${head_sha}&access_token=${GITCODE_TOKEN}"
# content 字段为 base64，解码后数行即得 position
```

`head_sha` 从 `GET /repos/{owner}/{repo}/pulls/{number}` 的 `head.sha` 获取。

## 3. 提交与回查

```bash
# 1. 防重：拉全量已有评论。列表默认分页 20 条，必须带 per_page
curl "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/${pr}/comments?access_token=${GITCODE_TOKEN}&per_page=100"

# 2. 提交
curl -X POST "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/${pr}/comments?access_token=${GITCODE_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"commit_id":"'"${head_sha}"'","path":"'"${path}"'","position":'"${position}"',"body":"..."}'

# 3. 回查：再次 GET 列表，确认新评论 comment_type=diff_comment
#    且 diff_position.start_new_line 等于目标行号
```

POST 返回 201 时响应体中 `comment_type`/`position` 等字段均为 null，不代表落点正确；
成败以 GET 回查为准。回查显示 `comment_type=pr_comment` 说明参数被降级，按第 1 节修正。

## 4. 删除

```bash
curl -X DELETE "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/comments/${note_id}?access_token=${GITCODE_TOKEN}"
```

`note_id` 用 GET 列表返回的**数字 id**，不是 POST 响应的 hex id。成功返回 HTTP 200，
单条 GET 回查 404。仅实测过删除本账号评论；删除他人评论取决于 Token 权限。

## 5. 安全边界

- 严禁向目标仓库发送测试/调试评论；参数问题在本地构造 JSON 排查。
- 每条评论是独立的写操作：写前按授权契约确认准确的 PR、行号与完整正文，写后逐条回查。
