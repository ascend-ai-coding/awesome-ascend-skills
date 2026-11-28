# Token 配置

GitCode API Token 的获取优先级和使用方式。

---

## 获取优先级

按以下优先级获取 Token：

1. **用户在请求中直接提供** — 用户明确在消息中给出了 token
2. **环境变量 `GITCODE_TOKEN`** — 从系统环境变量读取
3. **Git credential helper** — 按目标仓库上下文读取已保存的 HTTPS 凭据
4. **询问用户** — 以上均不可用时才提示配置

共享脚本使用 `resolve_token(cli_token, repo_url=<目标仓库或 Issue URL>, repository_root=<目标仓目录>)`；未提供 URL 时读取该目录的 `origin`。通过 `git credential fill` 查询 `gitcode.com` 的仓库路径和 URL 中已有的账号，兼容路径带/不带 `.git` 的保存方式，并遵循 Git 已配置的 helper 与账号选择。只做有超时的非交互查询，不直接读取凭据文件、不调用 approve/reject，不修改 Git 配置或凭据存储。

SSH remote 只用于确定仓库，仍需已有 HTTPS Token；SSH 密钥和 Git 提交身份不能转换成 API Token。helper 返回的 password 是待核验凭据，是否适用于 API、是否过期以及具体权限由目标仓库所需的只读接口核实，不以 Git 推送成功或 `/user` 的账号 scope 替代。凭据只在进程内使用，不写项目配置或报告；预检仅输出 `source: git_credential` 等可用性元数据，缺少凭据才进入原有等待流程。

---

## 使用方式

Token 支持以下两种传递方式：

### 方式一：`PRIVATE-TOKEN` Header（推荐）

```bash
curl -H "PRIVATE-TOKEN: {token}" "https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}"
```

### 方式二：`access_token` 查询参数

```bash
curl "https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}?access_token={token}"
```

---

## 注意事项

1. **权限确认**：先用目标仓库必需的轻量只读接口验证；不要用 `/user` 或 `/emails` 的
   403 判定 Token 整体无效，它们可能需要额外账号 scope
2. **安全保护**：不在日志中输出 token 明文
3. **失效处理**：目标仓库必需接口遇到 401/403 时重新进入 Token 预检，收到替换 Token
   后从失败请求恢复；同一会话正常阶段不重复询问
4. **授权边界**：Token 只表示具备接口凭据，不代表用户已授权评论、指派、push、PR
   或 CI 等外部写操作；写操作仍按 [authorization-contract.md](authorization-contract.md) 确认

---

## 当前用户 login（按需）

部分 skill（如 gitcode-issue-gen 的 Issue 自助 assign）需要知道 token 持有者的 GitCode login（即 `gitcode.com/<login>` 中的 `<login>`，也是 `@username` 中的 username）。

**统一通过 token 反查**，不要让用户额外配置环境变量：

```bash
curl -sS "https://api.gitcode.com/api/v5/user?access_token=${GITCODE_TOKEN}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('login',''))"
```

- token 已经唯一标识了认证主体，再让用户单独配置 username 是冗余且容易出错（GitCode login ≠ 昵称、≠ git config user.name、≠ PR 作者）
- `/user` 调用失败（401/403/网络）时：依赖该字段的步骤直接跳过并告知用户原因（多半 token 权限不足或网络异常），**不要**降级追问用户手填——根因解决不了，问也白问
