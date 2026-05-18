# Shell 安全审查详细参考

> 本文件包含 Shell/Bash 安全审查的完整代码示例和审查要点。由 SKILL.md 按需引用。

---

## 1. 脚本头部安全设置

```bash
# ✅ 每个脚本必须以此开头
#!/bin/bash
set -euo pipefail
# -e: 命令失败时立即退出
# -u: 引用未定义变量时报错
# -o pipefail: 管道中任一命令失败则整体失败
```

## 2. 变量引用（最常见的 Shell 安全问题）

```bash
# ❌ 不安全：未引用变量
rm -rf $dir/$file           # 如果 $dir 为空 → rm -rf /
cp $file /backup/           # 文件名含空格时出错
if [ $var = "yes" ]; then   # $var 为空时语法错误

# ✅ 安全：始终用双引号包裹变量
rm -rf "${dir:?}/${file:?}"   # :? 确保变量非空
cp "$file" /backup/
if [ "$var" = "yes" ]; then
```

## 3. 命令注入

```bash
# ❌ 不安全：eval 执行用户输入
eval "$user_input"
eval "echo $user_data"

# ❌ 不安全：反引号中的未过滤输入
result=$(echo $user_input | grep pattern)

# ✅ 安全：避免 eval，使用参数化
result=$(grep -F -- "$user_input" "$file")
# -- 防止参数被解释为选项
# -F 固定字符串匹配（避免正则注入）
```

## 4. 临时文件安全

```bash
# ❌ 不安全：可预测的临时文件
echo "$data" > /tmp/myapp.tmp  # 竞态条件 + 符号链接攻击

# ✅ 安全：使用 mktemp
tmpfile=$(mktemp /tmp/myapp.XXXXXX)
trap 'rm -f "$tmpfile"' EXIT  # 确保退出时清理
echo "$data" > "$tmpfile"

# ✅ 安全：临时目录
tmpdir=$(mktemp -d /tmp/myapp.XXXXXX)
trap 'rm -rf "$tmpdir"' EXIT
```

## 5. 权限与文件操作

```bash
# ❌ 不安全：过于宽松的权限
chmod 777 "$file"        # 任何人都可读写执行
chmod 666 "$config"      # 任何人都可读写

# ✅ 安全：最小权限原则
chmod 750 "$script"      # 所有者可执行，组可读
chmod 640 "$config"      # 所有者可读写，组可读
chmod 600 "$secret_key"  # 仅所有者可读写

# ❌ 不安全：不检查文件是否为符号链接
if [ -f "$file" ]; then
    cat "$file"  # 可能是符号链接指向敏感文件
fi

# ✅ 安全：检查符号链接
if [ -f "$file" ] && [ ! -L "$file" ]; then
    cat "$file"
fi
```

## 6. PATH 安全

```bash
# ❌ 不安全：依赖相对路径
curl http://example.com  # 如果 PATH 被劫持，可能执行恶意 curl

# ✅ 安全：使用绝对路径（关键命令）
/usr/bin/curl http://example.com

# ✅ 安全：脚本开头设置安全 PATH
export PATH="/usr/local/bin:/usr/bin:/bin"
```

## 7. 信号处理与清理

```bash
# ✅ 安全：使用 trap 确保清理
cleanup() {
    rm -f "$tmpfile"
    # 其他清理操作
}
trap cleanup EXIT INT TERM
```

## 8. 输入验证

```bash
# ❌ 不安全：未验证输入直接使用
read -r filename
cat "$filename"  # 路径遍历

# ✅ 安全：验证输入
read -r filename
# 只允许字母数字和下划线
if [[ ! "$filename" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo "Invalid filename" >&2
    exit 1
fi
# 确保在预期目录内
filepath="/data/uploads/$filename"
realpath=$(realpath "$filepath")
if [[ "$realpath" != /data/uploads/* ]]; then
    echo "Path traversal detected" >&2
    exit 1
fi
```

---

## Shell 安全工具

| 工具 | 用途 | 命令 |
|------|------|------|
| **ShellCheck** | Shell 脚本静态分析 | `shellcheck script.sh` |
| **shfmt** | Shell 格式化 | `shfmt -d script.sh` |
