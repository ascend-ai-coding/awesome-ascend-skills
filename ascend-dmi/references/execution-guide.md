# Ascend DMI 执行指南

## 执行方式

- **本地**：直接执行 `ascend-dmi [options]`
- **远程**：使用 `npu-remote-executor` skill 建立 SSH 连接后执行

## 处理交互式提示

部分命令会出现防呆提示（如 `Continue? [Y/N]`）。处理方式：

1. 如出现提示，停止并告知用户提示内容
2. 获取用户确认后，使用 `-q` 参数重新执行以跳过确认

需要 `-q` 的常见命令：`--bw`、`-f`、`-p`、`--dg`、`-r`

## JSON 输出

使用 `-fmt json` 获取结构化输出：
```bash
ascend-dmi --dg --se healthCheck -fmt json
```

## 故障排查

| 问题 | 检查命令 |
|------|---------|
| 命令未找到 | `source /usr/local/Ascend/toolbox/set_env.sh` |
| 权限不足 | `groups` / `ls -la /dev/davinci*` |
| 设备被占用 | `fuser -v /dev/davinci*` 或 `lsof /dev/davinci*` |
