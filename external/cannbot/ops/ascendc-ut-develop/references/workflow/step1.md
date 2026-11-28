## Step 1 了解当前仓库信息

### Step 1.1 阅读基础知识文档

如果文档 `references/${repo_type}/${repo_type}.md` 存在，立即阅读。

### Step 1.2 提取支持的 SoC 列表

```bash
# 优先使用 --list-socs（custom 工程）
bash ${repo_path}/build.sh --list-socs 2>/dev/null | grep '^ *- ' | sed 's/^ *- //'
# 若不支持 --list-socs，从变量提取（标准仓）
grep -oE 'SUPPORT_COMPUTE_UNIT[A-Z_]*=\([^)]+\)' ${repo_path}/build.sh | grep -o '"[^"]*"' | tr -d '"'
```

soc 和 arch 对应关系
| soc | arch |
|-----|------|
| ascend310p | arch20 |
| ascend910b | arch32 |
| ascend910_93 | arch32 |
| ascend950 | arch35 |

### Step 1.3 学习编译命令

```bash
bash build.sh -h
```

**明确以下场景的编译命令：**
- [ ] 单独编译运行 `${soc_type}` 模块的用例
- [ ] 仅编译 `${soc_type}` 模块，不运行用例（用于快速验证）
- [ ] 获取覆盖率

**⚠️ 重要：build.sh 参数使用规范**

| 参数 | 用途 | 注意事项 |
|-----|------|---------|
| 无 `--noexec` | 编译 + 运行一步完成 | **推荐日常使用**，能发现 CSV 格式等运行时错误 |
| `--noexec` | 仅编译，不运行 | 仅用于快速验证语法错误，**编译通过后必须再运行测试** |

**常见错误**：加了 `--noexec` 后忘记运行测试，导致 CSV 格式错误未被发现。

