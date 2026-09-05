# 代码处理：Git 历史精确追溯

## 读取时机

仅当用户指出具体代码、标识符、函数或选项，或根因需要确认引入提交时读取。

## Pickaxe 流程

1. 粗筛字符串出现次数变化的提交：

   ```bash
   git log --all --oneline -S "<标识符>"
   git log --all --reverse --format="%h %an %ae %ad %s" \
     --date=short -S "<标识符>"
   ```

2. 逐个验证真实 diff：

   ```bash
   git show <hash> --stat
   git show <hash> -- <path>
   git show <hash> -- <path> | grep "^[+-].*<标识符>"
   ```

3. 检查 `new file mode`、父提交和 commit body，识别 merge、squash、孤儿提交和
   整体重写造成的假新增：

   ```bash
   git log -1 --format="parents: %P" <hash>
   git log -1 --format="%b" <hash>
   git ls-tree <parent> <path>
   ```

4. 从最早的真实引入点向后串联演进提交，并提取关联 PR/Issue。

`-S` 只表示字符串出现次数变化，不等于真实引入。必须用实际 diff 和父提交树验证，
不得把 merge/squash 重带文件当作最早引入。

## 输出

记录候选提交、排除理由、真实引入提交、后续演进和关联 PR/Issue。无法确定时明确写
`unknown`，不要推测。
