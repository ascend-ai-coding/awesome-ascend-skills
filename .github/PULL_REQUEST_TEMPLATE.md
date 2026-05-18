## 变更描述

<!-- 请简要描述本次 PR 的变更内容 -->

### 变更类型
- [ ] 新增 Skill
- [ ] 更新现有 Skill
- [ ] 修复问题
- [ ] 其他（请说明）

### 涉及的 Skill
<!-- 如果是新增或修改 Skill，请填写 Skill 名称 -->
Skill 名称：

### 变更内容摘要
<!-- 请用 1-2 句话说明主要变更 -->


---

## 检查清单

提交 PR 前，请确认以下检查项已完成：

### 治理规则检查
- [ ] 已阅读 `docs/governance/skill-governance.md`
- [ ] 已明确本次变更属于哪个功能域（base / inference / training / profiling / ops / agent-tools / ai-for-science）
- [ ] 已明确本次变更是 bundle / domain skill set / leaf / router / external 哪一类
- [ ] 如为本地 skill，目录位于 `skills/<domain>/...`
- [ ] 如与其他 bundle 或 skill 容易混淆，已补充“怎么选”的边界说明

### SKILL.md 格式检查
- [ ] `name` 字段与叶子目录名匹配，或符合 nested / `ai-for-science` 命名规则
- [ ] `description` 字段不少于 20 个字符
- [ ] frontmatter 格式正确（以 `---` 开头和结尾）
- [ ] 包含 `description` 字段用于 Agent 匹配
- [ ] 正文中无 `[TODO]` 或 `[TODO:xxx]` 占位符

### 内容完整性检查
- [ ] 内部链接可正常访问（如 `references/xxx.md`）
- [ ] 代码块已标注语言（如 ```bash、```python）
- [ ] 表格格式正确（如有）

### 仓库更新检查
- [ ] 已添加到 `.claude-plugin/marketplace.json`
- [ ] 已更新 `README.md` 中对应的导航 / 安装入口 / decision tree（如适用）
- [ ] 如涉及开发目录导航，已同步更新 `skills/README.md` 或对应 `skills/<domain>/README.md`
- [ ] 如变更治理规则，已同步更新 `docs/governance/skill-governance.md`

---

## 测试说明

### 本地验证
运行以下命令进行本地验证：

```bash
# 验证所有 Skill 文件
python3 scripts/validate_skills.py

# 检查 frontmatter
find . -name "SKILL.md" -exec head -5 {} \; -print

# 检查 name 字段与目录/命名规则是否一致
find . -name "SKILL.md" | while read f; do
  dir=$(dirname "$f")
  name=$(grep "^name:" "$f" | cut -d: -f2 | tr -d ' ')
  echo "$dir -> name: $name"
done
```

### 本地测试截图

**请在提交 PR 前完成本地测试，并粘贴测试结果截图：**

#### 1. 验证脚本测试结果

<!-- 运行 `python3 scripts/validate_skills.py` 后粘贴截图 -->

```
<!-- 粘贴验证输出 -->
```

#### 2. Skill 功能测试（如适用）

<!-- 如果是新增或修改 Skill，请测试 Skill 的基本功能 -->

- [ ] 已在 AI Agent 中测试 Skill 触发
- [ ] 已验证 Skill 内容正确加载

**测试截图：**

<!-- 粘贴测试截图 -->

### CI 检查
提交 PR 后将自动运行以下检查：
1. **Validate SKILL.md files** - 验证所有 SKILL.md 文件格式
2. **Check frontmatter** - 检查 frontmatter 完整性
3. **Verify skill names** - 验证 name 字段与目录/命名规则匹配
4. **Check for broken internal links** - 检查内部链接是否损坏
5. **Check governance references** - 检查治理文档、README 与 PR 模板的关键引用是否存在

---

## 其他说明

<!-- 如有其他需要说明的内容，请在此填写 -->

### 关联 Issue
<!-- 如有关联的 Issue，请填写 Issue 编号 -->
Fixes #

### 截图（如适用）
<!-- 如需展示效果变更，请添加截图 -->
