## Step 4 补全用例，提升覆盖率

**如果用户没有明确说明跳过本步骤，不得跳过**

> **本步骤使用 Task 工具调用子 Agent 进行调试，避免污染上下文**

### Step 4.1: 获取当前覆盖率

```bash
# 标准仓
cd ${repo_path} && bash build.sh -u --ops=${op_name} --soc=${soc_type} --cov
# custom 工程
cd ${repo_path} && bash build.sh -u --soc=${soc_type} --cov
```

### Step 4.2: 阅读算子文档

```bash
# 查找并阅读算子目录下的 md 文档
find ${op_path} -name "*.md" -exec cat {} \;
```

### Step 4.3: 判断覆盖率类型

编译后需要先判断是**全局覆盖率**还是**单算子覆盖率**：

```bash
# 查看覆盖率报告包含的文件路径
lcov --list ops.info_filtered | head -50
```

| 覆盖率类型 | 文件路径特征 | 处理方式 |
|-----------|-------------|---------|
| 单算子覆盖率 | 仅包含当前算子路径 | 直接使用 |
| 全局覆盖率 | 包含多个算子路径 | 需要使用 `lcov --extract` 提取 |

详细提取方法见 [coverage-extraction.md](../coverage-guide/coverage-extraction.md)

### Step 4.4: 分析未覆盖代码

根据 `${interactive_mode}` 选择不同的分析方式：

#### 4.4a 交互模式

当 `${interactive_mode} == "interactive"` 时：

> **⚠️ 强制交互步骤 — 必须向用户询问，禁止跳过**
>
> 在交互模式下，**必须**通过 `question` 工具与用户确认目标文件，**不得**自动决定处理范围或直接生成报告。
>
> - 本步骤是交互模式的核心特征，跳过此步骤违背了用户选择交互模式的意图
> - 即使用户dismiss问卷，也必须记录用户的选择（包括"无选择"），不得自行假设处理范围

1. **获取文件覆盖率清单**
   ```bash
   # 提取每个文件的覆盖率，按覆盖率升序排列（未覆盖优先）
   lcov --list ops.info_filtered | grep -E "^\s*.+\.(cpp|h)$" | awk '{print $1, $NF}' | sort -t: -k2 -n
   ```

2. **存储文件覆盖率清单**
   ```bash
   # 存储到临时文件
   lcov --list ops.info_filtered | grep -E "^\s*.+\.(cpp|h)$" > /tmp/cannbot_${op_name}/coverage_files.txt
   ```

3. **⚠️ 询问用户选择目标文件（强制步骤，不可跳过）**
   
   **立即**使用 `question` 工具询问用户选择要提升覆盖率的文件：
   - 问卷配置参考 `assets/coverage_files_question.json`
   - 将不满足覆盖率标准的文件作为选项列出
   - 每个选项显示文件名和覆盖率
   - 用户选择的文件存储到 `/tmp/cannbot_${op_name}/target_files.txt`
   
   **如果用户 dismiss 问卷或选择"不处理"**：
   - 明确记录用户的选择状态（如"用户未选择目标文件，结束覆盖率提升流程"）
   - **不得**自行假设处理全部文件或继续后续步骤
   - 可询问用户是否只需生成报告或结束流程

4. **获取选中文件的未覆盖代码**
   ```bash
   # 针对每个目标文件提取未覆盖行
   for file in $(cat /tmp/cannbot_${op_name}/target_files.txt); do
     lcov --list ops.info_filtered | grep "${file}" | grep ":0"
   done
   ```

5. **分析分支条件** — 识别进入该分支需要的参数组合
6. **设计测试用例** — 参考 [test-implementation.md](../coverage-guide/test-implementation.md)

#### 4.4b 自动模式

当 `${interactive_mode} == "auto"` 时：

1. **获取未覆盖代码清单**
   ```bash
   lcov --list ops.info_filtered | grep ":0"
   ```
   
2. **分析分支条件** — 识别进入该分支需要的参数组合
3. **设计测试用例** — 参考 [test-implementation.md](../coverage-guide/test-implementation.md)

### Step 4.5: 迭代补充用例

**用例来源**：
- 已有算子文档中的示例
- 重构前的旧用例
- 算子其它组件的用例
- 未覆盖代码分析结果

**迭代策略**：

| 缺口类型 | 补充策略 |
|---------|---------|
| 异常分支 | 添加 ACLNN_ERR_*/GRAPH_FAILED 用例 |
| dtype分支 | 根据测试模块参考对应 guide 文档 |
| 边界条件 | 添加空tensor、大shape用例 |
| 格式分支 | 添加各 format 的测试用例 |

**dtype 分支补充策略（按测试模块）**：

| 测试模块 | 补充方法 | 参考 |
|---------|---------|------|
| opapi | 使用脚本提取 def 文件中的 dtype 组合 | 参考 [op-api-ut-guide.md](../ut-guide/op-api-ut-guide.md) 中的 "Dtype 排列组合校验" 章节 |
| ophost | 分析 tiling 实现中的 dtype 分支逻辑 | [op-host-ut-guide.md](../ut-guide/op-host-ut-guide.md) |
| opkernel | 分析 kernel 实现中的 dtype 分支逻辑 | [op-kernel-ut-guide.md](../ut-guide/op-kernel-ut-guide.md) |

**完成检查**：
- [ ] 所有新增测例通过
- [ ] 行覆盖率 >= 80%
- [ ] 函数覆盖率 >= 80%
