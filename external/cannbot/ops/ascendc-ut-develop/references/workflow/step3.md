## Step 3 重构 UT 框架

**用户没有明确提到“重构”时，跳过本步骤**

### Step 3.1 备份旧框架代码

将当前框架下 `ut/${test_model}` 目录下的 `.cpp`、`.h` 文件备份到系统临时目录：

```bash
mkdir -p /tmp/cannbot_${op_name}/backup
cp ${op_path}/tests/ut/${test_model}/*.cpp ${op_path}/tests/ut/${test_model}/*.h /tmp/cannbot_${op_name}/backup/
```

备份完成后，删除原有的测试文件。

### Step 3.2 搭建新框架，确保编译通过

> **本步骤使用 Task 工具调用子 Agent 进行调试，避免污染上下文**
> **不要读算子的定义和模板文件，分析算子的工作也交给子 Agent**

根据用户的需求，搭建新的 UT 框架，确认可编译通过

**算子的 UT 代码，需要放置在算子的目录下，而不是代码仓根目录下的 test 目录**

**注意，仅迁移一条预期执行成功的旧用例**，用于验证框架是否正常，**迁移所有用例是后续步骤的任务**

如果 `${repo_type}` 为 **"ops-transformer"** 且用户提及 **"CSV"**、**"CSV整改"**、**"CSV格式"**、**"CSV化"**，执行 **CSV 格式重构**:
1. 让子 Agent 参考 [ops-transformer.md](../ops-transformer/ops-transformer.md) 和 [csv-refactor-workflow.md](../ops-transformer/csv-refactor-workflow.md)
2. 让子 Agent 使用 `assets/csv-refactor/` 下的模板文件
3. 让子 Agent 查看详细规范 [csv-format-spec.md](../ops-transformer/csv-format-spec.md)

> **⚠️ 子 Agent 完成任务后，必须检查（编译 + 运行）：**
> - [ ] UT 可以编译通过
> - [ ] 所有用例运行显示 PASS
> 如果有误，要求子 Agent 修正

### Step 3.3 迁移所有旧用例

> **本步骤使用 Task 工具调用子 Agent 进行调试，分析算子的工作也交给子 Agent，避免污染上下文**

将旧框架中**所有**用例迁移至新框架，并确保用例**全部通过**

- **新增用例时，不得更改已迁移用例和公共代码，例如 CSV 表头**
- **每 5 条一组，确保编译运行通过后再迁移下一组**
- **所有旧用例都要完整迁移，迁移中出现了框架不适配的情况时，及时向用户寻求策略**

> **⚠️ 子 Agent 完成任务后，必须检查（编译 + 运行）：**
> - [ ] UT 可以编译通过
> - [ ] 运行测试验证所有用例 PASS
> - [ ] 新的用例集数量和重构前一致
> 如果有误，要求子 Agent 修正
