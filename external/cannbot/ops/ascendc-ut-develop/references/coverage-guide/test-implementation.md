# 测试用例实现指南

## 任务描述

为未覆盖的代码分支设计并实现具体的测试用例。

---

## 输入参数

| 参数 | 说明 |
|------|------|
| uncovered_lines | 未覆盖代码的文件和行号列表 |
| coverage_report | 当前覆盖率报告 |
| operator_info | 算子信息（名称、类型、数据类型等） |
| existing_tests | 已有测试用例列表 |
| soc_version | SOC版本列表 |
| output_dir | 输出目录 |

---

## 执行步骤

### 1. 分析未覆盖代码

对于每个未覆盖的代码区域：

1. **读取代码实现**：找到对应的源文件，理解代码逻辑
2. **分析分支条件**：识别进入该分支需要的参数组合
3. **确定进入条件**：
   - 数据类型要求（FP16/FP32/INT8等）
   - 张量形状要求（M, N, K维度）
   - 特殊参数要求（quantScale, antiQuantScale等）

### 2. 设计测试用例

根据 AscendC UT 规范设计测试用例。

**测试文件位置**：`{repo}/{category}/{operator_name}/tests/ut/op_host/{test_file}.cpp`

**测试用例模板**：

```cpp
// 测试用例命名规范：Test{OperatorName}{Scenario}
TEST_F({OperatorName}Test, TestScenario_{description}) {
    // 1. 准备输入参数
    int param1 = value1;
    int param2 = value2;

    // 2. 创建输入tensor
    auto x1 = CreateTensor({M, K}, dtype1);
    auto x2 = CreateTensor({K, N}, dtype2);

    // 3. 设置特殊参数（如有）
    // quantScale = xxx;

    // 4. 执行算子
    auto output = ExecuteOperator(x1, x2, param1, param2);

    // 5. 验证结果
    EXPECT_EQ(output.shape(), expected_shape);
}
```

### 3. 常见进入条件及参数设置

| 条件类型 | 示例 | 参数设置 |
|---------|------|---------|
| 数据类型分支 | if (dtype == INT8) | dtype=INT8 |
| 形状条件 | if (M >= 16 && K >= 64) | M=16, K=64 |
| 特殊参数 | if (hasQuantScale) | 提供quantScale参数 |
| 场景检测 | if (IsCapable(...)) | 满足IsCapable条件的参数 |

### 4. 验证覆盖效果

执行编译测试命令验证：

```bash
cd {PROJECT_DIR} && bash build.sh -u --ophost --ops='{operator_name}' --soc='{soc_version}' --cov 2>&1 | tee {output_dir}/log/round_N.log
```

**验证步骤**：
1. 确认编译成功
2. 确认所有测试通过
3. 对比覆盖率变化
4. 更新覆盖率报告

---

## 处理无法覆盖的代码

如果某些代码确实无法覆盖，分析原因：

| 类型 | 说明 | 示例 |
|------|------|------|
| 死代码 | 条件永远为真/假 | useMmOutputAsX1Input硬编码为true |
| 不参与UT | fallback实现 | fallback.cpp不在ophost测试范围 |
| 条件不可能满足 | 参数组合冲突 | 需要同时满足A和B，但A和B互斥 |

---

## 输出结果

输出以下信息供主智能体评估：

```markdown
# 测试用例实现报告

## 新增测试用例
- TestXxxScenario_1: 覆盖 xxx.cpp:123-145
- TestXxxScenario_2: 覆盖 xxx.cpp:200-210

## 覆盖率变化
- 调用前覆盖率：XX.XX%
- 调用后覆盖率：XX.XX%
- 提升：+X.XX%

## 测试结果
- 新增测例数：X
- 通过测例数：X
- 失败测例数：X（如有，说明原因）

## 无法覆盖代码（如有）
| 文件:行号 | 类型 | 原因 |
|-----------|------|------|
| xxx.cpp:789 | 死代码 | ... |
```

---

## 注意事项

- 测试用例必须符合项目规范
- 避免与现有测试用例重复
- 确保测试用例可维护性
- 保存所有修改到指定输出目录