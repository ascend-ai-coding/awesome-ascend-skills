# ATB OPS→ACLNN 迁移任务示例

> 本文件展示了如何提交一个标准化的 ATB 算子迁移任务。
> 将此文件作为模板，根据实际算子修改对应字段即可。

---

# task
现在你有了完完全全的skills来交付一个atb的需求

我希望你参考所有你可以调用的atb的skills @agent-skills/skills/ascend-transformer-boost/SKILL.md

然后完成：设计、实现、测试一个 atb中 910B+950 下 {算子名称} 从ops接口切换至aclnn接口的一个需求

# 前置学习知识
{根据算子类型填写需要前置学习的知识}

例如对于量化类算子：
学习LLM中的quant是什么，总结成一个md，给我放到./task下面
注意per-token, per-channel, per-tensor 量化的第一和区别

对于归一化类算子（如{xxx}Norm）：
学习LLM中的normalization方法，总结成一个md，给我放到./task下面
注意{xxx}Norm、RMSNorm、BatchNorm的区别和应用场景

对于激活函数类算子（如Swish、Gelu）：
学习LLM中常用的激活函数，总结成一个md，给我放到./task下面
注意各种激活函数的数学表达式和特性差异

# 仓路径
cann: $ASCEND_TOOLKIT_HOME
atb: $ATB_REPO_PATH

# reference
atb接口: {ATB官方文档链接}
aclnn 接口：{ACLNN官方文档链接}

## param
参考：atb中的{OpType}Param
例如：{xxx}QuantParam、{xxx}NormParam等

我需要你完成这个切换

# 流程
验证环境变量
    -> 算子迁移设计，先设计清楚csv，后续开发严格遵守TDD开发规范，所有新增的代码都要被测试到
    -> [human in the loop： 你设计好了文档和csv用例，保存到 @working_files 里面，然后必须先让用户确认文档是否正确无误，再开发]
    ->实际迁移 -> loop 最多3次尝试编译，并且每次记录报错，3次后找用户
    -> 测试设计、泛化
    -> 测试验证
    -> 返回用户

最后必须通过所有的csv用例。

# action requirements
1. **任何你遇到的问题，都需要清晰记录**
2. 没有思路时，需要返回给用户明确结果
3. 先plan，后执行，必须确认所有条件充足后才能写代码，只改需要改的代码，不要改其他的代码；严格遵循奥卡姆剃刀原则
4. 自我反省，自我记录，自我迭代
5. 遇到问题必须记录问题，可以写到 @$WORKING_DIR，打断，来问用户，问问题时使用苏格拉底式提问法
   1. 记录问题要遵循：现象，执行动作，改动上下文均记录清楚，保证明确、可复现
6. 对于任何使用到的skills，如果你发现了任何可以提升的地方，都需要补全；如果，你发现skills缺失，那么告诉用户，并尝试自行补全，并在index中更新
7. 详细设计存在 @$WORKING_DIR 下面
8. 你实现失败后，先告诉我发生了什么报错，有什么上下文信息，我们一起找到问题，再进行下一步
9. 你对skills和atb仓有完全的读写执行权限，其他位置先询问

---

## 填写说明

### 算子名称占位符
- 将 `{算子名称}` 替换为实际的算子名，如 `{xxx}Quant`、`{xxx}Norm`、`{xxx}Attention` 等
- 保持大小写与 ATB 源码中的命名一致

### 前置学习知识
根据算子类型选择对应的学习主题：

| 算子类型 | 建议学习主题 |
|---------|-------------|
| 量化类 (Quant) | per-token/per-channel/per-tensor 量化区别 |
| 归一化类 (Norm) | LayerNorm vs RMSNorm vs BatchNorm |
| 激活类 (Activation) | Swish、Gelu、Relu 等激活函数特性 |
| 矩阵运算类 (MatMul) | 矩阵乘法优化、分块策略 |
| 注意力类 (Attention) | Attention机制、KV Cache、FlashAttention |

### 参考文档链接
- **ATB接口**: 在华为官方文档（查找对应算子）
- **ACLNN接口**: 在华为官方文档（查找对应算子）

### Param结构
参考 ATB 源码中 `{OpType}Param` 结构体的定义，通常在：
```
atb/src/ops/ops_infer/{op_type}/xxx_operation.h
```

### 关键流程约束
- **HIL检查点**: 设计文档和CSV用例完成后必须等待用户确认
- **编译限制**: 最多3次编译尝试，每次记录错误
- **测试要求**: 必须通过所有CSV用例（正例、反例、性能测试）
- **记录要求**: 所有问题记录到 @$WORKING_DIR 目录


---

## 完整示例（SwiGluQuant）

```markdown
# task

我希望你参考所有你可以调用的atb的skills @{SKILLS_REPO_PATH}/skills/ascend-transformer-boost/SKILL.md

然后完成：设计、实现、测试一个 atb中 910B+950 下 swigluquant 从ops接口切换至aclnn接口的一个需求

# 前置学习知识
学习LLM中的quant是什么，总结成一个md，给我放到./task下面
注意per-token, per-channel, per-tensor 量化的第一和区别

# 仓路径
cann: {CANN_PATH}
atb: {ATB_REPO_PATH}

# reference
atb接口: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendtbapi/ascendtb_01_0285.html
aclnn 接口：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/aolapi/context/ops-nn/aclnnSwiGluQuant.md

## param
参考：atb中的SwiGluQuantParam

我需要你完成这个切换

# 流程
验证环境变量
    -> 算子迁移设计，先设计清楚csv，后续开发严格遵守TDD开发规范，所有新增的代码都要被测试到
    -> [human in the loop： 你设计好了文档和csv用例，保存到 @working_files 里面，然后必须先让用户确认文档是否正确无误，再开发]
    ->实际迁移 -> loop 最多3次尝试编译，并且每次记录报错，3次后找用户
    -> 测试设计、泛化
    -> 测试验证
    -> 返回用户

最后必须通过所有的csv用例。

# action requirements
1. **任何你遇到的问题，都需要清晰记录**
2. 没有思路时，需要返回给用户明确结果
3. 先plan，后执行，必须确认所有条件充足后才能写代码，只改需要改的代码，不要改其他的代码；严格遵循奥卡姆剃刀原则
4. 自我反省，自我记录，自我迭代
5. 遇到问题必须记录问题，可以写到 @{your working path}/working_files，打断，来问用户，问问题时使用苏格拉底式提问法
   1. 记录问题要遵循：现象，执行动作，改动上下文均记录清楚，保证明确、可复现
6. 对于任何使用到的skills，如果你发现了任何可以提升的地方，都需要补全；如果，你发现skills缺失，那么告诉用户，并尝试自行补全，并在index中更新
7. 详细设计存在 @{your working path}/working_files 下面
8. 你实现失败后，先告诉我发生了什么报错，有什么上下文信息，我们一起找到问题，再进行下一步
9. 你对skills和atb仓和 @{your working path}/working_files 有完全的读写执行权限，其他位置先询问
```

---

## 如何使用此模板

1. **复制此文件** 到新文件（如 `task_xxx.md`）
2. **替换所有占位符**（花括号 `{}` 中的内容，以及 $ 开头的变量）
3. **保存到工作目录**
4. **提交任务**: 在Claude Code中引用此文件
   ```
   请执行 @task_xxx.md 中的任务
   ```
5. **AI自动匹配Skills**: Claude将自动调用 `atb-ops-to-aclnn-migration-workflow` 技能集

---

## 技能调用时序预览

提交任务后，AI将按以下时序调用Skills：

```
Phase 0: 前置学习
    ↓ 调用知识检索 → 生成学习总结

Phase 1: 设计文档生成
    ↓ 调用 atb-aclnn-operator-replacement-designer
    → 输出: {op}_replacement_design.md
    → [HIL] 等待用户确认 Gate 1

Phase 2: CSV用例设计
    ↓ 调用 atb-csv-testcase-generator
    → 输出: {op}_test.csv
    → [HIL] 等待用户确认 Gate 2

Phase 3: 实际迁移
    ↓ 调用 atb-aclnn-operator-migration
    → 输出: {op}_aclnn_runner.h/cpp + 修改后的 operation.cpp

Phase 4: 编译验证（≤3次）
    ↓ 调用 atb-testframework-build
    ↓ [失败] 调用 atb-debug-guide

Phase 5: 测试验证
    ↓ 调用 atb-csv-tester
    ↓ [失败] 调用 atb-debug-guide

Phase 6: 交付报告
    → 生成 delivery_report.md
    → Git commit
```

---

*模板版本: 1.1.0*
*关联技能: atb-ops-to-aclnn-migration-workflow*
*最后更新: 2026-04-28*
