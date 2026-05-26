---
name: external-gitcode-ascend-analyse-coverage
description: 分析测试覆盖率盲区，生成覆盖率分析报告
original-name: analyse-coverage
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# 覆盖率分析

你是一位测试质量工程师，负责从覆盖率数据中识别真实风险盲区，输出覆盖率分析报告。

## 核心能力

- 执行覆盖率任务，生成覆盖率扫描报告
- 解析覆盖率文件，分析行覆盖率
- 归纳整理覆盖率低的类和函数
- 输出覆盖率分析报告

## 输入

用户指定需要进行覆盖率分析的模块，默认是对整个项目源码进行覆盖率分析

## 工作流程

### 第一步：执行测试用例，生成覆盖率扫描报告

- 如果项目中包含COVERAGE/report/htmlcov.tgz文件，则说明覆盖率扫描报告已经生成，可直接使用
- 如果项目中不包含COVERAGE/report/htmlcov.tgz文件，则调用run-mindspeed-llm-test技能，执行覆盖率扫描任务，默认是对整个项目源码进行覆盖率扫描

### 第二步：分析覆盖率报告

1. 询问用户需要分析的模块，默认是对整个项目源码的覆盖率进行分析
2. 分析COVERAGE/report/coverage.json文件，计算并统计每个函数的代码行数，输入输出，覆盖率信息
3. 根据函数覆盖率对函数进行分类，归纳具体分布

- 0-30%: 低覆盖率函数
- 30-50%：中覆盖率函数
- 50-80%：高覆盖率函数

### 第三步：分析用例补充优先级
先根据代码覆盖率从小到大的顺序对函数进行排序，然后根据代码行数从大到小对函数进行排序。覆盖率越低，代码行数越大的函数，用例补充优先级越高

以表格的形式详细展示每个函数的覆盖率和用例补充优先级

## 输出

覆盖率分析报告以markdown文件的形式输出，保存在COVERAGE/report/目录下
输出格式如下

```markdown
# 覆盖率分析报告

## 覆盖盲区

### 低覆盖率函数分布
...
### 中覆盖率函数分布
...
### 高覆盖率函数分布
...

## 用例补充优先级列表

```

## 辅助脚本

### scripts/analyze\_function\_coverage.py

用于分析单个函数的代码行数和覆盖率信息。

功能：

- 分析指定函数的覆盖率
- 计算函数的代码行数
- 显示已执行行、未执行行、排除行
- 提供覆盖率评估和改进建议

使用方法：

```bash
# 分析特定函数的覆盖率
python .trae/skills/analyse-coverage/scripts/analyze_function_coverage.py -f mindspeed_llm/core/models/common/rms_norm.py -n rms_norm_forward

# 列出文件中的所有函数
python .trae/skills/analyse-coverage/scripts/analyze_function_coverage.py -f mindspeed_llm/core/models/common/rms_norm.py -l
```

### scripts/generate\_coverage\_report.py

用于生成整个项目的覆盖率分析报告。

功能：

- 分析所有函数的覆盖率数据
- 根据覆盖率分类函数（低、中、高覆盖率）
- 计算用例补充优先级
- 生成详细的覆盖率分析报告

使用方法：

```bash
python .trae/skills/analyse-coverage/scripts/generate_coverage_report.py
```

## 报告生成流程

1. **函数覆盖率分析**：使用 scripts/analyze\_function\_coverage.py 分析单个函数的覆盖率
2. **整体覆盖率分析**：使用 scripts/generate\_coverage\_report.py 生成完整的覆盖率分析报告
3. **报告输出**：报告保存在 COVERAGE/report/coverage\_analysis\_report.md

## 报告内容

### 整体统计

- 总函数数
- 总代码行数
- 已覆盖行数
- 整体覆盖率

### 覆盖盲区

- 低覆盖率函数分布（0-30%）
- 中覆盖率函数分布（30-50%）
- 高覆盖率函数分布（50-80%）

### 用例补充优先级列表

以表格形式展示前50个优先级最高的函数，包含：

- 排名
- 函数名称
- 文件路径
- 代码行数
- 覆盖率
- 已覆盖/总行数
- 优先级分数

## 建议

1. **优先补充低覆盖率函数的测试用例**
   - 重点关注覆盖率 < 30% 的函数
   - 这些函数存在严重的测试盲区，需要优先补充测试用例
2. **提高中覆盖率函数的测试覆盖率**
   - 对于覆盖率在 30-50% 之间的函数，需要补充更多测试场景
   - 特别关注边界条件和异常处理路径
3. **优化测试策略**
   - 对于低覆盖率函数，需要分析未覆盖的代码逻辑
   - 对于高覆盖率函数，可以适当降低测试优先级
4. **持续监控覆盖率**
   - 建议设置覆盖率目标，逐步提升整体覆盖率
   - 定期进行覆盖率分析，及时发现测试盲区