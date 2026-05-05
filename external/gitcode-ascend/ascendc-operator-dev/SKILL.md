---
name: external-gitcode-ascend-ascendc-operator-dev
description: AscendC算子端到端开发编排器。当用户需要开发新算子、实现自定义算子、或完成从需求到测试的完整流程时使用。关键词：算子开发、operator
  development、端到端、完整流程、工作流编排、新建算子。
original-name: ascendc-operator-dev
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
license: UNKNOWN
---

# AscendC 算子端到端开发编排

**Skill类型**：流程导向型（七阶段工作流，子技能串行编排）

本 skill 编排七个子 skill，驱动 ascend-kernel 算子从零到生产可用。

## 核心原则

1. **七阶段串行**：工程初始化 → 设计文档 → 用例生成 → 代码生成&测试 → 接口文档 → 精度评估 → 性能评测，严格顺序执行
2. **子技能执行**：每个阶段 **MUST** 调用对应子 skill，不得自行实现
3. **阶段门控**：前一阶段检查点全部通过后才进入下一阶段
4. **设计驱动编码**：代码生成依赖设计文档中的 Tiling 策略和 UB 分配表
5. **自动化设计**：无需用户预先提供设计文档，设计阶段自动生成
6. **用例统一生成**：设计完成后立即生成测试用例文档，供后续精度评估和性能评测复用
7. **文档闭环**：编译测试通过后 **MUST** 生成 PyTorch 风格的中文接口文档，并在聊天界面展示
8. **精度闭环**：算子必须通过 ≥30 例全面精度评估才算完成
9. **性能闭环**：算子必须通过 msprof 性能对比评测，输出性能报告
10. **结果可视化**：Phase 4/5/6/7 的结果 **MUST** 以 Markdown 形式直接展示在聊天界面中，不要仅输出路径

## 可用子 Skill 清单

| Skill | 路径 | 职责 |
|-------|------|------|
| `ascendc-operator-project-init` | `ascendc-operator-project-init/SKILL.md` | 检测/创建 ascend-kernel 项目，生成算子骨架目录 |
| `ascendc-operator-design` | `ascendc-operator-design/SKILL.md` | 分析算子需求，生成设计文档（含 Tiling 策略、UB 分配表） |
| `ascendc-operator-testcase-gen` | `ascendc-operator-testcase-gen/SKILL.md` | 根据设计文档生成统一测试用例文档，供精度评估和性能评测复用 |
| `ascendc-operator-code-gen` | `ascendc-operator-code-gen/SKILL.md` | 根据设计文档生成 op_host/op_kernel 代码、框架适配、编译测试 |
| `ascendc-operator-compile-debug` | `ascendc-operator-compile-debug/SKILL.md` | 编译、安装 whl、生成测试文件、运行精度测试（由 code-gen 内部调用） |
| `ascendc-operator-doc-gen` | `ascendc-operator-doc-gen/SKILL.md` | 从源码提取接口信息，生成 PyTorch 风格中文 API 文档（必选阶段） |
| `ascendc-operator-precision-eval` | `ascendc-operator-precision-eval/SKILL.md` | 生成 ≥30 例精度测试、运行并输出精度验证报告（必选阶段） |
| `ascendc-operator-performance-eval` | `ascendc-operator-performance-eval/SKILL.md` | 使用 msprof 对比工程算子与原生算子性能，输出性能评测报告（必选阶段） |

## 工作流总览

```
Phase 1        Phase 2        Phase 3        Phase 4                      Phase 5        Phase 6         Phase 7
工程初始化  ──▶ 设计文档  ──▶ 用例生成  ──▶ 代码生成+框架适配+编译测试  ──▶ 接口文档  ──▶ 精度评估报告  ──▶ 性能评测报告
project-init   design         testcase-gen   code-gen → compile-debug      doc-gen        precision-eval  performance-eval

输入: 算子名称 + 功能描述                              输出: 生产可用算子 + 用例文档 + 接口文档 + 精度报告 + 性能报告
```

## 反模式清单（NEVER DO THESE）

- ❌ 不要跳过设计阶段直接写代码
- ❌ 不要跳过用例生成阶段，Phase 2 通过后必须执行 Phase 3（testcase-gen）
- ❌ 不要自行实现任何算子代码，必须调用子 skill
- ❌ 不要在代码生成之前修改框架文件（ops.h / register.cpp / CMakeLists.txt）
- ❌ 不要手动执行编译和测试，统一由 compile-debug skill 处理
- ❌ 不要引用不存在的 skill
- ❌ 不要跳过检查点验证
- ❌ 不要跳过接口文档阶段，Phase 4 通过后必须执行 Phase 5
- ❌ 不要跳过精度评估阶段，Phase 5 通过后必须执行 Phase 6
- ❌ 不要跳过性能评测阶段，Phase 6 通过后必须执行 Phase 7
- ❌ 不要使用非 msprof 的计时方式作为性能结论
- ❌ 精度评估和性能评测不要自行设计用例，必须先读取 testcase-gen 生成的用例文档

---

## Phase 0：需求收集

**目标**：确认算子开发所需的最小信息集，包括开发环境和算子需求

### Step 0.1：环境确认（MUST 在任何开发动作之前完成）

开发环境是所有后续阶段的前置依赖，**必须首先确认**。

#### CANN 环境

**自动检测流程**：

1. 检查环境变量 `ASCEND_HOME_PATH` 是否已设置（`echo $ASCEND_HOME_PATH`）
2. **若已设置**：直接使用，无需询问用户，将其作为 `CANN_PATH`
3. **若未设置**：**MUST** 向用户询问 CANN 安装路径（如 `/usr/local/Ascend/ascend-toolkit`）

**激活方式**：

```bash
source ${CANN_PATH}/*/set_env.sh
```

> 在每个需要编译或运行算子的 Shell 会话中，都必须先执行此激活命令。

#### Conda 环境

**自动检测流程**：

1. 检查当前是否已激活 conda 环境（`echo $CONDA_DEFAULT_ENV`）
2. **若已激活**（值非 `base` 且非空）：直接使用当前环境，无需询问用户
3. **若未激活或为 `base`**：**MUST** 向用户询问要使用的 conda 环境名称

**激活方式**：

```bash
conda activate <env_name>
```

> 在每个需要编译或运行算子的 Shell 会话中，都必须先激活 conda 环境。

#### 环境确认检查点

- [ ] CANN 路径已确定（自动检测或用户提供）
- [ ] `source ${CANN_PATH}/*/set_env.sh` 可正常执行
- [ ] Conda 环境已确定（自动检测或用户提供）
- [ ] `conda activate <env_name>` 可正常执行

### Step 0.2：算子需求收集

### 必须确认的信息

| 信息 | 格式要求 | 必填 | 说明 |
|------|----------|------|------|
| CANN 环境路径 | 绝对路径 | 是 | 自动检测 `$ASCEND_HOME_PATH`，未设置则询问用户 |
| Conda 环境名称 | 字符串 | 是 | 自动检测 `$CONDA_DEFAULT_ENV`，未激活则询问用户 |
| 算子名称 | snake_case | 是 | 如 `acosh`, `rms_norm`, `flash_attn` |
| 功能描述 | 文本/数学公式 | 是 | 如 "反双曲余弦 acosh(x) = ln(x + sqrt(x²-1))" |

**可选信息**（有默认值）：

| 信息 | 默认值 | 说明 |
|------|--------|------|
| 支持的数据类型 | float16, float32 | 可扩展 bfloat16 |
| SoC平台 | ascend910b | 通过平台 API 自动获取 |

### 决策树

| 用户请求 | 处理方式 |
|----------|---------|
| "生成 X 算子" / "开发 X 算子" | 先完成环境确认（Step 0.1），再从算子名推断功能，确认后直接执行全流程 |
| "帮我开发新算子"（无具体名称） | 先完成环境确认（Step 0.1），再询问算子名称和功能描述 |
| "继续算子开发" | 先完成环境确认（Step 0.1），再检查已有文件判断阶段，从中断处继续 |

### 验收标准

- [ ] CANN 环境路径已确定且可激活
- [ ] Conda 环境名称已确定且可激活
- [ ] 算子名称已确认（snake_case 格式）
- [ ] 功能描述已明确（含数学公式或计算逻辑）

---

## Phase 1：工程初始化

**调用 Skill**：`ascendc-operator-project-init`

### 执行内容

```
MANDATORY: 按 ascendc-operator-project-init skill 流程执行：
1. 检测 ascend-kernel 项目是否存在
2. 不存在则从模板复制
3. 在 csrc/ops/<op_name>/ 下创建算子骨架
4. 提示三处注册更新点
```

### 检查点

- [ ] ascend-kernel 项目存在（build.sh、CMakeLists.txt、csrc/）
- [ ] `csrc/ops/<op_name>/` 目录已创建
- [ ] 包含 `op_host/<op_name>.cpp`、`op_kernel/<op_name>.cpp`、`CMakeLists.txt`、`design.md`

**全部通过 → 进入 Phase 2**

---

## Phase 2：设计文档生成

**调用 Skill**：`ascendc-operator-design`

### 执行内容

```
MANDATORY: 按 ascendc-operator-design skill 流程执行：
1. 分析算子需求（名称、功能、数据类型）
2. 确定实现路径（AscendC Kernel / CATLASS / ACLNN）
3. 设计 Tiling 策略（Block级 + UB级）
4. 填写 UB 分配表，推导 bufferCoefficient
5. 生成完整设计文档到 csrc/ops/<op_name>/design.md
```

### 检查点

- [ ] `csrc/ops/<op_name>/design.md` 内容完整
- [ ] 包含函数签名和支持的数据类型
- [ ] 包含计算逻辑伪代码（AscendC API 调用序列）
- [ ] 包含 UB 分配表（列出所有 buffer 及总系数）
- [ ] 包含 bufferCoefficient（每种 dtype 的值）

**全部通过 → 进入 Phase 3**

---

## Phase 3：测试用例生成

**调用 Skill**：`ascendc-operator-testcase-gen`

### 执行内容

```
MANDATORY: 按 ascendc-operator-testcase-gen skill 流程执行：
1. 读取 csrc/ops/<op_name>/design.md，提取参数约束、支持的 dtype、典型 shape
2. 生成 TEST_SHAPES（常规 shape）、GENERAL_SHAPES（泛化 shape）、BOUNDARY_VALUES（边界值）
3. 生成算子标杆（CPU 参考实现、NPU 调用方式）
4. 输出用例文档到 csrc/ops/<op_name>/test/<op_name>-test-cases.md
```

### 检查点

- [ ] `csrc/ops/<op_name>/test/<op_name>-test-cases.md` 已生成
- [ ] 包含 SUPPORTED_DTYPES、TEST_SHAPES、GENERAL_SHAPES、BOUNDARY_VALUES
- [ ] 包含算子标杆（NPU 调用方式 + CPU 参考实现）
- [ ] shape 和参数值均在 design.md 约束范围内

**全部通过 → 进入 Phase 4**

---

## Phase 4：代码生成 + 框架适配 + 编译测试

**调用 Skill**：`ascendc-operator-code-gen`（内部自动调用 `ascendc-operator-compile-debug`）

### 执行内容

```
MANDATORY: 按 ascendc-operator-code-gen skill 流程执行：

阶段 1: 加载参考文档
  - 读取 references/GUIDE.md
  - 按算子类型加载对应 reference

阶段 2: 读取设计文档
  - 提取函数签名、UB 分配表、计算伪代码

阶段 3: 选择模板并生成代码
  - 选择 elementwise / row 模板
  - 生成 op_host/<op_name>.cpp（含 Tiling 计算逻辑）
  - 生成 op_kernel/<op_name>.cpp（含 Compute 计算逻辑）

阶段 4: 框架适配
  - 更新 csrc/ops.h（函数声明）
  - 更新 csrc/register.cpp（m.def + m.impl）
  - 更新 csrc/CMakeLists.txt（OP_SRCS + ascendc_library）

阶段 5: 编译安装与测试（调用 compile-debug skill）
  - ./build.sh 编译
  - pip install whl 安装
  - 生成 tests/test_<op_name>.py
  - 运行功能测试和精度测试
  - 编译/测试失败最多排错 3 次
```

### 检查点

- [ ] `op_host/<op_name>.cpp` 使用平台 API 获取硬件参数
- [ ] `op_kernel/<op_name>.cpp` 包含完整 CopyIn → Compute → CopyOut 流水线
- [ ] `ops.h` 已添加函数声明
- [ ] `register.cpp` 已添加 `m.def` 和 `m.impl`
- [ ] `csrc/CMakeLists.txt` 已添加 host 和 kernel 源文件
- [ ] 编译成功（whl 包已生成）
- [ ] 功能测试通过（exit code 0）
- [ ] 精度测试全部通过（pytest 全绿）

**全部通过 → 进入 Phase 5**

---

## Phase 5：接口文档生成

**调用 Skill**：`ascendc-operator-doc-gen`

### 执行内容

```
MANDATORY: 按 ascendc-operator-doc-gen skill 流程执行：

阶段 1: 信息提取
  - 从 register.cpp 提取 Python 调用签名（m.def schema）
  - 从 ops.h 提取 C++ 函数声明和返回类型
  - 从 design.md 提取算法描述、参数说明、dtype 支持、约束条件
  - 从 op_host 提取 TORCH_CHECK 约束
  - 从 tests/test_<op_name>.py 提取使用示例

阶段 2: 文档结构组装
  - 按 PyTorch 官方文档风格组装中文接口文档
  - 包含：标题签名 + 功能描述 + 参数说明 + 支持的数据类型 + Shape + 约束条件 + 使用示例 + 返回值

阶段 3: 文件生成
  - 生成 csrc/ops/<op_name>/README.md

阶段 4: 在交互界面展示完整文档内容
```

### 检查点

- [ ] 从源代码提取了完整的接口信息（签名、参数、dtype、shape、约束）
- [ ] README.md 包含完整的 7 个段落（标题签名 + 功能描述 + 参数说明 + 支持的数据类型 + Shape + 约束条件 + 使用示例 + 返回值）
- [ ] Python 调用签名与 `register.cpp` 的 `m.def` 一致
- [ ] 参数说明使用 PyTorch 文档风格，描述使用中文
- [ ] 使用示例中的代码可运行
- [ ] README.md 已写入 `csrc/ops/<op_name>/README.md`
- [ ] **接口文档已在聊天界面完整展示**

**全部通过 → 进入 Phase 6**

---

## Phase 6：精度评估报告

**调用 Skill**：`ascendc-operator-precision-eval`

### 执行内容

```
MANDATORY: 按 ascendc-operator-precision-eval skill 流程执行：

阶段 1: 加载用例文档 + 信息收集
  - 读取 csrc/ops/<op_name>/test/<op_name>-test-cases.md（testcase-gen 产出）
  - 提取 SUPPORTED_DTYPES、TEST_SHAPES、GENERAL_SHAPES、BOUNDARY_VALUES、算子标杆
  - 从已有代码补充提取精度阈值等信息

阶段 2: 用例适配（(shapes + boundary) × dtypes ≥ 30 例）
  - 直接复用 testcase-gen 的 TEST_SHAPES 和 BOUNDARY_VALUES
  - 每个 shape / 边界值遍历算子支持的全部 dtype

阶段 3: 测试脚本生成（输出到算子目录 csrc/ops/<op_name>/test/）
  - 基于模板生成 test_<op_name>_precision.py（pytest 格式）
  - 基于模板生成 run_<op_name>_precision_report.py（报告生成器）

阶段 4: 执行
  - 运行 pytest 全部通过
  - 运行报告生成器输出 JSON

阶段 5: 报告生成
  - 生成 <op_name>_precision_report.md（含常规 shape + 边界值表格 + 汇总 + 关键发现）
  - 向用户提示报告路径
```

### 检查点

- [ ] 用例数 = (shapes + boundary) × dtypes ≥ 30
- [ ] 算子支持的每种 dtype 都已测试
- [ ] pytest 精度测试全部通过
- [ ] JSON 报告生成（含 5 个精度指标: MaxAbsErr / MeanAbsErr / MaxRelErr / MeanRelErr / CosineSim）
- [ ] Markdown 报告生成于 `csrc/ops/<op_name>/test/<op_name>_precision_report.md`
- [ ] **精度测试结果已以 Markdown 表格形式展示在聊天界面**
- [ ] 已向用户提示精度报告路径

**全部通过 → 进入 Phase 7**

---

## Phase 7：性能评测报告

**调用 Skill**：`ascendc-operator-performance-eval`

### 执行内容

```
MANDATORY: 按 ascendc-operator-performance-eval skill 流程执行：

阶段 1: 加载用例文档 + 信息收集
  - 读取 csrc/ops/<op_name>/test/<op_name>-test-cases.md（testcase-gen 产出）
  - 提取 SUPPORTED_DTYPES、TEST_SHAPES、GENERAL_SHAPES、算子标杆
  - 从已有代码补充提取 OP Type 关键字等信息

阶段 2: 用例适配（JSONL 格式，≥8 case）
  - 从 testcase-gen 的 TEST_SHAPES + GENERAL_SHAPES 中选取代表性 shape
  - 覆盖算子支持的全部 dtype
  - 转换为 JSONL 格式

阶段 3: 脚本生成（输出到算子目录 csrc/ops/<op_name>/test/）
  - 基于模板生成 run_<op_name>_case.py（单 case msprof 执行器）
  - 基于模板生成 benchmark_<op_name>_msprof.py（总控脚本）
  - 生成 <op_name>_cases.jsonl

阶段 4: 执行采集
  - 运行总控脚本，每 case 20 次迭代（前 10 次预热）
  - 按 OP Type 从 op_summary_*.csv 提取 Task Duration(us) 和硬件指标
  - 输出 JSON 结果

阶段 5: 报告生成
  - 生成 <op_name>_perf_report.md（含结果表格 + 汇总 + 简短分析）
  - 向用户提示报告路径
```

### 检查点

- [ ] JSONL 用例覆盖多种 shape × dtype（≥ 8 case）
- [ ] 使用 `msprof` 采集，非其他计时方式
- [ ] 按 `OP Type` 筛选目标算子（非 Op Name）
- [ ] 20/10 预热/统计策略
- [ ] JSON 报告生成（含 Task Duration + 硬件指标）
- [ ] Markdown 报告生成于 `csrc/ops/<op_name>/test/<op_name>_perf_report.md`
- [ ] 报告包含简短分析（≥ 3 条结论）
- [ ] **性能测试结果已以 Markdown 表格形式展示在聊天界面**
- [ ] 已向用户提示性能报告路径

**全部通过 → 算子开发完成**

---

## 阶段间数据流

```
Phase 1 输出                    Phase 2 输入
  csrc/ops/<op_name>/    ────▶    算子名称、目录结构
  design.md (占位)

Phase 2 输出                    Phase 3 输入
  design.md (完整)       ────▶    参数约束、支持的 dtype、典型 shape
                                  → 生成统一测试用例文档

Phase 3 输出                    Phase 4 输入
  <op_name>-test-cases.md ────▶    design.md (完整)
  （用例文档，供后续复用）          函数签名、UB 分配表 → bufferCoefficient
                                  计算伪代码 → Compute 逻辑
                                  Tiling 策略 → Block/UB 切分参数

Phase 4 输出                    Phase 5 输入
  已安装的算子 whl        ────▶    register.cpp / ops.h / design.md /
  tests/test_<op_name>.py        op_host / test 文件
                                  → 提取接口信息生成文档

Phase 5 输出                    Phase 6 输入
  csrc/ops/<op>/README.md ────▶    <op_name>-test-cases.md（来自 Phase 3）
  接口文档完成                     算子名、调用方式、输入域约束
                                  支持的全部 dtype、精度阈值
                                  → 输出到 csrc/ops/<op_name>/test/

Phase 6 输出                    Phase 7 输入
  精度报告通过             ────▶    <op_name>-test-cases.md（来自 Phase 3）
  csrc/ops/<op>/test/            算子名、工程/原生调用方式
                                  支持的全部 dtype、OP Type 关键字
                                  → 输出到 csrc/ops/<op_name>/test/
```

## 状态跟踪表

| Phase | 前置条件 | 调用 Skill | 关键产出物 |
|-------|----------|------------|-----------|
| 0. 需求收集 | 无 | — | CANN 路径 + Conda 环境 + 算子名称 + 功能描述 |
| 1. 工程初始化 | Phase 0 | `ascendc-operator-project-init` | 算子骨架目录 |
| 2. 设计文档 | Phase 1 | `ascendc-operator-design` | design.md（含 Tiling + UB 分配表） |
| 3. 用例生成 | Phase 2 | `ascendc-operator-testcase-gen` | `<op_name>-test-cases.md`（统一用例文档） |
| 4. 代码&测试 | Phase 3 | `ascendc-operator-code-gen` → `compile-debug` | 可运行算子 + 基本测试通过 |
| 5. 接口文档 | Phase 4 | `ascendc-operator-doc-gen` | PyTorch 风格中文 API 文档 (README.md) |
| 6. 精度评估 | Phase 5 | `ascendc-operator-precision-eval` | ≥30 例精度测试 + 精度报告 |
| 7. 性能评测 | Phase 6 | `ascendc-operator-performance-eval` | msprof 性能对比 + 性能报告 |

## 错误恢复

### 从中断点恢复

当用户说"继续算子开发"时：

| 检测条件 | 判定阶段 | 恢复动作 |
|----------|----------|----------|
| `csrc/ops/<op_name>/` 不存在 | Phase 1 未完成 | 从 Phase 1 开始 |
| `design.md` 为占位或空 | Phase 2 未完成 | 从 Phase 2 开始 |
| `csrc/ops/<op_name>/test/<op_name>-test-cases.md` 不存在 | Phase 3 未完成 | 从 Phase 3 开始 |
| `op_host/` 仍为骨架代码 | Phase 4 未完成 | 从 Phase 4 开始 |
| whl 未生成 | Phase 4 编译未完成 | 从编译步骤恢复 |
| 基本测试未通过 | Phase 4 测试未完成 | 从测试步骤恢复 |
| `csrc/ops/<op_name>/README.md` 不存在 | Phase 5 未完成 | 从 Phase 5 开始 |
| `csrc/ops/<op_name>/test/` 无精度报告 | Phase 6 未开始 | 从 Phase 6 开始 |
| 精度报告不存在或精度测试未全部通过 | Phase 6 未完成 | 从 Phase 6 恢复 |
| 精度报告存在但性能报告不存在 | Phase 7 未开始 | 从 Phase 7 开始 |
| `<op_name>_perf_report.md` 不存在或不完整 | Phase 7 未完成 | 从 Phase 7 恢复 |

### 编译/测试失败

由 `ascendc-operator-compile-debug` skill 内部处理，最多排错 3 次。3 次仍失败则停止并向用户报告详细错误。
