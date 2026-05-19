# ATB Skills 开发者指南

本指南为ATB（Ascend Transformer Boost）开发者提供完整的工作流技能集介绍，帮助非ATB开发者和初学者快速理解每个技能的用途、使用场景和协作关系。

---

## 目录

- [背景价值](#背景价值)
- [快速开始](#快速开始)
- [技能总览](#技能总览)
- [详细功能介绍](#详细功能介绍)
- [使用场景](#使用场景)
- [架构与依赖](#架构与依赖)
- [使用技能](#使用技能)
- [提交新需求](#提交新需求)
- [术语表](#术语表)

---

## 背景价值

### 为什么需要这些技能

ATB算子迁移涉及多步骤、多文档、多工具，流程复杂容易出错：

- 需要同时掌握ATB层、ACLNN层、GE层、Runtime层知识
- 手工操作缺乏标准化，难以保证质量一致性
- 问题排查需要跨层调试能力
- Skills将专家经验沉淀为可复用的标准化流程

### OPS→ACLNN 迁移的技术价值

| 维度 | OPS接口 | ACLNN接口 |
|------|---------|-----------|
| 设备支持 | 通用昇腾设备（310P/310B/910A/910B/910C） | 更具有兼容性，包括新设备（950等） |
| 性能 | 标准实现 | 硬件深度优化，预计提升 |
| 维护 | 社区维护 | 昇腾官方持续优化 |
| 兼容性 | 历史兼容 | 新特性优先支持 |

### 昇腾生态中的位置

```
昇腾生态
├── CANN (基础软件栈)
├── ACLNN (算子API库)
├── ATB (Transformer加速库) ← atb Skills作用对象
│   └── 算子接入层
│       ├── OPS Runner (通用设备)
│       │     └── Kernel层
│       └── ACLNN Runner (新设备优化) ← 迁移目标（950等）
└── 上层框架 (MindIE/vLLM-Ascend等)
```

### 对于ATB的价值

**ATB（Ascend Transformer Boost）定位**：

- 昇腾平台的高性能Transformer加速库
- 专为昇腾NPU优化的算子实现
- 支持MindIE/vLLM-Ascend等主流框架

**OPS Runner vs ACLNN Runner**：

| 维度 | OPS Runner | ACLNN Runner |
|------|------------|--------------|
| 设备支持 | 通用昇腾设备 | 新设备专用优化（950等） |
| 实现方式 | 通用实现 | 硬件深度适配 |
| 新特性 | 特性化支持 | 优先支持 |

**Skills对ATB的价值**：

- 将专家经验沉淀为可执行流程
- 降低算子迁移的技术门槛
- 确保迁移质量一致性
- 加速新设备（950等）适配

---

## 快速开始

### 新用户5步入门

```
第1步: CANN环境安装 → cann-operator-env-config
       └─ 安装Toolkit+Kernels

第2步: NNAL安装 → atb-nnal-installer
       └─ 安装NNAL（ATB加速库）

第3步: 编译测试框架 → atb-testframework-build
       └─ 源替换，全量/增量编译

第4步: 算子设计/用例 → atb-aclnn-operator-replacement-designer
       └─ 生成7章设计文档
       └─ 用户确认Gate 1

第5步: 用例设计 → atb-csv-testcase-generator
       └─ 正例/反例/性能测试用例（910B/950）
       └─ 用户确认Gate 2

第6步: 实际迁移 → atb-aclnn-operator-migration
       └─ 实现ACLNN Runner（910B/950）

第7步: 编译验证 → atb-testframework-build
       └─ 最多3次尝试

第8步: 测试验证 → atb-csv-tester
       └─ 运行CSV用例（910B设备）
       └─ 950测试需用户在950设备上手动执行
```

### 常用快捷命令

```bash
# 环境检查
npu-smi info
source <CANN_PATH>/set_env.sh
source <ATB_REPO_PATH>/output/atb/set_env.sh

# 编译测试框架
cd <ATB_REPO_PATH>
bash scripts/build.sh testframework

# 运行CSV测试
cd tests/framework/python/CsvOpsTestTool
python3 atb_csv_ops_test.py -i <CSV_FILE_PATH> -n 1:5
```

### 遇到问题怎么办

| 问题类型 | 求助技能 |
|----------|----------|
| 环境/编译问题 | atb-debug-guide |
| 测试失败 | atb-debug-guide |
| 参数映射疑问 | atb-aclnn-operator-replacement-designer |
| 流程不清楚 | atb-ops-to-aclnn-migration-workflow |

---

## 技能总览

### Skills Index

| 技能目录 | 名称 | 类型 | 核心功能 | 前置依赖 | 典型耗时 |
|---------|------|------|---------|----------|----------|
| [atb-nnal-installer/](atb-nnal-installer/) | CANN NNAL 安装 | env-setup | NNAL 安装、环境变量配置（Toolkit+Kernels 由 cann-operator-env-config 安装） | cann-operator-env-config | 5-10分钟 |
| [atb-testframework-build/](atb-testframework-build/) | ATB 测试框架编译 | build | 全量编译/增量编译、GitHub→gitcode源替换 | atb-nnal-installer | 5-15分钟 |
| [atb-aclnn-operator-replacement-designer/](atb-aclnn-operator-replacement-designer/) | OPS→ACLNN 算子替换设计文档生成 | design | 生成7章结构化设计文档、参数映射分析 | 无 | 5-10分钟 |
| [atb-csv-testcase-generator/](atb-csv-testcase-generator/) | ATB CSV 测试用例生成 | testcase | 正例/反例/性能测试设计（910B/950）、CSV格式规范 | atb-aclnn-operator-replacement-designer | 5-10分钟 |
| [atb-aclnn-operator-migration/](atb-aclnn-operator-migration/) | OPS→ACLNN 算子迁移工具 | migration | 执行算子迁移、新设备（910B/950）启用ACLNN加速 | atb-nnal-installer + 设计文档 | 5-10分钟 |
| [atb-csv-tester/](atb-csv-tester/) | ATB CSV 测试执行 | test | 运行CSV格式ATB测试用例（910B）、结果解析 | atb-testframework-build + CSV用例 | 10-30分钟 |
| [atb-debug-guide/](atb-debug-guide/) | ATB 调试指南 | debug | 环境问题排查、内存错误、CSV测试失败诊断 | 无 | 按需调用 |
| [atb-ops-to-aclnn-migration-workflow/](atb-ops-to-aclnn-migration-workflow/) | 标准化工作流模板 | workflow | 7阶段流程定义、HIL Gate控制、技能调度 | 整合所有技能 | 参考模板 |

---

## 详细功能介绍

### atb-nnal-installer - CANN NNAL 安装

**技能定位**：安装 NNAL（ATB加速库），是整个 ATB 开发的基础

**核心能力**：

- 检查 NPU 驱动状态（`npu-smi info`）
- 支持两种 NNAL 安装方式：run 包安装、Docker 镜像提取
- NNAL 环境变量配置与持久化

**使用场景**：

- Toolkit+Kernels 已安装后，安装 NNAL
- 升级 NNAL 版本
- Docker 容器内快速部署 NNAL

**关键技术概念**：

- **Toolkit+Kernels**：由 `cann-operator-env-config` 技能安装，本技能的前置条件
- **NNAL**：ATB加速库，依赖 Toolkit 环境变量
- **芯片型号**：910B/950/910/310P/310B（Kernels 选择由 cann-operator-env-config 处理）

**与其他技能的关系**：

- 依赖 `cann-operator-env-config`（提供 Toolkit+Kernels）
- 是 `atb-testframework-build` 和 `atb-csv-tester` 的前置条件

---

### atb-testframework-build - ATB 测试框架编译

**技能定位**：编译ATB测试框架，为后续测试验证提供基础

**核心能力**：
- 自动检测GitHub→gitcode源替换需求
- 支持增量编译（仅修改文件）和全量编译（含第三方依赖）
- 自动检测CXX ABI版本（ABI_0/ABI_1）
- 编译产物验证（`libatb_test_framework.so`等）

**使用场景**：
- 首次编译ATB测试框架
- 修改代码后重新编译
- 第三方依赖更新后重建

**关键技术概念**：
- **CXX ABI**：C++应用程序二进制接口版本，影响库兼容性
- **3rdparty**：第三方依赖目录（nlohmannJson、Mind-KernelInfra、catlass等）
- **testframework**：测试框架构建目标，生成`libatb_test_framework.so`

**与其他技能的关系**：
- 依赖 `atb-nnal-installer`（需要CANN环境）
- 是 `atb-csv-tester` 的前置条件

---

### atb-aclnn-operator-replacement-designer - OPS→ACLNN 算子替换设计文档生成

**技能定位**：算子迁移的设计阶段工具，生成标准化设计文档

**核心能力**：
- 分析ATB和ACLNN接口文档，提取参数结构
- 建立ATB→ACLNN参数映射关系（直接映射/计算映射/废弃/新增）
- 生成7章结构化设计文档（需求、IR、规格、测试、实现、风险、结论）
- Human in the Loop（Gate 1）：用户确认设计文档

**使用场景**：
- 迁移前的设计阶段
- 评估迁移可行性
- 团队协作时统一设计方案

**关键技术概念**：
- **参数映射类型**：
  - 直接映射：参数名和含义完全一致
  - 计算映射：根据公式或规则转换（如`beginNormAxis`→`normalizedShape`）
  - 废弃参数：ACLNN不再需要
  - 新增参数：ACLNN新增，需设置默认值
- **HIL Gate 1**：设计文档确认检查点，用户确认后才进入下一阶段

**与其他技能的关系**：
- 独立技能，可在实际编码前提前完成
- 输出为 `atb-csv-testcase-generator` 提供输入（参数约束信息）

---

### atb-csv-testcase-generator - ATB CSV 测试用例生成

**技能定位**：TDD（测试驱动开发）核心工具，设计全面测试用例

**核心能力**：
- 正例设计：覆盖float16/bf16、主流模型shape、边界条件
- 反例设计：覆盖所有参数校验分支（数据类型/维度/参数错误）
- 性能测试设计：大/中/小shape覆盖
- CSV格式规范：21列标准格式，|分隔符

**使用场景**：
- 新算子测试用例生成
- 迁移后验证用例设计
- 回归测试用例补充

**关键技术概念**：
- **ExpectedError**：期望错误类型，如`NO_ERROR`、`I:ERROR_INVALID_TENSOR_DIM`
- `C:`前缀：CreateOperation参数配置错误
- `I:`前缀：InferShape输入张量错误
- `S:`前缀：Setup中outtensor/系统/内存错误
- **TestType**：测试类型，空为功能测试，`Performance`为性能测试
- **DataGenType/Range**：数据生成方式和范围（`customize`、`random`、`-2,2`）

**与其他技能的关系**：
- 依赖 `atb-aclnn-operator-replacement-designer`（参数约束信息）
- 输出为 `atb-csv-tester` 提供测试输入

---

### atb-aclnn-operator-migration - OPS→ACLNN 算子迁移工具

**技能定位**：实际执行算子代码迁移的核心工具

**核心能力**：
- 创建ACLNN Runner（继承`AclnnRunner`基类）
- 实现四大方法：
  - `BuildAclnnVariantPack`：处理输入输出张量转换
  - `SetAclNNWorkspaceExecutor`：配置工作空间和执行器
  - `LaunchAclnnKernel`：执行ACLNN计算
  - `LoadMethod`：动态加载ACLNN函数指针
- 设备检测：使用设备配置判断（如`GetSingleton<Config>().Is910B()`）
- 向后兼容：非新设备保持OPS实现

**使用场景**：
- 设计文档确认后的实际迁移
- 为新设备（910B/950/310P等）启用ACLNN加速

**关键技术概念**：
- **AclnnRunner**：ACLNN算子实现的基类，封装通用逻辑
- **四大方法**：ACLNN Runner必须实现的四个核心方法
- **LoadMethod**：动态加载`.so`库中的函数符号

**与其他技能的关系**：
- 依赖 `atb-nnal-installer` 和 设计文档
- 是迁移流程的核心执行技能

---

### atb-csv-tester - ATB CSV 测试执行

**技能定位**：执行CSV测试用例的专用工具

**核心能力**：
- 环境检查：验证CANN/ATB环境、测试框架库存在
- 测试执行：支持全部用例或指定范围（`-n 1:5`）
- 结果解析：区分正例通过、反例预期错误、意外失败
- CSV格式解析：21列标准格式支持

**使用场景**：
- 迁移完成后验证
- 回归测试
- 特定用例范围调试

**关键技术概念**：
- **CaseRange**：用例范围，格式`start:end`（如`1:5`表示用例1到5）
- **CSV格式**：21列，`|`分隔，包含CaseNum、CaseName、OpName、OpParam等

**与其他技能的关系**：
- 依赖 `atb-testframework-build`（测试框架）和 CSV用例文件
- 失败时可调用 `atb-debug-guide` 诊断

---

### atb-debug-guide - ATB 调试指南

**技能定位**：问题排查和调试的辅助工具

**核心能力**：
- ABI版本不匹配诊断（`cxx_abi_0` vs `cxx_abi_1`）
- 内存错误分析（`ERROR_OUT_OF_HOST_MEMORY`）
- Tensor维度错误诊断（`ERROR_INVALID_TENSOR_DIM`）
- 数据类型错误诊断（`KeyError: 'float32'`）
- ACLNN函数签名不匹配诊断（161001/161002错误码）
- 日志分析：ATB日志（`atb_*.log`）和下层日志（`plog-*.log`）

**使用场景**：
- 测试失败排查
- 编译错误定位
- 环境问题诊断
- 运行时错误分析

**关键技术概念**：
- **plog日志**：CANN下层组件日志（GE、RUNTIME、OP、ASCENDCL）
- **错误码分类**：
  - 161001：`ACLNN_ERR_PARAM_NULLPTR`，参数为空指针
  - 161002：`ACLNN_ERR_PARAM_INVALID`，参数类型/维度/取值无效

**与其他技能的关系**：
- 独立技能，被其他技能在失败时调用
- 可被任何阶段调用进行问题诊断

---

### atb-ops-to-aclnn-migration-workflow - 标准化工作流模板

**技能定位**：整合所有技能的7阶段标准化工作流模板

**核心能力**：
- 7阶段流程定义：前置学习→设计→用例→迁移→编译→测试→交付
- HIL Gate控制：Gate 1（设计确认）、Gate 2（用例确认）强制用户确认
- 3次编译规则：编译失败最多尝试3次，记录后询问用户
- 技能调度时序：明确各阶段调用哪个技能

**使用场景**：
- 提交迁移任务时的流程参考
- 理解完整迁移流程
- 新成员培训

**关键技术概念**：
- **Phase 0-6**：7个阶段的标准化流程
- **Gate 1/2**：Human in the Loop检查点，必须用户确认
- **3次编译规则**：第3次失败后停止并询问用户

**与其他技能的关系**：
- 整合所有其他8个技能
- 是任务提交的参考模板

---

## 架构与依赖

### Skill Dependency

```
ascend-transformer-boost (Index Skill)
│
├── atb-ops-to-aclnn-migration-workflow         ← 标准化工作流模板
│       │
├── atb-nnal-installer                 ← 依赖 cann-operator-env-config（提供 Toolkit+Kernels）
│       │
├── atb-testframework-build             ← 依赖 atb-nnal-installer
│       │
├── atb-aclnn-operator-replacement-designer  ← 独立，迁移前必需
│       │ (含 HIL 用户确认 Gate 1)
│       │
├── atb-csv-testcase-generator          ← 依赖设计文档
│       │ (含 HIL 用户确认 Gate 2)
│       │
├── atb-aclnn-operator-migration        ← 依赖 atb-nnal-installer + 设计文档
│       │ (依赖 Gate 1 & Gate 2 通过)
│       │
├── atb-csv-tester                      ← 依赖 atb-nnal-installer + 测试框架
│       │
└── atb-debug-guide                    ← 辅助技能，独立使用
```

### 典型迁移工作流程

```
[Phase 0] 前置知识学习
  └─ 阅读ATB/ACLNN接口文档
            │
[Phase 1] 设计文档生成
  └─ atb-aclnn-operator-replacement-designer
            │
     [Gate 1: 用户确认设计文档]
            │
[Phase 2] CSV用例设计
  └─ atb-csv-testcase-generator
            │
     [Gate 2: 用户确认用例]
            │
[Phase 3] 实际迁移
  └─ atb-aclnn-operator-migration
            │
[Phase 4] 编译验证
  └─ atb-testframework-build (≤3次)
     └─ [失败] atb-debug-guide
            │
[Phase 5] 测试验证
  └─ atb-csv-tester
     └─ [失败] atb-debug-guide
            │
[Phase 6] 交付报告
```

---

## 使用技能

### 如何调用技能

Skills被自动调用基于**关键词匹配**。当用户的输入包含触发关键词时，Agent会自动加载对应的Skill。

触发关键词示例：
- `ATB` / `昇腾` + `NPU` → 通用ATB相关
- `ATB` + `ACLNN` / `算子迁移` → 迁移相关技能
- `CANN` + `安装` / `910B` / `950` → 环境安装技能
- `ATB` + `CSV` / `测试` → 测试相关技能
- `调试` / `排查` / `错误` → atb-debug-guide

**示例对话**：
```
用户: 我需要把 SwigluQuant 算子从 OPS 接口迁移到 ACLNN 接口
Agent: 我将调用 atb-ops-to-aclnn-migration-workflow 技能...
```

### Checkpoint 机制

每个Skill定义了**检查点（Checkpoint）**来验证执行状态：

| Checkpoint Type | Description |
|-----------------|-------------|
| 自动检查 | 系统自动验证，如编译成功、文件存在 |
| 用户确认 | 需要人工确认，如设计文档审核（Gate 1/2） |

### Human in the Loop

**Human in the Loop（人在回路）**是AI工作流中的重要机制，指在关键决策点需要人类确认后才能继续执行。

**典型HIL场景**：
1. **设计文档确认（Gate 1）**：设计文档生成完成后，需用户确认参数映射、风险评估
2. **CSV用例确认（Gate 2）**：测试用例设计完成后，需用户确认覆盖范围
3. **3次编译失败后**：记录问题并询问用户，而非无限重试

---

## 提交新需求

### 任务文件模板

提交ATB OPS→ACLNN迁移任务时，建议使用以下结构：

```markdown
# task
完成：设计、实现、测试 atb中 910B+950 下 {算子名称} 从ops接口切换至aclnn接口的需求

# 前置学习知识
[根据算子类型填写，如quantization基础]

# 仓路径
cann: /usr/local/Ascend/ascend-toolkit/latest
atb: {atb仓库路径}

# reference
atb接口: {ATB文档链接}
aclnn接口: {ACLNN文档链接}

# 流程
验证环境变量 -> 算子迁移设计 -> [HIL确认] -> 实际迁移 -> 编译 -> 测试 -> 交付

# action requirements
1. 任何遇到的问题都需要清晰记录
2. 没有思路时需要返回给用户明确结果
3. 先plan，后执行，确认条件充足后才写代码
4. 自我反省，自我记录，自我迭代
5. 遇到问题必须记录并询问用户
```

完整模板参考：[atb-ops-to-aclnn-migration-workflow/TASK_TEMPLATE.md](atb-ops-to-aclnn-migration-workflow/TASK_TEMPLATE.md)

---

## 新增skill

### Directory Structure

```
skills/
└── ascend-transformer-boost/
    ├── SKILL.md                    ← 索引文件（必须）
    ├── skill-name-1/
    │   └── SKILL.md              ← 技能定义（必须）
    ├── skill-name-2/
    │   ├── SKILL.md
    │   ├── references/          ← 参考文档
    │   └── templates/           ← 模板文件
    └── ...
```

### SKILL.md Format

每个Skill必须包含 `SKILL.md` 文件，格式如下：

```markdown
---
name: skill-name
description: >
  Skill的简短描述
keywords:
    - keyword1
    - keyword2
metadata:
  author: author-name
  version: "1.0.0"
  created: "2026-04-17"
  skill-type: type (design/migration/test/...)
allowed-tools: Bash(*) Read(*) Write(*)
---

# Skill Name

## 概述
技能的详细介绍

## 参数约束
[必须从用户处获取的参数]

## 工作流程
[详细的任务执行步骤]

## 检查点
| 检查项 | 验证方式 |
|--------|----------|
| 检查点1 | 验证方式 |
```

### Naming Conventions

- **Skill目录**: 使烤串命名法（kebab-case）
  
  - ✅ `ascend-transformer-boost`
  - ❌ `ascend_transformer_boost`
  - ❌ `AscendTransformerBoost`
- **SKILL.md**: 必须命名为 `SKILL.md`（全大写）

### Updating Index

新增Skill后，必须更新 `{SKILLS根目录}/skills/ascend-transformer-boost/SKILL.md` 的技能索引表：

```markdown
## 技能索引表

| 技能目录 | 名称 | 类型 | 核心功能 |
|---------|------|------|---------|
| [skill-name/](skill-name/) | 新Skill名称 | type | 功能描述 |
```

---


## 术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| ATB | Ascend Transformer Boost | 昇腾Transformer加速库 |
| ACLNN | Ascend Computing Language Neural Network | 昇腾算子API库 |
| CANN | Compute Architecture for Neural Networks | 昇腾计算架构 |
| OPS | Operator | 算子实现接口（兼容原始ATB设备） |
| CSV | Comma-Separated Values | 逗号分隔值（测试用例格式，使用\|分隔） |
| TDD | Test-Driven Development | 测试驱动开发 |
| HIL | Human in the Loop | 人在回路 |
| Skill | - | 技能模块（ATB开发专用） |
| Runner | - | ATB算子执行器，封装具体实现 |
| AclnnRunner | - | ACLNN算子Runner的基类 |
| ABI | Application Binary Interface | 应用程序二进制接口 |
| Toolkit | - | CANN开发工具包 |
| Kernels | - | ATB算子层 |
| NNAL | - | ATB加速库+SiP加速库 |
| Gate | - | 工作流检查点 |

---

*版本: 2.1.0*  
*更新日期: 2026-04-28*  
*作者: ascend-transformer-boost-team & Claude Code /w opus4.7*
