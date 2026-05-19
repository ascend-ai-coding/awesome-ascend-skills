---
name: external-cannbot-ops-ascendc-code-review
description: Ascend C 代码检视技能。基于假设检验方法论对代码进行安全规范检视。当用户提供代码片段和检视规则描述时，可使用此技能进行代码检视，使用时必须明确提供：代码片段和检视规则描述。
original-name: ascendc-code-review
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# Ascend C 代码检视技能

## 核心原则

1. **合规映射** - 所有检视动作映射至编码规范具体条款，确保 100% 覆盖
2. **假设检验驱动** - 采用假设检验方法论，系统性收集证据判定风险
3. **可审计可复用** - 检视动作全程日志记录，证据链完整可追溯

## 调用接口

### 必需参数

调用此技能时，必须明确提供以下参数：

**参数1：代码片段**
- 待检视的代码内容
- 可以是函数、代码块、文件内容等

**参数2：检视规则描述**
- 明确指定要检视的问题类型
- 例如：`检查整数溢出`、`检查内存泄漏`、`检查空指针解引用`等
- 或指定具体的规范文件路径

**参数3：规范文件路径**（可选）
- 指定具体的编码规范文件路径
- 如果未指定，默认使用 `references/cpp-secure.md`（安全编码规范）

### 参数校验

如果缺少任何一个必需参数，应：
1. 明确通知用户缺少哪些参数
2. 放弃执行检视
3. 提示用户提供完整参数

## 规范文档体系

| 文档名称 | 文档路径 | 检视范围 | 适用场景 | 条款数 |
|---------|---------|---------|---------|-------|
| **C++ 安全编码规范** | `references/cpp-secure.md` | 数值安全、内存安全、输入验证、资源管理、并发安全 | C++ 代码安全性检视 | 47 条 |
| **Python 安全编码规范** | `references/python-secure.md` | 数值安全、异常处理、文件操作、序列化、命令执行、网络安全 | Python 代码安全性检视 | - |
| **C++ 代码风格规范** | `references/cpp-style.md` | 命名规则、格式规范、注释规范 | C++ 代码风格、可读性检视 | - |
| **C++ 通用编码规范** | `references/cpp-general.md` | 代码设计、数据类型、函数设计等 15 类规范 | C++ 代码质量、可维护性检视 | 46 条 |
| **安全编译规范** | `references/compile-secure.md` | ASLR、栈保护、GOT只读、立即绑定、堆栈不可执行 | 编译配置、构建脚本检视（仅 Tiling 侧） | 7 条 |
| **Ascend C API 最佳实践** | `references/ascendc-api.md` | API 黑名单、对齐要求、配对检查、限制约束 | Ascend C API 使用检视（仅 Kernel 侧） | 9 条 |
| **Ascend C 高性能编程** | `references/ascendc-perf.md` | 性能优化、精度标准、Tiling 设计 | Ascend C 高性能编程检视（仅 Kernel 侧） | 12 条 |
| **TOPK 问题清单** | `references/ascendc-topk.md` | Host侧高频问题、属性获取、特殊值处理、核间同步 | 算子开发重点检视 | 13 条 |
| **SIMT API C风格化规范** | `references/simt-api-analysis.md` | C++风格API转换为C风格API、变量名冲突、头文件位置 | SIMT kernel代码API检视（仅 Kernel 侧） | 13 条 |

## API 文档检索

> **重要**：检视 Kernel 代码涉及 API 用法时，必须使用 `/ascendc-docs-search` skill 查阅 API 文档，禁止凭记忆或推测判断。

### 检视时需查阅的核心 API

| API 类别 | 具体 API | 检视关注点 |
|---------|---------|-----------|
| **数据搬运** | `DataCopy`, `DataCopyPad` | 对齐要求、同步机制 |
| **内存管理** | `InitBuffer`, `AllocTensor`, `FreeTensor`, `EnQue`, `DeQue` | 配对要求 |
| **向量计算** | `Add`, `Sub`, `Mul`, `Div`, `Cast` | 参数限制、精度处理 |
| **归约操作** | `ReduceSum`, `ReduceMax` | 中间精度保护 |
| **SIMT 线程管理** | `GetThreadNum`, `GetThreadIdx`, `GetBlockIdx`, `GetBlockNum` | C++风格→C风格转换、维度选择 |
| **SIMT 数学运算** | `UintDiv`, `Min`, `Max`, `Floor`, `Ceil`, `Abs` | C++风格→C风格转换、UintDiv保留 |
| **SIMT 核函数调用** | `VF_CALL`, `Dim3` | C++风格→C风格转换 |
| **SIMT 原子操作** | `AtomicAdd`, `AtomicSub` | C++风格→C风格转换 |
| **SIMT 同步** | `ThreadBarrier` | C++风格→C风格转换 |

**查阅方法**：调用 `/ascendc-docs-search` skill，输入 API 名称即可获取官方文档。

## 检视流程（假设检验驱动）

### 阶段一：准备

1. 校验必需参数是否完整
2. 根据检视规则描述或指定路径读取编码规范文件
3. 确认待检视的代码片段
4. **选择适用的规范文档**：
   - C++ 代码安全性：使用 `cpp-secure.md`
   - Python 代码安全性：使用 `python-secure.md`
   - 代码风格质量：使用 `cpp-style.md` 或 `cpp-general.md`
   - 编译配置：使用 `compile-secure.md`
   - SIMT kernel API：使用 `simt-api-analysis.md`
5. **Kernel 代码检视前置学习**：
   - 若待检视代码涉及 Kernel 侧（Device 侧），必须使用 `/ascendc-docs-search` skill 查阅 API 文档
   - 常见需查阅的 API：`DataCopy`、`DataCopyPad`、`EnQue`、`DeQue`、`Cast` 等
   - SIMT kernel 代码需查阅：`GetThreadNum`、`GetThreadIdx`、`UintDiv`、`VF_CALL` 等
   - 禁止凭记忆或推测判断 API 用法正确性

### 阶段二：假设检验（核心）

**步骤1：代码段识别**

将目标代码划分为独立的代码段（函数、语句块、逻辑单元）

**步骤2：假设建立**

对每个代码段建立假设：
- **原假设 H0**：该代码段是安全的
- **备择假设 H1**：该代码段存在风险
- **自信值初始化**：0%

**步骤3：证据收集与评估**

按维度系统性寻找证据：

| 证据类型 | 分析动作 | 分值规则 |
|---------|---------|---------|
| **规范违反** | 对照规范条款识别违规点 | 有效证据 +40% |
| 上下文防御缺失 | 检查作用域内是否有防御代码 | 无防御 +30% |
| 函数调用链风险 | LSP/Grep 分析调用函数内部逻辑 | 发现风险 +25% |
| 数据流追踪风险 | 分析变量来源、运算过程 | 发现风险 +25% |

**分析要求**：
- 使用 LSP 获取代码中的符号定义，使用 Grep 查找依赖关系
- 风险代码必须检查是否在当前文件作用域内的其他位置进行防御
- 遇到函数调用，必须查看函数内部逻辑并综合判断
- 遇到存在风险的结构体或成员变量，必须查看定义和运算过程
- Ascend C 基础 API 用法，必须使用 `/ascendc-docs-search` skill 查阅官方文档，禁止凭记忆或推测
- **条例定制化要求**：若条例包含专属检视方法或强制要求，必须严格按该条例指引执行

**步骤4：证据有效性校验**

排除误报：
- 已在作用域其他位置进行有效防御 → 排除
- 上下文可证明不可能触发风险 → 排除

**步骤5：决策判断**

计算自信值并决策：
- 自信值 = Σ 有效证据分值
- **决策规则**：当自信值超过 60% 时，判定代码段存在风险，进入报告生成；否则继续检查下一代码段

### 阶段三：报告生成

1. 按代码检视报告模板格式生成报告
2. 每个问题详情前展示假设检验过程（证据链和自信值计算过程）
3. 明确标注违反规范的条款编号和来源文档

## 注意事项

1. 请先完整阅读对应的编码规范文件，并根据这些规范进行代码检视，其他的一概不统计
2. 检视过程中，存疑代码块存在函数调用行为，必须使用 LSP 或搜索工具进行深层次分析
3. **Kernel 代码检视前必须使用 `/ascendc-docs-search` skill 学习 Ascend C 基础 API 文档**：获取 DataCopy、EnQue/DeQue、Cast 等核心 API 的官方文档，禁止凭记忆或推测判断
4. **SIMT kernel 代码检视前必须查阅 `references/simt-api-analysis.md`**：了解 C++ 风格 API 到 C 风格 API 的转换规则，特别注意 UintDiv 必须保留
5. 不确定的编码规范问题，在输出报告中以存疑的形式进行列举，供用户自主判断
6. 检视报告中列举的问题代码不宜过长，描述清晰问题代码即可
7. 返回检视结果时，必须仔细检查结果中风险代码行的行数是否正确
8. 返回检视结果时，所有风险代码块都应该被引用，不能只展示一个行数

## 红线问题

### Host 侧红线问题
1. Host侧代码必须对除法、求余操作做除零保护
2. Host侧代码数组访问，必须进行越界保护
3. Host侧代码加法、乘法、减法操作，必须进行溢出和减翻保护
4. Host侧代码指针操作，必须先赋值后访问，并进行指针保护, 例如const bool* getLogtarget = attrs->GetAttrPointer<bool>(LOGTARGET_INDEX); this->logTarget_ =  *getLogtarget，此操作无指针保护
5. 变量使用前，必行进行有效初始化，例如类成员变量需要初始化
6. 申请资源，使用和释放必须匹配

### Kernel 侧红线问题（SIMT API）
1. SIMT kernel 代码必须将 C++ 风格 API 转换为 C 风格 API（如 `GetThreadNum` → `blockDim.x`）
2. `Simt::UintDiv` 必须保留，禁止转换（官方规定，无 C 风格替代）
3. 变量名不能命名为 `threadIdx`、`blockIdx`、`blockDim`、`gridDim`（与 C 风格 API 冲突）
4. 头文件 `simt_api/asc_simt.h` 必须在 namespace 外部
5. 转换后必须通过编译验证，确保无编译错误

> **TOPK 问题清单**详见：`references/ascendc-topk.md`（含 13 条高频检视问题，标注 Host/Kernel 适用范围）

> **SIMT API C风格化规范**详见：`references/simt-api-analysis.md`（含 13 条 API 转换规则，仅 Kernel 侧适用）

## 输出规范

### 检视报告必需包含

1. 检视类别（标注规范来源）
2. 风险点列表（含行号、代码片段）
3. 证据链（展示假设检验过程）
4. 建议修复方案
5. 规范条款引用（标注条款编号和来源文档）

### 输出格式

详见：agents/ascendc-ops-reviewer/style/code_review_summary_style.txt
