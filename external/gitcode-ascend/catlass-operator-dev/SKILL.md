---
name: external-gitcode-ascend-catlass-operator-dev
description: Catlass 算子端到端开发编排器。基于 ascend-kernel（csrc/ops），串联 catlass 设计、catlass-operator-code-gen
  与 ascendc 子 skill，完成从工程初始化到文档、精度、性能的闭环。关键词：Catlass、端到端、ascend-kernel、算子开发、工作流编排。
original-name: catlass-operator-dev
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-08'
synced-commit: 155ac37bd169ddb89479af528297cfb2237400aa
license: UNKNOWN
---

# Catlass 算子端到端开发编排

**Skill 类型**：流程导向型（六阶段工作流；**Catlass 源码准备**并入 Phase 1，子技能串行编排）

本 skill 编排 ascend-kernel 上 **Catlass** 算子从零到生产可用；**通用能力**（工程骨架、编译调试、接口文档、精度、性能）**复用 ascendc-*** 子 skill，**Catlass 专属**（源码树、设计、Device/Host 落地）使用 **catlass-*** 子 skill。

## 核心原则

1. **六阶段串行**：工程初始化（含 Catlass 源码）→ 设计文档 → 代码生成与编译测试 → 接口文档 → 精度评估 → 性能评测，严格顺序执行
2. **子技能执行**：每个阶段 **MUST** 打开并遵循对应子 skill，不得自行替代实现
3. **阶段门控**：前一阶段检查点全部通过后才进入下一阶段
4. **设计驱动编码**：代码生成依赖 **catlass-operator-design** 定稿的 `design.md` 与 **catlass/examples** 选型
5. **无需用户预先手写设计文档**：设计阶段由 **catlass-operator-design** 生成并落盘
6. **文档闭环**：编译测试通过后 **MUST** 生成 PyTorch 风格中文接口文档（Phase 4），并在聊天界面展示
7. **精度闭环**：算子必须通过 **≥30** 例全面精度评估（Phase 5）才算完成
8. **性能闭环**：算子必须完成 **torch_npu.profiler** 对比评测并输出性能报告（Phase 6）；结论以 **ascendc-operator-performance-eval** 为准
9. **结果可视化**：Phase 3/4/5/6 的关键结果 **MUST** 以 Markdown 等形式直接展示在聊天界面，不要仅输出路径
10. **算子命名**：`op_name`（snake_case）**必须**包含子串 **`catlass`**，与 ascend-kernel 内既有 Catlass 算子约定一致
11. **诚实停机**：因环境或依赖无法继续时，说明具体原因与已完成步骤后停止

## Catlass 编译与运行（易错摘要）

- **构建**：`BUILD_CATLASS_MODULE=ON`；CMake 使用含 **torch_npu** 的 Python（如 **`-DPYTHON_EXECUTABLE` / `ASCEND_BUILD_PYTHON`**）；**`CATLASS_ARCH`** 与芯片一致（见 **`catlass-operator-code-gen/references/compile-catlass.md`**）；CANN 可为 **bundle 根** + **`cann-*/set_env.sh`**。
- **pytest / torch_npu**：若报 **`ASCEND_RUNTIME_PATH`**：`export ASCEND_RUNTIME_PATH="${ASCEND_TOOLKIT_HOME}/runtime"`。
- **设计/代码**：与 **`catlass/include`**、**`catlass/examples`** 可对齐编译的示例一致，细则见 **`compile-catlass.md`**。

## 可用子 Skill 清单

| Skill | 路径 | 职责 |
|-------|------|------|
| `ascendc-operator-project-init` | `ascendc-operator-project-init/SKILL.md` | 检测/创建 ascend-kernel，在 `csrc/ops/<op_name>/` 生成算子骨架 |
| — | （Phase 1 内步骤） | 在 **ASCEND_KERNEL_ROOT** 克隆 **`catlass/`**（与 `csrc/` 同级），使 `include/`、`examples/` 可用 |
| `catlass-operator-design` | `catlass-operator-design/SKILL.md` | 将 Catlass 需求转为定稿设计文档（推荐 `csrc/ops/<op_name>/design.md`） |
| `catlass-operator-code-gen` | `catlass-operator-code-gen/SKILL.md` | 按 `design.md` 与 **catlass/examples** 落地 **op_host / op_kernel**、框架适配，并**内部调用**编译测试 skill |
| `ascendc-operator-compile-debug` | `ascendc-operator-compile-debug/SKILL.md` | 编译、安装 whl、生成/运行 `tests/test_<op_name>.py`（由 **catlass-operator-code-gen** 阶段 5 调用，勿单独跳过 code-gen 直接宣称完成） |
| `ascendc-operator-doc-gen` | `ascendc-operator-doc-gen/SKILL.md` | 生成 PyTorch 风格中文 API 文档 `README.md`（必选阶段） |
| `ascendc-operator-precision-eval` | `ascendc-operator-precision-eval/SKILL.md` | ≥30 例精度测试与精度验证报告（必选阶段） |
| `ascendc-operator-performance-eval` | `ascendc-operator-performance-eval/SKILL.md` | **JSONL** 用例 + **torch_npu.profiler**（warmup/active=5）+ `op_statistic.csv` 汇总，输出自定义 vs 标杆 Markdown 报告（必选阶段） |
| `catlass-operator-performance-optim` | `catlass-operator-performance-optim/SKILL.md` | 交付后可选：按 Catlass 文档做 tiling/性能迭代；代码变更后须回到 Phase 3 起复跑闭环 |

## 工程目录术语（与 AscendC 对齐）

| 术语 | 含义 |
|------|------|
| **ASCEND_KERNEL_ROOT** | ascend-kernel 根目录：含 `build.sh`、`CMakeLists.txt`、`csrc/` |
| **算子目录** | `<ASCEND_KERNEL_ROOT>/csrc/ops/<op_name>/` |
| **Catlass 源码** | `<ASCEND_KERNEL_ROOT>/catlass/`（**禁止**在 `csrc/ops/<op>/` 内克隆） |

## 工作流总览

```
┌─────────────────────────────┐   ┌──────────────┐   ┌───────────────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Phase 1                    │   │  Phase 2     │   │  Phase 3                  │   │  Phase 4         │   │  Phase 5         │   │  Phase 6         │
│  工程初始化 + Catlass 源码   │──▶│  Catlass 设计 │──▶│  代码生成+框架适配+编译测试 │──▶│  接口文档生成     │──▶│  精度评估报告     │──▶│  性能评测报告     │
│  project-init + clone      │   │  catlass-    │   │  catlass-code-gen →       │   │  doc-gen         │   │  precision-eval  │   │  performance-eval│
│  catlass                   │   │  design      │   │  compile-debug            │   │                  │   │                  │   │  (profiler)      │
└─────────────────────────────┘   └──────────────┘   └───────────────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘

输入: 算子名(含 catlass) + 功能描述 + 环境确认          输出: 可交付算子 + README + 精度报告 + profiler 性能报告
```

## 反模式清单（NEVER DO THESE）

- ❌ 不要跳过 **Catlass 源码准备**（无 **`catlass/include`**、**`catlass/examples`** 就做设计或代码生成）
- ❌ 不要在 **`csrc/ops/<op_name>/`** 内克隆 Catlass，必须在 **工程根** 下 `catlass/`
- ❌ 不要跳过设计阶段直接写 kernel/host
- ❌ 不要自行实现整套算子落地而不遵循 **catlass-operator-code-gen** 流程
- ❌ 不要在代码生成前擅自修改框架注册（以 project-init / code-gen 约定为准）
- ❌ 不要手动替代 **compile-debug** 所负责的编译安装与基础测试闭环（应通过 code-gen 阶段 5 触发）
- ❌ 不要跳过接口文档阶段（Phase 3 通过后必须 Phase 4）
- ❌ 不要跳过精度评估阶段（Phase 4 通过后必须 Phase 5）
- ❌ 不要跳过性能评测阶段（Phase 5 通过后必须 Phase 6）
- ❌ 不要使用与 **ascendc-operator-performance-eval** 不一致的采集方式作为最终性能结论
- ❌ 不要引用不存在的 skill

---

## Phase 0：需求收集

**目标**：确认 Catlass 算子开发的最小信息集与运行环境（与 **ascendc-operator-dev** Phase 0 对齐，并增加 Catlass 命名约束）。

### Step 0.1：环境确认（MUST 在任何开发动作之前完成）

#### CANN 环境

1. 检查 `ASCEND_HOME_PATH`（`echo $ASCEND_HOME_PATH`）
2. **已设置**：作为 `CANN_PATH`，无需重复询问
3. **未设置**：**MUST** 询问用户 CANN 路径（如 `/usr/local/Ascend/ascend-toolkit`）

```bash
source ${CANN_PATH}/*/set_env.sh
```

#### Conda 环境

1. 检查 `CONDA_DEFAULT_ENV`
2. **已激活且非 `base`**：直接使用
3. **未激活或为 `base`**：**MUST** 询问 conda 环境名

```bash
conda activate <env_name>
```

#### 环境确认检查点

- [ ] CANN 路径已确定且 `set_env.sh` 可执行
- [ ] Conda 环境已确定且可激活

### Step 0.2：算子需求收集

| 信息 | 格式要求 | 必填 | 说明 |
|------|----------|------|------|
| CANN 路径 | 绝对路径 | 是 | 同 ascendc，可自动检测 |
| Conda 环境 | 字符串 | 是 | 同 ascendc，可自动检测 |
| 算子名称 | snake_case，**含 `catlass`** | 是 | 如 `catlass_matmul_basic` |
| 功能描述 | 文本/公式/对标示例 | 是 | 与 Catlass 能力范围一致 |

**可选**：支持 dtype、SoC —— 默认值与 **catlass-operator-design** / 平台 API 一致即可。

### 决策树

| 用户请求 | 处理方式 |
|----------|----------|
| 「开发/生成某 Catlass 算子」 | 完成 Step 0.1 → 校验名称含 `catlass` → 确认功能 → 执行全流程 |
| 「继续 Catlass 算子开发」 | 完成 Step 0.1 → 按 **错误恢复** 检测当前阶段并续跑 |

### 验收标准

- [ ] CANN + Conda 已确认
- [ ] `op_name` 已确认且包含 **`catlass`**
- [ ] 功能描述明确

---

## Phase 1：工程初始化 + Catlass 源码准备

### Step 1.1：工程骨架

**调用 Skill**：`ascendc-operator-project-init`

```
MANDATORY: 按 ascendc-operator-project-init 执行：
1. 检测或创建 ascend-kernel
2. 在 csrc/ops/<op_name>/ 创建算子骨架
3. 提示注册更新点（后续由 catlass-operator-code-gen 落实）
```

**检查点（Step 1.1）**

- [ ] `ASCEND_KERNEL_ROOT` 含 `build.sh`、`CMakeLists.txt`、`csrc/`
- [ ] `csrc/ops/<op_name>/` 已创建，含占位 `design.md`、`op_host/`、`op_kernel/`、`CMakeLists.txt` 等（以该 skill 为准）

### Step 1.2：Catlass 源码

**本步骤不对应独立 skill文件**，但必须按下列要求执行。

**前置**：Step 1.1 完成

**执行内容**

1. 在 **`ASCEND_KERNEL_ROOT`** 下确保存在 **`catlass/`**，且含 **`catlass/include`**、**`catlass/examples`**
2. 若不存在：**MUST** 在工程根执行（**禁止**在 `csrc/ops/<op_name>/` 内克隆）  
   `git clone https://gitcode.com/cann/catlass.git catlass`

**检查点（Step 1.2）**

- [ ] `<ASCEND_KERNEL_ROOT>/catlass/include` 存在
- [ ] `<ASCEND_KERNEL_ROOT>/catlass/examples` 存在

**Phase 1 全部通过 → 进入 Phase 2**

---

## Phase 2：Catlass 设计文档

**调用 Skill**：`catlass-operator-design`

### 执行内容

```
MANDATORY: 按 catlass-operator-design 执行：
1. 分析需求与 Catlass 组件边界
2. 对齐 catlass/examples 与 catlass/include 的可实现路径
3. 定稿并落盘推荐路径：csrc/ops/<op_name>/design.md（与 doc-gen / precision-eval / performance-eval 读取一致）
```

### 检查点

- [ ] `csrc/ops/<op_name>/design.md` 已定稿（非空占位）
- [ ] 写清参考 example 路径、Kernel/Host 契约、dtype/shape 约束等（以 catlass-operator-design 为准）

**全部通过 → 进入 Phase 3**

---

## Phase 3：代码生成 + 框架适配 + 编译测试

**调用 Skill**：`catlass-operator-code-gen`（阶段 5 **MUST** 调用 `ascendc-operator-compile-debug`）

### 执行内容

```
MANDATORY: 按 catlass-operator-code-gen 执行（与 ascendc-operator-code-gen 阶段结构对齐）：

阶段 1: 加载 GUIDE / references（含 compile-catlass、与 ascendc code-gen 对齐章节）
阶段 2: 读取 design.md，锁定 catlass/examples 路径与类型系统
阶段 3: 生成 op_kernel + op_host，CMake 登记 Catlass 编译选项（BUILD_CATLASS_MODULE、CATLASS_ARCH 等见 compile-catlass.md）
阶段 4: 框架适配 — ops.h、register.cpp、csrc/CMakeLists.txt
阶段 5: 编译安装与测试 — 调用 ascendc-operator-compile-debug（build.sh、pip install、tests/test_<op_name>.py，失败排错以该 skill 为准）
```

### 检查点

- [ ] `op_host`、`op_kernel` 与 `design.md`、选定 example 一致
- [ ] 框架注册与仓库模板一致（`namespace ascend_kernel` 等）
- [ ] 编译成功，whl 可安装
- [ ] `tests/test_<op_name>.py` 存在且通过（exit code 0）
- [ ] **关键编译/测试结果在聊天中有摘要展示**

**全部通过 → 进入 Phase 4**

---

## Phase 4：接口文档生成

**调用 Skill**：`ascendc-operator-doc-gen`

### 执行内容

```
MANDATORY: 按 ascendc-operator-doc-gen 执行：
- 从 register.cpp、ops.h、design.md、op_host、tests 提取接口信息
- 生成 csrc/ops/<op_name>/README.md（PyTorch 风格中文）
- 在聊天界面展示文档要点或全文
```

### 检查点

- [ ] `README.md` 已写入算子目录
- [ ] 与 `m.def` / 实际 Python 调用一致
- [ ] **已在聊天界面展示**

**全部通过 → 进入 Phase 5**

---

## Phase 5：精度评估报告

**调用 Skill**：`ascendc-operator-precision-eval`

### 执行内容

```
MANDATORY: 按 ascendc-operator-precision-eval 执行：
- 用例数 ≥ 30，覆盖 shapes × dtypes × 边界
- 输出到 csrc/ops/<op_name>/test/，生成 Markdown 精度报告
- 在聊天界面展示总览、失败摘要与关键发现（不得仅给路径）
```

### 检查点

- [ ] pytest 精度用例全部通过
- [ ] `<op_name>_precision_report.md`（或该 skill 规定的报告名）已生成
- [ ] **聊天中已展示精度结果摘要**

**FAIL 闭环**：根因分析 → 修正设计（Phase 2）或代码（Phase 3）→ 再经 Phase 4、Phase 5 复测

**全部通过 → 进入 Phase 6**

---

## Phase 6：性能评测报告

**调用 Skill**：`ascendc-operator-performance-eval`

### 执行内容

```
MANDATORY: 以 ascendc-operator-performance-eval SKILL.md 为唯一细则：
- 在 csrc/ops/<op_name>/test/ 维护 JSONL 用例；生成前先读 design.md
- 使用 torch_npu.profiler，warmup=5、active=5
- 汇总 ASCEND_PROFILER_OUTPUT/op_statistic.csv 等指标，输出自定义算子 vs 标杆的 Markdown 报告
- 在聊天界面展示对比表与简要结论
```

### 检查点

- [ ] 用例与报告形态符合该 skill（含 DType、双路径对比等）
- [ ] 报告文件已落盘于算子 `test/` 目录
- [ ] **聊天中已展示性能摘要**

**全部通过 → Catlass 算子主流程完成**

---

## 交付后可选：性能优化

**调用 Skill**：`catlass-operator-performance-optim`

**须询问用户**是否进入调优；**不得**默认跳过询问。

- 用户同意 → 按 **catlass-operator-performance-optim** 修改 tiling/实现；**凡改代码** → 从 **Phase 3** 起复跑（Phase 3→4→5→6），直至再次达标
- 用户拒绝 → 结束

---

## 阶段间数据流

```
Phase 1 输出                         Phase 2 输入
  ascend-kernel + ops/<op>/骨架       算子名、catlass/ 可引用
  + catlass/include、examples   ────▶

Phase 2 输出                         Phase 3 输入
  design.md（定稿）            ────▶  example 路径、类型与 Host 契约

Phase 3 输出                         Phase 4 输入
  已安装 whl + test_<op>.py     ────▶  register.cpp / ops.h / design.md / op_host

Phase 4 输出                         Phase 5 输入
  README.md                    ────▶  接口、dtype、约束、调用方式

Phase 5 输出                         Phase 6 输入
  精度通过 + 报告                ────▶  算子名、标杆 API、JSONL 与 profiler 流程

Phase 6 输出
  性能报告（profiler）           ────▶  可选：用户确认后进入 catlass-operator-performance-optim
```

## 状态跟踪表

| Phase | 前置条件 | 调用 Skill / 动作 | 关键产出物 |
|-------|----------|---------------------|-----------|
| 0. 需求收集 | 无 | — | CANN + Conda + `op_name`（含 catlass）+ 功能描述 |
| 1. 工程 + Catlass | Phase 0 | `ascendc-operator-project-init` + 根目录 `catlass/` | 骨架 + Catlass 源码树 |
| 2. 设计 | Phase 1 | `catlass-operator-design` | `design.md` |
| 3. 代码与测试 | Phase 2 | `catlass-operator-code-gen` → `compile-debug` | 可运行算子 + 基础测试通过 |
| 4. 接口文档 | Phase 3 | `ascendc-operator-doc-gen` | `README.md` |
| 5. 精度评估 | Phase 4 | `ascendc-operator-precision-eval` | ≥30 例 + 精度报告 |
| 6. 性能评测 | Phase 5 | `ascendc-operator-performance-eval` | JSONL + profiler 报告 |
| （可选）调优 | Phase 6 + 用户确认 | `catlass-operator-performance-optim` | 迭代后的实现与报告 |

## 错误恢复

### 从中断点恢复

当用户说「继续 Catlass 算子开发」时：

| 检测条件 | 判定阶段 | 恢复动作 |
|----------|----------|----------|
| `csrc/ops/<op_name>/` 不存在 | Phase 1 未完成 | 从 Phase 1 Step 1.1 开始 |
| `catlass/examples` 不存在 | Phase 1 未完成 | 完成 Step 1.2 克隆 |
| `design.md` 为空或占位 | Phase 2 未完成 | 从 Phase 2 开始 |
| `op_host`/`op_kernel` 仍为骨架或与 design 不符 | Phase 3 未完成 | 从 Phase 3 开始 |
| whl 未安装或 `tests/test_<op_name>.py` 失败 | Phase 3 未完成 | 在 compile-debug 流程内恢复 |
| 无 `README.md` | Phase 4 未完成 | 从 Phase 4 开始 |
| `test/` 无精度报告或精度未全过 | Phase 5 未完成 | 从 Phase 5 恢复 |
| 无性能报告或不符合 performance-eval 要求 | Phase 6 未完成 | 从 Phase 6 恢复 |

### 编译/测试失败

由 **ascendc-operator-compile-debug**（经 **catlass-operator-code-gen** 触发）处理；重试与排错上限以 **compile-debug** skill 为准。
