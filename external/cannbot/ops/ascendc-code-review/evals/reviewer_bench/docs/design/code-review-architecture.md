# CANNBot Code Review 架构设计

## 一、设计目标

1. **Skill 独立可用** — 不装 Plugin 也能完整检视。`/ascendc-code-review` 自带完整串行流程，零外部依赖。
2. **Plugin 是并行加速层** — 复用 Skill 的方法论和规则文档，只替换执行载体（子 Agent 替代串行步骤）。Skill 和 Plugin 是同一套真相源的两条执行路径。
3. **opencode / Claude Code 双平台兼容** — 所有定义基于两平台共有原语：YAML frontmatter、markdown 文件、Agent 工具、SessionStart hook。不使用平台特有功能作为核心依赖。
4. **渐进式披露** — 每个步骤独立文件，Agent 逐文件 Read，按场景名前缀隔离上下文。不在单次 Read 中加载超过当前阶段需要的内容。
5. **规则数据驱动** — 规则文档自声明适用范围（`<适用>` 头），路由子 Agent 自动匹配。新增规则无需改任何流程代码。

---

## 二、目录结构

### 2.1 Skill 层（ascendc-code-review）

```
ascendc-code-review/
│
├── SKILL.md                             入口路由（34行）
│  作用：场景触发词 → scenario 文件路由
│  读到此文件的 Agent：主 Agent（编排层或 /skill 用户）
│
├── core/
│   └── methodology.md                   检视方法论（126行）
│      作用：假设检验流程、证据框架、置信度标准、红线问题、PR交叉验证
│      读到此文件的 Agent：检视子 Agent（阶段1被dispatch后）
│
├── workflows/                           场景编排（蓝图）
│   ├── file-review.md                   文件检视（~90行）
│   ├── pr-review.md                     PR检视（~88行）
│   └── design-consistency.md            设计一致性（~74行）
│  作用：定义阶段顺序、任务清单、上下文传递链、子Agent派发规则
│  读到此文件的 Agent：主 Agent（阶段0开始前）
│
├── steps/                               执行步骤（积木，12个）
│   │
│   │── 公共模块（common.* 前缀，跨场景复用）
│   ├── common.clause-routing.md         智能条例路由（~110行）
│   │    作用：分析代码特征 → 读<适用>头 → 声明式匹配 → 条例过滤
│   │          → 跨文档同类合并 → 分组排序+侧别标签 → 波次规划
│   │    读到此文件的 Agent：路由子Agent（general类型，阶段0并行派发）
│   │    模型：haiku（Claude Code）/ 默认（opencode）
│   │
│   ├── common.line-verify.md            行号校对（14行）
│   │    作用：Grep源码验证行号 → Read确认 → 修正偏差
│   │    读到此文件的 Agent：主Agent（阶段2）
│   │
│   ├── common.report-write.md           报告生成（~65行）
│   │    作用：置信度汇总 → 证据表 → 生成报告文件
│   │    读到此文件的 Agent：主Agent（最后阶段）
│   │
│   │── 文件检视场景（file-review.* 前缀）
│   ├── file-review.code-summarize.md    代码概要（~290行）
│   │    作用：派发指令 + 子Agent执行指南（8步流程+10张输出模板）
│   │    读到此文件的 Agent：主Agent（读派发指令）+ 概要子Agent（读执行指南）
│   │
│   ├── file-review.clause-review.md     检视prompt模板（33行）
│   │    作用：检视子Agent的prompt模板（侧别+条例ID+代码路径+执行要求）
│   │    读到此文件的 Agent：主Agent（取模板填充后派发）
│   │
│   │── PR检视场景（pr-review.* 前缀）
│   ├── pr-review.code-fetch.md          代码获取（20行）
│   │    作用：执行diff脚本 + clone完整源码仓
│   │    读到此文件的 Agent：主Agent（阶段0第一步，直接执行）
│   │
│   ├── pr-review.code-summarize.md      代码概要（~260行）
│   │    作用：派发指令 + 子Agent执行指南（PR模式：diff优先+完整源码仓）
│   │    读到此文件的 Agent：主Agent（读派发指令）+ 概要子Agent（读执行指南）
│   │
│   ├── pr-review.clause-review.md       检视prompt模板（37行）
│   │    作用：检视子Agent的prompt模板（+diff路径+完整源码+代码范围）
│   │    读到此文件的 Agent：主Agent（取模板填充后派发）
│   │
│   ├── pr-review.line-verify.md         行号校对PR版（22行）
│   │    作用：diff行号→实际行号 + 越界移除
│   │    读到此文件的 Agent：主Agent（阶段2）
│   │
│   │── 设计一致性场景（design-consistency.* 前缀）
│   ├── design-consistency.code-summarize.md  代码概要+设计映射
│   │    作用：派发指令 + 执行指南（读代码+读DESIGN.md→设计映射表）
│   │
│   ├── design-consistency.clause-review.md   S1-S7策略检查
│   │    作用：架构匹配/分支覆盖/API清单/数据流/参数语义/伪代码/约束合规
│   │
│   └── design-consistency.report-write.md    设计一致性报告格式
│
├── references/                           规则文档（10个，均有<适用>头）
│   ├── cpp-secure.md                     C++安全编码规范（31条）
│   │    语言:C++ 侧别:All,Tiling 领域:false 默认启用:true
│   │    条款级侧别标记：[适用:All] / [适用:Tiling]
│   │    含专属检视策略（如SEC-2.1三阶段类型分析、SEC-2.2三阶段类型分析）
│   │
│   ├── ascendc-api.md                    AscendC API最佳实践（10条）
│   │    语言:C++ 侧别:Kernel 领域:true
│   │    触发特性：DataCopy/AllocTensor/CrossCoreSetFlag等API
│   │    含逐API的文档查阅表（检视前必须使用/ascendc-docs-search）
│   │
│   ├── ascendc-perf.md                   AscendC高性能编程（12条+1交叉引用）
│   │    语言:C++ 侧别:All 领域:true
│   │    触发特征：AscendC::, pipe.InitBuffer, DataCopy, EnQue, DeQue
│   │    每个PERF条款含独立检视方法章节
│   │
│   ├── ascendc-topk.md                   TOPK高频问题清单（13条）
│   │    语言:C++ 侧别:All,Host,Kernel 领域:false
│   │    条款级侧别标记：[适用:All] / [适用:Host] / [适用:Kernel]
│   │    含交叉引用（如TOPK-1→SEC-3.5, TOPK-6→SEC-2.3）
│   │
│   ├── cpp-general.md                    C++通用编码规范（44条）
│   │    语言:C++ 侧别:All,Tiling 领域:false
│   │    条款级侧别标记：[适用:All] / [适用:Tiling] / [不适用]
│   │
│   ├── compile-secure.md                 安全编译选项（7条）
│   │    语言:Build 侧别:Tiling 领域:false
│   │    代码类型由文件扩展名决定（CMakeLists.txt/.cmake/Makefile）
│   │
│   ├── cpp-style.md                      C++代码风格（19条）
│   │    语言:C++ 侧别:All 领域:false 默认启用:false
│   │    唯一默认不启用的规则文件
│   │
│   ├── python-secure.md                  Python安全编码（28条）
│   │    语言:Python 侧别:N/A 领域:false
│   │
│   ├── simt-api-analysis.md              SIMT API C风格化（16条）
│   │    语言:C++ 侧别:Kernel 领域:true
│   │    触发：Simt::GetThreadNum/GetBlockIdx/VF_CALL/UintDiv等+simt_api/asc_simt.h
│   │    含逐API的头文件查阅表（检视前必须查阅asc-devkit头文件）
│   │
│   └── mc2-specific.md                   MC²领域规则（19条）
│       语言:C++ 侧别:Host,Kernel 领域:true
│       触发：hccl_/AllGather/SyncAll/expert/quant/AlltoAllV等
│       排除场景：纯通信无计算融合 / 纯计算无集合通信
│       含领域判定规则（C1/C2核心特征）+ PR差异→规则速查表
│
├── scripts/
│   ├── get_gitcode_pr_diff.py            GitCode PR diff获取
│   └── clone_pr_source.py                PR完整源码克隆
│
└── (style/ 目录已移除 — 报告格式统一在report-write内联)
```

### 2.2 Plugin 层（ops-code-reviewer）

```
ops-code-reviewer/
│
├── AGENTS.md                             入口触发（25行）
│  作用："加载ascendc-code-review skill，走scenario路由"
│  skill内的steps自动探测子Agent可用性实现并行/串行切换
│
├── agents/                               子Agent定义（薄壳）
│   ├── ascendc-code-summarizer.md        代码概要子Agent（48行）
│   │   输入输出契约 + 模式说明 + 指向steps/{scenario}.code-summarize.md
│   │   skills: ascendc-code-review（加载后获得路径解析）
│   │
│   └── ascendc-ops-reviewer.md           检视子Agent（85行）
│       输入输出契约 + 5步执行链：学方法论→取上下文→查API→提条例→假设检验
│       skills: ascendc-code-review（加载后获得路径解析）
│
├── hooks/
│   ├── hooks.json                        SessionStart hook配置
│   ├── run-hook.cmd                      Hook执行入口
│   └── session-start-reviewer            注入AGENTS.md到会话上下文
│
└── init.sh                               多平台安装脚本（656行）
    支持: opencode / claude / trae / cursor
    安装: skills软链接 + agents软链接 + AGENTS.md/CLAUDE.md
```

### 2.3 文件统计

| 目录 | 文件数 | 总行数（约） | 职责 |
|------|--------|-------------|------|
| core/ | 1 | 126 | 检视方法论（假设检验+证据框架+红线） |
| workflows/ | 3 | 252 | 场景编排蓝图 |
| steps/ | 12 | ~1200 | 执行步骤（派发指令+执行指南+prompt模板） |
| references/ | 10 | ~3500 | 规则知识库（均有<适用>头，共199条条例） |
| agents/ | 2 | 133 | 子Agent薄壳（输入输出契约+skill引用） |

---

## 三、场景流程详解

### 3.1 场景 A：文件检视

#### 触发
"检视代码"、"审核代码"、"检查规范"、"代码审查"、"帮我检视 xxx"

#### 任务清单
4 个固定任务（全部 pending）：
- 任务0：代码概要 + 条例分组（并行派发）
- 任务1：逐条检视（按波次派发检视子Agent）
- 任务2：行号校对
- 任务3：撰写报告

#### 阶段 0：代码概要 + 条例分组（并行）

**步骤 0.1 — 预处理**
1. 将任务0标记为 in_progress
2. 从代码文件路径提取算子名：取 `op_kernel/` 或 `op_host/` 的父目录名
3. 确认文件存在且可读

**步骤 0.2 — 并行派发**
在单个消息中同时派发两个子Agent：

子Agent A（代码概要）：
```
Agent({
  subagent_type: "ascendc-code-summarizer" 或 "general",
  description: "代码概要：梳理代码脉络",
  prompt: "代码概要生成（文件检视模式）

【输入】
- 代码文件路径：{code_file_path}
- 概要输出路径：{code_summary_output_path}

【执行要求】
1. 严格按本文件「子Agent执行指南」定义的8步流程执行
2. 概要写入输出路径，禁止跳过侧别识别
3. 返回结构化结果（含侧别识别）"
})
```

子Agent B（条例路由）：
```
Agent({
  subagent_type: "general",
  model: "haiku",
  description: "智能条例路由",
  prompt: "智能条例路由

【输入】
- 代码文件路径：{code_file_path}

【执行流程】
Step 1 — 分析代码特征（语言+侧别+领域关键词grep）
Step 2 — 读取所有references/*.md的<适用>头，声明式匹配
Step 3 — 条例级侧别过滤
Step 4 — 跨文档同类合并
Step 5 — 分组+排序+侧别标签
Step 6 — 波次计算"
})
```

**步骤 0.3 — 收集结果**
- 子Agent A → 侧别 + 概要路径（含6张索引表）
- 子Agent B → 分组规划表（波次、每组条例ID列表、每组代码范围标签）
- 将任务0标记为 done

#### 阶段 1：逐条检视

**步骤 1.1 — 准备**
1. 将任务1标记为 in_progress
2. Read `steps/file-review.clause-review.md` 获取 prompt 模板

**步骤 1.2 — 逐波派发**
按阶段0的分组规划表，逐波派发。每波在单个消息中并行调用 ≤10 个 Agent 工具：

```
每组的 prompt（由模板填充）：
【已由上游完成】
- 代码侧别识别：{Kernel侧/Tiling侧}
- 条款过滤：已按侧别过滤，保留以下条款
- 代码概要：{code_summary_path}

检视文件：{code_file_path}

检视条款：{条例ID-1} {条例标题}、{条例ID-2} {条例标题}

【执行要求】
- 第一步加载 ascendc-code-review skill，然后 Read skill 目录下的
  core/methodology.md 掌握假设检验方法、置信度标准和红线问题
- 若提供了代码概要，Read 获取全局视角
  （重点关注「API调用索引」和「跨文件防御摘要」）
- 对每条分配的条例，Read 对应规则文档的完整内容，
  重点关注其中的「检视策略」「检视方法」章节。
  若条例包含专属检视方法或强制要求，必须严格按该指引执行
- 若检视条款来自 ascendc-api / ascendc-perf / simt-api-analysis / mc2-specific，
  必须先使用 /ascendc-docs-search skill 查阅对应API的最新官方文档
- 严格按假设检验驱动流程执行（H0/H1、证据收集、自信值计算）
- 所有条款检视完成后直接输出逐条结果，禁止生成报告文件
```

子Agent类型优先级：`ascendc-ops-reviewer` → `general`（兜底）

每波完成后输出进度：
```
✅ 波次1 完成：组1-10 返回
   PASS: X 条 | FAIL: Y 条 | SUSPICIOUS: Z 条
```

**步骤 1.3 — 汇总**
所有波次完成后，汇总全部结果，将任务1标记为 done。

#### 阶段 2：行号校对

1. 将任务2标记为 in_progress
2. Read + 执行 `steps/common.line-verify.md`
3. 对所有 FAIL/SUSPICIOUS 发现：Grep 源文件 → Read 行号范围 → 验证代码匹配 → 修正偏差
4. 将任务2标记为 done

#### 阶段 3：撰写报告

1. 将任务3标记为 in_progress
2. Read + 执行 `steps/common.report-write.md`
3. 按置信度汇总（HIGH/MED/LOW），每个发现附假设检验证据表
4. 输出到 `./operators/{operator_name}/{source_file}_review_summary.md`
5. 将任务3标记为 done

#### 上下文传递链

```
                 ┌─ code-summarize → 侧别 + 概要路径
阶段0（并行） ───┤
                 └─ clause-routing → 分组规划表（波次+条例ID+侧别标签）
                         ↓
阶段1 → 逐条结果 (PASS/FAIL/SUSPICIOUS)
         ↓
阶段2 → 校对后的 FAIL/SUSPICIOUS
         ↓
阶段3 → 报告文件
```

---

### 3.2 场景 B：PR 检视

#### 触发
"检视 PR"、"审核 PR"、"帮我检视这个 PR"

#### 任务清单
4 个固定任务：
- 任务0：获取 diff + 代码概要 + 条例分组（code-fetch → 并行派发）
- 任务1：逐条检视
- 任务2：行号校对（PR版）
- 任务3：撰写报告

#### 阶段 0：获取 diff + 代码概要 + 条例分组

**步骤 0.1 — 获取代码**
1. 将任务0标记为 in_progress
2. 提取 PR 链接，判断托管平台（URL 含 `gitcode.com` → GitCode）
3. Read + 执行 `steps/pr-review.code-fetch.md`：
   - 定位 diff 脚本 → 执行获取 diff → 保存到 `./operators/.pr_diff/{pr_number}.diff`
   - clone 完整源码到 `./operators/.pr_repo/{pr_number}/`
   - 克隆失败则终止流程

**步骤 0.2 — 并行派发**
在单个消息中同时派发两个子Agent：

子Agent A（代码概要 — PR模式）：
```
传入：diff路径 + 完整源码路径 + 概要输出路径
模式：PR检视
→ 先Read diff了解变更范围
→ Read变更文件的完整源码
→ 对diff中的类成员变量，grep完整源码追溯声明→初始化→校验链
→ 输出含「变更文件概览」「逐文件分析」「跨文件关联」
→ 函数清单/API索引/常量清单标注「是否本次变更」
```

子Agent B（条例路由 — 同文件检视，增加diff输入）

**步骤 0.3 — 收集结果**
- 子Agent A → 侧别 + 概要路径（PR模式，含完整源码仓跨文件分析）
- 子Agent B → 分组规划表（每组含侧别标签：仅Kernel/仅Tiling/全部）
- 将任务0标记为 done

#### 阶段 1：逐条检视

流程同文件检视，差异：

1. prompt 模板额外传入：
   - `diff_file_path` — PR diff 文件路径
   - `repo_path` — 完整源码路径（变更文件的完整内容，用于确认变量来源/上游校验）
   - `检视代码范围` — 使用 routing 输出的侧别标签（仅Kernel/仅Tiling/全部）

2. 执行要求增加：
   - "先 Read diff 了解变更范围，再 Read 完整源码追溯变量来源"
   - methodology.md 中的 PR 交叉验证规则自动生效

3. 代码范围隔离：
   - 全 `[适用: Kernel]` 的组 → 只检视 `op_kernel/` 下文件
   - 全 `[适用: Tiling]/[适用: Host]` 的组 → 只检视 `op_host/` 下文件
   - 混合或含 `[适用: All]` → 检视全部变更文件

#### 阶段 2：行号校对（PR版）

Read + 执行 `steps/pr-review.line-verify.md`：
- diff 内行号 → Grep 完整源码 → 实际文件行号
- 报告中统一使用实际行号
- 校验行号是否在 diff 变更范围内
- 不在范围内的发现 → 判定为越界，移出报告

#### 阶段 3：撰写报告

同文件检视，输出路径为 `./operators/pr-{pr_number}/{pr_number}_review_summary.md`

#### 与文件检视的关键差异

| 差异点 | 文件检视 | PR检视 |
|--------|---------|--------|
| 阶段0多一步 | — | code-fetch获取diff+clone源码 |
| summarize输入 | 单文件路径 | diff + 完整源码仓 |
| summarize输出 | 单文件模板 | 多文件模板（变更概览+逐文件+跨文件关联+变更?列） |
| routing输入 | 代码文件路径 | +diff文件路径 |
| clause-review prompt | 文件路径 | +diff路径+完整源码路径+代码范围 |
| methodology | 标准流程 | +PR交叉验证规则 |
| line-verify | 源文件Grep | +diff行号→实际行号+越界移除 |
| 报告路径 | operators/{op}/ | operators/pr-{pr}/ |

---

### 3.3 场景 C：设计一致性

#### 触发
"设计实现一致性"、"设计一致性检查"、"对照 DESIGN.md"、"验证设计实现"

#### 任务清单
4 个固定任务（不含条例提取）：
- 任务0：获取代码+概要（含设计映射）
- 任务1：跳过条例提取
- 任务2：设计一致性检查（S1-S7）
- 任务3：行号校对
- 任务4：撰写设计一致性报告

#### 阶段 0：获取代码 + 概要（含设计映射）

1. 将任务0标记为 in_progress
2. 从代码文件路径提取算子名，确认代码文件 + DESIGN.md 存在
3. Read + 执行 `steps/design-consistency.code-summarize.md`：
   - 读代码 → 梳理脉络 → 读DESIGN.md → 提取设计要素 → Grep定位实现 → 对比判定
   - 概要末尾包含「设计映射」表（设计要素 | 设计描述 | 实现位置 | ✅/❌/⚠️/N/A）
4. 将任务0标记为 done

#### 阶段 1：跳过条例提取
将任务1直接标记为 done。不读取规范文档。

#### 阶段 2：设计一致性检查（S1-S7）

Read + 执行 `steps/design-consistency.clause-review.md`：

派发子Agent（general类型）执行7大策略：

| 策略 | 维度 | 检查内容 |
|------|------|---------|
| S1 | 架构匹配 | Kernel类型、硬件单元、流水线模式、存储层级是否与设计一致 |
| S2 | 分支覆盖 | 设计中的每个条件分支是否在代码中有对应实现 |
| S3 | API清单 | 设计的API映射是否使用正确API，是否使用黑名单接口 |
| S4 | 数据流追踪 | 数据输入到输出的完整路径是否与设计一致 |
| S5 | 参数语义 | 关键参数的含义和使用是否与设计一致 |
| S6 | 伪代码映射 | 设计伪代码的逻辑是否在代码中得到体现 |
| S7 | 约束合规 | 设计约束条件（对齐、取值范围、内存限制）是否被满足 |

#### 阶段 3-4：行号校对 + 撰写报告
- 行号校对：`steps/common.line-verify.md`
- 报告输出：`steps/design-consistency.report-write.md` → S1-S7判定表 + 总体评级
- 输出路径：`./operators/{operator_name}/{source_file}_design_consistency_review.md`

---

## 四、智能条例路由详解

路由子 Agent 是 `general` 类型，与 code-summarize 并行执行。使用 `model: "haiku"`（Claude Code）/ 默认模型（opencode，忽略 model 参数）。

### 4.1 路由算法

**Step 1 — 分析代码特征**

```
1.1 从文件路径判断：
  - .cpp/.h/.hpp → C++
  - .py → Python
  - CMakeLists.txt/.cmake/Makefile → Build
  - op_kernel/ → Kernel, op_host/ → Tiling

1.2 读取所有 references/*.md 的 <适用> 头，收集领域规则的触发关键词

1.3 用收集到的关键词 Grep 代码，记录命中情况
```

**Step 2 — 声明式匹配**

遍历 references/ 下所有 .md，读 `<适用>` 头：

| 匹配维度 | 规则 | 不通过处理 |
|---------|------|-----------|
| 语言 | 规则语言 = 代码语言 或 不限 | 跳过 |
| 侧别 | 规则侧别 = All 或 包含代码侧别 | 跳过 |
| 默认启用 | false → 跳过 | 跳过 |
| 领域规则 | 领域=false → 通过；领域=true且命中触发词 → 通过；未命中 → 跳过 | 跳过 |
| 排除场景 | 读<适用>头的排除场景字段动态判断 | 跳过 |

**Step 3 — 条例级侧别过滤**

读匹配文件的快速索引表：
- Kernel侧：保留 `[适用: All]` + `[适用: Kernel]`
- Tiling侧：保留 `[适用: All]` + `[适用: Tiling]`/`[适用: Host]`
- 混合侧别：保留全部，标注每条条例的适用侧别

**Step 4 — 跨文档同类合并**

按快速索引的「类别」字段（数值安全、内存安全、输入验证、API使用等）跨文档合并。常见合并示例：
- cpp-secure 数值安全 ↔ ascendc-topk 数值安全（除零、溢出）
- cpp-secure 内存安全 ↔ ascendc-topk 内存安全（野指针、数组越界）
- ascendc-api API使用 ↔ ascendc-perf 数据搬运（DataCopy相关）

**Step 5 — 分组 + 排序 + 侧别标签**

每组打上代码范围标签：
- 全部 `[适用: Kernel]` → `[仅Kernel]`：子Agent只检视 op_kernel/ 下文件
- 全部 `[适用: Tiling]/[适用: Host]` → `[仅Tiling]`：子Agent只检视 op_host/ 下文件
- 含 `[适用: All]` 或混合 → `[全部]`：子Agent检视全部文件

派发优先级（高危优先）：
1. 数值安全 + 内存安全 + 输入验证
2. API使用 + 数据搬运
3. 领域规则（simt-api-analysis / mc2-specific）
4. 性能（ascendc-perf 剩余）
5. 通用规范（cpp-general / compile-secure）
6. Python（python-secure）

**Step 6 — 波次计算**

总组数 ÷ 10，向上取整。每波 ≤10 组。

### 4.2 路由输出格式

```
📊 路由规划
代码语言: {C++/Python/Build/混合}
代码侧别: {Kernel/Tiling/混合}
领域关键词命中: {命中列表}
匹配规则文件: {文件列表}（共 N 条条例）
跳过文件: {文件}: {跳过原因}
同类合并: {X 组跨文档合并}

波次1（优先）:
  {组1（来源, 侧别标签）}: 条例ID {标题}, ...
     代码范围: {仅Kernel / 仅Tiling / 全部}
  {组2}: ...
     ...

波次2（如有）:
  ...

共 G 组，分 W 波。场景按每组的「代码范围」标签设置检视代码范围。
```

---

## 五、检视子 Agent 执行链

### 5.1 渐进式加载顺序

每个检视子 Agent 被 dispatch 后，按以下顺序逐步加载知识：

```
┌─ prompt 模板（33行）
│   条例ID列表 + 代码路径 + 执行要求
│
├─ [1] 加载 ascendc-code-review skill
│       → 获得 skill 路径 → 可解析 core/、references/ 相对路径
│
├─ [2] Read core/methodology.md（126行）
│       → 假设检验5步（代码段识别→假设建立→证据收集→校验→决策）
│       → 6正向证据 + 3负向证据
│       → 置信度表（HIGH≥80%, MED 70-80%, LOW<70%）
│       → 条款边界检查（禁止万能筐）
│       → 红线问题（Host侧6条 + Kernel侧5条SIMT规则）
│       → PR交叉验证矩阵（变量作除数/未初始化/外部输入）
│
├─ [3] Read 代码概要（code_summary.md）
│       → 6张索引表：函数清单 / API调用索引 / 常量清单
│                     变量溯源 / 跨文件防御摘要 / 代码关联
│
├─ [4] 逐条例 Read 对应规则文档完整内容
│       → 条款描述、错误示例、正确示例、注意事项
│       → 重点关注：「检视策略」「检视方法」章节
│       → 若条例含专属检视方法或强制要求，严格按该指引执行
│
├─ [5] API类条例：/ascendc-docs-search
│       查阅对应API的最新官方文档
│       确认参数限制、对齐要求、转换规则等
│
└─ [6] 逐条例执行假设检验 → 输出结构化结果
```

### 5.2 子Agent类型降级

```
ascendc-ops-reviewer（首选）
  → 自带skill，有完整5步执行链定义
  → 不可用时 ↓
general（兜底）
  → 无内置流程，完全依赖prompt模板中的执行要求
```

---

## 六、证据框架

### 6.1 正向证据（6种，增加置信度）

| 证据类型 | 分值 | 分析动作 |
|---------|------|---------|
| 规范违反 | +40% | 对照规范条款识别违规点 |
| 上下文防御缺失 | +30% | 检查作用域内是否有防御代码 |
| PR 归属 | +20% | 仅PR模式：问题代码在diff变更范围内 |
| 调用链风险 | +15% | LSP/Grep 分析调用函数内部逻辑 |
| 数据流风险 | +15% | 变量来源不明确、运算路径不可控 |
| 领域关联 | +10% | 代码命中规则文件的领域触发特征 |

### 6.2 负向证据（3种，降低置信度）

| 证据类型 | 分值 | 分析动作 |
|---------|------|---------|
| 防御存在 | -40% | 变量来源为硬件配置/编译期常量/TilingData已校验 |
| 上游校验 | -30% | 作用域外已有 OP_CHECK_IF / assert 等防护 |
| 范围外 | -50% | 仅PR模式：问题代码不在diff变更范围内 |

### 6.3 决策规则

```
步骤4 — 证据有效性校验：
  - 负向证据累加 ≤ -50% → 风险降级，不进入FAIL
  - 上下文可证明不可能触发风险 → 排除

步骤5 — 决策判断：
  - 自信值 = Σ正向证据分值 + Σ负向证据分值（负向为负数）
  - 自信值 ≥ 70% → 判定违规
```

### 6.4 置信度分级

| 等级 | 区间 | 分类 |
|------|------|------|
| HIGH | ≥80% | 发现问题 |
| MED | 70-80% | 需关注 |
| LOW | <70% | 疑似（不判定违规） |

### 6.5 分析要求

- 使用 LSP 获取符号定义，使用 Grep 查找依赖关系
- 风险代码必须检查是否在当前文件作用域内其他位置进行防御
- 遇到函数调用，必须查看函数内部逻辑并综合判断
- 利用 code-summarize 的变量溯源结果：来源类型为硬件配置/编译期常量/TilingData → 应用「防御存在」
- Kernel 代码涉及 API 用法时，使用 `/ascendc-docs-search` 查阅官方文档，禁止凭记忆或推测
- 若条例包含专属检视方法或强制要求，必须严格按该条例指引执行

---

## 七、代码概要输出规范

子 Agent 生成的 `code_summary.md` 包含以下结构化内容：

### 7.1 文件检视模式

```markdown
# 代码概要
算子: {name} | 功能: {实现目标} | 侧别: {Kernel/Tiling}

## 代码脉络
入口 → 数据流 → 计算核心 → 输出（含分支覆盖表）

## 变量溯源
| 变量 | 声明(文件:行) | 初始化(文件:行) | 校验(文件:行) | 来源类型 |

## 函数清单
| 函数 | 签名 | 行范围 | 角色 |

## API 调用索引
| API | 行号 | 上下文 |

## 常量清单
| 常量 | 值 | 位置(行) | 用途 |

## 跨文件防御摘要
| 关联文件 | 关键发现 | 位置(文件:行) | 影响范围 |

## 代码关联
上游文件 / 下游文件 / 高性能设计（仅Kernel侧）
```

### 7.2 PR 检视模式

```markdown
# PR 代码概要
PR: #{number} | 算子: {name} | 侧别: {Kernel/Tiling/混合}

## 变更文件概览
| # | 文件 | 侧别 | 变更类型 | 说明 |

## 文件1: {path}
变更概要 → 代码脉络 → 分支覆盖 → 变量溯源

## 函数清单（全变更文件汇总）
| 函数 | 签名 | 行范围 | 所属文件 | 角色 | 变更? |

## API 调用索引（全变更文件汇总）
| API | 行号 | 所属文件 | 上下文 | 变更? |

## 常量清单（全变更文件汇总）
| 常量 | 值 | 位置(文件:行) | 用途 | 变更? |

## 跨文件防御摘要
| 关联文件 | 关键发现 | 位置(文件:行) | 变更文件? | 影响范围 |

## 跨文件关联
Tiling→Kernel数据流 / 同步标志常量关联 / 模板派发链 / Include调用链
```

---

## 八、报告格式

### 8.1 标准报告（文件检视 / PR 检视）

```
# 代码检视报告

## 检视概览
- 代码文件 / 代码侧别 / 检视文档 / 总条例数 / 检视时间

## 检视统计
| 状态 | 条例数 | 占比 |
PASS / FAIL（发现问题）/ SUSPICIOUS（需关注）

## 发现问题（HIGH 置信度）
### [{条例ID}] {标题}
- 问题描述
- 代码片段（至少10行，标注行号）
- 假设检验证据：
  正向证据：
  | 证据类型 | 分值 | 证据描述 |
  负向证据：
  | 证据类型 | 分值 | 证据描述 |
  自信值 = Σ正向 + Σ负向 = {累计}% ≥ 70% → 判定违规
- 修复建议

## 需关注（MED 置信度）
（同上格式）

## 疑似（LOW 置信度）
（同上格式）

## 通过条例
{ID列表}
```

### 8.2 设计一致性报告

```
# 设计一致性检查报告

## 检视概览
代码文件 / 设计文档 / 检视时间

## 设计一致性检查
| 策略 | 维度 | 设计期望 | 实现实际 | 判定 |
S1-S7 × 4列

总体评级: 一致 / 部分一致 / 不一致
```

---

## 九、规则文档 `<适用>` 声明规范

### 9.1 格式

```markdown
<适用>
语言: {C++/Python/Build/不限}
侧别: {All/Kernel/Tiling/Host/N/A, 逗号分隔}
领域: {true/false}
触发: {领域规则的关键词列表，逗号分隔，领域=false时填—}
默认启用: {true/false}
排除场景: {可选，领域规则的排除条件，如MC2的纯通信/纯计算排除}
</适用>
```

### 9.2 各文件声明清单

| 文件 | 语言 | 侧别 | 领域 | 触发 | 默认启用 |
|------|------|------|------|------|---------|
| cpp-secure | C++ | All, Tiling | false | — | true |
| ascendc-api | C++ | Kernel | true | AscendC::, pipe.InitBuffer, DataCopy... | true |
| ascendc-perf | C++ | All | true | AscendC::, pipe.InitBuffer, DataCopy, EnQue... | true |
| ascendc-topk | C++ | All, Host, Kernel | false | — | true |
| cpp-general | C++ | All, Tiling | false | — | true |
| compile-secure | Build | Tiling | false | — | true |
| cpp-style | C++ | All | false | — | **false** |
| python-secure | Python | N/A | false | — | true |
| simt-api-analysis | C++ | Kernel | true | Simt::GetThreadNum, Simt::GetBlockIdx... | true |
| mc2-specific | C++ | Host, Kernel | true | hccl_, AllGather, SyncAll, expert, quant... | true |

### 9.3 新增规则示例

创建一个 `references/cuda-secure.md`：

```markdown
# CUDA 安全编码规范

<适用>
语言: C++
侧别: Kernel
领域: true
触发: cudaMalloc, cudaFree, cudaMemcpy, __global__, <<<>>>
默认启用: true
</适用>

## 快速索引
| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---------|---------|------|---------|---------|
| CUDA-1 | cudaMalloc返回值校验 | 内存安全 | 高 | [适用: Kernel] |
| CUDA-2 | 核函数启动配置校验 | 并发安全 | 高 | [适用: Kernel] |

## CUDA-1: cudaMalloc返回值校验
（详细规范内容...）
```

路由子 Agent 自动感知：
- Step 1.2：收集到触发关键词 `cudaMalloc, cudaFree, cudaMemcpy, __global__`
- Step 2：代码命中关键词 → 纳入匹配
- Step 3-6：按标准流程过滤、分组、排序

**零流程代码修改。**

---

## 十、渐进式披露原则

### 10.1 六条原则

1. **场景路由先行** — Agent 先看到编排蓝图（阶段数、任务清单、上下文传递），再逐阶段展开步骤细节
2. **步骤按场景隔离** — `file-review.*` 和 `pr-review.*` 各自独立命名前缀，不会交叉加载
3. **方法论延迟加载** — 检视子Agent被dispatch后才Read methodology.md（126行），编排Agent从不接触
4. **规则按需读取** — 路由子Agent只读 `<适用>` 头（~5行/文件），检视子Agent读取分配的条例详情（~30行/条例）
5. **场景负责派发，step提供模板** — scenario是唯一的dispatch编排者，step文件不含Agent()调用逻辑
6. **公共模块 common.* 前缀** — clause-routing、line-verify、report-write跨场景复用，语义明确

### 10.2 每层上下文预算

| 层 | 文件 | 行数 | Agent读入的上下文 |
|-----|------|------|-----------------|
| 入口 | SKILL.md | 34 | 触发词→场景路由表 |
| 蓝图 | workflows/xxx.md | ~90 | 4阶段任务清单+上下文传递链 |
| 派发 | steps/xxx.code-summarize.md | ~30（派发部分） | 子Agent调用参数 |
| 执行 | steps/xxx.code-summarize.md | ~260（执行指南） | 仅子Agent加载，不污染编排Agent |
| 模板 | steps/xxx.clause-review.md | ~35 | prompt模板，无dispatch逻辑 |
| 方法论 | core/methodology.md | 126 | 仅检视子Agent加载 |
| 规则 | references/xxx.md | 按需 | 路由子Agent读<适用>头，检视子Agent读条例详情 |

---

## 十一、子 Agent 类型与降级策略

| 步骤 | 首选 subagent_type | 兜底 | 模型 | 说明 |
|------|-------------------|------|------|------|
| code-summarize | ascendc-code-summarizer | general | 默认 | 首选有skill上下文可解析路径 |
| clause-routing | general（直接指定） | — | haiku | 路由任务轻量，haiku够用 |
| clause-review | ascendc-ops-reviewer | general | 默认 | 首选有完整5步执行链定义 |
| design-consistency | general（直接指定） | — | 默认 | 设计一致性检查无专用子Agent |

降级是自动的——直接尝试调用，失败则用下一级。不通过文件系统Glob检查来判断可用性。

---

## 十二、扩展接口

| 想做什么 | 创建/修改 | 不改什么 |
|---------|----------|---------|
| 加新编码规范 | `references/新规范.md`（含`<适用>`头和快速索引） | 不改任何routing/step/scenario |
| 加新检视步骤 | `steps/{scenario}.新步骤.md` | 不改已有step |
| 加新检视场景 | `workflows/新场景.md`（编排已有step） | 不改step和references |
| 加新子Agent类型 | `agents/新agent.md`（薄壳） | 不改skill |
| 改检视方法论 | `core/methodology.md` | 不改step |
| 改路由算法 | `steps/common.clause-routing.md` | 不改references |

---

## 十三、双平台兼容性

| 维度 | opencode | Claude Code | 处理方式 |
|------|----------|-------------|---------|
| 配置入口 | AGENTS.md | CLAUDE.md | init.sh自动选择 |
| 配置目录 | .opencode/ | .claude/ | init.sh自动选择 |
| Skill发现 | skills/目录 | skills/目录 | 相同 |
| Agent发现 | agents/目录 | agents/目录 | 相同 |
| Agent模型参数 | 忽略model参数 | 支持model: "haiku" | 直接写，opencode静默忽略 |
| 子Agent dispatch | Agent({subagent_type}) | Agent({subagent_type}) | 相同 |
| 所有权声明 | external_directory | permission | 相同语义 |
| SessionStart hook | 支持 | 支持 | 相同 |

---

## 十四、设计决策记录

1. **为什么不用 frontmatter 做 `<适用>` 声明** — frontmatter 在两个平台都可能被解析为 agent/skill 元数据，`<适用>` 自定义标签不会
2. **为什么 routing 用 general 而非专用子Agent** — 路由任务轻量（grep + 读头 + 匹配表），不值得维护专用子Agent。haiku省成本
3. **为什么保留两个 code-summarize 文件** — 虽然~80%内容相似，但分场景隔离避免了运行时交叉加载。维护成本可接受
4. **为什么 methodology 放 core/ 而非 references/** — references/ 是纯规则文档（都有 `<适用>` 头），methodology 是执行框架，语义不同
5. **为什么 cpp-style 默认不启用** — 命名/格式/注释类问题 clang-tidy/clang-format 可自动覆盖，只在用户显式要求时纳入
6. **为什么决策阈值是 70%** — 高于此值需要正向证据（规范违反+40% 或防御缺失+30%）配合其他正向证据才能触发，避免低质量告警
