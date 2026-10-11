# 文件检视代码概要

## 目的

派发一个子 Agent 读取 `file_input`，生成供 `plan-design` 和后续逐条检视共同使用的事实概要。

本步骤只整理代码或文档中能够直接确认的事实，不判定条例是否通过，不生成问题报告，也不修改待检视文件。

## 输入

- 待检视文件：`{file_input}`
- 概要输出路径：`{code_summary_output_path}`

## 派发要求

将上述输入传给代码概要子 Agent，并要求其完整执行本文件的“子 Agent 执行指南”。概要必须写入指定路径；子 Agent 完成后只需返回文件类型、代码侧别、算子名、功能概述和概要路径。

---

## 子 Agent 执行指南

以下步骤由代码概要子 Agent 执行。

### 1. 明确检视范围

枚举 `file_input` 中由用户明确指定的文件，形成“正式检视文件”列表。

允许读取下列仓库内关联文件以理解上下文：

- 被正式检视文件直接导入的公共模块；
- 可明确匹配的 `*_asc.py`、`*_kernel.py` 或公开 wrapper；
- 直接调用目标入口的仓库内代码；
- 与目标算子直接对应的测试文件。

关联文件只作为上下文，不得自动加入正式检视范围。概要中必须区分“正式检视文件”和“上下文文件”，后续条例只能对正式检视文件下结论。

### 2. 识别文件类型与代码侧别

逐文件识别，不得只根据文件名判断；路径和命名只能作为辅助证据。

| 文件类型 | 识别依据 | 代码侧别 |
|---|---|---|
| TileLang Kernel | 出现 `@tilelang.jit`、`T.Kernel`、`T.*`、`S.*`、`@T.prim_func` 或 `@T.macro` 等 TileLang DSL 特征 | Kernel |
| Python Host | 参数校验、输出分配、后端选择、Kernel 构建或调用、测试和 benchmark 等普通 Python 逻辑 | Host |
| 混合 Python | 同一检视范围同时包含 TileLang Kernel 和 Host 调用逻辑，或单文件同时承担两类职责 | 混合 |
| Markdown | `.md` 文档 | N/A |

多文件输入时，在概要中记录每个文件的类型和侧别，再给出本次输入的整体结论：

- 仅 Kernel 文件：整体侧别为 `Kernel`；
- 仅 Host 文件：整体侧别为 `Host`；
- 同时存在 Kernel 和 Host：整体侧别为 `混合`；
- 仅 Markdown：整体侧别为 `N/A`；
- 代码与 Markdown 混合：代码侧别按代码文件判定，Markdown 单独标记为 `N/A`。

### 3. 梳理 Python 与 TileLang 代码

仅对正式检视范围中的 Python 或 TileLang 文件执行本节。

#### 3.1 入口和调用关系

确认：

- 用户可调用的公开入口；
- Kernel builder 或编译入口；
- Host 到 Kernel 的调用关系；
- 关键辅助函数；
- 测试或上层 wrapper 如何调用目标代码。

使用 `rg` 在正式检视文件和必要的上下文文件中定位定义与调用位置。无法确认调用者时标记“未确认”，不得把函数名推测为公开入口。

#### 3.2 数据流和计算语义

按以下顺序梳理数据流：

```text
Host 输入和配置 → Kernel 构建参数 → Kernel 输入 Buffer → 搬运/计算 → 输出 Buffer → Host 返回值
```

记录：

- 输入输出数量、shape、dtype、stride 和业务角色；
- 主循环及其迭代含义；
- 主要数学运算；
- 输出写回位置；
- 尾块、mask、动态边界和空输入处理；
- 影响数据流的条件分支。

数学公式和业务含义必须有代码、测试或调用关系支撑。只有命名线索而无实现证据时，标记为“推测，待确认”。

#### 3.3 参数来源和防御

追踪会影响索引、容量、精度或分派的关键值：

- shape、dtype 和 stride；
- tile/block 大小和核数；
- 循环边界和索引上界；
- `num_stages`、Buffer 数量和内存层级；
- 算法模式、量化配置及其他编译期参数；
- 运行期标量和 `T.dynamic` 值。

对每个关键值记录定义、传递、实际校验和最终使用位置。值来自 Host、配置或硬件查询不代表已经校验；只有找到明确的断言、异常、条件分支或调用约束时，才记录为已防御。

#### 3.4 TileLang API 和执行结构索引

记录正式检视文件中实际出现的：

- `@tilelang.jit`、`@T.prim_func`、`@T.macro` 等装饰器；
- `T.Kernel`、`T.serial`、`T.Parallel`、`T.Pipelined`；
- `T.alloc_*`、`T.copy`、`T.gemm`、归约和原子操作；
- `T.call_extern`、同步操作和 `S.*`；
- 核数、线程或程序块映射；
- Buffer 的 shape、dtype、内存层级和生命周期；
- 流水阶段及显式数据依赖。

本步骤只建立 API 调用索引和代码事实，不调查 API 的外部语义，也不替代后续条例检视。

#### 3.5 性能结构事实

仅记录代码能够直接证明的结构：

- tile 或 block 如何切分；
- 单核或单程序实例处理范围；
- 是否存在多阶段流水、多级 Buffer、SIMD、SIMT、GEMM 或 persistent 循环；
- 各 Buffer 可由静态表达式计算出的容量；
- 尾块是否增加额外工作量。

没有 benchmark、profiling 或明确静态证据时，不得下“性能较差”“性能更优”等结论。

### 4. 梳理 Markdown 文档

仅对正式检视范围中的 Markdown 文件执行本节。Markdown 的代码侧别固定为 `N/A`，不生成 Kernel、Host、Buffer 或流水线分析。

记录：

- 文档用途及主要章节；
- 标题层级；
- 代码围栏和命令示例；
- 文件路径、相对链接和锚点；
- API、变量、状态名和强制级别术语；
- 对其他 `SKILL.md`、`workflows/`、`steps/`、`references/` 或脚本的引用；
- 多个 Markdown 文件之间的引用关系。

这里只建立供 `references/doc-style.md` 条例定位使用的索引，不判断文档是否违反规则。

### 5. 生成概要

将概要写入 `{code_summary_output_path}`。如果父目录不存在，先创建父目录。使用以下精简模板；某一类型不适用的章节直接省略，不生成大段空表。

```markdown
# 文件检视概要

## 检视范围

| 文件 | 文件类型 | 代码侧别 | 角色 | 范围 |
|---|---|---|---|---|
| {path} | TileLang/Python/Markdown | Kernel/Host/混合/N/A | {入口/Kernel/测试/文档等} | 正式检视 |
| {path} | {type} | {side} | {context role} | 仅上下文 |

整体文件类型: {Python/TileLang/Markdown/混合}
整体代码侧别: {Kernel/Host/混合/N/A}
算子或文档名称: {name}
功能概述: {有证据支撑的一句话说明}

## 代码脉络

> 仅代码输入生成。

入口与调用链: {入口 → builder/wrapper → Kernel}

数据流: {Host 输入 → Kernel 参数/Buffer → 搬运和计算 → 输出}

主要计算: {公式或实现语义；无法确认时明确标注}

### 关键分支与边界

| 条件 | 位置 | 触发场景 | 处理逻辑 |
|---|---|---|---|
| {condition} | {file:line} | {scenario} | {behavior} |

### 参数来源与防御

| 参数 | 定义/来源 | 校验位置 | 使用位置 | 已确认约束 |
|---|---|---|---|---|
| {name} | {file:line/expression} | {file:line 或未找到} | {file:line} | {constraint 或未确认} |

### 函数与调用索引

| 函数 | 位置 | 角色 | 直接调用者 |
|---|---|---|---|
| {name} | {file:start-end} | {入口/Kernel/辅助/测试} | {caller:line 或未确认} |

### TileLang API 索引

| API | 位置 | 调用上下文 |
|---|---|---|
| {API} | {file:line} | {简述参数和用途} |

### Buffer、切分与流水

| 对象/机制 | 位置 | shape/dtype/层级或配置 | 用途与依赖 |
|---|---|---|---|
| {buffer/pipeline/loop} | {file:line} | {facts} | {facts} |

## Markdown 索引

> 仅 Markdown 输入生成。

| 文档 | 用途 | 章节结构 | 代码/命令 | 路径、链接与引用 |
|---|---|---|---|---|
| {path} | {purpose} | {headings} | {locations} | {locations and targets} |

## 跨文件关系

| 源文件 | 目标文件 | 关系 | 位置 | 是否属于正式检视范围 |
|---|---|---|---|---|
| {source} | {target} | import/调用/参数传递/文档引用 | {file:line} | 是/否 |

## 未确认信息

- {缺失证据、未解析调用或无法确定的参数约束；没有则写“无”}
```

### 6. 返回结果

概要写入成功后，向主流程返回：

```text
文件类型: {Python/TileLang/Markdown/混合}
代码侧别: {Kernel/Host/混合/N/A}
算子名: {operator_name；纯文档输入使用文档或目录名称}
功能概述: {一句话}
概要路径: {code_summary_output_path}
```

## 约束

- 必须读取全部正式检视文件，并在概要中逐项列出。
- 允许读取关联文件作为上下文，但不得扩大正式检视范围。
- 每个侧别、调用关系、参数约束和业务结论都必须有源码或文档位置支撑。
- 无法确认的信息必须明确标记，不得补全或猜测。
- Markdown 的代码侧别必须为 `N/A`，不得归入 Host 或 Kernel。
- 不执行 API 预研、设计一致性检查、条例判定或性能结论评估。
- 只写入指定的概要文件，不修改 `file_input` 或上下文文件。
