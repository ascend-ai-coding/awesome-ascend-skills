---
name: external-cannbot-plugins-community-cannbot-knowledge-skills-ops-knowledge-vv-ingest
description: 为 AscendC 多模板 VV 算子从官方 golden 代码生成解耦的两层知识：算子特定 wiki（ops/{repo}/{category}/{op}.md，逐模板全链路
  + mermaid UB 布局图）和泛化优化点 runbook（runbooks/operator-optimization/vv-fusion-common.md，跨算子单一共享、增量合并的
  NPU 垂域优化点库）。仅适用于有多模板分发（TilingKey 多分支）的 VV 纯 Vector 算子。手动触发。
disable-model-invocation: true
argument-hint: 算子名 golden-GitCode-blob-or-tree-URL-at-commit
allowed-tools: Read, Glob, Grep, Write, Agent, Bash, WebFetch
original-name: ops-knowledge-vv-ingest
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# ops-knowledge-vv-ingest — 多模板分层知识生成器

为 AscendC 多模板 VV 算子，从官方 golden 代码生成**解耦的两层知识**：
- **泛化优化点 runbook** `runbooks/operator-optimization/vv-fusion-common.md`（**NPU 垂域优化点库**，扁平 `OPT-*`，跨算子单一共享、增量合并）；
- **算子特定知识** `ops/<repo>/<category>/<op>.md`（逐模板全链路 + §N.3 mermaid UB 内存布局图 + UB 占用公式）。

## 输入

- **算子名称**：如 `apply_rotary_pos_emb`、`softmax_v2`。
- **Golden GitCode URL@commit**：如 `https://gitcode.com/cann/ops-transformer/blob/<commit>/posembedding/apply_rotary_pos_emb/`。从中提取 `<repo>`（`ops-nn`/`ops-transformer`/`ops-cv`）、`<commit>`（固定 commit SHA）、`<category>`（算子类别，如 `posembedding`/`activation`）。
- **golden 源码一律从 GitCode HTTP 读取，不使用本地路径**：用 **WebFetch 抓 GitCode `blob/<commit>/<path>` 页**取源码（实测 `blob/` 页可取源码、`raw/` 返回 SPA 壳页不可用）。所有源码引用即这些 GitCode blob 永久链接（见「输出·源码引用格式」）。
- **输出根**（可选）：缺省 = **本仓库根**（`ops/`、`runbooks/` 直接在仓库根下）。

## 输出

**两层分层知识**（不是单文件 wiki），落到 `<输出根>/`：
- 算子特定：`ops/<repo>/<category>/<op_name>.md`（新建/覆盖该算子一份；`<repo>`：`ops-nn`→`nn`、`ops-transformer`→`transformer`、`ops-cv`→`cv`）。
- **平台→文件名约定（必守）**：从 golden 识别平台变体——**MemBase（910/910B/910_93，A2/A3）→ `<op_name>.md`**；**arch35（ascend950，A5）→ `<op_name>_a5.md`**。算子两版都有 → 各出一份、正文 `> **范围**` 互相指向（姊妹版互链）；**纯 A5 算子（仅 arch35，如 softmax_v2）只出 `<op_name>_a5.md`**。
- 泛化优化点：`runbooks/operator-optimization/vv-fusion-common.md`（**增量合并**进这同一份共享库，不新建带算子名的泛化文件）。
- 渐进导航 `index.md`：在算子所在目录及各上层补 `ops/<repo>/<category>/index.md`、`ops/<repo>/index.md`、`ops/index.md`（`kind: index` / `type: section_index|bundle_index` 形态；已存在则追加，不存在则创建；`dir/index.md` 跳目录、`file.md` 跳叶子）。

### Frontmatter（算子文件必加）

每个算子文件**必须**以以下 YAML frontmatter 开头：

```yaml
---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
category: <category>
resource: https://gitcode.com/cann/<repo>/tree/<commit>/<category>/<op_name>/
title: <op_name> 多模板分发与内存复用设计
description: <一句话概括：模板数 + 分发维度 + 含 mermaid 图>
tags: [<category>, <核心主题>, <其他主题标签>]
paradigms: [<范式，镜像 cannbot-skills 27 枚举>]
confidence: verified
status: verified
created_at: '<YYYY-MM-DD>T00:00:00Z'
updated_at: '<YYYY-MM-DD>T00:00:00Z'
---
```

runbook 使用 `kind: operator_optimization` / `type: optimization_runbook` profile；跨算子无单一 resource 时不强制填写 `resource`。各级 `index.md` 使用 `kind: index`，bundle 根 index.md 另注 `upstream_repo`/`upstream_ref`/`upstream_commit`。

### 源码引用格式（全部使用 GitCode 固定 commit HTTP 链接）

知识文件中**所有**对 golden 源码的引用必须使用 GitCode blob 永久链接，格式：

```
[`<相对路径>`](https://gitcode.com/cann/<repo>/blob/<commit>/<完整路径>)
```

带行范围的引用：

```
[`<文件名>:<行范围>`](https://gitcode.com/cann/<repo>/blob/<commit>/<完整路径>)
```

示例（§1 表格行）：
```
| [`op_host/arch35/softmax_v2_tiling.h`](https://gitcode.com/cann/ops-nn/blob/ffcbad031c7cb0acf02a2ae7160d22fe5ed097e2/activation/softmax_v2/op_host/arch35/softmax_v2_tiling.h) | 职责描述 |
```

**禁止**使用本地 `raw/` 路径或无链接的纯文件名。

### 目录结构（OKF 渐进式导航）

```
<输出根=仓库根>/
├── runbooks/
│   └── operator-optimization/
│       ├── index.md                  # okf.v1 section index
│       └── vv-fusion-common.md       # 泛化 OPT/AP/CT 库（跨算子单一共享）
└── ops/
    ├── index.md                      # → nn/index.md, transformer/index.md, cv/index.md
    ├── transformer/
    │   ├── index.md                  # → posembedding/index.md
    │   └── posembedding/
    │       ├── index.md              # → 两姊妹版
    │       ├── apply_rotary_pos_emb.md       # MemBase (A2/A3)
    │       └── apply_rotary_pos_emb_a5.md    # arch35 (A5) · 与 .md 互链
    └── nn/
        ├── index.md                  # → activation/index.md
        └── activation/
            ├── index.md              # → softmax_v2_a5.md
            └── softmax_v2_a5.md      # 纯 A5（仅 arch35）故带 _a5 后缀
```

## 工作流

按以下顺序逐阶段执行，每阶段完成后再进入下一阶段。

> 本 skill 宜在**干净上下文**中执行（每平台变体各一次，互不污染）；阶段 5'b 的 runbook 合并另起干净上下文子 Agent。**平台变体由阶段 1b 从 golden 自动判定**（不靠外部告知）：算子同时有 MemBase 与 arch35 两套实现时，各出一份姊妹版（`<op>.md` / `<op>_a5.md`）、每变体各起一个干净上下文执行；只有一套时只产对应一份。

### 阶段 0：通读两份章节骨架

读包内两份章节骨架（写作时严格对齐其章节顺序、详细程度、代码片段密度）：
1. [`STRUCTURE-operator.md`](STRUCTURE-operator.md)——算子卡骨架
2. [`STRUCTURE-runbook.md`](STRUCTURE-runbook.md)——泛化 runbook 骨架

核心立场：**算子文件忠实**（近无损记录该算子 golden 事实）⇄ **runbook 是 curated 库**（泛化、占位名骨架、同类归并、剔非优化点，不追求与某算子原文近无损）。

**优化维度（4 分类，每条 runbook OPT 必标一或多）**：

| 维度 | 含义 |
|------|------|
| **搬运** | DMA / MTE2·MTE3 吞吐：搬运量、连续 vs 碎、广播复用减搬运 |
| **计算** | Vector 指令吞吐/条数：repeatTimes 取大轴、归约算法、并行度 |
| **内存** | UB 占用 / buffer 复用 / 物化 vs 不物化 / 缓冲深度 |
| **精度** | dtype / cast 升降 / 数值稳定（减极值防溢出等）/ 数值误差 |

生成的两份必须对齐两份骨架的结构与详细程度。**注意**：runbook 是跨算子**共享累积**的，本次提取要**增量合并**（见阶段 5'b）而非覆盖。

### 阶段 1：扫描 Golden 代码结构（GitCode HTTP，非本地）

用 **WebFetch 抓 GitCode `blob/<commit>/<dir>/` 目录页**列出 golden 目录下所有文件，建立文件清单；逐个文件用 `blob/<commit>/<path>` 页 WebFetch 取源码（**要求逐字 + 行号返回，大文件按行段多次抓并核对完整性**——WebFetch 经小模型转换、勿轻信单次截断）。`file:line` 引用取自 blob 页行号。重点关注：
- `README.md` — **必读**：确认算子支持的平台范围（Ascend 950/A3/A2 等）、dtype、参数约束。范围声明必须对齐 README 的产品支持表。
- `op_host/` 下的 `*tiling.h`、`*tiling.cpp`（注意：arch35 算子可能在 `op_host/arch35/` 子目录下）
- `op_kernel/` 下的所有 `.h` 和 `.cpp` 文件（模板实现 + 路由入口，arch35 可能在 `op_kernel/arch35/` 子目录下）
- `op_host/op_api/` 下的算子注册文件

#### 1b. 自动判定平台变体与产出文件集（必做·只看 golden、不靠外部告知）

一个算子可能含 **MemBase（A2/A3，910 系）** 与 **arch35（A5，950）** 两套实现。**从 golden 实际存在的实现文件**判定有哪几套、决定产出哪几份、各自读哪些文件——**判定依据是实现是否真实存在，不是调用者口头说"这是 A5/A3"**：

- **MemBase（A2/A3）信号**：`op_kernel/` 下有**非 arch35** 的模板 `.h` + router `<op>.cpp`；`op_host/` 下有非 arch35 的 `*tiling.cpp/.h`；`op_host/config/` 含 `ascend910`/`ascend910b`/`ascend910_93` 等。命中 → 产出 **`<op>.md`**（只读这批非 arch35 文件）。
- **arch35（A5）信号**：`op_kernel/arch35/` 有模板 `.h`（router 常为 `<op>_apt.cpp` 或 apt 宏）；tiling 在 `op_host/arch35/` 或 `op_host/` 下 `*_arch35.cpp`；`op_host/config/ascend950`。命中 → 产出 **`<op>_a5.md`**（只读 arch35 那批文件）。
- **两套都在** → 产出两份姊妹版（`.md` + `_a5.md`），`> **范围**` 互链；**各自只读对应平台的 kernel/tiling 文件**、各走一遍阶段 2–5a（每变体各起一个干净上下文执行）。**只一套** → 只产对应一份（例：`softmax_v2` 只有 `op_kernel/arch35/`，故自动判为纯 A5 → 只产 `softmax_v2_a5.md`；不因外部是否提及 A5 而改变）。
- 输出 `已判定的变体清单`（每个变体：平台 / 文件名 / 对应的 kernel+tiling+router 文件集 / 对应 config）作为本阶段完成条件，后续阶段对每个变体分别执行。

### 阶段 2：识别分发模式并提取分发机制

AscendC 算子有两种主流多模板分发模式。先判断该算子属于哪种，再按对应方式提取。

#### 2a. 判断分发模式

两种模式在 **kernel 路由入口** 完全一致——都是 `TILING_KEY_IS(N)` 的 if-else 链（或等效宏展开）。差异只在 **host 侧 TilingKey 的选定机制**。

| 特征 | 模式 A：决策树型 | 模式 B：优先级+IsCapable 型 |
|------|-----------------|--------------------------|
| TilingKey 定义 | `*tiling.h` 中 `enum class`（小整数：1, 3, 4） | `*tiling.h` 中 `constexpr int64_t` 常量（大整数：500, 1000, 2000...） |
| TilingKey 选定 | **同一个** `Compute()` 函数内 if-else 分支，最终调用 `SetTilingKey()` | **每个模板类** 各自实现 `IsCapable()` + `GetTilingKey()`，框架按优先级依次尝试 |
| 优先级机制 | 无（TilingKey 唯一确定） | `constexpr int32_t TEMPLATE_*_PRIORITY` 常量，数值越小越优先 |
| 模板文件 | `op_kernel/*.h`（非 arch35 子目录） | `op_kernel/arch35/*.h`（arch35 子目录） |
| 路由入口 | `op_kernel/<op>.cpp`，`TILING_KEY_IS(N)` if-else | `op_kernel/<op>_apt.cpp`，`TILING_KEY_IS(KEY)` if-else 或宏展开 |
| 典型算子 | apply_rotary_pos_emb | softmax_v2 |

读取 `*tiling.h` 和路由入口文件后，根据上表判断模式。**路由入口的 `TILING_KEY_IS()` 分支数 = 模板数，在两种模式下提取方式相同。**

#### 2b. 提取 TilingKey 列表（两种模式通用）

从 `*tiling.h` 找到所有 TilingKey 常量定义（枚举值或 `constexpr int64_t`），从路由入口找到所有 `TILING_KEY_IS(N)` 分支或等效宏。建立 TilingKey → 模板类名 → 模板文件的映射表。

#### 2c. 模式 A 判定逻辑提取

读 `*tiling.cpp`，找到 `Compute()` 函数。跟踪每个 `SetTilingKey()` 的调用路径，提取触发条件。关注：
- 小 shape 路径的判定条件（通常涉及 `preCoreBatch` 和 UB 容量比较）
- 不同 dtype 对模板选择的影响（如 BF16 是否跳过某些模板）
- UB 余量相关的判定（如 `shengMte > 0`）
- 画 ASCII 决策树（if-else 嵌套结构）

#### 2d. 模式 B 判定逻辑提取

对于模式 B，分发由框架按优先级依次调用 `IsCapable()` 实现。提取步骤：
- 读 `*tiling.h`，找到所有 `TEMPLATE_*_PRIORITY` 常量，按数值从小到大排序（= 优先级从高到低）
- 对每个模板类，读其 `IsCapable()` 函数（通常在对应的 `*_tiling.cpp` 中），提取：
  - 检查了哪些条件（shape 维度、轴数、UB 容量、dtype）
  - 每个条件的阈值和含义
  - 如果 `IsCapable` 失败（return false），该模板被跳过，框架尝试下一个优先级
- 画 ASCII 优先级链：`Priority N → IsCapable? → YES: 选中 / NO: 尝试 Priority N+1`

#### 2e. 找 TilingData 结构（两种模式通用）

读 `*tiling.h`，找到 `BEGIN_TILING_DATA_DEF` 的每个结构体定义。注意模式 B 可能每个模板有独立的 TilingData 结构体（如 `SoftmaxV2ARTilingData`、`SoftmaxV2ARATilingData`）。对每个结构体：
- 列出所有字段名
- 理解每个字段的用途（搬运量、循环次数、偏移量、block 数等）
- 标注该 TilingData 对应哪个 TilingKey

### 阶段 3：逐模板提取 Kernel 实现

对每个 TilingKey 对应的模板文件，提取以下内容。注意：不同算子的函数命名和结构不同（RoPE 有 CopyInQK/ComputeTotary，softmax 有 ReduceMax/Exp/Sub/ReduceSum/Div），**不要套用 RoPE 的函数名**——从 golden 代码中读取实际的函数名和调用关系。

**核心原则：先识别计算管线，再识别哪些 API 影响模板分发。影响分模板的 API 是重中之重——它们的完整实现必须融入伪码。**

#### 3-0. 识别计算管线与分模板 API（新增，必做）

在逐模板提取前，先通读所有模板的 kernel 实现，识别两件事：

**a) 该算子的核心计算管线是什么？** 列出算法步骤序列。例如：
- softmax_v2: `CopyIn → Cast(fp16→fp32) → ReduceMax → Sub(x−max) → Exp → ReduceSum → Div(exp/sum) → Cast(fp32→fp16) → CopyOut`
- apply_rotary_pos_emb: `CopyInQK → Muls(sin,−1) → DataCopy rotate_half → Mul(q*cos) → Mul(rot*sin) → Add → CopyOut`

**b) 管线中哪些 API 在不同模板间有差异？** 这些就是"影响分模板的 API"。例如：
- softmax_v2: **ReduceMax/ReduceSum** 随 R 轴大小选择不同算法（编译期展开 / 运行期直接 / 二分累加 / 分块+缓存），这是模板分发的根因
- apply_rotary_pos_emb: **DataCopy 错位 vs 逐元素重排** 随 rotaryMode 选择，**Cast 有无** 随 dtype 选择

**c) 对每个分模板 API，从 golden 代码中提取完整实现，融入伪码。** 不是"提到名字 + 一行注释"，而是把函数的参数声明、关键分支（`if constexpr`、`MaskMergeMode` 差异化使用）、循环结构都写进伪码。伪码里的 API 调用要标注：
- 每个参数的含义（特别是决定行为的参数，如 `MaskMergeMode::ZEROING vs MERGING`、`src1RepStride=0`、`DIST_UNPACK_B16 vs DIST_NORM`）
- 该参数在不同模板间是否变化、如何变化
- 来源 golden 文件和行号

**3a. 类结构与成员** — 找到模板类声明，记录：
- 继承关系（基类提供了什么工具方法）
- 构造函数参数（管道是否外部传入，如 `TPipe* pipe` vs 内部声明）
- 模板参数 `<T1, T2>` 的含义（输入 dtype、计算 dtype）

**3b. TQue/TBuf 声明** — 找到类的成员变量区，记录：
- 每个 `TQue` 的 `QuePosition`（VECIN/VECOUT）和 buffer 深度（`bufferNum` 值）
- 每个 `TBuf` 的名称和用途
- 哪些 TBuf 是条件分配（`#if ORIG_DTYPE_QUERY == DT_BF16` 等编译期分支）
- `bufferNum` 值的选择原因（单缓冲 vs 双缓冲：跟循环结构的关系）
- 注意：模式 B（arch35）的算子可能使用不同的 buffer 类型或 API，如实记录

**3c. Init() — Buffer 分配 + 参数配置** — 找到初始化函数，记录：
- 每个 buffer 分配的第三个参数（大小来自 TilingData 的哪个字段）
- `DataCopyParams` 或等效搬运参数的配置（`blockCount/blockLen/srcStride/dstStride`）及其含义
- 重复/广播参数的配置（如 `src1RepStride=0` 的广播语义）
- 条件分配 buffer 的预处理宏

**3d. Process() — 循环结构** — 找到主循环入口，画出嵌套循环结构：
- 有几层循环
- 每层循环变量来自哪个 TilingData 字段
- 尾核 vs former 核的分流逻辑（如果存在）
- 如果循环结构简单（如单层 for），直接写明循环次数和每次处理的数据量

**3e. CopyIn / 数据搬运** — 找到 GM→UB 搬运代码，记录：
- 多路输入是否拼接搬运（dstStride 如何预留空间）
- 常量操作数（如 cos/sin、scale、bias）的搬运方式：一份还是每份复制
- 部分计算 / partial 分支处理
- 如果该算子只有单路输入（如 softmax 只有 x），记录搬运粒度和 stride 配置

**3f. 核心计算** — 找到计算的主函数，记录：
- 计算管线的步骤序列（如 softmax: SubMax → Exp → ReduceSum → Div → SubLog）
- 每一步使用的向量 API（Mul/Add/Muls/Cast/ReduceMax/Exp 等）
- 向量 API 的 `repeatTimes`、`src1RepStride` 等重复/广播参数配置
- 中间结果复用策略（哪些 TBuf 被多步计算覆盖重用）
- Cast 路径（如有）：CAST_NONE（升精度） vs CAST_ROUND（降精度回写）
- 尾块处理方式

**3g. CopyOut / 数据搬出** — 找到 UB→GM 搬出代码，记录：
- 搬出的 stride 配置
- 是否有多路输出需要拆分（如 Q/K 分离搬出）
- partial 分支处理

### 阶段 4：识别内存复用策略

对照 golden 代码，识别以下 4 种复用策略的具体应用位置：
1. **原地操作**：`Muls(dst, src, ...)` 中 `dst == src`
2. **跨迭代覆盖**：`mul1.Get<>()` 每轮循环复用同一块 TBuf
3. **预处理分支消除**：`#if ORIG_DTYPE_QUERY != DT_BF16` 等编译期分支
4. **Add 直写输出**：`Add(outUb, ...)` 的 dst 在 output Queue 上

**额外 · 标记高杠杆优化（为「预期收益」「双表示」做准备）**：提取时留意哪些优化点是该算子性能的**主导杠杆**——去掉它基线明显塌方（典型=避免在 UB 物化展开/广播复用减搬运、减少 DMA 量的优化）。这些点在 5a 的算子 trick 章节里：①「预期收益」字段——初版从 golden 无现成分数，多填「待轨迹验证」（不强造）；②**有信心是高收益时，配一张示意图（双表示）**强化表达；普通优化点保持单表示、避免膨胀。

### 阶段 5'：解耦落盘（产出两层）

把阶段 1–4 提取的内容按**判定准则**分流到两份：

1. **是 NPU 垂域优化点吗？** 判据——去掉该技巧、用最朴素写法，NPU 执行效率是否一样？一样 → C++ 技巧、不入 runbook（如 dtype 模板化 `<T1,T2>`）；不一样（影响搬运量/指令条数/UB/精度）→ 入库。否 → 不入 runbook。
2. **剥掉算子数学语义后还成立吗？** 成立 → 泛化进 runbook（占位名骨架），不成立 → 算子特定 trick / 模板细节。
3. **能迁移到别的 VV 算子吗？** 能 → `泛化层级: VV通用`；仅特定条件成立 → `泛化层级: 条件性`（迁移条件三字段写清失效边界）。

**5a · 算子特定文件** `ops/<repo>/<category>/<op>.md`：按 [`STRUCTURE-operator.md`](STRUCTURE-operator.md) 骨架写（okf.v1 operator frontmatter + 本算子速览 + §2 trick 前置 + §3 分发 + §4..N 模板含 §N.3 mermaid+UB 公式 + §N+1 对比总表 + 文末 `# 相关`；`## 反模式` 仅当有无法内联的反例时才写）。**§3.2「设计期决策」与「反模式」属通用优化点 → 上提 runbook**，本文件只留指针。每条 trick 必含「坏实践（反例）」字段（无则「待补充」）；实例化泛化点处加 `【实例化: → OPT-x（<…#锚点>）】`。知识文件头部不嵌元规则。

**5b · 优化点合并进 runbook** `runbooks/operator-optimization/vv-fusion-common.md`：按 [`STRUCTURE-runbook.md`](STRUCTURE-runbook.md) 骨架写（扁平 `OPT-*`、标签块、占位名骨架、迁移条件、坏实践必填、已知实例反链、CT/AP 区、`# 相关`）。**增量合并**：已有该优化点 → 只在「已知实例」append 反链；新 → 新增 `OPT-N`（ID 不复用、不重排）。runbook 头部 blockquote 只写一句「定位」，不嵌元规则。

**5c · 反模式 / 约束陷阱处理（内敛优先）**：负知识能内联进强关联正向优化点 → 内联到「坏实践」段；无对应正向优化点的纯坏写法 → 独立 `AP-N`；平台/API/DMA/精度约束陷阱 → runbook `## 约束与陷阱` 区 `CT-N`。**初版从 golden 不强造反例**——但「坏实践」字段必须存在（无则「待补充」）。`## 反模式` 专节**仅当有无法内联的反例时才写**（`AP-N` 条目），无则不显示该节、不留空占位。

**5d · 交叉引用（双向、相对路径+锚点）**：算子写 `【实例化: → OPT-x（<…#锚点>）】`，runbook「已知实例」写 `<op> §y（<…#锚点>）`；锚点 = 目标标题字面文本，URL 用尖括号 `<...>` 包裹。**强制脚本校验无悬空锚**（阶段 6'）。

**5'b · 去重合并用「干净上下文子 Agent」（runbook 已存在时必走）**：当 runbook 已存在、需合并本次优化点时，**不要由本提取 agent 内联合并**（上下文已被算子 golden 污染、易误泛化）。用 Agent 工具另起干净上下文子 Agent，只喂：①现有 runbook 全文、②本次候选优化点（已按 STRUCTURE-runbook 骨架、算子无关）、③STRUCTURE-runbook 骨架 + 本 SKILL 关键原则与 5b/5c 纪律。由它执行去重合并并回写，结果须过阶段 6'。

### 阶段 6'：解耦校验

**首选：直接跑包内校验脚本**（一条命令覆盖下方多数检查项——悬空锚、引用未定义、标题未瘦身、runbook 泄漏算子特定/golden 变量、算子文件泄漏泛化标签、frontmatter/mermaid/UB 公式/GitCode 引用/无 API 章节/导航 index/坏实践全覆盖/反模式专节）：
```bash
python3 scripts/validate_layered_knowledge.py --knowledge-root <知识库根> --ops <本轮新增/改动的算子 md...>
# --root 缺省=仓库根。--ops 只校验本轮生成的算子文件（避免误报仓内其它 ops）。退出码非 0（有 HARD）→ 修正后重跑至 0。
```

脚本不可用时，按下方逐项内联自检（任一不过则修正后重出）：

- [ ] **frontmatter**：算子文件使用 okf.v1 operator profile（见「输出·Frontmatter」）；runbook 使用 `operator_optimization` profile
- [ ] **源码引用** 全用 GitCode blob URL，无 `raw/` 路径
- [ ] **文末 `# 相关` 托管块**：ops 卡 + runbook 末尾有 okf:related 块；无残留「知识来源」节
- [ ] **分模板 API 完整融入伪码**（参数含义/模板间差异/行号）；**无独立 API 章节**
- [ ] **§N.3 mermaid block-beta + UB 总占用公式**（画法纪律见 [`STRUCTURE-operator.md`](STRUCTURE-operator.md)）；`#n` 与伪码同号
- [ ] **只收垂域优化点**：runbook 无纯 C++ 技巧（去掉它、朴素写法 NPU 效率一样→删）
- [ ] **标签块字段齐全**（OPT/runbook 标签块见 [`STRUCTURE-runbook.md`](STRUCTURE-runbook.md)）；**算子文件不泄漏 `优化维度`/`泛化层级`**
- [ ] **泛化骨架算子无关**：OPT 骨架只用占位名，grep golden 变量黑名单（脚本第 4 项）
- [ ] **坏实践段必填**：每条 OPT/trick 有「坏实践（反例）」（无则「待补充」）；**反模式专节**（ops `## 反模式` / runbook `## 反模式（AP-*）`）仅当有无法内联的反例时才写，无则不显示
- [ ] **交叉引用锚点闭合（双向、无悬空）——必须跑脚本**：章节前置/重编号极易 off-by-one 悬空
- [ ] **渐进导航** index.md 已补

### 阶段 7：重跑知识图谱（接入仓库 OKF 图谱）

新增/改动 wiki 卡片后，仓库的 `# 相关` 托管块不会自动更新——须重跑图谱（规范见目标知识库根目录的 `SPEC-Graph.md` / `CLAUDE.md`「图谱写入规范」）。从本 skill 目录执行，目标知识库用 `--knowledge-root` 指定：

1. `python3 ../ops-knowledge-ingest/scripts/okf_graph.py --knowledge-root <知识库根> candidates` — 确定性召回，产 `.build/candidates.json`。
2. **判定新增候选**：读 `.build/candidates.json`，对新 wiki 焦点卡的候选逐对判定 `related/type/reason`，写入 `graph/edge_judgments.json`（operator↔operator 用 `same_topic`；operator↔guide/api 用 `exemplifies`）。判定键 = `sorted([fp(a), fp(b)])` 用 `|` 连接，`fp` 来自 `okf_graph.card_fp_map(load_nodes())`。
3. `python3 ../ops-knowledge-ingest/scripts/okf_graph.py --knowledge-root <知识库根> inject` — 写入卡片 `# 相关` 托管块。
4. **inject 后刷新 `card_fp`**：inject 改变了卡片内容，须重算全部 `card_fp`（`okf_graph.card_fp_map(load_nodes())`）写回 `edge_judgments.json`，否则 verify 报 stale。
5. `python3 ../ops-knowledge-ingest/scripts/okf_graph.py --knowledge-root <知识库根> verify` — 必须 OK。

> 完成条件：`verify` OK，新 wiki 的 `# 相关` 块含 ≥1 条相关链接、无死链。
> 注：5a 落盘时已在每张 ops 卡 / runbook 文末**手写初版 `# 相关` 托管块**（「相关主题」=姊妹/成员卡，路径已知；替代旧「知识来源」节）；本阶段图谱**刷新并补全**该块（尤其「实践案例」=reference/api 卡，需图谱召回匹配）。`【实例化: → OPT】`/「已知实例」是**知识层手写交叉引用**，与图谱 `# 相关` 托管块各司其职、并存。

## 关键原则

- **一切引用必须来自 golden 代码**，不可从通用知识推断。每个断言都要标注 golden 出处（GitCode blob URL，含行范围）。
- **代码片段优先用伪码 + 注释**，不要整段复制 golden 代码。伪码保留关键变量名和 API 调用，标注 ★ 解释设计决策。
- **每个 API 入参的选择原因必须解释**：不是"设了什么值"，而是"为什么这样设"（关联 tiling 参数的推导过程）。
- **先识别计算管线、再识别分模板 API**：影响分模板的 API 是重中之重，其完整实现（参数含义 + 模板间差异 + golden 出处）必须融入伪码；不单列 API 汇总章节。
- **算子文件 §N.3 含 mermaid block-beta UB 布局图 + UB 总占用公式**：忠实 golden（真实 buffer 名/字节系数），算子文件专有；**绝不进 runbook**（runbook 仍占位名、无 golden 变量）。
- **坏实践 = 必填字段、内敛优先**：每条优化点/trick **必含**「坏实践（反例）」字段（错误 → 后果 → ✅ 正解即该优化点）；**初版从 golden 不强造反例**——无可填实例则写「待补充」预留，**不留空、不省略**（各条目字段一致）。反模式能内联进强关联正向优化点则内联到其「坏实践」；**无对应正向、无法内联**的反例写进 `## 反模式` 专节（`AP-N` 条目），**无此类反例则不写该节**；约束陷阱留 `CT-*`。
- **算子优化 trick 前置 + 预期收益 + 双表示**：算子文件「算子特定优化 trick」章节**前置**为 §2（高杠杆速览）。每条 trick 带**「预期收益」预留字段**（初版「待轨迹验证」、随轨迹动态更新填实测增益）；对**有信心是高收益**的优化点用**双表示（伪码+示意图）**强化表达，普通项单表示防膨胀。
- **双向链接构知识图谱**：两份互引一律写成**相对路径 + 章节锚点的可点击链接**（锚点=目标标题字面文本，URL 用尖括号 `<...>` 包裹），便于 Obsidian 构成知识图谱。被链接的标题须**短/稳/唯一**、不含破锚符号；生成后校验**无悬空锚、双向闭合**。
- **对比表要全面**：不遗漏任何模板间的差异维度，让模型能在设计阶段根据场景选择正确模板。
- **runbook 必须算子无关（质量硬否决项）**：条目原则与骨架/示意图都用占位名，**不得**以算子业务名（cos/sin/q/k…）或 golden 变量作主语；业务实例落算子文件、经「已知实例」反链补回。
- **runbook 纪律**（只收垂域优化点不收 C++ 技巧 / 算子文件忠实·runbook curated / 跨算子单一共享·增量合并；详见阶段 5b 与 [`STRUCTURE-runbook.md`](STRUCTURE-runbook.md)）：runbook 已存在时合并由**干净上下文子 Agent** 执行（阶段 5'b），不为每算子新建泛化文件。
