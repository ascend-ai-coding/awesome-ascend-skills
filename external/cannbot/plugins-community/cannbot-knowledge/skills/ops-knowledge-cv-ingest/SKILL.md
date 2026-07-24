---
name: external-cannbot-plugins-community-cannbot-knowledge-skills-ops-knowledge-cv-ingest
description: 为 AscendC CV（cube-vector）融合算子生成「cube↔vector 融合设计」wiki（手动触发）。
disable-model-invocation: true
original-name: ops-knowledge-cv-ingest
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# ops-knowledge-cv-ingest — CV 融合知识生成器

为 AscendC **CV 融合算子**（cube 类计算 + vector 类计算，经 GM workspace 衔接）生成「cube↔vector 融合设计」wiki。与 [`ops-knowledge-vv-ingest`](../ops-knowledge-vv-ingest/SKILL.md) 体裁不同、不混用：

- **vv**：纯向量算子，多模板分发设计
- **cv**：cube↔vector 协作算子，融合设计（结构随算子分发路径数自适应）

仅适用于 cube（matmul/conv 等）与 vector 经 GM workspace 衔接的融合算子；纯 Vector 多模板算子请用 vv skill。

## ⚠️ 泛化第一（最高优先级）

本 skill 面向**所有** CV 融合算子。[`templates/cv-template.md`](templates/cv-template.md) 内嵌的样例（取自 grouped_matmul_swiglu_quant_v2 A8W8）是**唯一现有的格式范式（exemplar），不是目标算子，绝不可把它的专属内容当模板硬编码**。仿照 vv skill「不要套用 RoPE 的函数名」：

- **不假设算子语义**：数学公式、算子名、输入输出语义一律从 golden 读取，不假设是 matmul+swiglu+quant。CV 融合 = cube 类计算（matmul/conv 等）+ vector 类计算（elementwise / reduce / activation / quant / cast 等）经 GM workspace 衔接。
- **不假设分发拓扑**：分发路径数、路径语义名（A8W8 等量化模式只是 gmm 的例子）、tilingKey 取值，全部从目标算子的 tiling / 路由代码读取。可能只有单一路径。
- **不假设流水线拓扑**：可能是单融合 kernel，也可能是多段（如 MSD pre/mid/post，或其它 N 段）经 workspace 串接；段数与命名从 golden 读取。
- **不套用具体命名/常量**：buffer 名（`xActQueue_`、`tmpBuf1_` 等）、API 调用序、字节系数、`RESRERVE_MEM_SIZE` 等常量、`ubFactorDimx` 分档阈值——都是 gmm 专属，必须按目标算子重新提取。
- **结构按算子自适应**：§0 分发总览与末章对比表**仅当存在多条流水线时**才生成；单路径算子省略这两节，退化为「计算公式 → Kernel → Tiling」三章。
- **exemplar 只学方法不抄内容**：阶段 0 读它是为了学 block-beta 画法、`#n` 跨节同号、`[入/出/移/算]` 标签、字节账方法、写作密度。

## 输入

用户提供两个参数：
- **算子名称**：如 `grouped_matmul_swiglu_quant`、`<任意 CV 融合算子>`
- **Golden 代码路径**：上游 GitCode 仓库 URL。从中提取 `<repo>`（`ops-nn`/`ops-transformer`/`ops-cv`）、`<commit>`（固定 commit SHA）、`<category>`。

## 输出

### Frontmatter（必加）

每个 wiki 文件**必须**以以下 YAML frontmatter 开头：

```yaml
---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
category: <category>
resource: https://gitcode.com/cann/<repo>/tree/<commit>/<category>/<op_name>/
title: <op_name> cube↔vector 融合设计
description: <一句话概括：引擎配比 + 数据流 + 含 mermaid 图>
tags: [<category>, <核心主题>, <其他主题标签>]
paradigms: [<范式，镜像 cannbot-skills 27 枚举>]
confidence: verified
status: verified
created_at: '<YYYY-MM-DD>T00:00:00Z'
updated_at: '<YYYY-MM-DD>T00:00:00Z'
---
```

### 文件位置与导航

单文件 wiki，保存到 `ops/<repo>/<category>/<op_name>.md`。`<repo>` 从 golden 仓库推导（`ops-nn` → `nn`、`ops-transformer` → `transformer`、`ops-cv` → `cv`），`<category>` 为算子类别目录。同时在该目录下创建/更新 `index.md` 追加导航链接，并在 `ops/<repo>/index.md` 和 `ops/index.md` 中补充层级链接。所有 `index.md` 间的跳转使用 `dir/index.md` 格式，叶子文件使用 `filename.md` 格式。

**同算子多平台版本互链**：若该算子已有其他平台版本 wiki，新 wiki 与旧 wiki 必须在正文开头的 `> **范围**` 声明里互相指向。

### 源码引用格式（GitCode 固定 commit HTTP 链接）

wiki 中**所有**对 golden 源码的引用必须使用 GitCode blob 永久链接：

```
[`<相对路径>`](https://gitcode.com/cann/<repo>/blob/<commit>/<完整路径>)
```

带行范围：

```
[`<文件名>:<行范围>`](https://gitcode.com/cann/<repo>/blob/<commit>/<完整路径>)
```

**禁止**使用本地 `raw/` 路径或无链接的纯文件名。

## 贯穿全文的严谨纪律（写进 wiki preamble，并在写作时逐条遵守）

1. 字节单位统一；每块子区字节相加 = 块总字节（逐块核对）。
2. 分配量 ≠ 实际数据：对齐冗余、预留未用都要标明。
3. 不张冠李戴：写清「读 workspace」还是「matmul 本身」等。
4. double buffer 是评估项，不是默认（看 InitBuffer 深度）。
5. 复用安全判据：同一子区相邻两次写之间须夹着对旧值的读。
6. 字节来源是 InitBuffer 公式，可追溯行号，不靠估。
7. 交代图/伪码的范围（哪一层循环、哪个域）。
8. 平实写法，不自造概念；引用源码用相对路径 + 行号。

## 工作流

按以下顺序逐阶段执行，每阶段完成后再进入下一阶段。**每阶段结尾的「完成条件」是 gate——未满足不进入下一阶段。**

### 阶段 0：通读参考范式

读 [`templates/cv-template.md`](templates/cv-template.md) 全文，吃透其**方法与构件**（不是内容）：
- 三章主结构（计算公式 / Kernel / Tiling）及每节 `<!--` 指导注释的写法
- block-beta UB 内存布局图的画法
- `#n` 在 §2.2.1 布局图与 §2.2.3 复用伪码之间**同号**的机制
- `[入] GM→UB · [出] UB→GM · [移] UB 内拷贝/广播 · [算] 向量计算` 标签体系
- 由 InitBuffer 实参逐块算字节、子区相加核对、反推 tile 的「字节账」方法
- 写作密度（每个 API 入参为何这样设都要解释）

> 完成条件：能口述三章结构、§2.1 五点串讲、§2.2.1 mermaid 画法、`#n` 同号规则、`[入/出/移/算]` 标签。

### 阶段 1：扫描 Golden 代码结构

列出 golden 目录下所有文件，建立文件清单。重点：
- `README.md` — **必读**：确认平台支持（Ascend 950/A3/A2 等）、dtype 矩阵、**支持的分发路径/量化模式**、shape/参数约束。wiki 的范围声明须对齐 README。
- `op_host/` 下 `*_tiling.{h,cpp}`（TilingData 结构、`Compute()`/`PostTiling()`/tilingKey 选定逻辑、cube/matmul tiling 的产生与覆盖）；`*_def.cpp`（dtype/format 矩阵）；`*_infershape.cpp`（shape 约束）。注意 arch35 算子可能在 `op_host/arch35/` 子目录。
- `op_kernel/` 下所有 `.h`/`.cpp`：融合 kernel 模板、`*_pipeline.h`、`*_utils.h`、路由入口 `.cpp`。注意多段流水线可能有 `*_{pre,mid,post}.h` 之类的分段文件；arch35 可能在 `op_kernel/arch35/`。
- `op_host/op_api/` 算子注册文件；`tests/` 下 golden / executor 脚本（数学参考实现）。

> 完成条件：文件清单覆盖 host/kernel 三类；README 功能与平台/路径支持已记录；cube/vector 两域的代码归属已分清。

### 阶段 2：识别分发路径与流水线拓扑（自适应）

从 tiling.cpp / 路由入口找全 tilingKey 常量及其选定逻辑：
- 建映射表 `tilingKey → 路径语义 → 流水线 kernel 文件`。**路径语义由代码决定**（可能是量化模式、shape 分档、weight 布局等），**不要预设为量化模式**。
- 对每条路径，判定其流水线是**单融合 kernel** 还是**多段**（经 workspace 串接的 N 段，段名从 golden 读）。
- 提取 host 如何选定 tilingKey（按 dtype 组合 / shape / attr 的决策逻辑），带 `文件:行号`。

**分支**：
- **>1 条路径** → 生成 wiki 的 **§0 分发总览**：分发映射表 + 选定决策（借用 vv 的分发概念，可用 ASCII 决策树或表格表达 host 如何选 tilingKey）。
- **仅 1 条路径** → 跳过 §0，直接进入三章主体。

> 完成条件：tilingKey→路径→kernel 映射表已建全；每条路径的流水线形态（单融合/多段）已判定；§0 是否生成已决。

### 阶段 3：提取算子计算公式（→ §1）

对齐 exemplar §1.1~1.4：

- **3a. §1.1 数学公式 / golden** — 写完整计算式（逐元素/归约/分块/分组等按算子实际）；定义每个维度符号；含 split/scale/量化时写清各步维度变化；附 golden/参考实现路径（tests 下 executor）。
- **3b. §1.2 Shape·dtype 速查表** — 每个输入/输出一行给 shape、dtype、format；shape 约束（各维上限、对齐、整除）单列并附校验处 `文件:行号`；**多路径算子**：dtype 矩阵需覆盖各路径差异（来源 `*_def.cpp`）。
- **3c. §1.3 计算图分解（AscendC API 接口）** — 把数学式落成真实 API 序列，按引擎分段（cube 段 / cube 结果到 vector 的搬运 / vector 段）；每步写真实 API 签名（含关键参数），不写数学伪名；无现成复合 API 的运算用基本 API 手工展开。
- **3d. §1.4 计算特征总结** — 一张表概括决定第 2 章循环与内存形态的要素：轴含义；各轴切分方式；引擎配比（AIC:AIV）；cube↔vector 数据交接方式；执行遍数（单遍/多遍）。

> 完成条件：四节齐全，每断言标注 `文件:行号`；§1.3 的 API 序列可追溯到 golden 代码。

### 阶段 4：逐流水线提取 Kernel（→ §2..N，每流水线一章）

对阶段 2 列出的**每条**流水线生成一章。**不要套用 gmm 的函数名/buffer 名/API 序/字节系数——从目标算子 golden 读实际结构。**每章包含：

**§x.1 Cube↔Vector 协作（经 GM workspace）** — 按数据流串讲（逐点只说一件事，整体交给末尾伪码）：
① 原始输入 shape；② cube 怎么切块（baseM×baseN 等）、跨核分发；③ 中转 workspace（形状=完整 cube 输出、dtype）；④ vector 怎么从 workspace 切块读（按哪个轴对齐）；⑤ 两域同步（轮次 / CrossCore flag / SyncAll / 防覆盖）。shape 用元素数（字节留到 §x.2.2）。末尾给 ①~⑤ 串接的伪码（标注 AIC/AIV 两域、循环范围）。

**§x.2 Vector Tile 循环与内存分配**：

- **§x.2.1 UBuf 内存布局（block-beta）** — 用 mermaid `block-beta`（**不用 SVG**）：
  - 第一行块头：每块标 `TQue/TBuf + 总字节`（宽度粗略区分大小，非比例，等大块等宽）。
  - 第二行：按偏移把块切子区（含 ReinterpretCast / 头尾分区 / 不同 dtype 视图），标子区字节。
  - 其下：按调用序纵向叠 `#n · API · 语义`；同一子区叠多框即复用。
  - `#n` 与 §x.2.3 操作数代入同号（硬约束）。
  - 末尾给「复用序（同子区自上而下）」列表 + 必要说明。
  - 所有行 span 之和必须 = `columns` 声明值（逐行核对）；末尾 `classDef` 着色。

- **§x.2.2 UB 总占用与切分约束** — 由各 buffer 的 InitBuffer 实参求总占用并反推切分上界：
  1. 逐块按 InitBuffer 公式写字节；子区字节相加须 = 块总字节（核对）。
  2. 各块相加成总公式。
  3. 由「总 ≤ UB_SIZE」反推一趟 tile 大小。
  说明是否开 double buffer 及原因；常量项（对齐/保留余量）如实标来源，不臆测。

- **§x.2.3 内存复用与计算过程** — 把内层循环体逐 API 展开，贯通「循环 / 数据拷贝 / 计算 / 复用」：
  1. 先写「派生」：各 LocalTensor 由哪块 Alloc / ReinterpretCast / 偏移得到。
  2. 按真实调用序逐行写，前缀 `/* #n */`、行末标 `[标签]` + 写入子区（`→子区(覆盖#k)`）；`#n` 唯一（连续同 buffer 原地链合并为一个号），与 §x.2.1 图同号。
  3. 如实保留循环结构（if 缓存 / 循环体缩进）。
  标签：`[入]` GM→UB · `[出]` UB→GM · `[移]` UB 内拷贝/广播 · `[算]` 向量计算。

**§x.3 该流水线 Tiling 参数** — 逐参数列「含义·第 2 章用处 | 计算方式」（固定值写常量，派生值写公式）；表后补：cube/matmul tiling 的产生与覆盖关系、核数下发（SetBlockDim）、workspace 计算、tilingKey 分发到本流水线的入口（带 `文件:行号`）。

**多段流水线**：§x.2 的 UB 分析按各段（如 pre/mid/post，或该算子实际段数）**分段重复**，并说清各段经 workspace 的数据交接。

> 完成条件：每条流水线一章齐全；§x.1 五点串讲 + 伪码完整；§x.2 三节齐全（mermaid span 核对、`#n` 同号、UB 公式可追溯）；多段流水线分段分析完整。

### 阶段 5：流水线对比总表（仅当 >1 条流水线）

借用 vv 对比表思想。对比维度从该算子**实际差异**中提取，例如：路径语义、tilingKey、单融合 vs 多段、workspace 占用、UB 主项系数、是否 double buffer、cube/vector 配比、设计复杂度（★☆☆ ~ ★★★）。**单流水线算子省略本节。**

> 完成条件（仅多流水线）：对比表覆盖所有流水线差异维度。

### 阶段 6：写出 wiki

按以下**自适应**结构落盘到输出路径：

```
# <op_name> cube↔vector 融合设计
preamble（8 条纪律 + 符号约定）
[§0 分发总览]                ← 仅多流水线时
§1 算子计算公式（1.1~1.4）
§2..N 流水线一/二/…           ← 单流水线则只一章
   x.1 Cube↔Vector 协作
   x.2 Vector Tile 循环与内存分配（x.2.1 block-beta / x.2.2 字节账 / x.2.3 复用伪码）
   x.3 Tiling 参数
[§末 流水线对比总表]          ← 仅多流水线时
```

符号约定示例（按算子调整）：`m`=一趟 UB 处理的行/token 数；`N`=中间全宽（若有 split 写清输出宽）；`m·N`=元素数；字节公式系数已含 dtype 字节（f32×4、int8×1 等）。

> 完成条件：wiki 落盘到 `ops/<repo>/<category>/<op_name>.md`；index.md 导航已补；结构按路径数自适应。

### 阶段 7：验证（自检清单）

逐条核对，**任一不通过则回到对应阶段修补**：

- [ ] **frontmatter** 字段齐全（schema_version/kind/type/source_family/category/resource/title/description/tags/paradigms/confidence/status/created_at/updated_at）
- [ ] **源码引用** 全部为 GitCode blob 永久链接，无 `raw/` 路径
- [ ] 每个源码引用带 `文件:行号`
- [ ] 每块子区字节相加 = 块总字节（逐块核对）
- [ ] `#n` 在 block-beta 图与伪码间一一对应、唯一
- [ ] 复用安全：同一子区相邻两次写之间夹着对旧值的读
- [ ] 字节来自 InitBuffer 公式可追溯，未臆测
- [ ] 标注 double buffer 与否及依据
- [ ] 图/伪码交代范围（哪层循环、哪个域）
- [ ] cube↔vector 协作交代了 workspace 形状/dtype 与同步机制
- [ ] 所有分发路径/流水线均有对应章节，§0 分发总览与逐章一致
- [ ] **导航** index.md 已补；多平台互链（若有姊妹版本）
- [ ] **泛化自检**：成文不含 gmm/swiglu 专属命名（除非目标算子确为它）；结构按算子路径数自适应（单路径无 §0/对比表）

### 阶段 8：重跑图谱

新增/升级为 verified 的 wiki 卡片后，必须重跑 `../ops-knowledge-ingest/scripts/okf_graph.py` 的图谱流程（规则见目标知识库根目录的 `SPEC-Graph.md` / `CLAUDE.md`「图谱写入规范」）：

1. `python3 ../ops-knowledge-ingest/scripts/okf_graph.py --knowledge-root <知识库根> candidates`
2. 判定新增候选写入 `graph/edge_judgments.json`（operator↔operator 用 `same_topic`；operator↔guide/api 用 `exemplifies`）
3. `python3 ../ops-knowledge-ingest/scripts/okf_graph.py --knowledge-root <知识库根> inject`
4. **inject 后刷新 `card_fp`**（`okf_graph.card_fp_map(load_nodes())`）写回 `edge_judgments.json`
5. `python3 ../ops-knowledge-ingest/scripts/okf_graph.py --knowledge-root <知识库根> verify` — 必须 OK

> 完成条件：`verify` OK，wiki 的 `# 相关` 块无死链。

## 关键原则

- **泛化第一**：适用于任意 CV 融合算子；exemplar 只学方法不抄内容；结构随算子自适应。
- **一切引用来自 golden 代码**，带 `文件:行号`，不靠通用知识推断。
- **代码用伪码 + 注释**，不整段复制；保留关键变量名与 API 调用，标注 ★ 解释设计决策。
- **每个 API 入参的选择原因必须解释**：不是「设了什么值」，而是「为什么这样设」（关联 tiling 参数推导）。
- **block-beta 用 mermaid（不用 SVG）；`#n` 跨节同号是硬约束**。
- **字节账逐块核对**；double buffer 是评估项不是默认。
- **cube↔vector 经 GM workspace 的协作与同步是 CV 融合的核心**，必须讲清。
