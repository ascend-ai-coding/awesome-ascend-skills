# 算子特定卡片骨架（`ops/<repo>/<category>/<op>.md`）

[`ops-knowledge-vv-ingest`](SKILL.md) **阶段 5a** 写作算子卡时加载本骨架。**泛化 runbook 骨架见** [`STRUCTURE-runbook.md`](STRUCTURE-runbook.md)；详细纪律 / 判定准则 / 提炼原则 / 字段 schema / 格式见 [`SKILL.md`](SKILL.md)（阶段 0 / 5a / 5c / 5d + 关键原则）。若目标知识库已有范式卡，可参考 `<知识库根>/ops/transformer/posembedding/apply_rotary_pos_emb.md`（MemBase A2/A3）、`<知识库根>/ops/transformer/posembedding/apply_rotary_pos_emb_a5.md`（arch35 A5）、`<知识库根>/ops/nn/activation/softmax_v2_a5.md` 的章节顺序、详细程度与代码片段密度。

**关键纪律**：「算子特定优化 trick」前置为 §2（高杠杆速览）；每模板 **§N.3 含三部分**（mermaid block-beta 布局图 / UB 总占用公式 / 复用序）；§N.4 伪码 `#n` 与 §N.3 mermaid 同号；全文源码引用一律 GitCode blob 永久链接；**不含**独立「API 速查/汇总/具体实现」章节（API 细节融入 §N.4 伪码）；**每条 trick 必含「坏实践（反例）」字段（无可填实例写「待补充」、不省略）**；卡片以 **文末 `# 相关` 托管块**收尾（替代旧「知识来源」节，golden 出处保留在 §1 行内 GitCode 链接 + frontmatter `resource`）。`## 反模式` 专节**仅当有无法内联的反例时才写**（`### AP-N` 条目），无则不显示该节。

```
---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
category: <category>
resource: https://gitcode.com/cann/<repo>/tree/<commit>/<category>/<op_name>/
title: <op_name> 多模板分发与内存复用设计
description: <模板数 + 分发维度 + 含 mermaid 图>
tags: [<category>, <核心主题>]
paradigms: [<范式，镜像 cannbot-skills 27 枚举>]
confidence: verified
status: verified
created_at: '<YYYY-MM-DD>T00:00:00Z'
updated_at: '<YYYY-MM-DD>T00:00:00Z'
---

# <op_name> 多模板分发与内存复用设计

> **目标读者**：模型在设计阶段（Phase 4）使用本文档...
> **范围**：从 README 确认平台支持（如 "arch35 RegBase 路径，支持 A2/A3/950" 或 "910B/910C MemBase 三模板"）。标注代码路径是否为该算子唯一实现；有姊妹平台版则 `> **范围**` 互相指向。
> **前置阅读**：（如有相关 guide）

---

## 本算子速览
（数学定义 / shape / dtype / layout / 平台 / 计算管线一行 / 模板数）

## 1. 源码路径与文件职责

### 1.1 Host 侧（Tiling 计算）
| 文件 | 职责 |  ← 文件名用 Markdown 链接指向 GitCode blob URL，职责列写关键函数/结构体/行号

### 1.2 Kernel 侧（模板实现）
| 文件 | TilingKey | 类名 | 职责 |  ← 文件列同上用 GitCode blob URL 链接

### 1.3 路由分发
| 文件 | 职责 |  ← 文件列同上用 GitCode blob URL 链接

---

## 2. 算子特定优化 trick（**正式章节·前置——高杠杆速览**，把藏在各模板伪代码里的优化抽出、集中展示）

该算子的关键优化原本散落在各模板伪代码里、不易看清；**前置**抽成正式条目，让设计阶段先抓住最影响性能/精度的几处优化。每条一个小节，**标签块字段各用 `- ` 列表项单独成行；标签块与各 `**小节**`、与正文之间一律空一行**（可读性硬规范，防 Markdown 软换行粘连）：
### 2.k <trick 名 —— 一句点明它优化了什么>

- **摘要**: <一句话：优化了什么 + 怎么做>（检索字段·agent 扫读即懂）
- **触发**: <本算子何时用到：场景/条件>（检索字段）
- **golden源**: <file:line 对应的 GitCode blob URL, ...>（该 trick 在 golden 的结构化出处，供进化期"实践 vs golden"代码比对机器提取）
- **实例化公共原则**: → `<知识库根>/runbooks/operator-optimization/vv-fusion-common.md#锚点` ｜ 无（纯算子语义）
- **预期收益**: <**预留字段**；初版多为「待轨迹验证」（golden 无差代码、无现成分数）；后续按 agent 实践轨迹动态更新填实测增益 + 〔来源〕 + 置信度>
- **迁移条件**: 适用 / 前提 / 失效（按情况精简）

**坏实践（反例）**：<**必填字段·每条 trick 都要有**；无可填实例则写「待补充」（不留空、不省略，保持各 trick 字段一致）；有则负知识优先内联此处——如规则重排用 `Gather`/`GetValue·SetValue` 逐元素离散访问，而非结构化 `DataCopy`（带 stride/错位）>

**示意图**：<仅**高收益且有信心**时（双表示）——ASCII 画清"物化 vs 广播""碎 DMA vs 连续"等权衡>

> 算子 trick 标签块**不带【优化维度】【泛化层级】**（runbook 专有，算子文件出现即泄漏）。

---

## 3. 多模板分发原则

### 3.1 分发维度
决策树/优先级链 ASCII 图（对齐 golden tiling.cpp Compute() 的分支逻辑 / IsCapable 优先级链）

### 3.2 设计阶段就必须决策（→ 上提 OPT-1，本文件只留一行指针）
关键原则 + ❌反模式 + ✅正确做法（正文写在 runbook）

### 3.3 子维度：dtype × 可用 TilingKey（算子事实；dtype 模板化 <T1,T2> 本身是 C++ 技巧、非泛化优化点；精度点实例化 → OPT-7）
| dtype | T2 | 可用 TilingKey | 说明 |

---

## 4-N. 模板一/二/...：<模板名>（TilingKey=N）

模板数量 = 路由入口中 `TILING_KEY_IS()` 的分支数。每个模板包含：
### N.1 触发条件（含 golden 代码片段和 GitCode 链接）
### N.2 Tiling 参数（该模板特有的参数，来自对应的 TilingData 结构体）
### N.3 UB 分配与内存复用

本节必须包含以下三部分：

**N.3.1 UBuf 内存布局（mermaid block-beta）** — 用 mermaid `block-beta` 图可视化 UB 上各 buffer 的布局与复用。画法纪律：
- 第一行块头：每块标 TQue/TBuf + depth（如 `TQue·2`）+ 总字节（含 dtype 字节系数，如 `2·m·R'·s B`）
- 第二行子区：按偏移把块切子区（含 ReinterpretCast / 头尾分区 / 不同 dtype 视图），标子区字节
- 其下操作步骤：按调用序纵向叠放，框内 `#n · API · 语义`（`#n` 与 §N.4 伪码同号）；同一子区叠多框即复用
- 末尾 `classDef` 着色：in（蓝）/ out（橙）/ calc（绿）/ mv（紫）/ cast（红）/ pass（灰）
- 所有行 span 之和必须 = `columns` 声明值（逐行核对）

**N.3.2 UB 总占用** — 由各 buffer 的 `InitBuffer` 实参写出字节公式：
- 逐块按 InitBuffer 公式写出字节；子区字节相加须 = 块总字节
- 各块相加成总公式；如有 double buffer，注明 `depth=2` 的影响
- 由「总 ≤ UB_SIZE」反推 tile 大小的公式（引用 tiling.cpp 行号）

**N.3.3 复用序与内存复用策略表** — 列出同子区自上而下的复用链，以及 4 种复用策略的具体应用位置

### N.4 计算流程伪码（ASCII 树状伪码，标注关键 API 入参含义，`#n` 与 §N.3 mermaid 图同号）
### N.5 路由代码（入口文件的 if-else 片段）

---

## <N+1>. 模板对比总表（算子事实；【实例化: → OPT-1】，对比维度清单是多模板路由的模板集合规划部分）

对比维度从该算子的实际差异中提取，至少包含：
- dtype 支持（哪些模板可用哪种 dtype）
- Queue 深度（单缓冲 vs 双缓冲）
- TBuf 数量（区分 FP16/FP32 和 BF16 路径）
- 循环层级（几层、是否含内循环）
- 数据排布方式（拼接/连续/分段）
- 核心计算的算法差异（如全载 vs 重计算）
- Cast 流水线（有无、升/降精度策略）
- 设计复杂度（★☆☆ ~ ★★★）

## 反模式（可选·仅当有无法内联的反例时）
（`### AP-N` 条目——无法内联到具体 trick 的纯坏写法。**无此类反例则不写本节**，不留空占位。可内联的反例进对应 trick 的「坏实践」字段）

---

<!-- okf:related:start -->

# 相关

（okf 图谱托管块。示例：`- 实践案例: <card>（相对路径到 reference/api 卡） — …` + `- 相关主题: <sibling>（相对路径到姊妹/同主题成员卡） — …`。**替代旧「知识来源」节**；golden 出处保留在 §1 行内 GitCode 链接 + frontmatter `resource`。5a 手写初版「相关主题」，阶段 7 图谱刷新补全「实践案例」）

<!-- okf:related:end -->
```
