> 通过父 Skill `ascendc-docs-gen` 按需加载本流程。`full` 指为本次选定接口生成完整文档，不代表全部 API 形态已完成验证。

# ACLNN API 文档生成

术语使用对应版本的公开官方资料，不能依据词表机械改写代码或技术含义。

根据当前接口证据生成文档。源码和已核验资料是事实依据；未发现证据时记录“尚未确认”，不推导为支持、不支持或具体取值。

## 输入与阶段关口

未提供算子源码目录时，请求实际路径并停在输入确认阶段，不遍历工作区或整机寻找同名算子。已给路径不可读时报告该路径的阻塞，不用其他算子替代；草拟边界仍按父入口处理。取得可读源码后，由主 Agent 执行预处理，不能将“尚无预处理产物”本身当作要求用户手动运行工具的理由。

按当前可执行阶段读取任务：先确认输入并执行预处理，产物有效后再读取 extract/compare，随后按依赖读取 formula/generate_skeleton，完整候选产生后再读取 review。未达到阶段关口时不提前加载全部 tasks、配置和领域参考；这不省略父入口要求先完整读取的写作指南、模板和示例。

执行前完整读取父 Skill 的 [源码提取指南](../source-extraction-guide.md)、[权威模板](../aclnn-api-doc-template.md) 和 [成品示例](../example-aclnn-md.md)。它们定义事实核验和成品结构；本模块补充阶段编排、证据绑定和单值产品标签要求，不另立成品模板。示例中的具体接口事实不能用于目标接口。

所有命令中的相对路径以本文件所在的模块根目录为准。Markdown 链接相对于所在文件解析。`tasks/` 定义各阶段任务，`tools/` 提供外部确定性工具，`config/` 保存生效配置，`references/` 提供按需参考。

## 运行依赖

Markdown 检查首次实际解析时从固定版本 CDN 下载解析器与上游许可证，按 [依赖清单](tools/scripts/lib/markdown-it-dependency.json) 校验 SHA-256 后加载；不使用 latest。两模块默认共享用户目录下 `.cache/canndoc-api-docs-gen` 缓存，也可用 `CANNDOC_DEPENDENCY_CACHE` 指定可写缓存根目录。有效缓存可离线使用，损坏或缺失时重新下载；网络、写入或校验失败则停止，不跳过文档检查。仅导入工具模块不会下载。

离线环境使用前，可联网执行 `node tools/scripts/lib/markdown-it-loader.js --prepare`，再将包含解析器和许可证的缓存提供给离线环境，并设置缓存根目录；也可用此命令提前准备当前用户缓存。首次无缓存运行需要访问清单中的 HTTPS 地址。

## 开发者资料交接

按[父入口](../../SKILL.md)的统一输入策略，由主 Agent 在预处理前完成资料处理；交接资料状态、已读文件定位、设计意图及尚未解决的缺项。咨询不进入本模块，实际生成不能用预处理结果代替开发者资料的读取。

进入 extract 时，明确传入目标接口、本轮已读资料的文件定位和设计意图，不能假定分析任务继承主 Agent 的对话历史。extract 将资料状态及“资料—源码对照表”写入已有的 `1_extract.md`，分别保留设计预期、源码事实、证据及待确认项；后续生成和审阅读取该对照，不以源码预处理产物代替开发者资料。过程报告只记录相关事实和定位，不复制整份资料。

## 执行步骤

1. 确定算子目录、项目根目录和目标接口。用户已指定接口时使用该接口；否则由父入口主 Agent 使用 `tools/scripts/detect-interfaces.js` 检测并按实际候选确认，避免启动交互式命令等待输入。可选择多个算子或接口，每个目标独立生成，过程产物保存在 `<operator-dir>/outputs/<interface>/`，正式文档交付到父 Skill 约定的 `<operator-dir>/docs/<interface>.md`。单次 preprocess 命令处理一个接口，批量入口处理选择集合。
2. 检查下表的全部预处理产物。外部预处理失败、存在 `0_*_FAILED.md` 或缺少必需输入时停止对应目标；分析子任务不临时补跑脚本；缺少本轮输入时返回主 Agent 执行预处理。
3. 执行提取与对比，再执行公式与骨架。外部编排在依赖满足时并行启动 extract/compare，以及 formula/generate_skeleton；工具不支持并行时如实记录执行方式。每次任务使用本轮产物，不能仅因文件存在复用旧结果。
4. 公式和骨架完成后，由父入口主 Agent 执行 `aclnn-finalize.mjs --prepare`，组装包括调用示例的完整候选文档。返回 review，读取完整候选文档并按需核对证据，输出绑定候选摘要的补丁。
5. 主 Agent 调用工具应用补丁并校验交付。报告实际产生的文件；只有最终合成及正式文件写入成功才声明文档已交付。资料—源码冲突的采用结论和待确认事项在交付说明中单列，冲突分析不混入 API 正文；必需技术事实缺失时仍按原有草稿规则处理。正式文档的相对链接按 docs 目录核对；outputs 中的时间戳副本用于保留审阅记录，不作为 README 链接目标。若同时请求 README，继续按父 Skill 的 README 模板在算子根目录生成或更新 README，逐一列出该算子的全部 ACLNN 入口与实际文档/示例链接，不因本次仅重生成其中一个接口而删去其他入口。

父入口主 Agent 负责执行预处理、候选组装和最终化工具；LLM 分析子任务负责推理、审阅和阶段文件。用户明确选择手动编排时，工具步骤由用户执行，按使用说明交接；工具不可用时报告当前阶段与缺失能力。外部命令见 [工具说明](tools/README.md)，完整示例见 [使用说明](references/usage.md)。

## 阶段与产物

文件编号用于命名，不代表严格串行顺序。此表是输入、输出和职责的权威清单。

| 阶段 | 输入/任务 | 输出 |
|---|---|---|
| 外部预处理 | 源码；`config/error_codes.yml`、`config/product_mapping.yml` | `0_extract_plan.json`、`0_prototype.md`、`0_cross_dir.md`、`0_chip_evidence.md`、`0_error_codes.md`、`0_call_example.md` |
| extract | `tasks/extract.md`、预处理证据、主 Agent 交接的设计意图与资料定位；按任务读取目标源码及依赖 | `1_extract.md`（含资料状态及资料—源码对照） |
| compare | `tasks/compare.md`、原型和关联接口代码 | `2_compare.md`，失败时 `2_compare_FAILED.md` |
| formula | `tasks/formula.md`、提取报告；计算语义缺口可定向核验已登记源码，新证据返回主 Agent 登记 | `3_formula.md` |
| generate_skeleton | `tasks/generate_skeleton.md`、提取/对比/原型 | `4_generate_skeleton.md` |
| 外部候选组装 | 原型、公式块、骨架、调用示例 | `4_generate.md`、`4_review_input.json` |
| review | `tasks/review.md`、完整候选、摘要、按需证据 | `5_review_patch.json` |
| 外部交付 | 当前候选和绑定其摘要的有效补丁 | `5_review.md`、过程目录中的时间戳副本、`<operator-dir>/docs/<interface>.md` |

compare 失败不必阻断生成：省略未证实的可选版本差异，原因留在过程报告。精确原型等关键事实缺失时不能交付完整文档。其他事实缺口在阶段报告中明确记录，不编造默认值。

## 外部组装与交付

预处理在读取证据前登记算子源码、已追踪外部依赖、示例和原型来源的文件摘要，保存在 `0_extract_plan.json` 的 `sourceSnapshot`。取证范围超出已登记文件时，主 Agent 用 `aclnn-preprocess.mjs --source-file <路径>` 重新预处理（可重复，相对于项目根目录）；工具会归档旧阶段产物，随后必须重新执行本轮提取、对比、公式、骨架和审阅，不沿用旧公式或候选。文件变化或旧会话缺少摘要时返回取证阶段，不在最终化时补记当前摘要。只有完整标题、适用章节、精确原型和目标调用示例均通过检查，才声明完整交付；缺少示例或必需事实时保留草稿。

父入口主 Agent（手动模式由用户）使用同一 `tools/aclnn-finalize.mjs` 入口：

- `--prepare` 替换骨架中的 `{{ACLNN_PROTOTYPE_BLOCK_PLACEHOLDER}}`、`{{ACLNN_FORMULA_BLOCK_PLACEHOLDER}}` 和 `{{ACLNN_EXECUTE_INTERFACE_BLOCK_PLACEHOLDER}}`，加入调用示例并规范化，产生完整候选与 SHA-256 摘要。此时产物仍是待审阅草稿。
- review 完整阅读候选，复制摘要到补丁的 `reviewedDocumentSha256`。正文修正使用 `documentEdits`；原型和示例必须准确对应目标 API。缺失或错误版本的示例不能冒充目标示例。
- 不带 `--prepare` 时检查候选与当前输入一致、补丁摘要一致，再应用编辑并执行产品、标签和公开输出边界校验。失败返回非零，不回退发布。输入或候选改变后须重新组装并审阅。

示例匹配按真实调用的目标接口及 GetWorkspaceSize 判断，不按文件名前缀选择其他版本。预处理扫描 `op_api/op_host` 的声明；opbuild 生成或安装头文件通过 `--prototype-manifest` 显式接入，来源格式与核验边界见 [原型来源](references/prototype-sources.md)。缺少同版本构建产物时报告原型证据不足，不能认定接口不存在，更不能根据 OpDef 猜原型。

## 参考资源

生成和审阅前读取 [产品名称与标签规范](references/product-document-format.md)。错误码符号和数值以 `config/error_codes.yml` 为准；产品映射以 `config/product_mapping.yml` 为准。

按目标特性读取，不将所有领域资料作为每次任务的前置输入：

- [官方概念与术语来源](references/official-sources.md)：涉及 dtype、广播、量化、稀疏模式或术语时按需查阅对应版本。
- `references/formula_style.md`：生成公式时使用。
- `references/chip_mapping.md`：解释产品映射规则，不能替代具体产品支持证据。
- [原型来源](references/prototype-sources.md)：目标接口由构建生成或需要接入安装头文件时读取。

仅需单独定位完整文档格式问题时，可使用 [只读诊断](references/document-checks.md)；它不用于强制迁移父 Skill 的局部修订。
