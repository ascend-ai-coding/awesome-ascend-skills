# 完整候选文档审阅

## 输入与边界

父入口主 Agent（手动模式由用户）完成 `tools/aclnn-finalize.mjs --prepare` 后，完整读取 `4_generate.md` 和 `4_review_input.json`。候选已经包含原型、公式、固定执行接口和调用示例；所有交付章节均在审阅范围内。

按需读取 `1_extract.md` 的语义基准和相关参数、`2_compare.md` 的差异结论、`3_formula.md` 的变量映射、`0_prototype.md` 的精确声明和 `0_call_example.md` 的示例。无需默认全文读取全部过程报告。证据不足时记录缺口或返回上游补证，不探索源码补造事实。

## 必查内容

- 原型、参数表、公式变量与目标 API 一致，参数 dtype、format、shape、空 Tensor 和连续性不能与提取证据相反。
- 核对 `1_extract.md` 的资料状态及“资料—源码对照”：已提供资料中的相关设计意图有处理结论，正文采用已验证的当前实现，冲突和未确认项在交接中保留。仅有注册声明不能证明具体接口拦截或错误码；缺少对应证据时不得把这些行为写成确定事实。
- 保留产品、重载、场景和跨参数条件；约束不能超出证据适用范围。缺少检查不等于支持，不以通用上限补当前 API 的未知范围。
- 依父 Skill 权威模板核对成品结构、产品 `<term>` 标记、原型展示和版本演进。compare 失败时省略未证实的版本差异，不引用内部失败文件。
- 调用示例必须调用精确目标 API 及对应 GetWorkspaceSize，不能借用其他版本的调用、参数、结果或行为。无可验证示例时记录缺失标记，文档为待补齐草稿，不声称完整交付。
- 审阅整篇文档中的内部证据、开发机路径和实现说明。改写为读者需要的接口行为，保留合法公开头文件、公开安装路径、原型、代码和许可证声明；不要按 `.h`、`.cpp` 等词面删除必要内容。
- 最终化工具的审阅提示需结合证据、数学含义和链接实际目标核对；同名不等于泄露。在交付说明中记录处理结论，不为消除提示改写技术含义。
- 读取 `references/product-document-format.md`，核对全文产品差异描述、标签范围、配对与唯一 ID；术语依据对应版本的公开官方资料，避免机械替换技术含义。

## 输出补丁

写入 `5_review_patch.json`，输出 JSON 对象本身。`reviewedDocumentSha256` 原样复制 `4_review_input.json` 的摘要；它绑定本次完整候选，不由模型猜测或重新计算。

```json
{
  "status": "pass",
  "reviewedDocumentSha256": "<4_review_input.json 中的摘要>",
  "issues": [],
  "documentEdits": [],
  "skeletonEdits": [],
  "formulaBlockEdits": [],
  "failureReason": ""
}
```

- `pass`：没有需要修改的问题，编辑数组均为空。
- `fix`：可以依据现有证据修正。每个 `documentEdits` 项包含 `section`、`oldText`、`newText`、`reason` 字符串，oldText 在 `4_generate.md` 中唯一匹配。保持原型、代码和公式字面量完整。
- `fail`：关键证据缺失、无法完成审阅或无法可靠构造补丁，填写 failureReason，停止交付。
- issues 项含 severity、section、message、basis；可选缺项可以记录为 warning，但不能将关键冲突降级为 warning 后放行。

`skeletonEdits` 与 `formulaBlockEdits` 仍支持对各自源产物的最小修正；优先使用 documentEdits，避免多个数组修改同一内容。模型不应用补丁、不写最终命名文件，由外部工具完成交付校验。
