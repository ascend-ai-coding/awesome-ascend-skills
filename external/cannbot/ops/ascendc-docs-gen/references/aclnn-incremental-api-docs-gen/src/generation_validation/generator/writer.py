# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Text writer - generates modified document based on evidence"""

import json
import re
from typing import Optional, Union

from src.llm.client import LLMClient
from src.logger import debug

from ..core.models import (
    ModificationResult,
    StyleExtraction,
    StyleSummary,
)
from ..style.loader import TemplateLoader


class TextWriter:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        template_loader: Optional[TemplateLoader] = None,
    ):
        self.llm = llm_client
        self.template_loader = template_loader

    def generate(
        self,
        original: Union[str, dict],
        evidences: dict = None,
        style_summary: StyleSummary = None,
        style_extraction: StyleExtraction = None,
        doc_part: str = "",
        sub_key: str = "",
        previous_issues: list = None,
        generated_doc: str = "",
        change_reason: Optional[dict] = None,
    ) -> ModificationResult:
        if self.llm is None:
            return ModificationResult(
                generated_text=original,
                evidence_usage=[],
                modification_summary={},
            )

        doc_part_label = self.template_loader.get_label(doc_part, sub_key)
        requirement = self.template_loader.get_requirement(doc_part, sub_key)
        style_template = self.template_loader.get_template_example(doc_part, sub_key)
        evidences_str = json.dumps(evidences, ensure_ascii=False, indent=2) if evidences else "无"

        change_reason_section = ""
        if change_reason and change_reason.get("keep"):
            reason_text = change_reason.get("reason", "")
            change_reason_section = f"""## 变更检测结论（代码变更 → 该章节被判定需要更新的理由）

**触发原因**：{reason_text}

> 写作指引：围绕上述理由组织变更，只调整与变更原因相关的部分；与理由无关的原文内容默认保留，不做无意义改写。
"""
        if doc_part == "summary" and change_reason_section and evidences and "公式" in evidences_str:
            change_reason_section += """> **计算公式注意**：计算公式是代码计算逻辑的直接映射，证据（尤其计算公式相关字段）显示计算顺序、参与项或常量与原文公式不一致，公式必须按证据同步更新。

"""

        issues_section = ""
        if previous_issues:
            # 保留真原文不动；上次生成结果单独给出，避免正确内容（如原样链接）
            # 只存在于问题清单里而无法被完整对照
            issues_str = json.dumps(previous_issues, ensure_ascii=False, indent=2)
            issues_section = f"""
## 上次生成结果（存在问题的版本，请在此版本基础上修正）

```markdown
{generated_doc}
```

## 上次校验发现的问题（必须修正）

以下是上一轮生成后被校验器指出的问题，请依据上方证据与"原文"逐一修正，避免引入新错误：

```json
{issues_str}
```
"""

        prompt = f"""你是一个增量文档更新助手。代码发生了变更，你需要用新证据更新旧文档。

## 一、原文（修改前的文档 — 基于旧代码编写）

以下是当前文档的已有内容，由旧版本的代码生成。其中的内容可能已过时，需要与证据比对后按需更新。

```markdown
{original}
```

## 二、证据（新代码的事实 — 最高数据优先级）

以下证据从**变更后的新代码**中提取，每条证据的 `content` 字段记录了新代码中某个属性的**真实值**。证据是所有事实数据的最终来源。

```json
{evidences_str}
```
{change_reason_section}{issues_section}
## 文档写作风格

### 原文整篇摘要（全局视角）
{style_summary}

### 文档风格约束
{style_extraction}

### 文档写作要求
1. 范围：只生成**当前文档部分**的内容（如：{doc_part_label}），而不是生成整个文档
2. **优先级规则**：**变更理由约束范围 > 证据值 > 模板格式 > 原文格式**
   - 变更理由：约束变更范围，只调整理由覆盖的部分，与理由无关的原文默认保留
   - 证据：提供所有事实数据（维度、数据类型、取值范围等）的最终真值
   - 模板：仅提供排版结构参考，其中的具体数据值是占位示例，不代表当前算子的真实值；HTML 表格的样式属性（width 总宽、col 列宽）不属于可靠拢项，一律以原文为准（见第 12 条）
   - 原文：证据未涉及的部分可保留，与证据冲突的部分必须用证据覆盖
3. 事实型证据（category=fact）的值必须在生成文档中**精确体现**
4. 描述型证据可自由组织语言
5. 保持与原文一致的写作风格
6. **否定式证据不得反转**：代码校验 `CHECK(值 == X) → 报错` 表示 X 被禁止，绝不能把 X 写成支持的取值
（如 `OP_TILING_CHECK(format == FRACTAL_NZ)` 报错意味着 FRACTAL_NZ **不被**支持）；
format/数据类型/shape 等字段，无证据表明变化时必须保持原文取值
7. **参数可选性同步**：当证据或变更理由表明某输入参数可不传（如代码仅在 `参数 != nullptr` 时才校验其取值）或未传入时按默认值处理时，参数表该参数的**必选标记必须同步改为可选**，并在使用说明中写明默认行为
——**变更理由中明确给出的默认值必须逐字体现在使用说明中**（如理由写"默认使用1.0"，使用说明必须含"不传入时默认为1.0"）；
同一参数在约束说明与参数表中的可选性表述必须一致——约束说明改为可选而参数表仍保留必选标记属于自相矛盾，必须同步
8. **参数按小节归属**：各接口函数说明中，执行接口小节（标题形如 `## aclnnXxx`，无 GetWorkspaceSize 后缀）的参数表
**只能包含执行参数**（workspace/workspaceSize/executor/stream）；
算子的输入/输出参数只能出现在对应的 `## aclnnXxxGetWorkspaceSize` 小节参数表中，禁止并入执行接口小节
9. **链接原样保留**：原文中已有的 .md 相对链接必须原样保留，不得增删或改写路径中的目录段（如 `docs/zh/context` 不得写成 `docs/context`），除非链接指向的内容已被删除
10. **公式类约束的最小增量修改**：证据中的计算公式若用代码内部变量表达，而原文档已有对应公式，必须以**原文公式的记号**为准，
仅把代码变更引入的差异项（新增/删除的项、变化的常量）施加到原文公式上（如在原公式末尾追加新增项）；禁止用代码表达式整体替换原文公式，
禁止引入原文档不存在的中间变量名（如 tokenNeedSizeDispatch 类内部计算量）作为文档公式符号——如必须表达，按其定义内联展开为文档符号。
代码新增的**对外数值常量**（如新增字节数、次数上限）必须写入，但以文档行文方式表达。原公式中 v1/v2 等变体间的记号差异（如 Align512 与 480Align512）和各产品条目的分立结构必须保持
11. **共享代码命名铁律**：昇腾算子仓库常用同一目录的共享代码实现多个 API 版本——证据中的文件名、结构体名、注册宏、内部函数名可能携带**旧版本号**
（如 `moe_distribute_combine_v2_tiling.cpp` / `MoeDistributeCombineV2TilingFunc` /
`IMPL_OP(MoeDistributeCombineV2)` 实现的正是本文档的 `aclnnMoeDistributeCombineV3`）。
这类版本差异属共享实现常态，**不是**本文档算子名的错误：
    - **文档的算子名/API 版本是权威**：正文中的算子名一律保持本文档版本，禁止把证据中的旧版本算子名、结构体名、注册宏名、文件名写入文档，也不得据此改写文档中已有的算子名；
    - **kernel 内部变量名不得作为文档符号**（不限于公式）：约束描述、shape 说明等一切场景，禁止引入 kernel 内部计算量/结构体字段名（如 `maxSharedGroupNum` 类），
    如需表达其约束的量，按其定义内联展开为文档既有符号（与第 10 条公式规则同理）；
    - **其他算子变体的内容禁止引入**：共享代码文件中可能包含其他算子变体（如 RMSNorm 融合系）的参数与约束，只采用与本文档算子及其 aclnn 接口签名相关的证据内容
12. **HTML表格样式原样保留**：原文中已有的 HTML 表格标签（`<table>`/`<colgroup>`/`<col>`）
及其 style 属性（table-layout、width 总宽、各 col 列宽）必须**逐字符原样保留**，
禁止按模板样例或"模板具体要求"中的列宽/总宽重排、取整或省略
（如原文 `width: 1154px` 不得改成 `1000px`，原文各 col 列宽 `153px/121px/880px` 不得替换为模板默认值 `180px/100px/720px`）；
仅当证据表明参数行/列结构发生变化时，才允许在原文样式基础上做对应的最小调整；仅当原文该处不存在表格时，才按"模板具体要求"新建表格

## 模板参考（仅用于排版和结构）

除修改现有不符内容外，还需尽可能向标准模板靠拢（仅指章节结构、字段构成与行文组织；原文已有 HTML 表格的样式属性按第 12 条原样保留，不参与靠拢）。

### 模板样例
```markdown
{style_template}
```

### 模板具体要求
{requirement}

## 输出格式

### **evidence_conflict 判断标准**：
- 仅当证据间存在**明确且不可调和**的矛盾、无法同时满足所有证据时，才标记为 evidence_conflict
- 典型场景：
  - 部分证据已更新（如头文件中新增了 BF16 数据类型支持），但其他证据仍引用旧值（如参数校验代码中未包含 BF16），且差异无法用注释滞后等原因解释
  - 两个同等权威的证据对同一事实给出**明确相反**的描述（如一个明确支持 INT8，另一个明确排除了 INT8）
  - 证据要求的修改与文档中其他引用不一致，重新生成无法同时满足所有证据
- 不同来源的证据对同一事实给出不同的值（如参数名、数据类型支持列表、返回值等），仅当差异**明确且不可调和**、无法用表述详略/注释滞后/作用域不同等合理原因解释时，才标记为 evidence_conflict；
即使依据优先级规则可以选出"更权威"的一方，此类明确冲突仍须标记。因为这种不一致通常意味着代码修改不完整，文档无法确定应该采用哪个值
- **若冲突不能完全确定**（如差异可能仅因表述详略不同、注释未同步、不同文件关注点不同），**不要标记** evidence_conflict，视为正常证据差异，按优先级规则选用值即可
- 若判断为 evidence_conflict，need_change 仍可为 true（尝试生成），但需在 conflict_details 中说明冲突点
- conflict_details 中的每一条冲突描述都**必须**注明对应的源文件名（如 aclnn_abs.h），让读者能直接定位到具体文件

### 请严格按以下格式输出：

<analysis>
{{
    "need_change": true/false,
    "reason": "判断理由（如：证据与原文一致无需修改 / 证据间存在冲突无法确定正确内容 / 需要修改xxx）",
    "evidence_conflict": false,
    "conflict_details": []
}}
</analysis>

<changes>
（变更点列表，JSON数组。若 need_change 为 false，输出空数组 []）
[
  {{"type": "新增/删除/修改", "location": "位置", "content": "变更内容"}}
]
</changes>

<document>
（修改后的当前章节完整内容，仅包含本章节标题与正文，不包含其他章节，纯markdown文本）
</document>
"""
        debug(f"[Generating]\n{prompt}")
        raw = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是一个增量文档更新助手。你的核心任务是：代码发生了变更，原文档基于旧代码编写，证据来自变更后的新代码。"
                        "你必须用证据中的新事实覆盖原文中受影响的旧值。证据中的事实值（维度、数据类型、取值范围、格式等）始终高于原文和模板中的旧值。"
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )

        analysis_match = re.search(r"<analysis>(.*?)</analysis>", raw, re.DOTALL)
        doc_match = re.search(r"<document>(.*?)</document>", raw, re.DOTALL)
        changes_match = re.search(r"<changes>(.*?)</changes>", raw, re.DOTALL)

        analysis = {}
        if analysis_match:
            try:
                analysis = json.loads(analysis_match.group(1))
            except json.JSONDecodeError:
                analysis = {}

        generated_text = doc_match.group(1).strip() if doc_match else raw
        changes = []
        if changes_match:
            try:
                changes = json.loads(changes_match.group(1))
            except json.JSONDecodeError:
                changes = []

        return ModificationResult(
            generated_text=generated_text,
            evidence_usage=[],
            modification_summary={
                "summary": f"基于 {len(evidences) if evidences else 0} 个证据生成",
                "changes": changes,
                "analysis": analysis,
            },
        )
