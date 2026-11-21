# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""LLM-based change evaluator to filter out trivial/small changes that don't require documentation updates."""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from src.generation_validation.config import format_doc_part, format_doc_parts
from src.llm.client import LLMClient
from src.logger import debug, error, info, warning

_TILING_FILE_PATTERN = re.compile(r"(op_host/.*tiling.*\.cpp|tiling.*\.cpp|tilling\.h)")


SYSTEM_PROMPT = """你是一个专业的代码与文档关联性分析专家。

你的任务是判断用户修改代码后，是否需要同步更新对应文档部分（doc part/subpart）。

你需要基于以下信息做出判断：
1. 用户实际修改了哪些文件的哪些内容（git diff格式的变更详情）
2. 变更所影响的文档部分（doc_part）
3. 文件与文档部分的映射关系

对于每个doc_part，你需要判断"是否有必要修改对应文档"，输出格式为JSON数组。

判断标准（默认 keep: true，仅以下明确情况才标记 keep: false）：

keep: false 的情况（仅限以下，其他一律 keep: true）：
- 仅修改了空白字符、缩进、换行
- 仅修改了代码注释内容
- 仅添加了调试日志（如 printf、LOG）
- 仅修改了函数体内部实现逻辑，且函数声明（函数名、参数名、参数类型、返回值）完全未变
  **例外**：如果修改的常量或变量约束了对外的维度数、shape范围、dtype、取值范围等数值约束，则 constraint 和 sub_function_hash 均需 keep: true

keep: true 的情况（包括但不限于）：
- 函数名发生变化（如 aclnnXxx → aclnnXxx_new）
- 函数参数名发生变化（如 out → out_new）
- 函数参数类型或数量变化
- 返回值类型变化
- 新增或删除函数
- 头文件（.h）中任何声明级别的变化
- 约束条件、支持的数据类型、支持的格式等变化
- 算子对输入的接受范围变化（如运行时将输入 dtype 重解释为另一种 dtype、开始接受以 UINT8 等类型存储的 FP8 视图输入），
即使函数签名与原校验代码未变 → 同时影响 constraint 和 sub_function_hash
- 维度数限制、shape 范围、dtype/格式约束等数值常量变化（如 MAX_DIM_LEN、MIN_SHAPE 等）→ 同时影响 constraint 和 sub_function_hash
- 新增或删除子函数
- 示例代码（example）文件内容变化
- 任何用户在文档中能看到的内容发生变化

## doc_part 代号与文档章节的对应关系
- "support" → "## 产品支持情况"（产品/芯片/架构支持矩阵）
- "summary" → "## 功能说明"（算子功能描述、算法说明）
- "function_proto" → "## 函数原型"（接口声明、函数签名）
- "sub_function_hash" → "## aclnnXxx" / "## aclnnXxxGetWorkspaceSize"（子函数参数说明表，包含：描述、输入/输出、
使用说明、数据类型、数据格式、维度shape、非连续Tensor，以及错误码说明。前缀匹配所有 ## aclnn 开头的章节。约束值变更时需联带更新表中对应的具体数值列。）
- "constraint" → "## 约束说明"（数据类型、shape、取值范围等约束条件）
- "example" → "## 调用示例"（调用示例代码）

注意：只评估文档内容是否需要变更，不考虑代码正确性。如果不确定，标记为 keep: true。

## doc_part 联动规则
- 维度数限制、shape 范围、dtype 约束等数值变化时，constraint（约束说明文本）和 sub_function_hash（参数说明表中的 shape/dtype 列）记录的是同一信息的两种展示形式，两者必须联带更新。
- 支持的格式（format/dmat）变化时，constraint 和 sub_function_hash（参数说明表中的 dmat 列）联带更新。
- 评估 reason 必须显式点名计算公式相关变化：代码变更涉及计算逻辑（计算顺序、新增/删除参与项、常量变化）时，
reason 必须写明"涉及计算公式/计算顺序变化"及具体内容——该 reason 是下游生成环节确定修改范围的约束，reason 未提公式时生成环节会默认保留原文档公式。"""


USER_PROMPT_TEMPLATE = """请分析以下代码变更，判断每个affected doc_part是否需要更新对应文档。

## 文档内容
``` markdown
{doc_content}
```

## 变更详情（changed_detail）
{changed_detail}

## 受影响的doc_part列表
{affected_subparts}

## 文件与doc_part映射关系
{file_subpart_mapping}

请按照以下JSON格式返回结果：
```json
{{
  "evaluation": [
    {{
      "doc_part": "support",
      "keep": false,
      "reason": "只修改了注释和换行，不影响文档内容"
    }},
    {{
      "doc_part": "constraint",
      "keep": true,
      "reason": "函数参数从3个变为4个，约束条件发生变化"
    }}
  ]
}}
```

其中：
- "doc_part": 需要评估的doc_part名称
- "keep": true表示需要修改对应文档，false表示不需要
- "reason": 简要说明判断理由

如果某个doc_part没有受到任何实质性影响，保持其keep为false。"""


def _build_user_prompt(
    doc_content: str,
    changed_detail: Dict[str, Dict],
    affected_subparts: List[str],
    file_subpart_mapping: Dict[str, List],
) -> str:
    changed_detail_json = json.dumps(changed_detail, ensure_ascii=False, indent=2)
    affected_subparts_json = json.dumps(affected_subparts, ensure_ascii=False, indent=2)
    filtered_mapping = {}
    for k, v in file_subpart_mapping.items():
        if k not in affected_subparts:
            continue
        cleaned = []
        for file_entry in v:
            if isinstance(file_entry, dict):
                cleaned.append(file_entry["file_name"])
            else:
                cleaned.append(file_entry)
        filtered_mapping[k] = cleaned
    file_subpart_json = json.dumps(filtered_mapping, ensure_ascii=False, indent=2)

    return USER_PROMPT_TEMPLATE.format(
        doc_content=doc_content,
        changed_detail=changed_detail_json,
        affected_subparts=affected_subparts_json,
        file_subpart_mapping=file_subpart_json,
    )


def evaluate_changes(
    doc_content: str,
    affected_subparts: List[str],
    changed_detail: Dict[str, Dict],
    file_subpart_mapping: Dict[str, List],
) -> List[str]:
    """
    使用LLM判断哪些affected_subparts真正需要更新文档。

    Args:
        affected_subparts: run_detection返回的doc_subpart_todo_list
        changed_detail: file_change_detector返回的详细变更信息
        file_subpart_mapping: 文件与doc_part的映射关系

    Returns:
        verified_todo_list: LLM验证后确认真实需要更新的doc_part列表
    """
    llm_client = LLMClient()

    user_prompt = _build_user_prompt(doc_content, changed_detail, affected_subparts, file_subpart_mapping)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = llm_client.chat_with_json(messages, temperature=0.0)
        if not isinstance(response, dict):
            warning(f"LLM返回了非字典类型: {type(response)}")
            return list(affected_subparts), {}
        evaluation = response.get("evaluation", [])
        if evaluation is None:
            evaluation = []

        verified_todo_list = []
        all_reasons = {}
        for item in evaluation:
            doc_part = item.get("doc_part", "")
            keep = item.get("keep", True)
            reason = item.get("reason", "")

            all_reasons[doc_part] = {
                "keep": keep,
                "reason": reason,
                "source": "general",
            }
            if keep:
                verified_todo_list.append(doc_part)
                info(f"[KEEP] {doc_part}: {reason}")
            else:
                info(f"[SKIP] {doc_part}: {reason}")

        info(
            f"LLM评估完成: 原始{len(affected_subparts)}个doc_part, "
            f"保留{len(verified_todo_list)}个, 过滤{len(affected_subparts) - len(verified_todo_list)}个"
        )
        return verified_todo_list, all_reasons

    except Exception as e:
        import traceback

        error(f"LLM评估失败，返回全部affected_subparts: {e}\n{traceback.format_exc()}")
        return list(affected_subparts), {}


def _is_tiling_file(file_path: str) -> bool:
    return bool(_TILING_FILE_PATTERN.search(file_path.replace("\\", "/")))


TILING_SYSTEM_PROMPT = """你是一个专业的昇腾算子 tiling 代码变更分析专家。

你的任务是分析 tiling 文件的代码变更（git diff 格式），判断每处变更是否影响文档，以及影响哪些文档章节（doc_part）。

## 可用的 doc_part 列表
- "constraint": 约束说明（数据类型、shape、取值范围等约束条件）
- "sub_function_hash": 子函数列表/签名变化
- "support": 产品/架构支持情况
- "summary": 功能说明/算法描述
- "function_proto": 函数原型（接口声明）
- "example": 调用示例代码

## 总体原则
仅修改函数体内部实现逻辑且对外语义不变的变更（如重构、变量重命名、拆分函数、调整判断顺序），不影响文档。如果不确定，标记为 keep: true。

**校验宏语义（方向极易读反）**：`OP_TILING_CHECK(cond, log, ret)` / `CHECK` 的第一个参数是**报错条件**——cond 成立才报错。
如 `OP_TILING_CHECK(format == FORMAT_FRACTAL_NZ, "xxFormat is invalid")`
表示 FRACTAL_NZ **被禁止**（支持 ND 等），不得描述为"要求/必须 FRACTAL_NZ"；
`CHECK(dtype != DT_FLOAT, ...)` 表示必须为 FLOAT。描述约束变化时必须按此方向表述。

## 判断规则（严格按以下规则分类）

### 1. OP_TILING_CHECK / OP_LOGE 变更
- 约束条件变化（新增/修改/删除 check 条件、放宽/收紧约束）→ "constraint"
- 涉及到新增/修改错误码 → "sub_function_hash"
- 同时涉及约束变化和错误码 → "constraint" + "sub_function_hash"
- 修改 check 但约束语义不变（重构拆分、变量重命名、调整判断顺序）→ 不影响文档

### 2. 业务容器变更
承载业务语义的集合（*_MAP/*_map/*_SET/*_set/std::vector 白名单/std::array 配置/unordered_map 等）：
- 新增或删除条目 → 按容器类别影响对应 doc_part
- 仅重构容器结构但内容不变（如 vector 改 array、调整初始化方式）→ 不影响文档
容器类别与 doc_part 映射：
- 量化/数据类型相关（如 QUANT_MODE_MAP/DTYPE_MAP/SUPPORTED_DTYPE_SET 等）→ "sub_function_hash" + "constraint"
- 架构/产品相关（如 ARCH_MAP/SUPPORTED_ARCH_SET 等）→ "support"
- 功能/算法相关（如 OP_MODE_MAP/ALGO_SET/FEATURE_SET 等）→ "summary" + "constraint"
- 其他业务集合 → 不影响文档

### 3. CheckTensorShape* 变更
- 基本范围变更（取值区间、维度数限制）→ "sub_function_hash"
- 特定场景约束（如 quantMode=4 且 FP4 时 H 偶数）→ "constraint"
- 仅重构实现但检查逻辑不变 → 不影响文档

### 4. constexpr 常量变更
- 新增或修改 constexpr 常量，且该常量表示数值约束（如最大维度数、最小shape等）→ "constraint"
- 仅内部使用的常量 → 不影响文档

### 5. enum 变更
- 新增 enum 值，且对应新功能/新模式 → 不影响文档
- 新增 enum 值改变了已有约束语义 → "constraint"

### 6. 架构条件编译
- 新增或删除 #ifdef ASCEND_xxx / 架构条件编译分支 → "support"
- 仅调整已有分支内部实现逻辑 → 不影响文档

### 7. 仅内部重构
- 函数拆分、enum 重命名但语义不变、代码格式调整 → 不影响文档

## 输出要求
对每个受影响的 doc_part，输出 keep: true 并说明理由。如果某类变更不影响文档，输出 keep: false。
如果不确定，标记为 keep: true。"""

TILING_USER_PROMPT_TEMPLATE = """请分析以下 tiling 文件的代码变更，判断每处变更影响哪些文档章节。

## 变更的 tiling 文件及 diff
{tiling_diffs}

请按照以下JSON格式返回结果：
```json
{{
  "evaluation": [
    {{
      "doc_part": "constraint",
      "keep": true,
      "reason": "新增了 OP_TILING_CHECK 检查 inputDtype 必须为 FLOAT，新增约束条件"
    }},
    {{
      "doc_part": "support",
      "keep": false,
      "reason": "无架构条件编译变更"
    }}
  ]
}}
```"""


def analyze_tiling_changes(
    changed_detail: Dict[str, Dict],
    console: Optional[Any] = None,
) -> Tuple[Set[str], Dict[str, Dict]]:
    tiling_diffs = {}
    for file_path, detail in changed_detail.items():
        if _is_tiling_file(file_path):
            tiling_diffs[file_path] = detail

    if not tiling_diffs:
        info("无 tiling 文件变更，跳过 tiling 分析")
        return set(), {}

    info(f"检测到 {len(tiling_diffs)} 个 tiling 文件变更: {list(tiling_diffs.keys())}")

    diffs_text = ""
    for file_path, detail in tiling_diffs.items():
        diffs_text += f"\n### {file_path} (type: {detail.get('type', 'unknown')})\n"
        diffs_text += f"```diff\n{detail.get('diff', '(no diff)')}\n```\n"

    user_prompt = TILING_USER_PROMPT_TEMPLATE.format(tiling_diffs=diffs_text)
    messages = [
        {"role": "system", "content": TILING_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    llm_client = LLMClient()
    try:
        response = llm_client.chat_with_json(messages, temperature=0.0)
        if not isinstance(response, dict):
            warning(f"Tiling LLM返回了非字典类型: {type(response)}")
            return set(), {}

        evaluation = response.get("evaluation", [])
        if evaluation is None:
            evaluation = []

        affected = set()
        all_reasons = {}
        for item in evaluation:
            doc_part = item.get("doc_part", "")
            keep = item.get("keep", False)
            reason = item.get("reason", "")
            if keep and doc_part:
                affected.add(doc_part)
                all_reasons[doc_part] = {
                    "keep": keep,
                    "reason": reason,
                    "source": "tiling",
                }
                info(f"[TILING-KEEP] {doc_part}: {reason}")
            else:
                info(f"[TILING-SKIP] {doc_part}: {reason}")

        info(f"Tiling 分析完成: {len(affected)} 个 doc_part 受影响: {affected}")
        return affected, all_reasons

    except Exception as e:
        import traceback

        error(f"Tiling LLM分析失败: {e}\n{traceback.format_exc()}")
        return set(), {}


def run_evaluation(
    evidence_dir: str,
    before_dir: str,
    after_dir: str,
    doc_content: str,
    fine_grained: bool = True,
    console: Optional[Any] = None,
    repo_before_dir: str = None,
    repo_after_dir: str = None,
) -> Tuple[List[str], Dict[str, Dict], List[str], List[str]]:
    """
    完整的变更评估流程：运行检测 + LLM过滤。

    Args:
        evidence_dir: 证据目录路径
        before_dir: 修改前代码目录
        after_dir: 修改后代码目录
        fine_grained: 是否启用细粒度检测
        console: 可选，rich console 对象，用于输出结果
        repo_before_dir: 全仓修改前路径（用于检测 repo_other 依赖变更）
        repo_after_dir: 全仓修改后路径（用于检测 repo_other 依赖变更）

    Returns:
        verified_todo_list: LLM验证后的待处理doc_part列表
        all_reasons: 所有doc_part的评估理由，格式为 {doc_part: {"keep": bool, "reason": str}}
        modified_file_dir_list: 有改动的文件的相对路径列表
        repo_other_changed_files: 全仓范围内算子目录之外的其他变更文件列表
    """
    from pprint import pprint

    from src.change_detection.doc_part_change_detection.docpart_file_extractor import (
        DocpartFileExtractor,
    )
    from src.change_detection.doc_part_change_detection.subpart_mapper import (
        run_detection,
    )

    extractor = DocpartFileExtractor(evidence_dir, fine_grained)
    if fine_grained:
        file_subpart_mapping = extractor.extract_fine_grained_mappings()
    else:
        file_subpart_mapping = extractor.extract_mappings()

    (
        affected_subparts,
        changed_detail,
        mapped_files,
        unmatched_files,
        repo_other_changed_files,
    ) = run_detection(
        evidence_dir,
        before_dir,
        after_dir,
        fine_grained=fine_grained,
        enable_llm_filter=True,
        console=console,
        repo_before_dir=repo_before_dir,
        repo_after_dir=repo_after_dir,
    )

    if "example" in affected_subparts and "example" not in file_subpart_mapping:
        from src.change_detection.doc_part_change_detection.subpart_mapper import (
            _is_example_file,
        )

        example_file_list = [f for f in list(changed_detail.keys()) + mapped_files if _is_example_file(f)]
        example_file_list = sorted(set(example_file_list))
        if fine_grained:
            file_subpart_mapping["example"] = [{"file_name": f, "line_range": None} for f in example_file_list]
        else:
            file_subpart_mapping["example"] = example_file_list

    modified_file_dir_list = [str(Path(after_dir) / f) for f in mapped_files]
    unmatched_file_dir_list = [str(Path(after_dir) / f) for f in unmatched_files if f.endswith((".h", ".cpp"))]

    unmatched_set = set(unmatched_files)
    matched_changed_detail = {f: d for f, d in changed_detail.items() if f not in unmatched_set}

    has_tiling = any(_is_tiling_file(f) for f in matched_changed_detail)

    if not affected_subparts and not has_tiling:
        console.print("\n  [yellow]无受影响的 doc part 且无 tiling 变更，跳过 LLM 变更评估[/yellow]")
        info("affected_subparts 为空且无 tiling 变更，跳过 LLM 验证")
        return (
            [],
            {},
            modified_file_dir_list + unmatched_file_dir_list,
            repo_other_changed_files,
        )

    console.print(
        Panel(
            f"关联 doc parts: {format_doc_parts(affected_subparts)}\nTiling 变更: {has_tiling}",
            title="[yellow]LLM 变更评估[/yellow]",
            border_style="yellow",
        )
    )

    evaluate_result = []
    all_reasons = {}
    tiling_affected = set()
    tiling_reasons = {}

    task_labels = []
    if affected_subparts:
        task_labels.append("evaluate")
    if has_tiling:
        task_labels.append("tiling")

    prog_console = (
        console
        if (console is not None and getattr(console, "file", None) is not None and console.file.isatty())
        else Console()
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=prog_console,
        disable=not sys.stdout.isatty(),
    ) as progress:
        overall = progress.add_task("  LLM 变更评估", total=len(task_labels))

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            if "evaluate" in task_labels:
                futures[
                    executor.submit(
                        evaluate_changes,
                        doc_content=doc_content,
                        affected_subparts=affected_subparts,
                        changed_detail=matched_changed_detail,
                        file_subpart_mapping=file_subpart_mapping,
                    )
                ] = "evaluate"
            if "tiling" in task_labels:
                futures[
                    executor.submit(
                        analyze_tiling_changes,
                        changed_detail=matched_changed_detail,
                        console=console,
                    )
                ] = "tiling"

            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    result = future.result()
                    if task_name == "evaluate":
                        evaluate_result, all_reasons = result
                    elif task_name == "tiling":
                        tiling_affected, tiling_reasons = result
                except Exception as e:
                    error(f"{task_name} 任务异常: {e}")
                progress.update(overall, advance=1, description=f"  LLM 变更评估 ({task_name} 完成)")

    final_set = set(evaluate_result) | tiling_affected
    verified_todo_list = sorted(final_set)

    conflicts = set(all_reasons.keys()) & set(tiling_reasons.keys())
    for dp in conflicts:
        ev_keep = all_reasons[dp]["keep"]
        if not ev_keep:
            info(f"冲突解决: {dp} evaluate=SKIP 但 tiling=KEEP → 最终 KEEP")

    merged_reasons = dict(all_reasons)
    for dp, detail in tiling_reasons.items():
        if dp in merged_reasons and merged_reasons[dp]["keep"] and detail["keep"]:
            merged_reason = f"【整体评估】{merged_reasons[dp]['reason']}【Tiling评估】{detail['reason']}"
            merged_reasons[dp] = {
                "keep": True,
                "reason": merged_reason,
                "source": "both",
            }
        elif dp not in merged_reasons or not merged_reasons[dp]["keep"]:
            merged_reasons[dp] = detail

    # Display the same final decisions and reasons that are returned for JSON output.
    table = Table(show_header=True, header_style="bold cyan", show_lines=True, expand=True)
    table.add_column("文档章节", width=14)
    table.add_column("是否影响", width=8, justify="center")
    table.add_column("评估理由", ratio=1, no_wrap=False, overflow="fold")
    for doc_part, detail in merged_reasons.items():
        tag = Text.from_markup("[red]是[/red]" if detail["keep"] else "[green]否[/green]")
        table.add_row(format_doc_part(doc_part), tag, detail["reason"])
    console.print(table)

    skipped = [p for p in affected_subparts if p not in final_set]
    if skipped:
        console.print(f"  [dim]已过滤 ({len(skipped)}):[/dim] {format_doc_parts(skipped)}")
    console.print(
        f"  [bold green]最终保留 ({len(verified_todo_list)}):[/bold green] {format_doc_parts(verified_todo_list)}"
    )
    info(f"有改动的文件 ({len(modified_file_dir_list)}):")
    for f in modified_file_dir_list:
        info(f"  - {f}")
    if unmatched_file_dir_list:
        info(f"未匹配到 subpart 的变更文件 ({len(unmatched_file_dir_list)}):")
        for f in unmatched_file_dir_list:
            info(f"  - {f}")
    return (
        verified_todo_list,
        merged_reasons,
        modified_file_dir_list + unmatched_file_dir_list,
        repo_other_changed_files,
    )


if __name__ == "__main__":
    from src.generation_validation import config

    evidence_dir = "data/outputs/snapshot/add/base/anchor"
    before_dir = "data/outputs/snapshot/operator_storage/add/base"
    after_dir = "data/test/ops-math-master/math/add_modified"
    console = Console()
    verified_todo_list, all_reasons, modified_file_dir_list, repo_other_mod_file = run_evaluation(
        str(evidence_dir),
        str(before_dir),
        str(after_dir),
        "",
        fine_grained=True,
        console=Console(),
        repo_before_dir="data/test/ops-math-master",
        repo_after_dir="data/test/ops-math-master_modified",
    )
    console.print(f"verified_todo_list: {verified_todo_list}")
    console.print(f"all_reasons: {all_reasons}")
    console.print(f"modified_file_dir_list: {modified_file_dir_list}")
    console.print(f"repo_other_mod_file: {repo_other_mod_file}")
