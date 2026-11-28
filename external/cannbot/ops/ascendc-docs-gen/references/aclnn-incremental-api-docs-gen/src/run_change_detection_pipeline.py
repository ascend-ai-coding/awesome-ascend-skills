# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Change Detection Pipeline - 融合变更检测与文档生成的非交互式流水线 (Stage 2-6)。

将 run_all_cli.py 的阶段2-6逻辑与 change_deteciton_cli() 合并为单一脚本,
供 VSCode 扩展以子进程方式调用。workspace_dir 为绝对路径, 其余路径均为相对路径。

输出标记:
  - ===change-detection-result=== / ===/change-detection-result=== (由 change_deteciton_cli 透传)
  - ==result== / ==/result== (最终 markdown 列表 JSON)
  - DONE 文件 (成功) / FAILED 文件 (异常); 运行开始时清理旧标记, 结束后出现的标记必属本次运行
"""

import argparse
import json
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console

# 设置路径: 添加 cann-document-generation/（仓库根目录）到搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation_validation.config import format_doc_part
from src.settings import apply_settings, load_settings
from src.snapshot_construction.anchor_construction.common.doc_selector import (
    select_doc_file,
)
from src.snapshot_construction.pub_ref import PubRefConfig


def _configure_ssl_cert() -> None:
    """SSL 证书配置: 如果证书不存在则禁用 SSL 验证（原 import 期全局改写，移入函数保留语义）。"""
    try:
        from src.llm import config as llm_config
    except ImportError:
        return
    if llm_config.SSL_CERT is not None and not Path(str(llm_config.SSL_CERT)).exists():
        print(f"[run_change_detection_pipeline] SSL cert not found: {llm_config.SSL_CERT}, disabling")
        llm_config.SSL_CERT = None


def _unlink_quietly(path: Path) -> None:
    """删除哑文件, 不存在时静默跳过。"""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _clear_sentinel_files(workspace_dir: str, data_path: str) -> Path:
    """运行开始时清理旧 DONE/FAILED 哑文件, 返回解析后的 data_path。

    哑文件语义: 运行期间无标记 = 进行中; 运行结束后出现的标记必属本次运行。
    """
    data_path_resolved = (Path(workspace_dir).resolve() / data_path).resolve()
    data_path_resolved.mkdir(parents=True, exist_ok=True)
    _unlink_quietly(data_path_resolved / "DONE")
    _unlink_quietly(data_path_resolved / "FAILED")
    return data_path_resolved


def _output_result(results: dict, data_path_resolved: Path):
    """输出结果标记并写入 DONE 文件"""
    print("\n=== Pipeline Complete ===")
    print("==result==")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("==/result==")

    _unlink_quietly(data_path_resolved / "FAILED")  # 终态互斥: 删除相反状态旧标记
    done_file = data_path_resolved / "DONE"
    done_file.write_text(
        json.dumps({"completedAt": datetime.now().isoformat()}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"DONE file written: {done_file}")


def run_pipeline(
    snapshot_id: str,
    workspace_dir: str,
    original_op_dir: str,
    modified_op_dir: str,
    data_path: str,
    operator_version: str = "base",
    verbose: bool = False,
):
    """执行阶段2-6的变更检测+文档生成流水线"""
    from src.change_detection.doc_part_change_detection.change_evaluator import (
        run_evaluation,
    )
    from src.change_detection.merge_evidence import merge_redirect_to_anchor
    from src.change_detection.run_change_detection_cli import change_deteciton_cli
    from src.generation_validation.style.summarizer import gen_doc_summarize
    from src.generation_validation.ui.cli import (
        _is_validation_passed,
        resolve_evidence_file,
        run_full_pipeline,
    )
    from src.logger import setup_logger
    from src.md_fill.main_fill import update_md_with_dict

    setup_logger(level=logging.INFO, console=False)
    _configure_ssl_cert()

    # ── 路径解析 ─────────────────────────────────────────────────
    workspace = Path(workspace_dir).resolve()
    data_path_resolved = (workspace / data_path).resolve()
    data_path_resolved.mkdir(parents=True, exist_ok=True)
    original_op_dir_resolved = (workspace / original_op_dir).resolve()
    modified_op_dir_resolved = (workspace / modified_op_dir).resolve()

    selected_op = original_op_dir_resolved.name
    print(f"  Workspace:  {workspace}")
    print(f"  Operator:   {selected_op}")
    print(f"  Data path:  {data_path_resolved}")

    results = {
        "snapshotId": snapshot_id,
        "version": operator_version,
        "markdownFiles": [],
        "noChangeParts": [],
        "skippedParts": [],
        "generatedAt": None,
    }

    md_stems_dir = data_path_resolved / selected_op

    if not md_stems_dir.exists():
        raise Exception(f"算子快照目录不存在: {md_stems_dir}")

    # ── 选择算子 md_stem (与 run_all_cli.py 非交互模式一致) ─────────
    available_md_stems = sorted(d.name for d in md_stems_dir.iterdir() if d.is_dir())
    if not available_md_stems:
        raise Exception(f"未找到可用的算子 md_stem: {md_stems_dir}")

    if len(available_md_stems) == 1:
        selected_md_stem = available_md_stems[0]
    elif operator_version in available_md_stems:
        selected_md_stem = operator_version
    else:
        print(f"  md_stem '{operator_version}' not in {available_md_stems}, fallback to first")
        selected_md_stem = available_md_stems[0]
    print(f"  md_stem:   {selected_md_stem}")

    # ── 配置各阶段路径 ───────────────────────────────────────────
    version_dir = md_stems_dir / selected_md_stem
    snapshot_op_dir = version_dir
    evidence_dir = version_dir / "anchor"
    source_docs = snapshot_op_dir / "docs"
    summary_dir = version_dir / "style"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 修改后算子直接使用 (无需版本解析, 与 run_all_cli.py 一致)
    modified_op_dir_md_stem = modified_op_dir_resolved

    # ── 选择文档文件 ─────────────────────────────────────────────
    if not source_docs.exists():
        raise Exception(f"文档目录不存在: {source_docs}")

    doc_files = sorted(source_docs.glob("*.md"))
    if not doc_files:
        raise Exception(f"未找到 .md 文件: {source_docs}")

    selected_file = select_doc_file(doc_files, selected_md_stem=selected_md_stem)
    if selected_file is None:
        raise Exception(f"未找到以 aclnn 开头的文档文件: {source_docs}")
    doc_content = Path(selected_file).read_text(encoding="utf-8")

    # ===== 阶段2: 生成文档摘要 =====
    print("\n=== Stage 2: 生成接口文档摘要 ===")
    summary_file_dir = summary_dir / f"{selected_md_stem}-summary.json"
    gen_doc_summarize(doc_content, str(summary_file_dir), force=False)
    print(f"Summary saved: {summary_file_dir}")

    # ===== 阶段3: 变更评估 =====
    print("\n=== Stage 3: 评估受变更影响的关联接口文档 ===")
    print(f"  Evidence dir:   {evidence_dir}")
    print(f"  Before dir:     {snapshot_op_dir}")
    print(f"  After dir:      {modified_op_dir_md_stem}")

    console = Console()
    verified_todo_list, _all_reasons, modified_file_dir_list, _repo_other_changed_files = run_evaluation(
        evidence_dir=str(evidence_dir),
        before_dir=str(snapshot_op_dir),
        after_dir=str(modified_op_dir_md_stem),
        doc_content=doc_content,
        fine_grained=True,
        console=console,
    )
    doc_subpart_todo_list = verified_todo_list
    print(f"  Todo list: {doc_subpart_todo_list}")

    # 如果无需变更, 跳过阶段4-6, 输出空结果
    if not doc_subpart_todo_list:
        print("\nNo document changes detected, skipping Stage 4-6")
        results["generatedAt"] = datetime.now().isoformat()
        _output_result(results, data_path_resolved)
        return results

    # ===== 阶段4: 证据重检测 (change_deteciton_cli + merge) =====
    print("\n=== Stage 4: 构建新文档与代码模型 ===")
    re_detection_output_path = version_dir / "re_detection"
    print(f"  Output path: {re_detection_output_path}")
    print(f"  Todo list:   {doc_subpart_todo_list}")

    # 过滤 “example”
    redirection_doc_parts = [p for p in doc_subpart_todo_list if p != "example"]

    # 传入绝对路径, 不传 workspace_dir (默认 None, 与 run_all_cli.py 一致)
    change_deteciton_cli(
        str(snapshot_op_dir),
        str(modified_op_dir_md_stem),
        str(data_path_resolved),
        selected_md_stem,
        str(re_detection_output_path),
        redirection_doc_parts,
        selected_op,
        file_dir_list=modified_file_dir_list,
        verbose=verbose,
    )
    merge_redirect_to_anchor(re_detection_output_path, redirection_doc_parts, verbose=verbose)
    print("Re-detection completed")

    # ===== 阶段5: 生成与验证 =====
    print("\n=== Stage 5: 文档生成并验证 ===")
    backfill_map = {}
    evidence_dir_path = re_detection_output_path / "evidences"

    def _process_doc_part(doc_part: str):
        """返回 (doc_part, generated_text, status, reason)。

        status: ok=可回填 / no_change=LLM 判定无需修改(合法结果, 保留原章节) /
                skipped=无证据文件 / failed=硬失败(run_full_pipeline 内部
                MAX_REGENERATION_RETRIES 重试已耗尽后的最终失败)。
        """
        print(f"\nProcessing doc_part: {format_doc_part(doc_part)}")
        evidence_file = resolve_evidence_file(evidence_dir_path, doc_part)

        if evidence_file is None and doc_part != "example":
            print(f"  Warning: No evidence file for {format_doc_part(doc_part)}, skipping")
            return doc_part, None, "skipped", "No evidence file"

        result = run_full_pipeline(
            evidence_file,
            doc_part,
            None,  # log_capture
            style_dir=summary_file_dir,
            modified_op_dir=str(modified_op_dir_md_stem),
            operator_name=selected_op,
        )

        # 只允许实际成功的生成结果进入回填; 失败在这里意味着内部重试已耗尽
        if not result.success:
            reason = f"生成失败 (内部重试已耗尽): {result.error or '未知原因'}"
            print(f"  FAIL: {format_doc_part(doc_part)}: {reason}")
            return doc_part, None, "failed", reason
        if result.evidence_conflict:
            reason = f"证据冲突, 内容不可信: {result.conflict_details or '详见验证日志'}"
            print(f"  FAIL: {format_doc_part(doc_part)}: {reason}")
            return doc_part, None, "failed", reason
        if result.no_change_needed:
            reason = result.conflict_details or "LLM 判断无需修改"
            print(f"  NO-CHANGE: {format_doc_part(doc_part)}: {reason}")
            return doc_part, None, "no_change", reason

        generated_text = "".join(item["generated_text"] for item in result.generated)
        if not generated_text.strip():
            reason = "生成内容为空, 拒绝空内容回填 (避免清空原章节)"
            print(f"  FAIL: {format_doc_part(doc_part)}: {reason}")
            return doc_part, None, "failed", reason

        # 重试耗尽后验证仍不合格的内容不可信, 视为硬失败
        # (与 run_all_cli 的 post_analyze 阻止回填口径一致; 谓词复用生成模块统一实现)
        failed_items = [v for v in (result.validation or []) if not _is_validation_passed(v)]
        if failed_items:
            reason = f"验证未通过 (重试已耗尽, {len(failed_items)} 个验证项失败)"
            print(f"  FAIL: {format_doc_part(doc_part)}: {reason}")
            return doc_part, None, "failed", reason

        print(f"  Generated {len(generated_text)} chars")
        return doc_part, generated_text, "ok", ""

    max_workers = PubRefConfig.DOC_PART_MAX_WORKERS
    failures: list[str] = []
    no_change_parts: list[str] = []
    skipped_parts: list[str] = []
    print(f"  Processing {len(doc_subpart_todo_list)} doc_parts in parallel (max_workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_doc_part, dp): dp for dp in doc_subpart_todo_list}
        for future in as_completed(futures):
            dp = futures[future]
            try:
                doc_part, generated_text, status, reason = future.result()
            except Exception as e:
                # 不再静默吞异常: 聚合后统一抛出, 由入口写 FAILED
                print(f"  Error processing doc_part {format_doc_part(dp)}: {e}")
                failures.append(f"{format_doc_part(dp)}: 处理异常: {e}")
                continue
            if status == "ok":
                backfill_map[doc_part] = generated_text
            elif status == "no_change":
                no_change_parts.append(doc_part)
            elif status == "skipped":
                skipped_parts.append(doc_part)
            else:
                failures.append(f"{format_doc_part(doc_part)}: {reason}")

    # all-or-nothing: 任一硬失败 → 聚合抛出 → main() 写 FAILED、exit 1, 不做部分回填
    if failures:
        raise RuntimeError(f"{len(failures)} 个 doc_part 生成失败 (内部重试已耗尽): " + "; ".join(failures))
    results["noChangeParts"] = no_change_parts
    results["skippedParts"] = skipped_parts

    # ===== 阶段6: 回填到 Markdown =====
    print("\n=== Stage 6: Fill Back to Markdown ===")

    if selected_file and backfill_map:
        base_file = selected_file
        head_md = modified_op_dir_resolved / "docs" / selected_file.name
        if head_md.is_file():
            base_file = head_md
            print(f"  Backfill base (head): {head_md}")
        else:
            print(f"  Head md not found ({head_md}), fallback to snapshot base: {selected_file}")

        new_op_doc_dir = original_op_dir_resolved / "docs" / f"[new]{selected_md_stem}.md"
        update_md_with_dict(backfill_map, str(base_file), str(new_op_doc_dir))
        print(f"New markdown generated: {new_op_doc_dir}")

        rel_md_path = selected_file.relative_to(snapshot_op_dir)
        original_path = modified_op_dir_resolved / rel_md_path
        results["markdownFiles"].append(
            {
                "docPart": "full",
                "fileName": f"[new]{selected_md_stem}.md",
                "originalPath": str(original_path),
                "generatedPath": str(new_op_doc_dir),
            }
        )

    results["generatedAt"] = datetime.now().isoformat()
    _output_result(results, data_path_resolved)
    return results


def main():
    parser = argparse.ArgumentParser(description="Change Detection Pipeline - 融合变更检测与文档生成 (Stage 2-6)")
    parser.add_argument("--snapshot-id", required=True, help="快照 ID")
    parser.add_argument("--workspace-dir", required=True, help="工作区根目录 (绝对路径)")
    parser.add_argument("--original-op-dir", required=True, help="修改前算子目录 (相对于 workspace-dir)")
    parser.add_argument("--modified-op-dir", required=True, help="修改后算子目录 (相对于 workspace-dir)")
    parser.add_argument("--data-path", required=True, help="数据输出路径 (相对于 workspace-dir)")
    parser.add_argument("--operator-version", default="base", help="算子版本 (默认: base)")
    parser.add_argument("--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    try:
        apply_settings(load_settings())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    # 运行开始清理旧哑文件 (DONE/FAILED), 避免上次运行残留标记被误判为本次状态
    _clear_sentinel_files(args.workspace_dir, args.data_path)

    try:
        run_pipeline(
            snapshot_id=args.snapshot_id,
            workspace_dir=args.workspace_dir,
            original_op_dir=args.original_op_dir,
            modified_op_dir=args.modified_op_dir,
            data_path=args.data_path,
            operator_version=args.operator_version,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"\nPipeline FAILED: {e}")
        traceback.print_exc()

        # 写入 FAILED 文件 (终态互斥: 先删除旧 DONE)
        workspace = Path(args.workspace_dir).resolve()
        data_path_resolved = (workspace / args.data_path).resolve()
        data_path_resolved.mkdir(parents=True, exist_ok=True)
        _unlink_quietly(data_path_resolved / "DONE")
        failed_file = data_path_resolved / "FAILED"
        failed_file.write_text(
            json.dumps(
                {"error": str(e), "failedAt": datetime.now().isoformat()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"FAILED file written: {failed_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
