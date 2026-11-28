# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import argparse
import json
import os
import re
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.settings import apply_settings, load_settings
from src.snapshot_construction.anchor_construction.common.file_discovery import (
    find_all_aclnn_docs,
)
from src.snapshot_construction.anchor_construction.common.md_resolver import (
    resolve_md_to_source_files,
)
from src.snapshot_construction.snapshot import load_snapshot, prepare_operator_data


def _discover_md_docs(operator_path: Path) -> list[Path]:
    """Discover all aclnn*.md docs in operator directory.

    find_all_aclnn_docs uses fname.startswith("aclnn") which misses
    prefix-marked files like [new]aclnnAbs.md. This supplements by
    also scanning for [ -prefixed aclnn md files.
    """
    md_files = find_all_aclnn_docs(operator_path)
    seen = set(md_files)
    if operator_path.exists():
        for dirpath, _, filenames in os.walk(operator_path):
            for fname in filenames:
                if fname.startswith("[") and fname.endswith(".md") and "aclnn" in fname.lower():
                    p = Path(dirpath) / fname
                    if p not in seen:
                        md_files.append(p)
                        seen.add(p)
    return md_files


def main(operator_path, data_path, work_dir):
    try:
        apply_settings(load_settings())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    operator_path = f"{work_dir}/{operator_path}"
    console = Console()
    console.print("  > 构建基线文档与代码模型")

    echo_info = {}
    operator_name = os.path.basename(operator_path.rstrip("/"))
    operator_out = f"{data_path}/{operator_name}"

    # Discover all aclnn*.md files in the operator directory
    md_files = _discover_md_docs(Path(operator_path))

    # Prepare operator-level data once (doc_json, code_map, tree_id)
    if md_files:
        console.print("    - [cyan]公共部分[/cyan]")
        prepare_operator_data(operator_path, operator_out)

    for md_file in md_files:
        console.print(f"    - [cyan]{md_file.name}[/cyan]")

        # Resolve source files for this md by naming convention
        source_file_paths = resolve_md_to_source_files(md_file, Path(operator_path))
        source_files = [str(f) for f in source_file_paths]

        # Derive md_stem for output path (same logic as PathConfig.md_stem).
        # Preserve the original filename verbatim — keep '&' (multi-API) and
        # '[...]' prefix markers (e.g. [new]aclnnAbs) so select_doc_file's
        # f.stem == selected_md_stem match stays consistent with docs/ filenames.
        md_stem = Path(md_file).stem

        # Build snapshot for this md
        echo_info[md_stem] = load_snapshot(
            operator_path,  # REAL operator dir (not staged)
            f"{data_path}/{operator_name}/{md_stem}",  # per-md output path
            ops_root=work_dir,  # REAL repo root (not staged)
            md_file=str(md_file),
            source_files=source_files,
            data_root=operator_out,
            skip_data_prep=True,
        )

    print("===snapshot-result===")
    print(json.dumps(echo_info, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operator Snapshot Tool")
    parser.add_argument(
        "--operator_path",
        help="Path to the operator directory",
        default="data/ops-math/math/abs",
    )
    parser.add_argument("--data_path", help="Path to the data output directory", default="out")
    parser.add_argument(
        "--workspace_dir",
        help="Path to the work directory",
        default=os.environ.get("CANN_OPS_ROOT"),
    )
    args = parser.parse_args()
    if not args.workspace_dir:
        raise SystemExit("未提供工作目录: 请通过 --workspace_dir 参数或 CANN_OPS_ROOT 环境变量指定 ops 仓根目录")
    main(
        f"{args.operator_path}",
        f"{args.workspace_dir}/{args.data_path}",
        f"{args.workspace_dir}",
    )


# 示例: python src/main_snapshot.py --operator_path=math/abs --workspace_dir=<你的ops仓根目录> --data_path=out
