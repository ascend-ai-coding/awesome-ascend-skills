# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
File anchor 依赖拓展后处理。

读取 anchor_dir 下 5 份 anchor JSON，对每个 type=="file" 的原 anchor，
用 include_analyzer 做传递闭包展开，把 ① 自身目录 + ② 仓内其他依赖文件
也组装为 file anchor（标记 dep_class 区分 original/self/repo_other），
回填 evidence_id 到 doc.sub_part（constraint/support），回写 JSON。

传递闭包下同一依赖文件可能被多个原 anchor 触发，去重后累积 dep_from_anchor_ids。
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional

from src.snapshot_construction.anchor_construction.common.include_analyzer import (
    extract_includes,
    classify_include,
    build_transitive,
    display_to_abs,
    build_file_index,
)
from src.snapshot_construction.anchor_construction.common.anchor_builder import (
    build_file_anchor,
)
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    get_registry,
)
from src.snapshot_construction.pub_ref import PubRefConfig

logger = logging.getLogger(__name__)


class _SeqCounter:
    """续号计数器：从现有 max seq 接续，按 shared_counters 决定 anchor/evidence 是否同号。"""

    def __init__(self, start_anchor: int, start_evidence: int, shared: bool):
        self.anchor = start_anchor
        self.evidence = start_evidence
        self.shared = shared

    def next(self) -> tuple[int, int]:
        if self.shared:
            self.anchor += 1
            self.evidence = self.anchor
        else:
            self.anchor += 1
            self.evidence += 1
        return self.anchor, self.evidence


def _scan_max_seqs(anchor_tuple: list[dict]) -> tuple[int, int]:
    """扫现有 anchor_id / evidence_id 的最大数字序号，用于续号。"""
    max_anchor = 0
    max_evidence = 0
    for a in anchor_tuple:
        m = re.search(r"(\d+)$", a.get("anchor_id", ""))
        if m:
            max_anchor = max(max_anchor, int(m.group(1)))
        for ev in a.get("evidence", []):
            m = re.search(r"(\d+)$", ev.get("evidence_id", ""))
            if m:
                max_evidence = max(max_evidence, int(m.group(1)))
    return max_anchor, max_evidence


def _make_ids(
    config, anchor_seq: int, evidence_seq: int, anchor_type: str = "file"
) -> tuple[str, str]:
    """按 doc_part 的 id_config 格式生成 anchor_id + evidence_id。"""
    ic = config.id_config
    anchor_id = (
        f"anchor_{anchor_type}_{anchor_seq:03d}"
        if ic.type_prefix_in_anchor
        else f"anchor_{anchor_seq:03d}"
    )
    evidence_id = (
        f"ev_{anchor_type}_{evidence_seq:03d}"
        if ic.type_prefix_in_evidence
        else f"ev_{evidence_seq:03d}"
    )
    return anchor_id, evidence_id


def _get_paired_display(display: str) -> Optional[str]:
    """对 .h/.cpp display path 返回配对文件路径（同 stem 同目录），无配对返回 None。"""
    p = Path(display)
    suffix = p.suffix.lower()
    if suffix in (".h", ".hpp", ".hxx"):
        return str(p.with_suffix(".cpp"))
    if suffix in (".cpp", ".cc", ".cxx"):
        return str(p.with_suffix(".h"))
    return None


def _build_dep_anchor(
    display: str,
    ops_root: Path,
    dep_class: str,
    src_anchor_id: str,
    src_gene_target: list,
    src_tree_id: str,
    counter: _SeqCounter,
    config,
    dep_file_to_anchor: dict,
    data_loader=None,
    original_paths=None,
    file_index=None,
) -> tuple[Optional[dict], Optional[str]]:
    """组装一个依赖 file anchor。同 display 已组装过则只累积 dep_from。

    self 类用 data_loader 加载 code_map/tree_id；repo_other 类留空（不在算子目录内）。
    """
    if display in dep_file_to_anchor:
        existing = dep_file_to_anchor[display]
        existing["anchor_information"]["dep_from_anchor_ids"].append(src_anchor_id)
        return None, existing["anchor_id"]

    dep_abs = display_to_abs(display, ops_root, file_index=file_index)
    if dep_abs is None or not dep_abs.is_file():
        return None, None

    # 依赖文件已是原 anchor 则跳过创建，返回已有 anchor_id 建立引用
    if original_paths:
        dep_resolved = str(dep_abs.resolve())
        if dep_resolved in original_paths:
            return None, original_paths[dep_resolved]

    anchor_seq, evidence_seq = counter.next()
    anchor_id, evidence_id = _make_ids(config, anchor_seq, evidence_seq, "file")

    try:
        file_content = dep_abs.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        file_content = ""

    if dep_class == "self" and data_loader is not None:
        code_map = data_loader.load_code_map(dep_abs.name)
        tree_id = data_loader.load_tree_id()
        file_func = data_loader.load_file_func(dep_abs.name)
    elif dep_class == "self":
        code_map = {"nodes": [], "edges": []}
        tree_id = src_tree_id
        file_func = ""
    else:
        code_map = {"nodes": [], "edges": []}
        tree_id = ""
        file_func = ""

    new_anchor = build_file_anchor(
        anchor_id=anchor_id,
        file_path=str(dep_abs),
        file_name=dep_abs.name,
        code_map=code_map,
        file_content=file_content,
        file_func=file_func,
        tree_id=tree_id,
        gene_deter="暂空",
        gene_target=src_gene_target,
        evidence_id=evidence_id,
        note="以下为file型标准结构",
    )
    # 拼错兼容字段（下游 change_detection 强制要求，见 file_extractor.py:135）
    new_anchor["evidence"][0]["envidence_id"] = evidence_id
    # 依赖标记
    new_anchor["anchor_information"]["dep_class"] = dep_class
    new_anchor["anchor_information"]["dep_from_anchor_ids"] = [src_anchor_id]

    dep_file_to_anchor[display] = new_anchor
    return new_anchor, anchor_id


def expand_file_deps(
    ops_root,
    op_dir,
    anchor_dir,
    depth_limit: Optional[int] = None,
    data_loader=None,
    expand_dir_cpp: bool = True,
    max_companions: int = 200,
) -> None:
    """对 anchor_dir 下 5 份 JSON 的 file anchor 做依赖拓展。

    Args:
        ops_root: 仓根目录（staged 仓根，如 operator_storage）
        op_dir: 算子目录路径（staged 算子目录，如 operator_storage/add/base）
        anchor_dir: 5 份 anchor JSON 所在目录
        depth_limit: 传递闭包深度限制（None=无限）
        expand_dir_cpp: 目录伴生展开（同目录 .cpp 实现纳入闭包并展开其 include），
            使仅被 .cpp 引用/自注册的文件（kernel 入口、tiling 模板等）可被检出
        max_companions: 目录伴生文件总数上限（防爆兜底）
    """
    ops_root_raw = Path(ops_root)
    ops_root = ops_root_raw.resolve()
    op_dir = Path(op_dir).resolve()
    anchor_dir = Path(anchor_dir)

    registry = get_registry()
    transitive_cache: dict = {}  # 跨文件共享解析缓存
    file_index = build_file_index(ops_root, op_dir)

    for doc_part in PubRefConfig.DOC_PARTS:
        config = registry.get_config(doc_part)
        json_path = anchor_dir / config.output_filename
        if not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        anchor_tuple = data.get("anchor_tuple", [])

        max_anchor, max_evidence = _scan_max_seqs(anchor_tuple)
        counter = _SeqCounter(
            max_anchor, max_evidence, config.id_config.shared_counters
        )

        new_anchors = []
        dep_file_to_anchor: dict = {}  # display → anchor dict（同份 JSON 内去重）

        # 预扫描原 file anchor 路径，用于跳过已存在的依赖文件
        original_paths: dict[str, str] = {}  # resolve 路径 → anchor_id
        for _a in anchor_tuple:
            _i = _a.get("anchor_information", {})
            if _i.get("type") == "file":
                _fp = _i.get("level1", {}).get("file_path", "")
                if _fp:
                    try:
                        original_paths[str(Path(_fp).resolve())] = _a["anchor_id"]
                    except Exception:
                        pass

        for anchor in anchor_tuple:
            info = anchor.get("anchor_information", {})
            if info.get("type") != "file":
                continue

            # 原 anchor 补 original 标记
            info["dep_class"] = "original"

            file_path = info.get("level1", {}).get("file_path", "")
            abs_path = Path(file_path)
            if not abs_path.is_file():
                logger.debug(f"跳过不存在的文件: {file_path}")
                continue

            # 提取直接依赖并分类
            file_dir = abs_path.parent
            direct_self, direct_repo_other, direct_external = [], [], []
            for inc, qt in extract_includes(abs_path):
                if qt == "angle":
                    continue
                cat, display = classify_include(
                    inc, file_dir, ops_root, op_dir, file_index=file_index
                )
                if cat == "self":
                    direct_self.append(display)
                elif cat == "repo_other":
                    direct_repo_other.append({"原始引用": inc, "解析路径": display})
                else:
                    direct_external.append(inc)

            # 传递闭包展开
            transitive = build_transitive(
                direct_self,
                direct_repo_other,
                direct_external,
                ops_root,
                op_dir,
                depth_limit,
                file_cache=transitive_cache,
                file_index=file_index,
                expand_dir_cpp=expand_dir_cpp,
                max_companions=max_companions,
            )

            # 源 anchor 的 gene_target / tree_id（复制给新 anchor）
            src_gene_target = (
                anchor.get("evidence", [{}])[0].get("gene_rule", {}).get("target", [])
            )
            src_tree_id = info.get("level3", {}).get("tree_id", "")

            dep_to_ids = []

            # ① 自身目录依赖
            for display in transitive["①_自身目录"]:
                new_anchor, new_id = _build_dep_anchor(
                    display,
                    ops_root_raw,
                    "self",
                    anchor["anchor_id"],
                    src_gene_target,
                    src_tree_id,
                    counter,
                    config,
                    dep_file_to_anchor,
                    data_loader=data_loader,
                    original_paths=original_paths,
                    file_index=file_index,
                )
                if new_anchor is not None:
                    new_anchors.append(new_anchor)
                if new_id:
                    dep_to_ids.append(new_id)

                paired = _get_paired_display(display)
                if paired:
                    p_anchor, p_id = _build_dep_anchor(
                        paired,
                        ops_root_raw,
                        "self",
                        anchor["anchor_id"],
                        src_gene_target,
                        src_tree_id,
                        counter,
                        config,
                        dep_file_to_anchor,
                        data_loader=data_loader,
                        original_paths=original_paths,
                        file_index=file_index,
                    )
                    if p_anchor is not None:
                        new_anchors.append(p_anchor)
                    if p_id:
                        dep_to_ids.append(p_id)

            # ② 仓内其他依赖
            for item in transitive["②_仓内其他"]:
                display = item["解析路径"]
                new_anchor, new_id = _build_dep_anchor(
                    display,
                    ops_root_raw,
                    "repo_other",
                    anchor["anchor_id"],
                    src_gene_target,
                    src_tree_id,
                    counter,
                    config,
                    dep_file_to_anchor,
                    data_loader=data_loader,
                    original_paths=original_paths,
                    file_index=file_index,
                )
                if new_anchor is not None:
                    new_anchors.append(new_anchor)
                if new_id:
                    dep_to_ids.append(new_id)

                paired = _get_paired_display(display)
                if paired:
                    p_anchor, p_id = _build_dep_anchor(
                        paired,
                        ops_root_raw,
                        "repo_other",
                        anchor["anchor_id"],
                        src_gene_target,
                        src_tree_id,
                        counter,
                        config,
                        dep_file_to_anchor,
                        data_loader=data_loader,
                        original_paths=original_paths,
                        file_index=file_index,
                    )
                    if p_anchor is not None:
                        new_anchors.append(p_anchor)
                    if p_id:
                        dep_to_ids.append(p_id)

            # 原 anchor 记录它依赖的新 anchor
            info["dep_to_anchor_ids"] = dep_to_ids

        # 追加新 anchor
        anchor_tuple.extend(new_anchors)

        # 回填 evidence_id（仅 constraint / support，与 pipeline._build_doc_structure 行为对齐）
        sub_part = data.get("doc", {}).get("sub_part", {})
        if doc_part == "constraint":
            cl = sub_part.get("constrain_clarify", [{}])
            if cl:
                cl[0].setdefault("evidence_id", []).extend(
                    a["evidence"][0]["evidence_id"] for a in new_anchors
                )
        elif doc_part == "support":
            sm = sub_part.get("support_matrix", [{}])
            if sm:
                sm[0].setdefault("evidence_ids", []).extend(
                    a["evidence"][0]["evidence_id"] for a in new_anchors
                )

        # 回写
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(
            f"expand_file_deps: {doc_part} 新增 {len(new_anchors)} 个依赖 file anchor"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="file anchor 依赖拓展后处理")
    parser.add_argument("--ops-root", required=True, help="ops-* 仓库根目录")
    parser.add_argument("--op-dir", required=True, help="算子目录路径")
    parser.add_argument("--anchor-dir", required=True, help="5 份 anchor JSON 所在目录")
    parser.add_argument("--depth", type=int, default=None, help="传递闭包深度限制")
    args = parser.parse_args()
    expand_file_deps(args.ops_root, args.op_dir, args.anchor_dir, args.depth)
