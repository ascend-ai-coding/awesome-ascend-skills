# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Unified anchor builder module.

Provides AnchorTuple dataclass and unified build_*_anchor functions for all anchor types.
All functions return dict for JSON serialization compatibility.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from src.snapshot_construction.anchor_construction.common.config import (
    ANCHOR_TYPE_FILE,
    ANCHOR_TYPE_FUNCTION,
    ANCHOR_TYPE_SECTION,
    ANCHOR_TYPE_NOTE,
)


@dataclass
class AnchorTuple:
    """
    Unified anchor tuple structure.

    Attributes:
        anchor_id: Unique identifier (format: anchor_{type}_{seq:03d})
        anchor_information: Dict with type, level1, level2, level3 (optional)
        evidence: List of evidence dicts, each with evidence_id key
        note: Optional note string
    """

    anchor_id: str
    anchor_information: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result = asdict(self)
        # Remove None values for cleaner output
        if result["note"] is None:
            del result["note"]
        return result


def build_file_anchor(
    anchor_id: str,
    file_path: str,
    file_name: str,
    code_map: Dict[str, Any],
    file_content: str,
    file_func: str,
    tree_id: str,
    gene_deter: str,
    gene_target: List[List[str]],
    evidence_id: Optional[str] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a file-type anchor.

    Args:
        anchor_id: Anchor identifier (format: anchor_file_{seq:03d})
        file_path: Absolute file path
        file_name: Base file name
        code_map: Code map dict
        file_content: Full file content
        file_func: File function description (file_功能)
        tree_id: Tree identifier
        gene_deter: Generation determination
        gene_target: Generation target (e.g., [["support_matrix"]])
        evidence_id: Optional evidence identifier (format: ev_{seq:03d})
        note: Optional note string

    Returns:
        Dict with anchor structure compatible with downstream change_detection
    """
    evidence_entry = {
        "evidence_information": {
            "file_path": file_path,
            "file_name": file_name,
            "file_功能": file_func,
            "file_content": file_content,
        },
        "gene_rule": {
            "deter": gene_deter,
            "target": gene_target,
        },
    }
    if evidence_id is not None:
        evidence_entry["evidence_id"] = evidence_id

    result = {
        "anchor_id": anchor_id,
        "anchor_information": {
            "type": ANCHOR_TYPE_FILE,
            "level1": {
                "file_path": file_path,
                "file_name": file_name,
            },
            "level2": {
                "code_map": code_map,
                "token_hash": file_content,
                "file_功能": file_func,
            },
            "level3": {
                "tree_id": tree_id,
            },
        },
        "evidence": [evidence_entry],
    }
    if note is not None:
        result["note"] = note
    return result


def build_function_anchor(
    anchor_id: str,
    file_path: str,
    function_name: str,
    function_sign: str,
    calls: List[str],
    called_by: List[str],
    returns: List[str],
    gene_deter: str,
    gene_target: List[List[str]],
    evidence_id: Optional[str] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a function-type anchor.

    Args:
        anchor_id: Anchor identifier (format: anchor_function_{seq:03d})
        file_path: File path containing the function
        function_name: Function name
        function_sign: Full function signature
        calls: List of functions this function calls
        called_by: List of functions that call this function
        returns: List of return-related nodes
        gene_deter: Generation determination
        gene_target: Generation target
        evidence_id: Optional evidence identifier (format: ev_{seq:03d})
        note: Optional note string

    Returns:
        Dict with anchor structure compatible with downstream change_detection
    """
    evidence_entry = {
        "evidence_information": {
            "file_path": file_path,
            "content": [
                {
                    "function_hash": function_sign,
                    "full_function": "",
                }
            ],
            "call": calls,
            "callee": called_by,
            "return": returns,
        },
        "gene_rule": {
            "deter": gene_deter,
            "target": gene_target,
        },
    }
    if evidence_id is not None:
        evidence_entry["evidence_id"] = evidence_id

    result = {
        "anchor_id": anchor_id,
        "anchor_information": {
            "type": ANCHOR_TYPE_FUNCTION,
            "level1": {
                "file_path": file_path,
                "function_name": function_name,
            },
            "level2": {
                "call": calls,
                "callee": called_by,
                "return": returns,
            },
            "level3": {
                "function_sign": function_sign,
            },
        },
        "evidence": [evidence_entry],
    }
    if note is not None:
        result["note"] = note
    return result


def build_section_anchor(
    anchor_id: str,
    file_path: str,
    content: str,
    start_line: int,
    end_line: int,
    token_hash: str,
    gene_deter: str,
    gene_target: List[List[str]],
    evidence_id: Optional[str] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a section-type anchor.

    Args:
        anchor_id: Anchor identifier (format: anchor_section_{seq:03d})
        file_path: File path
        content: Section content
        start_line: Start line number
        end_line: End line number
        token_hash: Token hash (typically same as content)
        gene_deter: Generation determination
        gene_target: Generation target
        evidence_id: Optional evidence identifier (format: ev_{seq:03d})
        note: Optional note string

    Returns:
        Dict with anchor structure compatible with downstream change_detection
    """
    evidence_entry = {
        "evidence_information": {
            "file_path": file_path,
            "content": content,
        },
        "gene_rule": {
            "deter": gene_deter,
            "target": gene_target,
        },
    }
    if evidence_id is not None:
        evidence_entry["evidence_id"] = evidence_id

    result = {
        "anchor_id": anchor_id,
        "anchor_information": {
            "type": ANCHOR_TYPE_SECTION,
            "level1": {
                "file_path": file_path,
                "start_line": str(start_line),
                "end_line": str(end_line),
                "token_hash": token_hash,
            },
            "level2": {
                "token_hash": token_hash,
            },
        },
        "evidence": [evidence_entry],
    }
    if note is not None:
        result["note"] = note
    return result


def build_note_anchor(
    anchor_id: str,
    file_path: str,
    behind_code: str,
    note_hash: str,
    token_hash: str,
    gene_deter: str,
    gene_target: List[List[str]],
    evidence_id: Optional[str] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a note-type anchor.

    Args:
        anchor_id: Anchor identifier (format: anchor_note_{seq:03d})
        file_path: File path
        behind_code: Code behind the note
        note_hash: Note hash
        token_hash: Token hash
        gene_deter: Generation determination
        gene_target: Generation target
        evidence_id: Optional evidence identifier (format: ev_{seq:03d})
        note: Optional note string

    Returns:
        Dict with anchor structure compatible with downstream change_detection
    """
    evidence_entry = {
        "evidence_information": {
            "file_path": file_path,
            "content": note_hash,
            "behind_code": behind_code,
        },
        "gene_rule": {
            "deter": gene_deter,
            "target": gene_target,
        },
    }
    if evidence_id is not None:
        evidence_entry["evidence_id"] = evidence_id

    result = {
        "anchor_id": anchor_id,
        "anchor_information": {
            "type": ANCHOR_TYPE_NOTE,
            "level1": {
                "file_path": file_path,
                "behind_code": behind_code,
            },
            "level2": {
                "token_hash": token_hash,
                "note_hash": note_hash,
            },
        },
        "evidence": [evidence_entry],
    }
    if note is not None:
        result["note"] = note
    return result


def build_evidence(
    evidence_id: str,
    evidence_info: Dict[str, Any],
    gene_rule: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an evidence dict.

    Args:
        evidence_id: Evidence identifier (format: ev_{seq:03d})
        evidence_info: Evidence information dict (file_path, content, etc.)
        gene_rule: Gene rule dict with 'deter' and 'target' keys

    Returns:
        Dict with evidence_id, evidence_information, and gene_rule keys
    """
    return {
        "evidence_id": evidence_id,
        "evidence_information": evidence_info,
        "gene_rule": gene_rule,
    }


if __name__ == "__main__":
    # Self-test
    print("Testing build_file_anchor...")
    r = build_file_anchor(
        anchor_id="anchor_file_001",
        file_path="/op/abs/aclnn_abs.h",
        file_name="aclnn_abs.h",
        code_map={},
        file_content="code",
        file_func="func_desc",
        tree_id="tree_desc",
        gene_deter="deter",
        gene_target=[["support_matrix"]],
    )
    assert r["anchor_information"]["type"] == "file"
    assert r["anchor_id"] == "anchor_file_001"
    print(f"  anchor_id: {r['anchor_id']}")
    print(f"  type: {r['anchor_information']['type']}")
    print("  OK")

    print("\nTesting build_function_anchor...")
    r = build_function_anchor(
        anchor_id="anchor_function_001",
        file_path="/op/abs/aclnn_abs.cpp",
        function_name="aclnnAbs",
        function_sign="int aclnnAbs()",
        calls=["helper1"],
        called_by=["main"],
        returns=["int"],
        gene_deter="deter",
        gene_target=[["support_matrix"]],
    )
    assert r["anchor_information"]["type"] == "function"
    print(f"  anchor_id: {r['anchor_id']}")
    print(f"  type: {r['anchor_information']['type']}")
    print("  OK")

    print("\nTesting build_section_anchor...")
    r = build_section_anchor(
        anchor_id="anchor_section_001",
        file_path="/op/abs/aclnn_abs.cpp",
        content="int x = 1;",
        start_line=10,
        end_line=15,
        token_hash="hash123",
        gene_deter="deter",
        gene_target=[["support_matrix"]],
    )
    assert r["anchor_information"]["type"] == "section"
    print(f"  anchor_id: {r['anchor_id']}")
    print(f"  type: {r['anchor_information']['type']}")
    print("  OK")

    print("\nTesting build_note_anchor...")
    r = build_note_anchor(
        anchor_id="anchor_note_001",
        file_path="/op/abs/aclnn_abs.cpp",
        behind_code="int x = 1;",
        note_hash="note123",
        token_hash="hash123",
        gene_deter="deter",
        gene_target=[["support_matrix"]],
    )
    assert r["anchor_information"]["type"] == "note"
    print(f"  anchor_id: {r['anchor_id']}")
    print(f"  type: {r['anchor_information']['type']}")
    print("  OK")

    print("\nTesting build_evidence...")
    e = build_evidence(
        evidence_id="ev_001",
        evidence_info={"file_path": "/op/abs/aclnn_abs.h", "content": "code"},
        gene_rule={"deter": "deter", "target": [["support_matrix"]]},
    )
    assert e["evidence_id"] == "ev_001"
    print(f"  evidence_id: {e['evidence_id']}")
    print("  OK")

    print("\nTesting AnchorTuple...")
    anchor = AnchorTuple(
        anchor_id="anchor_file_001",
        anchor_information={"type": "file", "level1": {}, "level2": {}, "level3": {}},
        evidence=[],
        note="test note",
    )
    d = anchor.to_dict()
    assert d["anchor_id"] == "anchor_file_001"
    assert d["note"] == "test note"
    print(f"  anchor_id: {d['anchor_id']}")
    print("  OK")

    print("\nAll tests passed!")
