# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Pipeline helper functions - shared setup and utilities."""

import json
from pathlib import Path

from src.llm import LLMClient
from src.logger import highlight, section

from .. import config
from .stages import EvidenceStage


def setup_pipeline(evidence_json: str, llm_client=None, deduplicate: bool = False) -> tuple:
    """Setup pipeline: initialize LLM, load and parse evidence.

    Args:
        evidence_json: Path to evidence JSON file.
        llm_client: Optional LLM client. If None, creates new one.
        deduplicate: If True, deduplicate anchors by file_path before parsing.

    Returns:
        tuple: (evidence_stage, raw_input, parsed, ranked, llm, evidence_path)
    """
    evidence_path = Path(evidence_json) if Path(evidence_json).is_absolute() else config.PROJECT_ROOT / evidence_json

    llm = llm_client or LLMClient()

    highlight(f"  evidence_path: {evidence_path}")
    highlight(f"  LLM model: {llm.model}")

    section("步骤 1: 加载代码证据")
    evidence_stage = EvidenceStage(llm)
    raw_input = evidence_stage.load_evidence_file(evidence_path)

    highlight(f"  doc_part: {raw_input.doc.doc_part}")
    highlight(f"  anchors: {len(raw_input.anchor_tuple)}")

    parsed, ranked = evidence_stage.parse_and_rank(raw_input, deduplicate=deduplicate)
    highlight(f"  total evidences: {parsed.total_evidences}")

    targets = parsed.evidences_by_target
    highlight(f"  evidences_by_target: {', '.join(f'{k}({len(v)})' for k, v in targets.items())}")

    return evidence_stage, raw_input, parsed, ranked, llm, evidence_path


def get_output_dir() -> Path:
    """Get or create output directory."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return config.OUTPUT_DIR


def save_json(data, path: Path):
    """Save data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
