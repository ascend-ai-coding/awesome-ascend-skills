#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
"""_baseline_integrity.py — shared baseline anchor verification.

Imported by verify.py and benchmark.py. Checks that the working directory's
{op_name}.py matches the sha256 recorded in {work_dir}/output/.baseline_anchor.json
(at the time Phase 1 freeze ran). If the anchor is missing or the digest differs,
the script refuses to run (exit 3 / 4) to prevent comparing against a tampered
baseline.

Exit codes:
  3 — anchor file missing (Phase 1 did not run freeze_baseline.py)
  4 — anchor digest mismatch (baseline file was tampered with after freeze)

Directory layout (must match CLAUDE.md Phase 1):
  {work_dir}/
    {op_name}.py                       ← file being verified
    output/
      .baseline_anchor.json            ← freeze-time digest
      iter_N/verify/                   ← where verify.py runs (verify_dir)
      opt_iter_N/verify/
      iter_N/perf_result.json          ← where benchmark.py runs (verify_dir)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ANCHOR_FILENAME = ".baseline_anchor.json"

# 退出码：3 = 锚缺失/损坏（Phase 1 未 freeze）；4 = 基线/源被篡改
ANCHOR_MISSING_EXIT_CODE = 3
BASELINE_TAMPERED_EXIT_CODE = 4


class BaselineGateError(Exception):
    """基线闸门未通过时抛出，由调用方 main() 统一捕获并退出，

    避免在内部函数中调用 sys.exit（对齐 benchmark.VerifyGateError 约定）。
    """

    def __init__(self, message: str = "", exit_code: int = BASELINE_TAMPERED_EXIT_CODE):
        super().__init__(message)
        self.exit_code = exit_code


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_work_dir(verify_dir: str | os.PathLike) -> Path:
    """Recover {work_dir} from a verify_dir.

    verify_dir is one of:
      {work_dir}/output/iter_{N}/verify/
      {work_dir}/output/opt_iter_{N}/verify/
      {work_dir}/output/                  (when caller passes output dir directly)
      {work_dir}/                         (when caller passes work dir directly)

    Strategy: walk upward until we find a sibling `output/` directory whose
    parent contains {op_name}.py. Falls back to assuming verify_dir is the
    work_dir itself (no output/ in path).
    """
    verify_path = Path(verify_dir).resolve()
    # Walk up: look for a directory containing "output/" subdir
    candidate = verify_path
    for _ in range(6):  # cap depth to avoid runaway
        if (candidate / "output").is_dir():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    # Fallback: assume the verify_dir IS the work_dir
    return verify_path


def _load_anchor(anchor_path: Path) -> dict:
    """加载并校验锚文件，返回 anchor dict。

    锚缺失 / JSON 损坏 / 缺 torch_sha256 字段时抛出 BaselineGateError(exit_code=3)。
    """
    if not anchor_path.is_file():
        logger.error(
            "[基线闸门] 锚文件缺失: %s. Phase 1 未执行 freeze_baseline.py。"
            "请在 Phase 1 末尾调用 freeze_baseline.py 落锚。",
            anchor_path,
        )
        raise BaselineGateError("锚文件缺失", exit_code=ANCHOR_MISSING_EXIT_CODE)

    try:
        with anchor_path.open("r", encoding="utf-8") as f:
            anchor = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("[基线闸门] 锚文件损坏: %s (%s)", anchor_path, e)
        raise BaselineGateError("锚文件损坏", exit_code=ANCHOR_MISSING_EXIT_CODE) from e

    if not anchor.get("torch_sha256"):
        logger.error("[基线闸门] 锚文件缺少 torch_sha256 字段: %s", anchor_path)
        raise BaselineGateError("锚文件缺少 torch_sha256", exit_code=ANCHOR_MISSING_EXIT_CODE)

    return anchor


def _check_source_gate(anchor: dict, actual_hash: str) -> None:
    """若锚记录了 user-source sha256（mode=user freeze），复核源文件未漂移。

    捕获用户原始 benchmark 在 freeze 与 verify/benchmark 之间被篡改/删除的情况。
    任一不一致抛出 BaselineGateError(exit_code=4)。
    """
    source_hash_expected = anchor.get("source_sha256")
    source_path_str = anchor.get("source_path")
    if not (source_hash_expected and source_path_str):
        return

    source_path = Path(source_path_str)
    if not source_path.is_file():
        logger.error(
            "[基线闸门] 用户源 benchmark 已不存在: %s. freeze 时存在，现已被删除或移动。",
            source_path,
        )
        raise BaselineGateError("用户源缺失", exit_code=BASELINE_TAMPERED_EXIT_CODE)

    source_hash_actual = _sha256_of(source_path)
    if source_hash_actual != source_hash_expected:
        logger.error(
            "[基线闸门] 用户源 benchmark 被篡改: %s. freeze 时 sha256=%s..., 现在 sha256=%s....",
            source_path,
            source_hash_expected[:12],
            source_hash_actual[:12],
        )
        raise BaselineGateError("用户源被篡改", exit_code=BASELINE_TAMPERED_EXIT_CODE)

    # And the work-dir copy must still equal the source byte-for-byte
    if actual_hash != source_hash_actual:
        logger.error(
            "[基线闸门] 工作目录副本与用户源 benchmark 不一致: 副本 sha256=%s..., 源 sha256=%s....",
            actual_hash[:12],
            source_hash_actual[:12],
        )
        raise BaselineGateError("副本与源不一致", exit_code=BASELINE_TAMPERED_EXIT_CODE)


def _check_baseline_integrity(verify_dir: str | os.PathLike, op_name: str) -> None:
    """Verify the baseline anchor matches the current {op_name}.py.

    Raises BaselineGateError(exit_code=3) if anchor missing/corrupt,
    BaselineGateError(exit_code=4) if the baseline or source digest mismatches.
    Returns silently on success.
    """
    work_dir = _resolve_work_dir(verify_dir)
    anchor_path = work_dir / "output" / ANCHOR_FILENAME
    torch_path = work_dir / f"{op_name}.py"

    anchor = _load_anchor(anchor_path)
    expected_hash = anchor["torch_sha256"]

    if not torch_path.is_file():
        logger.error(
            "[基线闸门] 基线文件不存在: %s. 工作目录可能已被破坏。",
            torch_path,
        )
        raise BaselineGateError("基线文件不存在", exit_code=BASELINE_TAMPERED_EXIT_CODE)

    actual_hash = _sha256_of(torch_path)
    if actual_hash != expected_hash:
        logger.error(
            "[基线闸门] 基线文件被篡改: %s. 期望 sha256=%s..., 实际 sha256=%s.... "
            "Phase 1 freeze 之后禁止修改 {op_name}.py。请回滚到 freeze 时的状态。",
            torch_path,
            expected_hash[:12],
            actual_hash[:12],
        )
        raise BaselineGateError("基线文件被篡改", exit_code=BASELINE_TAMPERED_EXIT_CODE)

    _check_source_gate(anchor, actual_hash)

    logger.info(
        "[基线闸门] OK: %s 哈希匹配 freeze 锚 (%s...)",
        torch_path,
        expected_hash[:12],
    )
