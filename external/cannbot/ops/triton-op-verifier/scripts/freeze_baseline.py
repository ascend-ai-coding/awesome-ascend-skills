#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
"""freeze_baseline.py — Phase 1 baseline anchor generator.

Computes sha256 of {work_dir}/{op_name}.py and writes the digest to
{work_dir}/output/.baseline_anchor.json. Downstream verify.py / benchmark.py
read this anchor and refuse to run if the file's hash changes.

Exit codes:
  0 — anchor written successfully
  1 — anchor already exists (refuse to overwrite; protects against hiding tampering)
  2 — target file {op_name}.py not found
  3 — invalid arguments / IO error writing anchor
  5 — source_path provided (mode=user) but work-dir copy sha256 != source sha256
      (Agent rewrote the baseline instead of byte-level cp; refuse to freeze)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from pathlib import Path


logger = logging.getLogger(__name__)

ANCHOR_FILENAME = ".baseline_anchor.json"
VALID_MODES = ("auto", "user")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze baseline anchor for an operator task.")
    parser.add_argument("--op_name", required=True, help="Operator name (without .py suffix)")
    parser.add_argument("--work_dir", required=True, help="Working directory containing {op_name}.py")
    parser.add_argument(
        "--mode",
        default="user",
        choices=VALID_MODES,
        help="Source mode: 'auto' for Agent-generated benchmark (Mode B fallback), "
        "'user' for user-provided benchmark. When mode=user, --source_path is required.",
    )
    parser.add_argument(
        "--source_path",
        default=None,
        help="Absolute path to the original user-provided benchmark .py. "
        "Required when mode=user: freeze will refuse if work-dir copy sha256 != source sha256 "
        "(Agent rewrote the baseline). Optional for mode=auto (Agent-generated benchmark).",
    )
    return parser


def _resolve_source_digest(mode: str, source_path: str | None) -> tuple[str | None, str | None, int]:
    """Compute the user-source digest for mode=user.

    Returns (source_digest, source_path_resolved, error_code); error_code == 0 on success.
    Mode-user invariant: work-dir copy must equal user's source byte-for-byte, which
    prevents the Agent from rewriting the baseline to "fix" a buggy benchmark.
    """
    if mode != "user":
        return None, None, 0
    if not source_path:
        logger.error(
            "[freeze] ERROR: --source_path is required when --mode=user. "
            "User-provided benchmark must be anchored against its original file."
        )
        return None, None, 3
    src = Path(source_path).resolve()
    if not src.is_file():
        logger.error("[freeze] ERROR: source_path not found: %s", src)
        return None, None, 2
    return sha256_of(src), str(src), 0


def _build_record(args, digest: str, target: Path, source_digest: str | None,
                  source_path_resolved: str | None) -> dict:
    record = {
        "op_name": args.op_name,
        "torch_sha256": digest,
        "torch_path": str(target),
        "source_mode": args.mode,
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    if source_digest is not None:
        record["source_sha256"] = source_digest
        record["source_path"] = source_path_resolved
    return record


def _write_anchor(anchor_path: Path, record: dict) -> int:
    """Atomically write the anchor JSON; return 0 on success, 3 on IO error."""
    tmp = anchor_path.with_suffix(".json.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, anchor_path)
    except OSError as e:
        logger.error("[freeze] ERROR: failed to write anchor %s: %s", anchor_path, e)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return 3
    return 0


def main() -> int:
    args = _build_parser().parse_args()

    work_dir = Path(args.work_dir).resolve()
    target = work_dir / f"{args.op_name}.py"
    output_dir = work_dir / "output"
    anchor_path = output_dir / ANCHOR_FILENAME

    if not target.is_file():
        logger.error("[freeze] ERROR: target file not found: %s", target)
        return 2

    if anchor_path.exists():
        # Refuse to overwrite — re-freezing could hide tampering or be abused
        # to "bless" a modified file. Phase 1 must freeze exactly once.
        logger.error(
            "[freeze] ERROR: anchor already exists at %s. Refusing to overwrite. "
            "To re-freeze, delete the anchor manually (and audit why).",
            anchor_path,
        )
        return 1

    source_digest, source_path_resolved, err = _resolve_source_digest(args.mode, args.source_path)
    if err:
        return err

    digest = sha256_of(target)

    if source_digest is not None and digest != source_digest:
        logger.error(
            "[freeze] ERROR: work-dir copy sha256 != source sha256.\n"
            "          work-dir copy: %s sha256=%s...\n"
            "          source:        %s sha256=%s...\n"
            "          Agent rewrote the baseline instead of byte-level cp. "
            "Refusing to freeze. If the source has a real bug, mark "
            "baseline_buggy=true and fail the task instead of patching the baseline.",
            target, digest[:12], source_path_resolved, source_digest[:12],
        )
        return 5

    output_dir.mkdir(parents=True, exist_ok=True)
    record = _build_record(args, digest, target, source_digest, source_path_resolved)
    err = _write_anchor(anchor_path, record)
    if err:
        return err

    logger.info("[freeze] OK: anchor written to %s", anchor_path)
    logger.info("          sha256=%s", digest)
    logger.info("          mode=%s", args.mode)
    if source_digest is not None:
        logger.info("          source_sha256=%s", source_digest)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
