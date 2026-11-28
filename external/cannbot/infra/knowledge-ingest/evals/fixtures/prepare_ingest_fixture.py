# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Prepare the pinned, writable knowledge sample for the ingest evaluation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = "https://gitcode.com/cann/cannbot-knowledge.git"
REVISION = "61aa0949368c8dd019230480229e636cbf466c2c"
SKILLS = ".agents/skills"
CARDS = (
    "knowledge/ops/ascendc/guides/knowledge_index_stale_valueerror_rebuild_first_not_bypass.md",
    "knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/intra_core_sync/hard_event.md",
    "knowledge/ops/ascendc/runbooks/runtime/catlass_resource_memory_model_event_ids.md",
    "knowledge/ops/ascendc/runbooks/runtime/co1_ready_reuse_event_diagnostic.md",
    "knowledge/ops/ascendc/runbooks/runtime/syncall_requires_batch_schedule_mode.md",
    "knowledge/ops/ascendc/runbooks/compilation/mix_syncall_deadlock_pure_aiv_escape.md",
    "knowledge/ops/ascendc/runbooks/compilation/manual_crosscore_flag_handoff_mixed_aic_aiv.md",
    "knowledge/ops/ascendc/runbooks/compilation/ol_35_pipebarrier_only_supports_mte2_v_mte3_not_pipe_s.md",
    "knowledge/ops/ascendc/runbooks/runtime/group_2d_core_split_numblocks_align_schedule_mode.md",
)


DATA_COPY_PAD_CARDS = (
    "knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_pad_gm_to_ub.md",
    "knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_pad_ub_to_gm.md",
)


def git(source, *args):
    return subprocess.check_output(
        ["git", "-C", source, *args],
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        timeout=60,
    )


def native(root, script, *args):
    venv = root / ".venv" / "bin" / "python"
    python = venv if venv.is_file() else sys.executable
    return subprocess.check_output(
        [python, "-B", root / SKILLS / script, "--knowledge-root", root, *args],
        cwd=root,
        timeout=60,
    )


def checkout_sample(source, root, cards):
    paths = (
        "AGENTS.md",
        "LICENSE",
        "requirements.txt",
        "governance",
        f"{SKILLS}/knowledge-query/scripts",
        f"{SKILLS}/knowledge-ingest/scripts",
        f"{SKILLS}/knowledge-lint/scripts",
        *cards,
    )
    entries = git(source, "ls-tree", "-rz", REVISION, "--", *paths)
    for entry in entries.split(b"\0"):
        if entry and not entry.startswith((b"100644 blob ", b"100755 blob ")):
            raise ValueError("Pinned fixture inputs must be regular files")
    git(source, "--work-tree", root, "checkout", REVISION, "--", *paths)


def fetch_source(source):
    source.mkdir()
    git(source, "init", "--quiet")
    git(source, "fetch", "--quiet", "--no-tags", "--depth=1", REPOSITORY, REVISION)
    if git(source, "rev-parse", "FETCH_HEAD").decode().strip() != REVISION:
        raise ValueError("Unexpected knowledge revision")


def prepare_runtime(root):
    setup = Path(__file__).resolve().parents[2] / "scripts" / "setup_knowledge.py"
    result = subprocess.check_output(
        [sys.executable, "-B", setup, "--knowledge-root", root], cwd=root, timeout=110
    )
    if Path(json.loads(result)["root"]) != root:
        raise ValueError("The native installer returned an unexpected knowledge root")


def prepare(workspace, cards=CARDS):
    target = workspace / "knowledge-fixture"
    config = workspace / ".cannbot" / "knowledge.env"
    if target.exists() or target.is_symlink() or config.exists() or config.is_symlink():
        raise ValueError("Expected a workspace without a prior knowledge fixture")
    with tempfile.TemporaryDirectory(
        prefix=".ingest-fixture-", dir=workspace
    ) as scratch:
        source = Path(scratch) / "source"
        fetch_source(source)
        root = Path(scratch) / "repository"
        root.mkdir()
        checkout_sample(source, root, cards)
        root.rename(target)
        try:
            prepare_runtime(target)
            native(
                target, "knowledge-ingest/scripts/knowledge_ingest.py", "sync-indexes"
            )
            native(
                target,
                "knowledge-query/scripts/knowledge_index.py",
                "verify",
                "--level",
                "full",
            )
            config.parent.mkdir(exist_ok=True)
            config.write_text(
                "CANNBOT_KNOWLEDGE_ROOT=../knowledge-fixture\n", encoding="utf-8"
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            shutil.rmtree(target)
            raise
    print(f"Prepared {len(cards)} original knowledge cards from {REVISION}")


def changed_card(workspace, root):
    tracked = set(CARDS)
    actual = {
        p.relative_to(root).as_posix()
        for p in (root / "knowledge").rglob("*.md")
        if p.name != "index.md"
    }
    if actual != tracked:
        raise ValueError("The existing knowledge cards must be preserved")
    with tempfile.TemporaryDirectory(
        prefix=".ingest-receipt-", dir=workspace
    ) as scratch:
        source = Path(scratch) / "source"
        fetch_source(source)
        changed = []
        for rel in CARDS:
            path = root / rel
            if path.resolve() != path or not path.is_file():
                raise ValueError("Expected a regular card inside the fixture")
            before = git(source, "show", f"{REVISION}:{rel}")
            after = path.read_bytes()
            if before != after:
                changed.append((rel, before, after))
    if len(changed) != 1:
        raise ValueError("Expected one existing card to be updated")
    rel, before, after = changed[0]
    markers = (b"sync_all.md", b"knowledge card changed after index build")
    if any(marker in before or marker not in after for marker in markers):
        raise ValueError("The changed card lacks the new source evidence")
    return rel


def verify(workspace):
    root = workspace / "knowledge-fixture"
    if root.resolve() != root or not root.is_dir():
        raise ValueError("Expected the prepared knowledge fixture")
    rel = changed_card(workspace, root)
    native(
        root,
        "knowledge-lint/scripts/knowledge_lint.py",
        "--card",
        rel,
        "--strict-navigation",
    )
    native(
        root, "knowledge-ingest/scripts/knowledge_ingest.py", "sync-indexes", "--check"
    )
    native(
        root, "knowledge-query/scripts/knowledge_index.py", "verify", "--level", "full"
    )
    results = json.loads(
        native(
            root,
            "knowledge-query/scripts/knowledge_query.py",
            "search",
            "--platform",
            "a2",
            "--status",
            "all",
            "--query",
            "knowledge card changed after index build",
        )
    )
    handle = Path(rel).relative_to("knowledge").with_suffix("").as_posix()
    if not any(card.get("path") == handle for card in results["results"]):
        raise ValueError("A2 search did not return the updated original card")
    print(
        json.dumps(
            {"result": "KNOWLEDGE_FIXTURE_VERIFIED", "card": rel, "revision": REVISION}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--verify", action="store_true", help="Validate the edited fixture"
    )
    modes.add_argument(
        "--data-copy-pad", action="store_true", help="Prepare the DataCopyPad query sample"
    )
    args = parser.parse_args()
    if args.verify:
        verify(Path.cwd())
    else:
        prepare(Path.cwd(), DATA_COPY_PAD_CARDS if args.data_copy_pad else CARDS)
