#!/usr/bin/env python3
"""Verify verl + optional vLLM-Ascend patches for train-inference consistency dump."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ENGINE_MARKERS = [
    "_ensure_debugger",
    "_log_update_actor_step",
    "DUMP_ON",
    "update_actor_log.jsonl",
    "PrecisionDebugger",
]

VERL_CHECKS: list[tuple[str, list[str], bool]] = [
    (
        "verl/trainer/ppo/ray_trainer.py",
        ["PROMPTS_ONLY", "compute_prompts_only", "responses_len"],
        True,
    ),
    (
        "verl/utils/debug/metrics.py",
        ["PROMPTS_ONLY", "skipping debug metrics"],
        True,
    ),
    (
        "verl/trainer/ppo/rollout_corr_helper.py",
        ["PROMPTS_ONLY", "compute_rollout_correction_and_add_to_batch"],
        True,
    ),
]

REQUEST_ID_CHECKS: list[tuple[str, list[str]]] = [
    ("verl/workers/rollout/llm_server.py", ['extra_fields["request_id"]', "vllm_request_id"]),
    ("verl/experimental/agent_loop/agent_loop.py", ['extra_fields["request_id"]', "vllm_request_id"]),
]

VLLM_CHECKS: list[tuple[str, list[str]]] = [
    (
        "vllm_ascend/worker/dispatch_logger.py",
        ["class DispatchLogger", "dispatch_log.jsonl"],
    ),
    (
        "vllm_ascend/worker/model_runner_v1.py",
        ["DispatchLogger", "_finish_msprobe_step", "_dispatch_logger"],
    ),
]

BACKEND_FILES = {
    "fsdp": "verl/workers/engine/fsdp/transformer_impl.py",
    "megatron": "verl/workers/engine/megatron/transformer_impl.py",
}


def check_markers(root: Path, rel: str, markers: list[str]) -> tuple[bool, list[str]]:
    path = root / rel
    if not path.is_file():
        return False, [f"missing file: {path}"]

    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [m for m in markers if m not in text]
    if missing:
        return False, [f"missing markers: {', '.join(missing)}"]
    return True, []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", required=True)
    parser.add_argument("--backend", choices=["fsdp", "megatron"], default="fsdp")
    parser.add_argument("--vllm-ascend-root", default="")
    parser.add_argument("--skip-vllm", action="store_true")
    args = parser.parse_args()

    verl_root = Path(args.verl_root)
    failed = False

    engine_rel = BACKEND_FILES[args.backend]
    ok, errors = check_markers(verl_root, engine_rel, ENGINE_MARKERS)
    if ok and args.backend == "fsdp":
        text = (verl_root / engine_rel).read_text(encoding="utf-8", errors="replace")
        if "forward_backward_batch" not in text or "_debugger.start" not in text:
            ok, errors = False, ["FSDP: expected dump hooks in forward_backward_batch"]
    if ok and args.backend == "megatron":
        text = (verl_root / engine_rel).read_text(encoding="utf-8", errors="replace")
        if "_finish_dump_step" not in text and "_dump_should_record" not in text:
            ok, errors = False, ["Megatron: expected _finish_dump_step hooks"]

    if ok:
        print(f"PASS: {verl_root / engine_rel}")
    else:
        failed = True
        print(f"FAIL: {verl_root / engine_rel}")
        for err in errors:
            print(f"  - {err}")

    for rel, markers, required in VERL_CHECKS:
        ok, errors = check_markers(verl_root, rel, markers)
        if ok:
            print(f"PASS: {verl_root / rel}")
        elif required:
            failed = True
            print(f"FAIL: {verl_root / rel}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"SKIP (optional): {verl_root / rel}")

    request_id_ok = False
    for rel, markers in REQUEST_ID_CHECKS:
        ok, errors = check_markers(verl_root, rel, markers)
        if ok:
            request_id_ok = True
            print(f"PASS: {verl_root / rel} (request_id)")
        else:
            print(f"INFO: {verl_root / rel} not patched ({errors[0] if errors else 'missing'})")
    if not request_id_ok:
        failed = True
        print("FAIL: request_id injection missing in both llm_server.py and agent_loop.py")

    if not args.skip_vllm and args.vllm_ascend_root:
        vllm_root = Path(args.vllm_ascend_root)
        for rel, markers in VLLM_CHECKS:
            ok, errors = check_markers(vllm_root, rel, markers)
            if ok:
                print(f"PASS: {vllm_root / rel}")
            else:
                failed = True
                print(f"FAIL: {vllm_root / rel}")
                for err in errors:
                    print(f"  - {err}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
