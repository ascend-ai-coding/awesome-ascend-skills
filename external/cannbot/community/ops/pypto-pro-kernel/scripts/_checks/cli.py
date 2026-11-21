# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Kernel skill checks and a source-gated command runner.

Invoke from any directory with `python <skill>/scripts/check.py`. Defaults are resolved relative to the skill; explicit input paths are resolved relative to the caller's current directory. Put path-override options before the subcommand.

`rules` runs static checks; `coverage` verifies API example assignments; `catalog` reconciles the matmul/algorithm index; `links` checks local Markdown targets and anchors; `surface` searches API usage; `api` reads declaration stubs; `srcrefs` reports suspicious source-line references. `suggest` proposes bounded source-pattern comparisons; `experiment` maintains explicit best/current scratch state. The default command runs rules and knowledge admission, including coverage, catalog, links and API freshness. Those checks do not import PyPTO or use a board. `run` checks the named source files, then executes the caller's explicit command in the current environment.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections import Counter, deque
from pathlib import Path
from .actions import GOALS

from . import api as api_table
from . import api_stub
from . import catalog
from . import explain
from . import links
from . import name_resolution
from . import (
    feedback,
    query,
    knowledge,
    knowledge_admission,
    knowledge_evaluation,
    ub_l1,
    group_depth,
    native,
    runtime_diagnostics,
    evaluation,
    suggest,
    experiment,
    vf_dataflow,
    host_io,
    execution_status,
    task_context,
    claims,
)
from . import coverage as coverage_rule
from . import (
    api_policy,
    declaration,
    hardware,
    init_output,
    mutex_overlap,
    simt,
    srcrefs,
    synchronisation,
    tile_shape,
    valid_shape,
    register_alias,
    register_dtype,
    register_init,
    gm_scalar,
    vf_scope,
    vf_control_flow,
    gemm,
    scalar_annotation,
    allocation_scope,
    stage_handover,
    work_distribution,
)
from .source import Finding, reuse_parsed_sources
from .paths import DOC_ROOTS, EXAMPLES_ROOT, PYPTO_ROOT

DEFAULT_SCOPE = EXAMPLES_ROOT


def python_files(targets: list[Path]) -> list[Path]:
    found: list[Path] = []
    for target in targets:
        if target.is_dir():
            found += [
                path
                for path in sorted(target.rglob("*.py"))
                if "__pycache__" not in path.parts
            ]
        else:
            found.append(target)
    return found


@reuse_parsed_sources()
def run_rules(
    targets: list[Path], *, pypto_root: Path = PYPTO_ROOT, examples_root: Path = EXAMPLES_ROOT
) -> list[Finding]:
    """Apply declarations to public examples and native rules to every selected Python file."""
    findings: list[Finding] = []
    files = python_files(targets)
    root = Path(examples_root).resolve()
    example_kernels = set()
    if any(path.name == "kernel.py" and path.resolve().is_relative_to(root) for path in files):
        example_kernels = {
            path.resolve()
            for path in catalog.public_kernels(root, groups=("basic_api", *catalog.GROUPS))
            if path.resolve().is_relative_to(root)
        }
    for path in files:
        findings += name_resolution.check(path)
        if path.resolve() in example_kernels:
            findings += declaration.check(path)
        findings += hardware.check(path)
        findings += tile_shape.check(path)
        findings += valid_shape.check(path)
        findings += register_alias.check(path)
        findings += register_dtype.check(path)
        findings += register_init.check(path)
        findings += vf_dataflow.check(path, pypto_root=pypto_root)
        findings += gm_scalar.check(path)
        findings += vf_scope.check(path, pypto_root=pypto_root)
        findings += vf_control_flow.check(path)
        findings += work_distribution.check(path)
        findings += api_policy.check(path, pypto_root=pypto_root)
        findings += gemm.check(path)
        findings += allocation_scope.check(path)
        findings += group_depth.check(path, pypto_root=pypto_root)
        findings += scalar_annotation.check(path, pypto_root=pypto_root)
        findings += synchronisation.check(path, pypto_root=pypto_root)
        findings += stage_handover.check(path)
        findings += init_output.check(path)
        findings += mutex_overlap.check(path)
        findings += simt.check(path)
        findings += ub_l1.check(path, pypto_root=pypto_root)
    return findings


def report_rules(
    targets: list[Path], *, pypto_root: Path = PYPTO_ROOT, examples_root: Path = EXAMPLES_ROOT
) -> bool:
    findings = run_rules(targets, pypto_root=pypto_root, examples_root=examples_root)
    violations = [
        finding for finding in findings if not finding.waived and not finding.advisory
    ]
    waived = [finding for finding in findings if finding.waived]
    advisory = [
        finding for finding in findings if finding.advisory and not finding.waived
    ]
    for finding in findings:
        print(finding)
    name_scopes = [name_resolution.inspect(path)[1] for path in python_files(targets)]
    print(
        f"Python 名字（Pyflakes {name_resolution.backend()[1]}）："
        f"{sum(s['checked'] for s in name_scopes)} 个文件已检查，"
        f"{sum(s['unverified'] for s in name_scopes)} 个文件未判定；"
        f"{sum(f.rule == name_resolution.RULE for f in violations)} 处未定义名字"
    )
    for reason in sorted({s["reason"] for s in name_scopes if s["unverified"]}):
        print(f"  名字检查未判定：{reason}")
    ranks = [tile_shape.scope_counts(path) for path in python_files(targets)]
    api_counts = [
        api_policy.inspect(path, pypto_root=pypto_root)[1]
        for path in python_files(targets)
    ]
    print(
        f"api policy：{sum(c['vf'] for c in api_counts)} 处 VF 调用，"
        f"{sum(c['unverified'] for c in api_counts)} 处名称未完整核对；"
        f"{sum(c['tile'] for c in api_counts)} 处禁用的 tile Vector 计算"
    )
    print(
        f"tile rank：{sum(c for c, _ in ranks)} 处字面量已检查，"
        f"{sum(u for _, u in ranks)} 处非字面量未判定；"
        f"{sum(f.rule == tile_shape.RULE for f in violations)} 处违规"
    )
    dst_counts = [register_init.scope_counts(path) for path in python_files(targets)]
    print(
        f"VF 目标读取：{sum(c for c, _ in dst_counts)} 处直接 FMA 赋值已检查，"
        f"{sum(u for _, u in dst_counts)} 处调用形式/命名空间未判定；"
        f"{sum(f.rule == register_init.RULE for f in violations)} 处违规"
    )
    allocation_counts = [
        allocation_scope.scope_counts(path) for path in python_files(targets)
    ]
    print(
        f"分配作用域：{sum(c for c, _ in allocation_counts)} 处直接调用已检查，"
        f"{sum(u for _, u in allocation_counts)} 处上下文/命名空间未判定；"
        f"{sum(f.rule == allocation_scope.RULE for f in violations)} 处违规"
    )
    depth_counts = [
        group_depth.inspect(path, pypto_root=pypto_root)[1]
        for path in python_files(targets)
    ]
    print(
        f"group depth：{sum(c['checked'] for c in depth_counts)} 处已检查，"
        f"{sum(c['unverified'] for c in depth_counts)} 处未判定；"
        f"{sum(f.rule == group_depth.RULE for f in violations)} 处违规"
    )
    print(
        f"rules：{len(python_files(targets))} 个文件，{len(violations)} 处违规"
        f"（另有 {len(waived)} 处按 `反例：` 声明放行，{len(advisory)} 处需核查提示）"
    )
    return not violations


def rules_result(targets, *, pypto_root, examples_root=EXAMPLES_ROOT, findings=None):
    files = python_files(targets)
    result = feedback.envelope("source_check", pypto_root, files)
    findings = (
        run_rules(targets, pypto_root=pypto_root, examples_root=examples_root)
        if findings is None else findings
    )
    result["diagnostics"] = [feedback.finding_dict(f) for f in findings]
    transfers, resolved, unverified = [], 0, 0
    for path in files:
        _, facts, counts = ub_l1.inspect(path, pypto_root=pypto_root)
        transfers += facts
        resolved += counts["resolved"]
        unverified += counts["unverified"]
    result["coverage"] = {
        "source_files": len(files),
        "ub_l1": {
            "resolved": resolved,
            "unverified": unverified,
            "limits": (
                "Direct source descriptors only; helpers, shadowing, producer bytes and synchronization are not proved."
            ),
        },
        "lexical_names": [name_resolution.inspect(path)[1] for path in files],
        "group_depth": [
            {
                "file": str(path.resolve()),
                **group_depth.inspect(path, pypto_root=pypto_root)[1],
            }
            for path in files
        ],
    }
    chains = [vf_dataflow.inspect(path, pypto_root=pypto_root) for path in files]
    result["coverage"]["vf_dataflow"] = [
        {"file": str(path.resolve()), **chain[2]} for path, chain in zip(files, chains)
    ]
    result["coverage"]["synchronisation"] = [
        {
            "file": str(path.resolve()),
            **synchronisation.inspect(path, pypto_root=pypto_root)[1],
        }
        for path in files
    ]
    result["dataflow_reviews"] = [
        {
            "file": str(path.resolve()),
            "source_sha256": feedback.file_identity(path)["sha256"],
            **review,
        }
        for path, chain in zip(files, chains)
        for review in chain[1]
    ][:8]
    result["transfers"] = transfers[:12]
    result["omitted_transfer_sites"] = max(0, len(transfers) - 12)
    result["context"] = query.context(
        query.ids_for_rules(f.rule for f in findings), examples_root=examples_root
    )
    return feedback.finish(result)


def source_hashes(targets: list[Path]) -> dict[Path, str]:
    return {
        path.resolve(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in python_files(targets)
    }


def _run_with_tail(command, *, json_output=False):
    """Preserve live stdout/stderr and retain a bounded diagnostic tail."""
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    tails = [deque(maxlen=200), deque(maxlen=200)]

    def forward(stream, destination, tail):
        try:
            while chunk := stream.readline(4096):
                destination.write(chunk)
                destination.flush()
                tail.append(chunk)
        finally:
            stream.close()

    readers = [
        threading.Thread(target=forward, args=(stream, destination, tail), daemon=True)
        for stream, destination, tail in zip(
            (proc.stdout, proc.stderr),
            (sys.stderr if json_output else sys.stdout, sys.stderr),
            tails,
        )
    ]
    for reader in readers:
        reader.start()
    try:
        status = proc.wait()
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    finally:
        for reader in readers:
            reader.join(timeout=1)
    return status, "\n".join("".join(tail) for tail in tails)


def run_command(
    targets: list[Path],
    command: list[str],
    *,
    pypto_root: Path,
    examples_root=EXAMPLES_ROOT,
    native_spec=None,
    json_output=False,
    details=False,
    feedback_file=None,
) -> int:
    """Gate one explicit command; do not infer numerical success from its output."""
    before = source_hashes(targets)
    if feedback_file is not None:
        feedback_file = feedback_file.resolve()
        if feedback_file in before or (
            native_spec is not None and feedback_file == native_spec.resolve()
        ):
            raise ValueError("The feedback file cannot overwrite a checked source or native spec")
        if feedback_file.suffix == ".py" and any(
            path.is_dir() and feedback_file.is_relative_to(path.resolve())
            for path in targets
        ):
            raise ValueError("The feedback file cannot add a Python source to the checked set")
        feedback_file.parent.mkdir(parents=True, exist_ok=True)
        feedback_file.unlink(missing_ok=True)
    spec_before = feedback.file_identity(native_spec) if native_spec else None
    result = rules_result(targets, pypto_root=pypto_root, examples_root=examples_root)
    result["command"] = {"argv": command, "started": False, "exit_code": None}
    review_sources = list(before)

    def emit():
        feedback.finish(result)
        if (
            result["status"] == "failed"
            and not result["context"]
            and any(
                d["rule"]
                in {"command-failed", "native-rejection", "framework-rejection"}
                for d in result["diagnostics"]
            )
            and not any(d["rule"] == "source-changed" for d in result["diagnostics"])
        ):
            try:
                if source_hashes(targets) == before and (
                    native_spec is None
                    or feedback.file_identity(native_spec) == spec_before
                ):
                    routes, omitted = query.review_routes(
                        review_sources, pypto_root=pypto_root
                    )
                    if (
                        routes
                        and source_hashes(targets) == before
                        and (
                            native_spec is None
                            or feedback.file_identity(native_spec) == spec_before
                        )
                    ):
                        (
                            result["related_contracts"],
                            result["omitted_related_contracts"],
                        ) = routes, omitted
                        result["contract_review_scope"] = (
                            "Conditional reviews of statically resolved source"
                            " references; not a diagnosis, executed-call proof or"
                            " inferred numerical root cause. Query a matching owner before a broad search."
                        )
            except (OSError, SyntaxError):
                pass  # A missing or changed source cannot supply a review route.
        if feedback_file is not None:
            payload = (
                json.dumps(feedback.presentation(result, details=details), ensure_ascii=False)
                + "\n"
            ).encode("utf-8")
            with tempfile.TemporaryDirectory(
                prefix=".kernel-feedback-", dir=feedback_file.parent
            ) as directory:
                temporary = Path(directory) / "feedback.json"
                temporary.write_bytes(payload)
                temporary.chmod(0o600)
                os.replace(temporary, feedback_file)
            if not json_output:
                print(
                    "kernel_feedback_file "
                    + json.dumps(
                        {
                            "path": str(feedback_file),
                            "sha256": feedback.digest(payload),
                            "status": result["status"],
                            "command": {
                                key: result["command"][key]
                                for key in ("started", "exit_code")
                            },
                            "validation": result["validation"],
                            "diagnostic_counts": dict(
                                Counter(item["severity"] for item in result["diagnostics"])
                            ),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                return
        if json_output:
            feedback.print_result(result, details=details)
        else:
            print(
                "kernel_feedback "
                + json.dumps(
                    feedback.presentation(result, details=details),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                flush=True,
            )

    if not before:
        print("run: no Python sources to check", file=sys.stderr)
        return 2
    if not json_output:
        for item in result["diagnostics"]:
            loc = item["location"]
            print(f"{loc['file']}:{loc['line']}: {item['rule']}: {item['message']}")
    if result["status"] == "failed":
        print("run: static checks failed; command was not started", file=sys.stderr)
        emit()
        return 1
    if source_hashes(targets) != before:
        print(
            "run: sources changed during preflight; command was not started",
            file=sys.stderr,
        )
        result["diagnostics"].append(
            feedback.diagnostic(
                "source-changed", "error", "Sources changed during preflight."
            )
        )
        emit()
        return 1
    if native_spec:
        _, entry, _ = native.read_spec(native_spec)
        if entry not in before:
            result["diagnostics"].append(
                feedback.diagnostic(
                    "native-entry-unchecked",
                    "error",
                    "The native spec entry must be included in --source.",
                )
            )
            emit()
            return 1
        review_sources = [entry]
        with contextlib.redirect_stdout(sys.stderr):
            parsed = native.report(
                native_spec, pypto_root=pypto_root, examples_root=examples_root
            )
        result["native_preflight"] = {k: parsed[k] for k in ("status", "specialization", "coverage")}
        result["identity"]["installed_runtime"] = parsed["identity"][
            "installed_runtime"
        ]
        result["validation"].update(parsed["validation"])
        result["diagnostics"] += parsed["diagnostics"]
        if parsed["context"]:
            result["context"] = parsed["context"]
        if parsed["status"] != "passed":
            result["stage"], result["status"] = parsed["stage"], parsed["status"]
            emit()
            return feedback.exit_status(parsed)
        if (
            source_hashes(targets) != before
            or feedback.file_identity(native_spec) != spec_before
        ):
            result["diagnostics"].append(
                feedback.diagnostic(
                    "source-changed",
                    "error",
                    "Sources or native spec changed during native preflight.",
                )
            )
            emit()
            return 1
    for path, digest in before.items():
        if not json_output:
            print(f"source_sha256 {digest}  {path}", flush=True)
    sys.stdout.flush()
    try:
        result["stage"] = "execution"
        return_code, output_tail = _run_with_tail(command, json_output=json_output)
        result["command"]["started"] = True
    except OSError as exc:
        print(f"run: could not start command: {exc}", file=sys.stderr)
        result["diagnostics"].append(
            feedback.diagnostic("command-start-failed", "error", str(exc))
        )
        emit()
        return 1
    status = return_code if return_code >= 0 else 128 - return_code
    result["command"]["exit_code"] = status
    if not json_output:
        print(f"command_exit={status}", flush=True)
    if status:
        diagnostics, contexts = runtime_diagnostics.from_output(
            output_tail,
            examples_root=examples_root,
            transfers=result.get("transfers", []),
        )
        result["diagnostics"] += diagnostics or [
            feedback.diagnostic(
                "command-failed",
                "error",
                "The command failed; no specific diagnostic matched.",
                evidence={"output_tail": output_tail[-8000:]},
                next_action="Inspect the original output; no root cause was inferred.",
            )
        ]
        if contexts:
            result["context"] = contexts
        modules = {d["evidence"].get("module") for d in diagnostics}
        if modules & {"PRO_CODEGEN", "PRO_COMPILE"}:
            result["validation"]["compilation"] = "failed"
    diagnostic_body = "\n".join(runtime_diagnostics.bodies(output_tail))
    if status and explain._RANK.search(diagnostic_body):
        for path in before:
            try:
                current = explain.inspect(path)
                if current["sha256"] != before[path]:
                    continue
                symptoms = explain.rank_diagnostics(current, diagnostic_body)
                symptoms = [
                    s
                    for s in symptoms
                    if s["declarations"] or s["same_name_allocations"]
                ]
                if symptoms:
                    contract = {
                        "source_sha256": current["sha256"],
                        "source": str(path),
                        "scope": "bounded command-output tail; name matches require launch verification",
                        "symptoms": symptoms,
                    }
                    result.setdefault("source_contract_diagnostics", []).append(
                        contract
                    )
                    if not json_output:
                        print(
                            "source_contract_diagnostic "
                            + json.dumps(contract, ensure_ascii=False),
                            flush=True,
                        )
            except (OSError, SyntaxError, ValueError):
                pass  # Source drift is separately reported below.
    try:
        changed = (
            source_hashes(targets) != before
            or native_spec is not None
            and feedback.file_identity(native_spec) != spec_before
        )
    except OSError:
        changed = True
    if changed:
        print(
            "run: sources changed during execution; result is not bound to the checked source",
            file=sys.stderr,
        )
        status = status or 1
        result["diagnostics"].append(
            feedback.diagnostic(
                "source-changed",
                "error",
                "Sources or native spec changed during execution; validation is invalidated.",
            )
        )
        result["validation"] = {key: "invalidated" for key in result["validation"]}
    emit()
    return status


def report_coverage(examples_root: Path = EXAMPLES_ROOT) -> bool:
    table = coverage_rule.collect(examples_root)
    problems = table.problems()
    print(
        f"coverage：§八点二 {len(table.assigned)} 项 ↔ {len(table.written)} 个样例，"
        f"{len(problems)} 处对不上"
    )
    for line in problems:
        print(f"  {coverage_rule.RULE}: {line}")
    for example, apis in table.pending().items():
        print(f"  待写：`{example}` 被指派 {len(apis)} 项 —— {', '.join(apis)}")
    return not problems


def report_catalog(examples_root: Path = EXAMPLES_ROOT) -> bool:
    result = catalog.collect(examples_root)
    print(
        f"catalog：{len(result.examples)} 个样例，{len(result.indexed)} 个已收录，"
        f"{len(result.findings)} 处对不上"
    )
    for finding in result.findings:
        print(finding)
    return not result.findings


def report_links(targets: tuple[Path, ...] | list[Path] = DOC_ROOTS) -> bool:
    result = links.collect(targets)
    print(
        f"links: {len(result.documents)} Markdown files, {result.local_links} local links, "
        f"{result.checked_fragments} Markdown fragments checked; {len(result.findings)} findings"
    )
    print(
        f"  {result.external_links} external links not fetched; "
        f"{result.unverified_fragments} non-Markdown fragments unverified"
    )
    for finding in result.findings:
        print(finding)
    return not result.findings


def report_srcrefs(
    doc_roots: tuple[Path, ...] = DOC_ROOTS, pypto_root: Path = PYPTO_ROOT
) -> bool:
    """Check source-line references in documentation. **This is a report, not a gate** (the heuristic criterion was 48%
    accurate in the first item-by-item review), so it is not part of `rules` and does not affect the exit code of
    `python scripts/check.py` with no arguments.
    """
    docs = sorted(path for root in doc_roots for path in root.rglob("*.md"))
    suspects, checked, skipped, waived = srcrefs.collect(docs, pypto_root)
    print(
        f"srcrefs：核了 {checked} 处行号引用，{len(suspects)} 处可疑"
        f"（{waived} 处按 `VERIFIED` 手核放行，"
        f"另有 {skipped} 处无法判定：同名文件多于一个、或那一行没有可核的标识符）"
    )
    for suspect in suspects:
        print(f"  {suspect}")
    if suspects:
        print(
            f"  **逐条看，别批量改号**——修法三选一见 `{Path(srcrefs.__file__).resolve()}` 的 docstring"
        )
    return True


def report_surface(examples_root: Path = EXAMPLES_ROOT) -> bool:
    index = coverage_rule.surface(examples_root)
    for namespace in coverage_rule.NAMESPACES:
        apis = index[namespace]
        print(f"# {namespace}.* —— {len(apis)} 个 API")
        for api, examples in apis.items():
            print(f"{namespace}.{api}\t{' '.join(examples)}")
        print()
    return True


def _query_filters(raw):
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("Knowledge filters must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cards-root",
        type=Path,
        help="Optional knowledge-cards directory containing the owner's index.md; requires requirements-cards.txt",
    )
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=EXAMPLES_ROOT,
        help="Examples directory (including basic_api); defaults to the skill's bundled examples/",
    )
    parser.add_argument(
        "--pypto-root",
        type=Path,
        default=PYPTO_ROOT,
        help="PyPTO source repository; defaults to PYPTO_SRC or pypto/ beside the skill",
    )
    sub = parser.add_subparsers(dest="command")
    for operation in ("search", "read"):
        io = sub.add_parser(
            operation, help="Bounded source search/read with explicit paths"
        )
        io.add_argument(
            "--path", action="append", type=Path, required=operation == "search"
        )
        io.add_argument("--policy", type=Path)
        if operation == "search":
            io.add_argument("--pattern")
            io.add_argument("--glob")
            io.add_argument("--files", action="store_true")
            io.add_argument("--ignore-case", action="store_true")
            io.add_argument(
                "--fixed",
                action="store_true",
                help="Search literal text instead of a regular expression, for example 'vf.full('",
            )
        else:
            io.add_argument(
                "--request",
                action="append",
                help="Per-file JSON range; repeat instead of --path",
            )
            io.add_argument(
                "--start-line",
                type=int,
                default=1,
                help="Absolute source line; raw text and numbered_text retain source positions for citations",
            )
            io.add_argument("--max-lines", type=int)
            io.add_argument("--offset-bytes", type=int)
            io.add_argument("--source-sha256")
            io.add_argument(
                "--fragment",
                help="Markdown heading slug or explicit anchor; keeps continuation inside that section",
            )
    status = sub.add_parser(
        "status",
        help="Wait on one existing source-bound execution receipt; never launch",
    )
    status.add_argument("--receipt", type=Path, required=True)
    status.add_argument("--request-id", required=True)
    status.add_argument("--source", type=Path, required=True)
    status.add_argument("--wait-seconds", type=float, default=20)
    status.add_argument("--policy", type=Path)
    hook = sub.add_parser(
        "host-hook", help="Codex scoped-read PreToolUse adapter, reading JSON on stdin"
    )
    hook.add_argument("--policy", type=Path, required=True)
    hook.add_argument("--counters", type=Path)
    hook.add_argument(
        "--config",
        action="store_true",
        help="Print the hook configuration without enabling it",
    )
    recommendations = sub.add_parser(
        "suggest",
        help="Bounded source-pattern hypotheses with optional measured task costs",
    )
    recommendations.add_argument("--source", type=Path, action="append", required=True)
    recommendations.add_argument("--measurements", type=Path)
    recommendations.add_argument("--limit", type=int, default=3)
    recommendations.add_argument("--details", action="store_true")
    recommendations.add_argument(
        "--goal",
        choices=GOALS,
        help="Explicit current action; adds scoped guidance without execution",
    )
    progress = sub.add_parser(
        "experiment",
        help="Maintain one best snapshot and one current trial in explicit scratch",
    )
    progress.add_argument("--state", type=Path, required=True)
    actions = progress.add_mutually_exclusive_group()
    actions.add_argument(
        "--init",
        type=Path,
        help="Initialize from a frozen plan; never overwrite an existing state",
    )
    actions.add_argument(
        "--select",
        type=Path,
        dest="selection",
        help="Replace the current trial from a selection spec",
    )
    actions.add_argument(
        "--observe",
        type=Path,
        dest="observation",
        help="Import an explicit verifier observation",
    )
    actions.add_argument(
        "--export-best",
        type=Path,
        help="Restore verified best source bytes into a new directory",
    )
    assertion = sub.add_parser(
        "claim",
        help="Check a scoped performance assertion using shared experiment evidence gates; no execution",
    )
    assertion.add_argument("--spec", type=Path, required=True)
    evaluate = sub.add_parser(
        "evaluate",
        help="Bounded UB→L1 tool tasks or matched external agent observations",
    )
    evaluate.add_argument(
        "--checker",
        type=Path,
        help="Checker entry to evaluate, including a frozen baseline",
    )
    evaluate.add_argument(
        "--split", choices=("all", "development", "heldout"), default="all"
    )
    evaluate.add_argument(
        "--observations",
        type=Path,
        help="Compare external model observations instead of running tool tasks",
    )
    evaluate.add_argument(
        "--prepare",
        type=Path,
        help="Prepare a blind repair task in a new scratch directory",
    )
    evaluate.add_argument(
        "--events",
        type=Path,
        help="Read actual host JSONL tool/token telemetry without grading correctness",
    )
    evaluate.add_argument(
        "--host-counters",
        type=Path,
        help="Matching audited host counters, including pre-tool attempts omitted from JSONL",
    )
    evaluate.add_argument(
        "--knowledge-cases",
        type=Path,
        help="Grade a frozen retrieval benchmark through the shared query CLI",
    )
    lookup = sub.add_parser(
        "query",
        help="Discover knowledge objects, exact public symbols and maintained scenarios",
    )
    lookup.add_argument("text", nargs="?")
    lookup.add_argument(
        "--kind",
        dest="kinds",
        action="append",
        choices=knowledge.KINDS,
        help="Filter knowledge kinds; repeat to combine",
    )
    lookup.add_argument(
        "--filters",
        type=_query_filters,
        help=(
            "JSON routing conditions: source_memory, destination_memory, dtype, layout, source_dtype, destination_dtype"
        ),
    )
    lookup.add_argument(
        "--limit",
        type=int,
        help=f"Knowledge objects per page, 1..{knowledge.MAX_LIMIT}; default {knowledge.DEFAULT_LIMIT}",
    )
    lookup.add_argument(
        "--cursor",
        help="Continue with the same text, kinds, filters and selected sources",
    )
    lookup.add_argument(
        "--reference-sha256",
        help="Pin a document/docstring or example kernel resolution to its discovered source SHA",
    )
    lookup.add_argument(
        "--task",
        type=Path,
        help="Inspect a pypto-task-context/1 JSON inventory; use positional TEXT for a knowledge question",
    )
    lookup.add_argument(
        "--validate",
        action="store_true",
        help="Read-only knowledge admission, separate from discovery and task inventory",
    )
    lookup.add_argument(
        "--policy", type=Path, help="Fixed roots and reply budget for --task"
    )
    lookup.add_argument("--source", type=Path)
    lookup.add_argument("--format", choices=("text", "json"), default="text")
    lookup.add_argument(
        "--details",
        action="store_true",
        help="Include full identity and descriptor fields",
    )
    lookup.add_argument(
        "--goal",
        choices=GOALS,
        help="Explicit current action, or use the controller inventory goal",
    )
    semantic = sub.add_parser(
        "native",
        help="Capture and inspect native IR from an explicit metadata spec; no launch",
    )
    semantic.add_argument("--spec", type=Path, required=True)
    semantic.add_argument("--format", choices=("text", "json"), default="json")
    semantic.add_argument(
        "--details",
        action="store_true",
        help="Include full identity and transfer descriptor fields",
    )
    diagnostic = sub.add_parser(
        "explain",
        help="Source-bound contracts and work-site differences; no code execution",
    )
    diagnostic.add_argument("--source", type=Path, required=True)
    diagnostic.add_argument("--baseline", type=Path)
    diagnostic.add_argument("--error-log", type=Path)
    diagnostic.add_argument(
        "--function", help="Optional function name or qualified nested name"
    )
    rules = sub.add_parser("rules", help="Static rules (gate)")
    rules.add_argument("paths", nargs="*", type=Path, help=f"Default: {DEFAULT_SCOPE}/")
    rules.add_argument("--format", choices=("text", "json"), default="text")
    rules.add_argument(
        "--details",
        action="store_true",
        help="Include full identity and transfer descriptor fields in JSON",
    )
    execute = sub.add_parser(
        "run", help="Static checks and source hashes, then an explicit command"
    )
    execute.add_argument(
        "--source",
        dest="sources",
        action="append",
        type=Path,
        required=True,
        help="Python file or directory to check; repeat for kernel, wrapper and tests",
    )
    execute.add_argument(
        "--native-spec",
        type=Path,
        help="Require matching native parse/IR preflight before the command",
    )
    execute.add_argument("--format", choices=("text", "json"), default="text")
    execute.add_argument(
        "--feedback-file",
        type=Path,
        help="Write final structured feedback to this file; text mode prints a compact receipt",
    )
    execute.add_argument(
        "--details",
        action="store_true",
        help="Include full identity and transfer descriptor fields",
    )
    execute.add_argument(
        "argv", nargs=argparse.REMAINDER, help="-- command [arguments...]"
    )
    sub.add_parser("coverage", help="Reconcile the 81-item table with example code")
    sub.add_parser(
        "catalog",
        help="Reconcile matmul/algorithm examples with catalog selection tables",
    )
    doc_links = sub.add_parser(
        "links", help="Check local inline Markdown links and heading fragments (gate)"
    )
    doc_links.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Markdown files/directories; defaults to this skill, including references",
    )
    sub.add_parser("surface", help="Reverse index: API → examples that use it")
    api = sub.add_parser(
        "api", help="Live declaration signatures or generated agent .pyi reference"
    )
    api.add_argument("namespaces", nargs="*", help="pl / vf / pl.simt; defaults to all")
    pyi = api.add_mutually_exclusive_group()
    pyi.add_argument(
        "--write-pyi",
        action="store_true",
        help="Regenerate references/api/pypto_pro_language.pyi",
    )
    pyi.add_argument(
        "--check-pyi",
        action="store_true",
        help="Check the generated reference against selected source",
    )
    sub.add_parser(
        "srcrefs",
        help="Reconcile source-line references in docs (report; may include false positives)",
    )
    args, unexpected = parser.parse_known_args(argv)
    if unexpected:
        message = "unrecognized arguments: " + " ".join(unexpected)
        if (
            args.command == "search"
            and args.pattern
            and all(not value.startswith("-") for value in unexpected)
        ):
            message += (
                "\nIf these words belong to the search pattern, quote the whole value: "
                "--pattern 'whole phrase'. For literal text also use --fixed. No source search ran."
            )
        parser.error(message)

    if args.command == "host-hook":
        from . import host_hooks

        if args.config:
            host_io.policy(args.policy)
            print(
                json.dumps(
                    host_hooks.configuration(args.policy, counters=args.counters)
                )
            )
            return 0
        payload = {}
        try:
            raw = sys.stdin.read(65537)
            if len(raw.encode()) > 65536:
                raise ValueError("Hook input exceeds 64 KiB")
            payload = json.loads(raw)
            result = host_hooks.pre_tool(payload, policy_path=args.policy)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Context hook could not validate this read: "
                    + str(exc)[:300],
                }
            }
        if (
            args.counters
            and isinstance(payload, dict)
            and isinstance(payload.get("session_id"), str)
        ):
            host_hooks.count(args.counters, payload, result)
        print(json.dumps(result))
        return 0

    if args.command in {"search", "read", "status"}:
        config = None
        try:
            config = host_io.policy(
                args.policy,
                pypto_root=args.pypto_root,
                examples_root=args.examples_root,
                cards_root=args.cards_root,
            )
            if args.command == "status":
                if args.policy:
                    host_io.checked_path(args.source, config)
                    host_io.checked_path(args.receipt, config, must_exist=False)
                    pinned = config["receipts"].get(str(args.receipt.resolve()))
                    if pinned and (
                        pinned["request_id"] != args.request_id
                        or pinned["source"] != str(args.source.resolve())
                    ):
                        raise ValueError(
                            "Status arguments do not match the controller-pinned request/source"
                        )
                result = execution_status.inspect(
                    args.receipt,
                    request_id=args.request_id,
                    source=args.source,
                    wait_seconds=args.wait_seconds,
                )
                print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
                return 0 if result["terminal"] else 2
            if args.command == "read":
                if args.request:
                    if (
                        args.path
                        or args.start_line != 1
                        or any(
                            value is not None
                            for value in (
                                args.max_lines,
                                args.offset_bytes,
                                args.source_sha256,
                                args.fragment,
                            )
                        )
                    ):
                        raise ValueError(
                            "Use per-request ranges or --path with global ranges, not both"
                        )
                    result = host_io.read_requests(
                        [json.loads(item) for item in args.request], config=config
                    )
                else:
                    result = host_io.read(
                        args.path,
                        config=config,
                        start_line=args.start_line,
                        max_lines=args.max_lines,
                        offset_bytes=args.offset_bytes,
                        source_sha256=args.source_sha256,
                        fragment=args.fragment,
                    )
            else:
                result = host_io.search(
                    args.path,
                    config=config,
                    pattern=args.pattern,
                    glob=args.glob,
                    files=args.files,
                    ignore_case=args.ignore_case,
                    fixed=args.fixed,
                )
            print(host_io.encode(result))
            return 0
        except (OSError, ValueError, TypeError, subprocess.TimeoutExpired) as exc:
            result = host_io.failure(
                args.command,
                exc,
                limit=config["limits"]["max_output_bytes"]
                if config
                else host_io.DEFAULTS["max_output_bytes"],
            )
            print(host_io.encode(result))
            return 2

    if args.command == "claim":
        try:
            result = claims.report(args.spec)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return feedback.exit_status(result)

    if args.command in {"suggest", "experiment"}:
        try:
            if args.command == "suggest":
                result = suggest.report(
                    python_files(args.source),
                    measurements=args.measurements,
                    limit=args.limit,
                    pypto_root=args.pypto_root,
                    goal=args.goal,
                    cards_root=args.cards_root,
                )
                feedback.print_result(result, details=args.details)
                return feedback.exit_status(result)
            result = experiment.run(
                args.state,
                init=args.init,
                selection=args.selection,
                observation=args.observation,
                export_best=args.export_best,
                pypto_root=args.pypto_root,
            )
        except (OSError, SyntaxError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return 0

    if args.command == "evaluate":
        try:
            if args.host_counters and not args.events:
                parser.error("--host-counters requires --events")
            if args.knowledge_cases and any(
                (args.events, args.prepare, args.observations, args.host_counters)
            ):
                parser.error(
                    "--knowledge-cases is separate from task/event/observation options"
                )
            if args.knowledge_cases:
                result = knowledge_evaluation.run(
                    args.knowledge_cases,
                    checker=args.checker,
                    split=args.split,
                    pypto_root=args.pypto_root,
                    examples_root=args.examples_root,
                )
            elif args.events:
                if (
                    args.prepare
                    or args.observations
                    or args.checker
                    or args.split != "all"
                ):
                    parser.error("--events is separate from task/comparison options")
                result = evaluation.event_metrics(
                    args.events, host_counters=args.host_counters
                )
            elif args.prepare:
                if args.observations or args.checker or args.split != "all":
                    parser.error(
                        "--prepare is separate from evaluation/comparison options"
                    )
                result = evaluation.prepare_agent_task(args.prepare)
            elif args.observations:
                if args.checker or args.split != "all":
                    parser.error("--observations is separate from tool-suite options")
                result = evaluation.compare_observations(args.observations)
            else:
                result = evaluation.run(
                    checker=args.checker, split=args.split, pypto_root=args.pypto_root
                )
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.events and result["status"] != "complete":
            return 2
        if args.knowledge_cases and result["status"] == "unverified":
            return 2
        return 1 if result.get("status") == "failed" else 0

    if args.command in {"query", "native"}:
        try:
            if args.command == "query":
                if args.validate:
                    if any(
                        value is not None
                        for value in (
                            args.text,
                            args.source,
                            args.task,
                            args.kinds,
                            args.filters,
                            args.limit,
                            args.cursor,
                            args.reference_sha256,
                            args.goal,
                            args.policy,
                        )
                    ):
                        raise ValueError(
                            "--validate, task inventory and knowledge discovery are separate operations"
                        )
                    result = knowledge_admission.report(
                        pypto_root=args.pypto_root,
                        examples_root=args.examples_root,
                        cards_root=args.cards_root,
                    )
                elif args.task:
                    if any(
                        value is not None
                        for value in (
                            args.text,
                            args.source,
                            args.kinds,
                            args.filters,
                            args.limit,
                            args.cursor,
                            args.reference_sha256,
                        )
                    ):
                        raise ValueError(
                            "--task inventory and knowledge queries are separate operations"
                        )
                    config = host_io.policy(
                        args.policy,
                        pypto_root=args.pypto_root,
                        examples_root=args.examples_root,
                    )
                    try:
                        result = task_context.report(
                            args.task, config=config, goal=args.goal
                        )
                    except (OSError, ValueError, TypeError) as exc:
                        result = host_io.failure(
                            "task_context",
                            exc,
                            limit=config["limits"]["max_output_bytes"],
                        )
                    print(host_io.encode(result))
                    return 2 if result["status"] == "unverified" else 0
                else:
                    if args.policy:
                        raise ValueError("--policy applies only to --task inventories")
                    result = query.report(
                        args.text,
                        source=args.source,
                        pypto_root=args.pypto_root,
                        examples_root=args.examples_root,
                        goal=args.goal,
                        kinds=args.kinds,
                        filters=args.filters,
                        limit=args.limit,
                        cursor=args.cursor,
                        reference_sha256=args.reference_sha256,
                        cards_root=args.cards_root,
                    )
            else:
                with contextlib.redirect_stdout(sys.stderr):
                    result = native.report(
                        args.spec,
                        pypto_root=args.pypto_root,
                        examples_root=args.examples_root,
                    )
        except (OSError, SyntaxError, ValueError) as exc:
            if args.command == "native":
                parser.error(str(exc))
            result = query.failure(exc, pypto_root=args.pypto_root)
        feedback.print_result(
            result, json_output=args.format == "json", details=args.details
        )
        return feedback.exit_status(result)

    if args.command == "explain":
        try:
            result = explain.report(
                args.source,
                baseline=args.baseline,
                error_log=args.error_log,
                function=args.function,
                pypto_root=args.pypto_root,
                examples_root=args.examples_root,
            )
        except (OSError, SyntaxError, ValueError) as exc:
            print(f"explain: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "api":
        if args.write_pyi or args.check_pyi:
            if args.namespaces:
                parser.error(
                    "--write-pyi / --check-pyi cover the whole public API; omit namespace filters"
                )
            return (
                0
                if api_stub.report(write=args.write_pyi, pypto_root=args.pypto_root)
                else 1
            )
        return (
            0
            if api_table.report(args.namespaces or None, pypto_root=args.pypto_root)
            else 1
        )

    if args.command == "surface":
        return 0 if report_surface(args.examples_root) else 1

    if args.command == "srcrefs":
        return 0 if report_srcrefs(pypto_root=args.pypto_root) else 1

    if args.command == "coverage":
        return 0 if report_coverage(args.examples_root) else 1

    if args.command == "catalog":
        return 0 if report_catalog(args.examples_root) else 1

    if args.command == "links":
        targets = args.paths or DOC_ROOTS
        if any(
            not path.exists() or (path.is_file() and path.suffix.lower() != ".md")
            for path in targets
        ):
            parser.error("links expects existing Markdown files or directories")
        return 0 if report_links(targets) else 1

    given = (
        args.sources if args.command == "run" else getattr(args, "paths", None)
    ) or []
    targets = given or [args.examples_root]
    missing = [path for path in targets if not path.exists()]
    if missing:
        parser.error(
            "路径不存在："
            + "、".join(str(path) for path in missing)
            + "；请传入 rules 的目标路径或用 --examples-root 指定样例目录"
        )

    if args.command == "run":
        command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        if not command:
            parser.error("run requires an explicit command after --")
        try:
            return run_command(
                targets,
                command,
                pypto_root=args.pypto_root,
                examples_root=args.examples_root,
                native_spec=args.native_spec,
                json_output=args.format == "json",
                details=args.details,
                feedback_file=args.feedback_file,
            )
        except (OSError, SyntaxError, ValueError) as exc:
            parser.error(str(exc))

    if args.command == "rules" and args.format == "json":
        result = rules_result(
            targets, pypto_root=args.pypto_root, examples_root=args.examples_root
        )
        feedback.print_result(result, details=args.details)
        return feedback.exit_status(result)

    clean = report_rules(
        targets, pypto_root=args.pypto_root, examples_root=args.examples_root
    )
    if args.command is None:
        print()
        try:
            result = knowledge_admission.report(
                pypto_root=args.pypto_root,
                examples_root=args.examples_root,
                cards_root=args.cards_root,
            )
        except (OSError, SyntaxError, ValueError) as exc:
            result = query.failure(exc, pypto_root=args.pypto_root)
        feedback.print_result(result, json_output=False)
        return 1 if not clean else feedback.exit_status(result)
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
