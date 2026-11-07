# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Kernel skill checks and a source-gated command runner.

Invoke from any directory with `python <skill>/scripts/check.py`. Defaults are resolved relative to the skill; explicit
input paths are resolved relative to the caller's current directory. Put path-override options before the subcommand.

`rules` runs static checks; `coverage` verifies API example assignments; `catalog` reconciles the matmul/algorithm
index; `links` checks local Markdown targets and anchors; `surface` searches API usage; `api` reads declaration stubs;
`srcrefs` reports suspicious source-line references. The default command runs rules, coverage and catalog checks. Those
checks do not import PyPTO or use a board. `run` checks the named source files, then executes the caller's explicit
command in the current environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from . import api as api_table
from . import api_stub
from . import catalog
from . import explain
from . import links
from . import name_resolution
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
from .source import Finding
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


def run_rules(targets: list[Path], *, pypto_root: Path = PYPTO_ROOT) -> list[Finding]:
    """Run `writing-style-declaration` only on `kernel.py` (the declaration is an example convention); run hardware,
    synchronization, and SIMT rules on every `.py` file (a user's kernel file may have any name).
    """
    findings: list[Finding] = []
    for path in python_files(targets):
        findings += name_resolution.check(path)
        if path.name == "kernel.py":
            findings += declaration.check(path)
        findings += hardware.check(path)
        findings += tile_shape.check(path)
        findings += valid_shape.check(path)
        findings += register_alias.check(path)
        findings += register_dtype.check(path)
        findings += register_init.check(path)
        findings += gm_scalar.check(path)
        findings += vf_scope.check(path, pypto_root=pypto_root)
        findings += vf_control_flow.check(path)
        findings += work_distribution.check(path)
        findings += api_policy.check(path, pypto_root=pypto_root)
        findings += gemm.check(path)
        findings += allocation_scope.check(path)
        findings += scalar_annotation.check(path)
        findings += synchronisation.check(path)
        findings += stage_handover.check(path)
        findings += init_output.check(path)
        findings += mutex_overlap.check(path)
        findings += simt.check(path)
    return findings


def report_rules(targets: list[Path], *, pypto_root: Path = PYPTO_ROOT) -> bool:
    findings = run_rules(targets, pypto_root=pypto_root)
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
    print(
        f"rules：{len(python_files(targets))} 个文件，{len(violations)} 处违规"
        f"（另有 {len(waived)} 处按 `反例：` 声明放行，{len(advisory)} 处需核查提示）"
    )
    return not violations


def source_hashes(targets: list[Path]) -> dict[Path, str]:
    return {
        path.resolve(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in python_files(targets)
    }


def _run_with_tail(command):
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
            (proc.stdout, proc.stderr), (sys.stdout, sys.stderr), tails
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


def run_command(targets: list[Path], command: list[str], *, pypto_root: Path) -> int:
    """Gate one explicit command; do not infer numerical success from its output."""
    before = source_hashes(targets)
    if not before:
        print("run: no Python sources to check", file=sys.stderr)
        return 2
    if not report_rules(targets, pypto_root=pypto_root):
        print("run: static checks failed; command was not started", file=sys.stderr)
        return 1
    if source_hashes(targets) != before:
        print(
            "run: sources changed during preflight; command was not started",
            file=sys.stderr,
        )
        return 1
    for path, digest in before.items():
        print(f"source_sha256 {digest}  {path}", flush=True)
    sys.stdout.flush()
    try:
        return_code, output_tail = _run_with_tail(command)
    except OSError as exc:
        print(f"run: could not start command: {exc}", file=sys.stderr)
        return 1
    status = return_code if return_code >= 0 else 128 - return_code
    print(f"command_exit={status}", flush=True)
    if status and explain._RANK.search(output_tail):
        for path in before:
            try:
                current = explain.inspect(path)
                if current["sha256"] != before[path]:
                    continue
                symptoms = explain.rank_diagnostics(current, output_tail)
                symptoms = [
                    s
                    for s in symptoms
                    if s["declarations"] or s["same_name_allocations"]
                ]
                if symptoms:
                    print(
                        "source_contract_diagnostic "
                        + json.dumps(
                            {
                                "source_sha256": current["sha256"],
                                "source": str(path),
                                "scope": "bounded command-output tail; name matches require launch verification",
                                "symptoms": symptoms,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            except (OSError, SyntaxError, ValueError):
                pass  # Source drift is separately reported below.
    try:
        changed = source_hashes(targets) != before
    except OSError:
        changed = True
    if changed:
        print(
            "run: sources changed during execution; result is not bound to the checked source",
            file=sys.stderr,
        )
        status = status or 1
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    args = parser.parse_args(argv)

    if args.command == "explain":
        try:
            result = explain.report(
                args.source,
                baseline=args.baseline,
                error_log=args.error_log,
                function=args.function,
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
        return run_command(targets, command, pypto_root=args.pypto_root)

    clean = report_rules(targets, pypto_root=args.pypto_root)
    if args.command is None:
        print()
        clean = report_coverage(args.examples_root) and clean
        clean = report_catalog(args.examples_root) and clean
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
