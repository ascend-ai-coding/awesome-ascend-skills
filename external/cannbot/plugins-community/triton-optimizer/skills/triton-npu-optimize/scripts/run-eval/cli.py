# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterator, Literal, Protocol, TextIO, cast

from result_payload import ResultPayload

SCRIPT_DIR = Path(__file__).resolve().parent
_RUN_BENCH_HINT = "Hint: use `compare-perf` to inspect this perf artifact instead of reading it directly."
_BLOCKS_PARALLEL_ENV = "TRITON_ALL_BLOCKS_PARALLEL"
_BLOCKS_PARALLEL_UNSAFE_VALUE = "1"
_BLOCKS_PARALLEL_SAFE_VALUE = "0"
_OPTIMIZE_DELETE_PT_FILES_ENV = "TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES"
_PT_CLEANUP_MODES = frozenset({"never", "round", "run-test"})
_LEGACY_ROUND_CLEANUP_VALUES = frozenset({"1", "true", "yes", "on"})
_LEGACY_NEVER_CLEANUP_VALUES = frozenset({"0", "false", "no", "off"})
PtCleanupMode = Literal["never", "round", "run-test"]


def _profile_bench_hint(profile_dir: Path) -> str:
    return (
        "Hint: rerun the bundled `profile-report` helper for this "
        f"`--profile-dir {profile_dir}` if you need the summary again; "
        "if that is not enough, inspect the raw files in this profile directory directly."
    )


def _pt_cleanup_mode() -> PtCleanupMode:
    raw_value = os.environ.get(_OPTIMIZE_DELETE_PT_FILES_ENV)
    if raw_value is None:
        return "round"
    value = raw_value.strip().lower()
    if value in _PT_CLEANUP_MODES:
        return cast(PtCleanupMode, value)
    if value in _LEGACY_ROUND_CLEANUP_VALUES:
        return "round"
    if value in _LEGACY_NEVER_CLEANUP_VALUES:
        return "never"
    return "round"


def _is_ordinary_pt_result_file(path: Path) -> bool:
    name_lower = path.name.lower()
    return name_lower == "test_result.pt" or name_lower.endswith("_result.pt")


def _cleanup_pt_file(pt_file: Path) -> str | None:
    if not pt_file.is_file() or not _is_ordinary_pt_result_file(pt_file):
        return None
    try:
        pt_file.unlink()
        return pt_file.name
    except OSError:
        return None


def _cleanup_run_test_pt_files(paths: tuple[Path | None, ...]) -> list[str]:
    if _pt_cleanup_mode() != "run-test":
        return []
    cleaned: list[str] = []
    seen: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        cleaned_name = _cleanup_pt_file(resolved_path)
        if cleaned_name is not None:
            cleaned.append(str(resolved_path))
    return cleaned


@contextlib.contextmanager
def _guard_operator_execution_env(command: str) -> Iterator[None]:
    if command not in {
        "run-test-baseline",
        "run-test-optimize",
        "run-bench",
        "profile-bench",
    }:
        yield
        return
    previous = os.environ.get(_BLOCKS_PARALLEL_ENV)
    if previous != _BLOCKS_PARALLEL_UNSAFE_VALUE:
        yield
        return
    os.environ[_BLOCKS_PARALLEL_ENV] = _BLOCKS_PARALLEL_SAFE_VALUE
    try:
        yield
    finally:
        os.environ[_BLOCKS_PARALLEL_ENV] = previous


class ParseMetadataFn(Protocol):
    def __call__(self, path: Path) -> dict[str, str]:
        ...


class ResolveRemoteExecutionFn(Protocol):
    def __call__(
        self,
        explicit_remote: str | None,
        explicit_remote_workdir: str | None,
    ) -> tuple[str | None, str | None]:
        ...


class RunLocalTestFn(Protocol):
    def __call__(
        self,
        test_file: Path,
        operator_file: Path,
        test_mode: str,
        *,
        verbose: bool = False,
    ) -> tuple[ResultPayload, Path | None]:
        ...


class RunRemoteTestFn(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> tuple[ResultPayload, Path | None, str]:
        ...


class RunLocalBenchFn(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> tuple[ResultPayload, Path | None]:
        ...


class RunRemoteBenchFn(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> tuple[ResultPayload, Path | None, str]:
        ...


class CompareResultFn(Protocol):
    def __call__(self, ref_result: Path, new_result: Path) -> int:
        ...


class CompareRemoteResultFn(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> int:
        ...


class ComparePerfFn(Protocol):
    def __call__(
        self,
        baseline_perf: Path,
        compare_perf: Path,
        *,
        skip_latency_errors: bool = False,
        metric_source: str = "auto",
    ) -> int:
        ...


class RunLocalProfileBenchFn(Protocol):
    def __call__(
        self,
        bench_file: Path,
        operator_file: Path,
        case_id: str | None = None,
        kernel_name: str | None = None,
    ) -> tuple[ResultPayload, Path | None]:
        ...


class RunRemoteProfileBenchFn(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> tuple[ResultPayload, Path | None, str]:
        ...


class ProfileSummaryModule(Protocol):
    def build_report(
        self,
        profile_path: str | Path,
        target_op: str | None = None,
        top_count: int = 5,
        output_format: str = "markdown",
    ) -> str:
        ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_test_baseline = subparsers.add_parser("run-test-baseline")
    _add_run_test_arguments(run_test_baseline)

    run_test_optimize = subparsers.add_parser("run-test-optimize")
    _add_run_test_arguments(run_test_optimize)

    run_bench = subparsers.add_parser("run-bench")
    run_bench.add_argument("--bench-file", required=True)
    run_bench.add_argument("--operator-file", required=True)
    run_bench.add_argument("--baseline-operator-file")
    run_bench.add_argument("--skip-latency-errors", "--skip-error", dest="skip_latency_errors", action="store_true")
    run_bench.add_argument(
        "--metric-source",
        default="auto",
        choices=["auto", "kernel", "total-op", "all"],
    )
    run_bench.add_argument("--output")
    run_bench.add_argument("--remote")
    run_bench.add_argument("--remote-workdir")
    run_bench.add_argument("--keep-remote-workdir", action="store_true")
    run_bench.add_argument("--verbose", action="store_true")
    run_bench.add_argument("--bench-mode", choices=["torch-npu-profiler", "perf-counter"])
    run_bench.add_argument("--npu-devices")

    profile_bench = subparsers.add_parser("profile-bench")
    profile_bench.add_argument("--bench-file", required=True)
    profile_bench.add_argument("--operator-file", required=True)
    profile_bench.add_argument("--case-id")
    profile_bench.add_argument("--kernel-name", help=argparse.SUPPRESS)
    profile_bench.add_argument("--target-op")
    profile_bench.add_argument("--remote")
    profile_bench.add_argument("--remote-workdir")
    profile_bench.add_argument("--keep-remote-workdir", action="store_true")
    profile_bench.add_argument("--verbose", action="store_true")

    profile_report = subparsers.add_parser("profile-report")
    profile_report.add_argument("--profile-dir", required=True)
    profile_report.add_argument("--target-op")
    profile_report.add_argument("--format", choices=["markdown", "json"], default="markdown")
    profile_report.add_argument("--top", type=int, default=5)

    compare_perf = subparsers.add_parser("compare-perf")
    compare_perf.add_argument("--baseline", required=True)
    compare_perf.add_argument("--compare", required=True)
    compare_perf.add_argument("--skip-latency-errors", "--skip-error", dest="skip_latency_errors", action="store_true")
    compare_perf.add_argument(
        "--metric-source",
        default="auto",
        choices=["auto", "kernel", "total-op", "all"],
    )

    return parser


def _add_run_test_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--operator-file", required=True)
    parser.add_argument("--ref-result", "--baseline-result", dest="ref_result")
    parser.add_argument("--ref-operator-file", "--baseline-operator-file", dest="ref_operator_file")
    parser.add_argument("--remote")
    parser.add_argument("--remote-workdir")
    parser.add_argument("--keep-remote-workdir", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--test-mode", choices=["standalone", "differential"])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    with _guard_operator_execution_env(args.command):
        return _dispatch_command(parser, args)


def _dispatch_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    remote, remote_workdir = _resolve_remote_execution(args)
    if args.command == "compare-perf":
        return _dispatch_compare_perf(parser, args)
    if args.command in {"run-test-baseline", "run-test-optimize"}:
        return _dispatch_run_test(parser, args, remote, remote_workdir)
    if args.command == "profile-bench":
        return _dispatch_profile_bench(parser, args, remote, remote_workdir)
    if args.command == "profile-report":
        return _dispatch_profile_report(parser, args)
    return _dispatch_run_bench(parser, args, remote, remote_workdir)


def _dispatch_compare_perf(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    compare_perf_files = _load_compare_perf_function()
    baseline_perf = _resolve_existing_path(parser, args.baseline, "Baseline perf")
    compare_perf = _resolve_existing_path(parser, args.compare, "Compare perf")
    return compare_perf_files(
        baseline_perf,
        compare_perf,
        skip_latency_errors=args.skip_latency_errors,
        metric_source=args.metric_source,
    )


def _dispatch_run_test(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    remote: str | None,
    remote_workdir: str | None,
) -> int:
    _parse_test_metadata, run_local_test, run_remote_test = _load_test_functions()
    test_file = _resolve_existing_path(parser, args.test_file, "Test file")
    operator_file = _resolve_existing_path(parser, args.operator_file, "Operator file")
    ref_result = _resolve_optional_existing_path(parser, getattr(args, "ref_result", None), "Reference result")
    ref_operator_file = _resolve_optional_existing_path(
        parser, getattr(args, "ref_operator_file", None), "Reference operator file"
    )
    resolved_test_mode = args.test_mode or _resolve_test_mode_from_metadata(test_file)
    try:
        ref_result = _resolve_run_test_comparison_inputs(
            parser, args, resolved_test_mode, ref_result, ref_operator_file, test_file,
            run_local_test, run_remote_test, remote, remote_workdir,
            optimize_mode=args.command == "run-test-optimize",
        )
        result, archived_result, remote_workspace = _run_test_once(
            args, test_file, operator_file, resolved_test_mode, run_local_test, run_remote_test,
            remote, remote_workdir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _render_result(result, skip_stdout=remote is not None)
    print(f"Return code: {result['return_code']}")
    final_code = _finish_run_test(args, result, archived_result, ref_result, remote, remote_workdir)
    if remote is not None and args.keep_remote_workdir:
        print(f"Remote workspace: {remote_workspace}")
    return final_code


def _run_test_once(*call_args: object, **_call_kwargs: object) -> tuple[ResultPayload, Path | None, str | None]:
    args, test_file, operator_file, test_mode, run_local_test, run_remote_test, remote, remote_workdir = call_args
    args = cast(argparse.Namespace, args)
    if remote is not None:
        result, archived_result, workspace = run_remote_test(
            test_file, operator_file, test_mode, remote, remote_workdir,
            keep_remote_workdir=args.keep_remote_workdir, verbose=args.verbose, stderr=sys.stderr,
        )
        return result, archived_result, workspace
    result, archived_result = run_local_test(test_file, operator_file, test_mode, verbose=args.verbose)
    return result, archived_result, None


def _finish_run_test(*call_args: object, **_call_kwargs: object) -> int:
    args, result, archived_result, ref_result, remote, remote_workdir = call_args
    args = cast(argparse.Namespace, args)
    final_code = int(result["return_code"])
    if archived_result is None:
        if ref_result is not None:
            print(
                "Differential run-test did not produce an archived result required for automatic comparison.",
                file=sys.stderr,
            )
            return 1
        return final_code
    print(f"Archived result: {archived_result}")
    if ref_result is not None:
        final_code = _compare_run_test_result(
            ref_result, archived_result, remote, remote_workdir, verbose=bool(args.verbose), stderr=sys.stderr
        )
    if args.command == "run-test-optimize":
        _cleanup_run_test_pt_files((archived_result,))
    return final_code


def _dispatch_profile_bench(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    remote: str | None,
    remote_workdir: str | None,
) -> int:
    run_local, run_remote = _load_profile_functions()
    bench_file = _resolve_existing_path(parser, args.bench_file, "Bench file")
    operator_file = _resolve_existing_path(parser, args.operator_file, "Operator file")
    try:
        result, profile_dir, remote_workspace = _run_profile_bench_once(
            args, bench_file, operator_file, run_local, run_remote, remote, remote_workdir
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _render_result(result, skip_stdout=remote is not None and args.verbose)
    print(f"Return code: {result['return_code']}")
    if profile_dir is not None:
        print(f"Profile directory: {profile_dir}")
        print(_build_profile_report(profile_dir, args.target_op))
        print(_profile_bench_hint(profile_dir))
    if remote is not None and args.keep_remote_workdir:
        print(f"Remote workspace: {remote_workspace}")
    return int(result["return_code"])


def _run_profile_bench_once(
    *call_args: object,
    **_call_kwargs: object,
) -> tuple[ResultPayload, Path | None, str | None]:
    args, bench_file, operator_file, run_local, run_remote, remote, remote_workdir = call_args
    args = cast(argparse.Namespace, args)
    if remote is not None:
        result, profile_dir, workspace = run_remote(
            bench_file, operator_file, remote, remote_workdir, case_id=args.case_id, kernel_name=args.kernel_name,
            keep_remote_workdir=args.keep_remote_workdir, verbose=args.verbose, stderr=sys.stderr,
        )
        return result, profile_dir, workspace
    result, profile_dir = run_local(bench_file, operator_file, case_id=args.case_id, kernel_name=args.kernel_name)
    return result, profile_dir, None


def _dispatch_profile_report(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    report = _build_profile_report(
        _resolve_existing_path(parser, args.profile_dir, "Profile directory"), args.target_op,
        top_count=args.top, output_format=args.format,
    )
    print(report)
    return 0


def _dispatch_run_bench(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    remote: str | None,
    remote_workdir: str | None,
) -> int:
    bench_file, operator_file, baseline_operator_file = _resolve_bench_paths(parser, args)
    _metadata, run_local_bench, run_remote_bench = _load_bench_functions()
    baseline_perf_path = _prepare_baseline_bench(
        args,
        bench_file,
        baseline_operator_file,
        run_local_bench,
        run_remote_bench,
        remote,
        remote_workdir,
    )
    if baseline_perf_path is _BASELINE_BENCH_FAILED:
        return 1
    return _run_and_report_bench(
        args,
        bench_file,
        operator_file,
        baseline_perf_path,
        run_local_bench,
        run_remote_bench,
        remote,
        remote_workdir,
    )


_BASELINE_BENCH_FAILED = object()


def _resolve_bench_paths(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None]:
    bench_file = _resolve_existing_path(parser, args.bench_file, "Bench file")
    operator_file = _resolve_existing_path(parser, args.operator_file, "Operator file")
    baseline_operator_file = _resolve_optional_existing_path(
        parser,
        getattr(args, "baseline_operator_file", None),
        "Baseline operator file",
    )
    return bench_file, operator_file, baseline_operator_file


def _prepare_baseline_bench(*call_args: object, **_call_kwargs: object) -> Path | None | object:
    args, bench_file, baseline_operator_file, run_local_bench, run_remote_bench, remote, remote_workdir = call_args
    args = cast(argparse.Namespace, args)
    resolved_bench_mode = args.bench_mode or "torch-npu-profiler"
    baseline_perf_path = (
        _derived_perf_path(baseline_operator_file) if baseline_operator_file is not None else None
    )
    if baseline_perf_path is not None and baseline_perf_path.exists():
        print(f"Baseline perf file: {baseline_perf_path}")
    try:
        if (
            baseline_operator_file is not None
            and baseline_perf_path is not None
            and not baseline_perf_path.exists()
        ):
            baseline_result, generated_perf, workspace = _run_bench_once(
                run_local_bench,
                run_remote_bench,
                bench_file,
                baseline_operator_file,
                resolved_bench_mode,
                remote,
                remote_workdir,
                npu_devices=args.npu_devices,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
                output=None,
            )
            if int(baseline_result["return_code"]) != 0:
                _render_result(baseline_result, skip_stdout=remote is not None and args.verbose)
            if generated_perf is not None:
                baseline_perf_path = generated_perf
                print(f"Baseline perf file: {baseline_perf_path}")
            if int(baseline_result["return_code"]) != 0 or generated_perf is None:
                _print_remote_workspace(args, remote, workspace)
                return _BASELINE_BENCH_FAILED
            _print_remote_workspace(args, remote, workspace)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return _BASELINE_BENCH_FAILED
    return baseline_perf_path


def _run_and_report_bench(
    *call_args: object,
    **_call_kwargs: object,
) -> int:
    (
        args, bench_file, operator_file, baseline_perf_path, run_local_bench,
        run_remote_bench, remote, remote_workdir,
    ) = call_args
    args = cast(argparse.Namespace, args)
    resolved_bench_mode = args.bench_mode or "torch-npu-profiler"
    try:
        result, perf_path, workspace = _run_bench_once(
            run_local_bench,
            run_remote_bench,
            bench_file,
            operator_file,
            resolved_bench_mode,
            remote,
            remote_workdir,
            npu_devices=args.npu_devices,
            keep_remote_workdir=args.keep_remote_workdir,
            verbose=args.verbose,
            stderr=sys.stderr,
            output=args.output,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if result["return_code"] != 0:
        _render_result(result, skip_stdout=remote is not None and args.verbose)
    _print_remote_workspace(args, remote, workspace)
    if perf_path is not None:
        print(f"Perf file: {perf_path}")
        if baseline_perf_path is None:
            print(_RUN_BENCH_HINT)
        else:
            compare_perf_files = _load_compare_perf_function()
            return compare_perf_files(
                baseline_perf_path,
                perf_path,
                skip_latency_errors=args.skip_latency_errors,
                metric_source=args.metric_source,
            )
    return int(result["return_code"])


def _print_remote_workspace(
    args: argparse.Namespace,
    remote: str | None,
    workspace: str | None,
) -> None:
    if remote is not None and args.keep_remote_workdir and workspace is not None:
        print(f"Remote workspace: {workspace}")


def _resolve_existing_path(
    parser: argparse.ArgumentParser,
    raw_path: str,
    label: str,
) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        parser.error(f"{label} path does not exist: {path}")
    return path


def _resolve_optional_existing_path(
    parser: argparse.ArgumentParser,
    raw_path: str | None,
    label: str,
) -> Path | None:
    if raw_path is None:
        return None
    return _resolve_existing_path(parser, raw_path, label)


def _derived_result_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_result.pt"


def _derived_perf_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_perf.txt"


def _resolve_test_mode_from_metadata(test_file: Path) -> str:
    parse_test_metadata = _load_test_functions()[0]
    metadata = parse_test_metadata(test_file)
    mode = metadata.get("test-mode")
    if mode not in {"standalone", "differential"}:
        raise ValueError(f"Test metadata is missing required 'test-mode' entry: {test_file}")
    return mode


def _resolve_run_test_comparison_inputs(
    *call_args: object,
    **call_kwargs: object,
) -> Path | None:
    (
        parser, args, resolved_test_mode, ref_result, ref_operator_file, test_file,
        run_local_test, run_remote_test, remote, remote_workdir,
    ) = call_args
    args = cast(argparse.Namespace, args)
    optimize_mode = bool(call_kwargs["optimize_mode"])
    _validate_run_test_comparison_inputs(
        parser,
        resolved_test_mode,
        ref_result,
        ref_operator_file,
        optimize_mode=optimize_mode,
    )
    if ref_operator_file is None:
        return ref_result

    return _resolve_ref_operator_result(
        test_file,
        ref_operator_file,
        resolved_test_mode,
        run_local_test,
        run_remote_test,
        remote,
        remote_workdir,
        keep_remote_workdir=bool(args.keep_remote_workdir),
        verbose=bool(args.verbose),
    )


def _validate_run_test_comparison_inputs(
    parser: argparse.ArgumentParser,
    resolved_test_mode: str,
    ref_result: Path | None,
    ref_operator_file: Path | None,
    *,
    optimize_mode: bool,
) -> None:
    if optimize_mode:
        if ref_result is not None and ref_operator_file is not None:
            parser.error(
                "run-test-optimize differential mode requires exactly one of "
                "--ref-result or --ref-operator-file"
            )
        if resolved_test_mode == "differential" and ref_result is None and ref_operator_file is None:
            parser.error(
                "run-test-optimize differential mode requires exactly one of "
                "--ref-result or --ref-operator-file"
            )
    elif ref_result is not None and ref_operator_file is not None:
        parser.error("run-test-baseline differential mode accepts at most one of --ref-result or --ref-operator-file")

    if ref_result is not None and resolved_test_mode != "differential":
        parser.error("--ref-result is supported only with --test-mode differential")
    if ref_operator_file is not None and resolved_test_mode != "differential":
        parser.error("--ref-operator-file is supported only with --test-mode differential")


def _resolve_ref_operator_result(
    *call_args: object,
    **call_kwargs: object,
) -> Path:
    (
        test_file, ref_operator_file, resolved_test_mode, run_local_test,
        run_remote_test, remote, remote_workdir,
    ) = call_args
    keep_remote_workdir = bool(call_kwargs["keep_remote_workdir"])
    verbose = bool(call_kwargs["verbose"])
    derived_ref_result = _derived_result_path(ref_operator_file)
    if derived_ref_result.exists():
        return derived_ref_result

    ref_mode = resolved_test_mode
    if remote is not None:
        try:
            ref_run_result, archived_result, remote_workspace = run_remote_test(
                test_file,
                ref_operator_file,
                ref_mode,
                remote,
                remote_workdir,
                keep_remote_workdir=keep_remote_workdir,
                verbose=verbose,
                stderr=sys.stderr,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc
        _render_ref_run_result(
            ref_run_result,
            archived_result,
            remote_workspace=remote_workspace if keep_remote_workdir else None,
            skip_stdout=True,
        )
        _raise_if_ref_run_failed(ref_run_result, archived_result)
        return derived_ref_result

    try:
        ref_run_result, archived_result = run_local_test(
            test_file,
            ref_operator_file,
            ref_mode,
            verbose=verbose,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    _render_ref_run_result(ref_run_result, archived_result, remote_workspace=None, skip_stdout=False)
    _raise_if_ref_run_failed(ref_run_result, archived_result)
    return derived_ref_result


def _render_ref_run_result(
    ref_run_result: ResultPayload,
    archived_result: Path | None,
    *,
    remote_workspace: str | None,
    skip_stdout: bool = False,
) -> None:
    _render_result(ref_run_result, skip_stdout=skip_stdout)
    print(f"Return code: {ref_run_result['return_code']}")
    if archived_result is not None:
        print(f"Archived result: {archived_result}")
    if remote_workspace is not None:
        print(f"Remote workspace: {remote_workspace}")


def _raise_if_ref_run_failed(
    ref_run_result: ResultPayload,
    archived_result: Path | None,
) -> None:
    if int(ref_run_result["return_code"]) != 0 or archived_result is None:
        raise RuntimeError("reference operator run did not produce a successful archived result")


def _render_result(result: ResultPayload, skip_stdout: bool) -> None:
    stdout = result["stdout"]
    stderr = result["stderr"]
    if stdout and not skip_stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")


def _compare_run_test_result(*call_args: object, **call_kwargs: object) -> int:
    ref_result, archived_result, remote, remote_workdir = call_args
    verbose = bool(call_kwargs["verbose"])
    stderr = cast(TextIO | None, call_kwargs["stderr"])
    compare_result_files, compare_remote_result_files = _load_compare_result_functions()
    if remote is None:
        return compare_result_files(ref_result, archived_result)
    try:
        return compare_remote_result_files(
            ref_result,
            archived_result,
            remote,
            remote_workdir,
            verbose=verbose,
            stderr=stderr,
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=stderr or sys.stderr)
        return 1


def _resolve_remote_execution(args: argparse.Namespace) -> tuple[str | None, str | None]:
    resolve_remote_execution = _load_remote_execution_function()
    return resolve_remote_execution(
        getattr(args, "remote", None),
        getattr(args, "remote_workdir", None),
    )


def _load_test_functions() -> tuple[ParseMetadataFn, RunLocalTestFn, RunRemoteTestFn]:
    with _script_dir_on_path():
        module = importlib.import_module("test_runner")

    return (
        cast(ParseMetadataFn, getattr(module, "parse_test_metadata")),
        cast(RunLocalTestFn, getattr(module, "run_local_test")),
        cast(RunRemoteTestFn, getattr(module, "run_remote_test")),
    )


def _load_bench_functions() -> tuple[ParseMetadataFn, RunLocalBenchFn, RunRemoteBenchFn]:
    with _script_dir_on_path():
        from bench_contract import parse_bench_metadata
        from bench_runner import run_local_bench, run_remote_bench

    return (
        cast(ParseMetadataFn, parse_bench_metadata),
        cast(RunLocalBenchFn, run_local_bench),
        cast(RunRemoteBenchFn, run_remote_bench),
    )


def _load_compare_result_functions() -> tuple[CompareResultFn, CompareRemoteResultFn]:
    with _script_dir_on_path():
        module = importlib.import_module("compare_result")

    return (
        cast(CompareResultFn, getattr(module, "compare_result_files")),
        cast(CompareRemoteResultFn, getattr(module, "compare_remote_result_files")),
    )


def _load_compare_perf_function() -> ComparePerfFn:
    with _script_dir_on_path():
        from perf_artifacts import compare_perf_files

    return cast(ComparePerfFn, compare_perf_files)


def _run_bench_once(
    *call_args: object,
    **call_kwargs: object,
) -> tuple[ResultPayload, Path | None, str | None]:
    (
        run_local_bench, run_remote_bench, bench_file, operator_file,
        resolved_bench_mode, remote, remote_workdir,
    ) = call_args
    npu_devices = call_kwargs["npu_devices"]
    keep_remote_workdir = call_kwargs["keep_remote_workdir"]
    verbose = call_kwargs["verbose"]
    stderr = call_kwargs["stderr"]
    output = call_kwargs["output"]
    if remote is not None:
        result, perf_path, remote_workspace = run_remote_bench(
            bench_file,
            operator_file,
            resolved_bench_mode,
            remote,
            remote_workdir,
            npu_devices,
            keep_remote_workdir=keep_remote_workdir,
            verbose=verbose,
            stderr=stderr,
            output=output,
        )
        return result, perf_path, remote_workspace
    result, perf_path = run_local_bench(
        bench_file,
        operator_file,
        resolved_bench_mode,
        npu_devices,
        output=output,
    )
    return result, perf_path, None


def _load_remote_execution_function() -> ResolveRemoteExecutionFn:
    with _script_dir_on_path():
        from remote_execution_env import resolve_remote_execution

    return cast(ResolveRemoteExecutionFn, resolve_remote_execution)


def _load_profile_functions() -> tuple[RunLocalProfileBenchFn, RunRemoteProfileBenchFn]:
    with _script_dir_on_path():
        module = importlib.import_module("profile_runner")

    return (
        cast(RunLocalProfileBenchFn, getattr(module, "run_local_profile_bench")),
        cast(RunRemoteProfileBenchFn, getattr(module, "run_remote_profile_bench")),
    )


def _build_profile_report(
    profile_dir: Path,
    target_op: str | None = None,
    top_count: int = 5,
    output_format: str = "markdown",
) -> str:
    script = SCRIPT_DIR.parents[1] / "triton-npu-profile-operator" / "scripts" / "reporter.py"
    with _temporary_sys_path_entry(str(script.parent)):
        spec = importlib.util.spec_from_file_location("profile_reporter_runtime", script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load profile reporter script: {script}")
        loaded_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded_module)
    module = cast(ProfileSummaryModule, loaded_module)
    return module.build_report(
        profile_dir,
        target_op=target_op,
        top_count=top_count,
        output_format=output_format,
    )


@contextlib.contextmanager
def _temporary_sys_path_entry(path: str) -> Iterator[None]:
    added = False
    if path not in sys.path:
        sys.path.insert(0, path)
        added = True
    try:
        yield
    finally:
        if added:
            sys.path.remove(path)


@contextlib.contextmanager
def _script_dir_on_path() -> Iterator[None]:
    with _temporary_sys_path_entry(str(SCRIPT_DIR)):
        yield


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
