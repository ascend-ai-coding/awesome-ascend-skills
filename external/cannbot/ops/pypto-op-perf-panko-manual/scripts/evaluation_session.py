# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""One device-owning worker for correctness and measurement across dispatches."""

import argparse
import builtins
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import sys
import traceback

import session_worker

CONFIG_FIELDS = (
    "op",
    "op_dir",
    "op_file",
    "device",
    "test_command",
    "lint_root",
    "lint_stage",
    "warmup",
    "runs",
    "agg",
    "eval_timeout_s",
    "evaluation_mode",
)


def close_session(op_dir):
    """Release only the worker identified by this operator's private session."""
    root = Path(op_dir).resolve() / "optimization" / ".evaluation"
    session_worker.close_worker(root)


def clear_terminal_marker(op_dir):
    """A newly initialized search must not inherit the previous run's terminal marker."""
    (Path(op_dir).resolve() / "optimization" / "controller-terminal.json").unlink(
        missing_ok=True
    )


def reset_session(op_dir):
    """Start fresh search evidence without deleting any earlier cache or profiler files."""
    close_session(op_dir)
    clear_terminal_marker(op_dir)
    root = Path(op_dir).resolve() / "optimization" / ".evaluation"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    session_worker.write_json(root / "epoch.json", dict(id=os.urandom(16).hex()))


def request(evaluator):
    """Evaluate fresh inputs through a reusable worker; the caller stays CPU-only."""
    _command(evaluator.test_command, evaluator.device)
    config = {key: getattr(evaluator, key) for key in CONFIG_FIELDS}
    config.update(cwd=os.getcwd(), worker_module="evaluation_session")
    for key in ("op_dir", "op_file", "lint_root"):
        if config[key]:
            config[key] = str(Path(config[key]).resolve())
    op_dir = Path(config["op_dir"])
    fingerprint = context_fingerprint(config)
    root = op_dir / "optimization" / ".evaluation"
    cached = evaluator.evaluation_mode in ("cached", "prepared")
    source = Path(config["op_file"]).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    target = (
        root
        / "results"
        / (hashlib.sha256((fingerprint + digest).encode()).hexdigest() + ".json")
    )
    if cached and target.exists():
        result = _cached_result(target, config)
        if result is not None:
            return result
    try:
        if cached:
            compile(source, config["op_file"], "exec")
        result = session_worker.dispatch(
            root, config, fingerprint, {}, evaluator.eval_timeout_s
        )
    except SyntaxError:
        result = {"test_rc": 1, "test_output": traceback.format_exc()}
    if cached:
        result["cache_hit"] = False
        if "metrics" in result:
            result["metrics"]["evaluation_cache_hit"] = False
        if source != Path(config["op_file"]).read_bytes():
            close_session(op_dir)
            return {
                "test_rc": 1,
                "test_output": "Candidate source changed during evaluation",
            }
        if fingerprint != context_fingerprint(config):
            close_session(op_dir)
            return {
                "test_rc": 1,
                "test_output": "Evaluation context changed during evaluation",
            }
        if reusable_result(result, config):
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            session_worker.write_json(target, result)
        if static_failure(result):
            close_session(op_dir)
    return result


def _cached_result(target, config):
    try:
        result = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict) or not reusable_result(result, config):
        return None
    result["cache_hit"] = True
    if "metrics" in result:
        result["metrics"]["evaluation_cache_hit"] = True
    return result


def context_fingerprint(config):
    """Invalidate evaluation evidence when its inputs or runtime change."""
    folders = [Path(__file__).parent, Path(config["op_dir"])]
    if config.get("lint_root"):
        folders.append(Path(config["lint_root"]))
    epoch = Path(config["op_dir"]) / "optimization" / ".evaluation" / "epoch.json"
    context = session_worker.DependencyContext(
        tuple(folders),
        epoch,
        config.get("op_file"),
        ignored_names=("tuning_notes.md",),
        ignored_dirs=("optimization", "output", "prof", "__pycache__"),
    )
    return session_worker.context_fingerprint(config, context)


def static_failure(result):
    """Retain exact-source syntax/capacity failures; lint faults remain retryable."""
    from bayesian_optimization.evaluator import ENV_FAULT_MARKERS

    text = "".join(
        result.get(key, "") for key in ("test_output", "lint_output", "measure_output")
    )
    if (
        result.get("lint_rc")
        or result.get("test_rc") == 124
        or any(marker in text for marker in ENV_FAULT_MARKERS)
    ):
        return False
    return "SyntaxError:" in text or any(
        marker in text
        for marker in ("TensorErr::TENSOR_MEMORY_ALLOCATION", "exceeds MEM_UB size")
    )


def reusable_result(result, config):
    """Keep original verdicts and complete profiler evidence, never transient faults."""
    if static_failure(result):
        return True
    metrics = result.get("metrics", {})
    return (
        result.get("test_rc") == 0
        and result.get("lint_rc") == 0
        and isinstance(metrics.get("p"), (int, float))
        and math.isfinite(metrics["p"])
        and metrics["p"] > 0
        and metrics.get("raw_count") == config["warmup"] + config["runs"]
    )


def _command(command, device):
    from bayesian_optimization.evaluator import command_steps

    steps, cwd = command_steps(command)
    for argv, env in steps:
        executable = shutil.which(argv[0], path=env.get("PATH"))
        if (
            not executable
            or Path(executable).resolve() != Path(sys.executable).resolve()
        ):
            raise ValueError(
                "Resident tests must use the evaluation Python interpreter"
            )
        if len(argv) < 2 or (argv[1].startswith("-") and argv[1] not in ("-m", "-c")):
            raise ValueError("Resident tests support Python scripts, -m, or -c")
        if env.get("TILE_FWK_DEVICE_ID", str(device)) != str(device):
            raise ValueError("Test command device differs from evaluation device")
    return steps, cwd


def _python_step():
    try:
        if sys.argv[0] == "-m":
            sys.argv = sys.argv[1:]
            runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
        elif sys.argv[0] == "-c":
            source = sys.argv[1]
            sys.argv = ["-c", *sys.argv[2:]]
            exec(compile(source, "<test-command>", "exec"), {"__name__": "__main__"})
        else:
            sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
            runpy.run_path(sys.argv[0], run_name="__main__")
    except builtins.SystemExit as error:
        if error.code is None or isinstance(error.code, int):
            return error.code or 0
        traceback.print_exception(error, limit=0)
        return 1
    return 0


def run_python_tests(command, device):
    """Run every Python step, retaining imported JIT objects and restoring process state."""
    steps, cwd = _command(command, device)
    previous_cwd, previous_argv = os.getcwd(), sys.argv
    previous_path, previous_env = sys.path[:], os.environ.copy()
    inherited_paths = {
        str(Path(entry or previous_cwd).resolve())
        for entry in previous_env.get("PYTHONPATH", "").split(os.pathsep)
    }
    base_path = [
        entry
        for entry in previous_path
        if entry and str(Path(entry).resolve()) not in inherited_paths
    ]
    try:
        if cwd:
            os.chdir(cwd)
        for argv, env in steps:
            os.environ.clear()
            os.environ.update(previous_env | env)
            command_paths = [
                str(Path(entry or os.getcwd()).resolve())
                for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            ]
            sys.path = [os.getcwd(), *command_paths, *base_path]
            sys.argv = argv[1:]
            code = _python_step()
            if code:
                return code
        return 0
    finally:
        os.chdir(previous_cwd)
        sys.argv, sys.path = previous_argv, previous_path
        os.environ.clear()
        os.environ.update(previous_env)


def evaluate_candidate(config, prepared=None):
    """Execute the original gates and measure with their implementation module."""
    import measure_latency

    op_dir = Path(config["op_dir"])
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename and op_dir in Path(filename).resolve().parents:
            del sys.modules[name]
    if str(op_dir) not in sys.path:
        sys.path.insert(0, str(op_dir))
    op_file = Path(config["op_file"])
    spec = importlib.util.spec_from_file_location(op_file.stem, op_file)
    implementation = importlib.util.module_from_spec(spec)
    sys.modules[op_file.stem] = implementation
    exec(compile(op_file.read_text(), str(op_file), "exec"), implementation.__dict__)
    result = {}

    def test():
        try:
            if prepared is None:
                result["test_rc"] = run_python_tests(
                    config["test_command"], config["device"]
                )
            else:
                with prepared.correctness(config):
                    result["test_rc"] = run_python_tests(
                        config["test_command"], config["device"]
                    )
        except Exception:
            traceback.print_exc()
            result["test_rc"] = 1

    result["test_output"] = measure_latency.capture_output(test, 1)
    if result["test_rc"]:
        return result
    from bayesian_optimization.evaluator import RealEvaluator

    # The worker invokes the original lint command, rather than consuming a client response.
    evaluator = RealEvaluator(
        **{
            k: v
            for k, v in config.items()
            if k in CONFIG_FIELDS and k != "evaluation_mode"
        }
    )
    lint_ok, result["lint_output"] = evaluator.lint_result()
    result["lint_rc"] = 0 if lint_ok else 1
    if not result["lint_rc"]:
        try:
            inputs = prepared.measurement(config) if prepared is not None else None
            run_once = measure_latency.build_run_once(
                config["op_dir"], config["op"], implementation, config["device"], inputs
            )
            result["metrics"] = measure_latency.measure(
                run_once,
                config["warmup"],
                config["runs"],
                config["agg"],
                op_dir=config["op_dir"],
            )
            if prepared is not None:
                result["metrics"]["preparation_reuse"] = {
                    "inputs": prepared.random.hits if prepared.random else 0,
                    "references": prepared.reference.hits if prepared.reference else 0,
                    "measurement_inputs": prepared.inputs.hits
                    if prepared.inputs
                    else 0,
                }
        except Exception:
            result["measure_output"] = traceback.format_exc()
    return result


def create_handler(config):
    """Prepare evaluator-local state once for the resident worker."""
    os.environ["TILE_FWK_DEVICE_ID"] = str(config["device"])
    prepared = None
    if config.get("evaluation_mode") == "prepared":
        prepared = importlib.import_module("prepared_inputs").PreparedInputs()
    return lambda payload: evaluate_candidate(config, prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close", action="store_true")
    parser.add_argument("--reset", action="store_true")
    for name in ("op-dir", "op", "op-file", "test-command", "lint-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--device", default=os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    parser.add_argument(
        "--mode",
        choices=("subprocess", "resident", "cached", "prepared"),
        default="resident",
    )
    parser.add_argument("--eval-timeout-s", type=float, default=300)
    args = parser.parse_args()
    if args.op_dir and (args.close or args.reset):
        reset_session(args.op_dir) if args.reset else close_session(args.op_dir)
    else:
        if not all((args.op_dir, args.op, args.op_file, args.test_command)):
            parser.error("evaluation requires op-dir, op, op-file, and test-command")
        from bayesian_optimization.evaluator import RealEvaluator

        evaluator = RealEvaluator(
            args.op,
            args.op_dir,
            args.op_file,
            args.device,
            args.test_command,
            lint_root=args.lint_root,
            evaluation_mode=args.mode,
            eval_timeout_s=args.eval_timeout_s,
        )
        s, metrics = evaluator.evaluate_current()
        json.dump(dict(s=s, metrics=metrics), sys.stdout)
        if not s:
            close_session(args.op_dir)


if __name__ == "__main__":
    main()
