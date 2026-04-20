#!/usr/bin/env python3
import argparse
import ast
import json
import math
import pathlib
import re
from typing import Any


LIST_RE = re.compile(r"^\s*\[.*\]\s*$")
MEAN_DIFF_RE = re.compile(r"mean abs diff:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)


def maybe_parse_list(line: str) -> list[Any] | None:
    if not LIST_RE.match(line):
        return None
    try:
        value = ast.literal_eval(line.strip())
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, list) else None


def parse_result_log(path: pathlib.Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.rstrip("\n") for line in text.splitlines()]

    parsed_lists: list[list[Any]] = []
    mean_abs_diff = None
    saw_error_hint = False

    for line in lines:
        maybe_list = maybe_parse_list(line)
        if maybe_list is not None:
            parsed_lists.append(maybe_list)
        match = MEAN_DIFF_RE.search(line)
        if match:
            mean_abs_diff = float(match.group(1))
        if "error, please check log" in line.lower():
            saw_error_hint = True

    latest_rewards = parsed_lists[-2] if len(parsed_lists) >= 2 else None
    baseline_rewards = parsed_lists[-1] if len(parsed_lists) >= 1 else None
    same_length = (
        latest_rewards is not None
        and baseline_rewards is not None
        and len(latest_rewards) == len(baseline_rewards)
    )

    status = "unknown"
    reason = "Could not find a decisive comparison result."
    if mean_abs_diff is not None:
        if math.isclose(mean_abs_diff, 0.0, abs_tol=0.0):
            status = "pass"
            reason = "mean abs diff is exactly zero."
        else:
            status = "accuracy_regression"
            reason = "mean abs diff is non-zero."
    elif saw_error_hint:
        status = "train_error"
        reason = "Comparison log reported an error and asked to inspect the training log."
    elif latest_rewards is not None and baseline_rewards is not None and not same_length:
        status = "train_error"
        reason = "Found two reward lists with different lengths."

    return {
        "path": str(path),
        "status": status,
        "reason": reason,
        "mean_abs_diff": mean_abs_diff,
        "latest_rewards": latest_rewards,
        "baseline_rewards": baseline_rewards,
        "same_length": same_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse the msverl comparison result log.")
    parser.add_argument(
        "path",
        nargs="?",
        default="/home/st_daily_verl/msverl.log",
        help="Path to the comparison log.",
    )
    args = parser.parse_args()

    result = parse_result_log(pathlib.Path(args.path))
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
