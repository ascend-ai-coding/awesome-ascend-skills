#!/usr/bin/env python3
"""validate_params.py — Validate vllm bench serve parameter combinations.

Usage:
    python3 validate_params.py --backend openai-chat --endpoint /v1/chat/completions --dataset-name random [--random-input-len N] ...

Output: JSON {"valid": bool, "warnings": [...], "errors": [...]}
"""

import argparse
import json
import os
import sys

# Allow importing common.py from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    ALL_BACKENDS, ALL_DATASETS, BACKEND_ENDPOINT_MAP,
    DATASET_BACKEND_COMPAT, DATASETS_REQUIRING_PATH,
)


def validate(args):
    errors = []
    warnings = []

    # Check backend is valid
    if args.backend not in ALL_BACKENDS:
        errors.append(f"Unknown backend: '{args.backend}'. Valid: {ALL_BACKENDS}")

    # Check dataset is valid
    if args.dataset_name not in ALL_DATASETS:
        errors.append(f"Unknown dataset: '{args.dataset_name}'. Valid: {ALL_DATASETS}")

    # Check dataset-backend compatibility
    compat = DATASET_BACKEND_COMPAT.get(args.dataset_name)
    if compat is not None and args.backend not in compat:
        errors.append(
            f"Dataset '{args.dataset_name}' is not compatible with backend '{args.backend}'. "
            f"Compatible backends: {compat}"
        )

    # Check dataset-path requirement
    if args.dataset_name in DATASETS_REQUIRING_PATH and not args.dataset_path:
        if args.dataset_name == "sharegpt":
            warnings.append(
                "ShareGPT dataset without --dataset-path: will attempt auto-download from HuggingFace."
            )
        else:
            errors.append(
                f"Dataset '{args.dataset_name}' requires --dataset-path."
            )

    # Check goodput format
    if args.goodput:
        valid_metrics = {"ttft", "tpot", "e2el"}
        for g in args.goodput:
            parts = g.split(":")
            if len(parts) != 2:
                errors.append(f"Invalid --goodput format: '{g}'. Expected 'METRIC:VALUE_MS'.")
            else:
                metric, value = parts
                if metric not in valid_metrics:
                    errors.append(
                        f"Invalid goodput metric: '{metric}'. Valid: {valid_metrics}"
                    )
                try:
                    float(value)
                except ValueError:
                    errors.append(f"Invalid goodput value: '{value}'. Must be a number (ms).")

    # Warn on low num_prompts
    if args.num_prompts and args.num_prompts < 100:
        warnings.append(
            f"--num-prompts={args.num_prompts} is low for P99 statistics. "
            f"Consider using at least 100 for meaningful percentile metrics."
        )

    # Check ramp-up consistency
    if args.ramp_up_strategy and args.request_rate:
        errors.append(
            "--ramp-up-strategy and --request-rate are mutually exclusive."
        )

    # Check --endpoint matches backend
    expected_endpoint = BACKEND_ENDPOINT_MAP.get(args.backend)
    if expected_endpoint and hasattr(args, "endpoint") and args.endpoint:
        if args.endpoint != expected_endpoint:
            errors.append(
                f"--endpoint '{args.endpoint}' does not match backend '{args.backend}'. "
                f"Expected: '{expected_endpoint}'"
            )
    elif expected_endpoint and expected_endpoint != "/v1/completions":
        # Backend needs non-default endpoint but none specified
        if not (hasattr(args, "endpoint") and args.endpoint):
            warnings.append(
                f"Backend '{args.backend}' requires --endpoint {expected_endpoint}. "
                f"The default /v1/completions will cause a URL validation error."
            )

    # Sonnet deprecation
    if args.dataset_name == "sonnet":
        warnings.append("Sonnet dataset is deprecated. Consider using 'random' instead.")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate vllm bench serve parameters")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--endpoint", default=None, help="API endpoint")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--input-len", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--request-rate", type=float, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--goodput", nargs="*", default=None)
    parser.add_argument("--ramp-up-strategy", default=None)

    args = parser.parse_args()
    errors, warnings = validate(args)

    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
