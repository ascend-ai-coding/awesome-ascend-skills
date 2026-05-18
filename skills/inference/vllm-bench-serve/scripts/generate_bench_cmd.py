#!/usr/bin/env python3
"""generate_bench_cmd.py — Generate vllm bench serve commands with archival flags.

Usage:
    # From CLI args:
    python3 generate_bench_cmd.py --base-url http://ip:port --model /path/to/weights \
        --served-model-name MODEL_NAME --backend openai-chat \
        --dataset-name random --random-input-len 1024 --random-output-len 128 \
        --max-concurrency 8 --num-prompts 500

    # From config JSON:
    python3 generate_bench_cmd.py --config config.json

Output: Complete vllm bench serve command string to stdout.
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Allow importing common.py from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BACKEND_ENDPOINT_MAP


def sanitize_model_name(model: str, max_len: int = 30) -> str:
    """Extract short model name for filenames."""
    # Take last path segment
    short = model.split("/")[-1]
    # Replace problematic characters
    short = short.replace("/", "-").replace(" ", "_")
    return short[:max_len]


def generate_filename(model: str, dataset: str, backend: str) -> str:
    """Generate standardized result filename."""
    model_short = sanitize_model_name(model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"bench_{model_short}_{dataset}_{backend}_{timestamp}.json"


def generate_command(params: dict) -> str:
    """Generate the complete vllm bench serve command."""
    parts = ["vllm bench serve"]

    # Connection
    if params.get("base_url"):
        parts.append(f"--base-url {params['base_url']}")
    else:
        if params.get("host"):
            parts.append(f"--host {params['host']}")
        if params.get("port"):
            parts.append(f"--port {params['port']}")

    # Model (--model = weight path for tokenizer, --served-model-name = API model name)
    if params.get("model"):
        parts.append(f"--model {params['model']}")
    if params.get("served_model_name"):
        parts.append(f"--served-model-name {params['served_model_name']}")

    # Backend + Endpoint
    backend = params.get("backend", "openai-chat")
    parts.append(f"--backend {backend}")
    # --endpoint defaults to /v1/completions in vllm bench serve, so we must
    # explicitly set it for any backend that uses a different endpoint.
    endpoint = params.get("endpoint") or BACKEND_ENDPOINT_MAP.get(backend, "/v1/completions")
    parts.append(f"--endpoint {endpoint}")

    # Dataset
    dataset = params.get("dataset_name", "random")
    parts.append(f"--dataset-name {dataset}")
    if params.get("dataset_path"):
        parts.append(f"--dataset-path {params['dataset_path']}")

    # Dataset-specific length params — use correct per-dataset flag names
    input_len = params.get("input_len") or params.get("random_input_len")
    output_len = params.get("output_len") or params.get("random_output_len")

    if dataset in ("random", "random-mm", "random-rerank"):
        if input_len:
            parts.append(f"--random-input-len {input_len}")
        if output_len:
            parts.append(f"--random-output-len {output_len}")
    elif dataset == "sonnet":
        if input_len:
            parts.append(f"--sonnet-input-len {input_len}")
        if output_len:
            parts.append(f"--sonnet-output-len {output_len}")
    elif dataset == "sharegpt":
        if output_len:
            parts.append(f"--sharegpt-output-len {output_len}")
    elif dataset == "custom" or dataset == "custom_mm":
        if output_len:
            parts.append(f"--custom-output-len {output_len}")
    elif dataset == "hf":
        if output_len:
            parts.append(f"--hf-output-len {output_len}")
    elif dataset == "spec_bench":
        if output_len:
            parts.append(f"--spec-bench-output-len {output_len}")
    elif dataset == "prefix_repetition":
        if output_len:
            parts.append(f"--prefix-repetition-output-len {output_len}")
    else:
        # Fallback to generic aliases
        if input_len:
            parts.append(f"--input-len {input_len}")
        if output_len:
            parts.append(f"--output-len {output_len}")
    if params.get("random_range_ratio") is not None:
        parts.append(f"--random-range-ratio {params['random_range_ratio']}")
    if params.get("random_prefix_len"):
        parts.append(f"--random-prefix-len {params['random_prefix_len']}")
    if params.get("random_batch_size"):
        parts.append(f"--random-batch-size {params['random_batch_size']}")

    # Load control
    if params.get("num_prompts"):
        parts.append(f"--num-prompts {params['num_prompts']}")
    if params.get("max_concurrency"):
        parts.append(f"--max-concurrency {params['max_concurrency']}")
    if params.get("request_rate") is not None:
        parts.append(f"--request-rate {params['request_rate']}")
    if params.get("num_warmups"):
        parts.append(f"--num-warmups {params['num_warmups']}")

    # Metrics
    percentile_metrics = params.get("percentile_metrics", "ttft,tpot,itl,e2el")
    metric_percentiles = params.get("metric_percentiles", "50,90,95,99")
    parts.append(f'--percentile-metrics "{percentile_metrics}"')
    parts.append(f'--metric-percentiles "{metric_percentiles}"')

    # Goodput / SLO
    if params.get("goodput"):
        for g in params["goodput"]:
            parts.append(f'--goodput "{g}"')

    # Extra params
    if params.get("extra_args"):
        parts.append(params["extra_args"])

    # Result archival (ALWAYS applied)
    parts.append("--save-result")
    parts.append("--save-detailed")

    result_dir = params.get("result_dir", "./bench_results/single")
    parts.append(f"--result-dir {result_dir}")

    model_name = params.get("model", "unknown")
    result_filename = params.get(
        "result_filename",
        generate_filename(model_name, dataset, backend),
    )
    parts.append(f"--result-filename {result_filename}")

    return " \\\n  ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate vllm bench serve command")
    parser.add_argument("--config", help="JSON config file path")

    # Direct CLI params (used when --config is not provided)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--backend", default="openai-chat")
    parser.add_argument("--endpoint", default=None, help="API endpoint (auto-set from backend if not specified)")
    parser.add_argument("--dataset-name", default="random")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--input-len", type=int, default=None, help="General input length (alias)")
    parser.add_argument("--output-len", type=int, default=None, help="General output length (alias)")
    parser.add_argument("--random-input-len", type=int, default=None, help="Random dataset input length")
    parser.add_argument("--random-output-len", type=int, default=None, help="Random dataset output length")
    parser.add_argument("--random-range-ratio", type=float, default=None)
    parser.add_argument("--random-prefix-len", type=int, default=None)
    parser.add_argument("--random-batch-size", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--request-rate", type=float, default=None)
    parser.add_argument("--num-warmups", type=int, default=None)
    parser.add_argument("--goodput", nargs="*", default=None)
    parser.add_argument("--percentile-metrics", default="ttft,tpot,itl,e2el")
    parser.add_argument("--metric-percentiles", default="50,90,95,99")
    parser.add_argument("--result-dir", default="./bench_results/single")
    parser.add_argument("--result-filename", default=None)
    parser.add_argument("--extra-args", default=None)

    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        with open(args.config) as f:
            params = json.load(f)
    else:
        params = {
            "base_url": args.base_url,
            "host": args.host,
            "port": args.port,
            "model": args.model,
            "served_model_name": args.served_model_name,
            "backend": args.backend,
            "endpoint": args.endpoint,
            "dataset_name": args.dataset_name,
            "dataset_path": args.dataset_path,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            "random_prefix_len": args.random_prefix_len,
            "random_batch_size": args.random_batch_size,
            "num_prompts": args.num_prompts,
            "max_concurrency": args.max_concurrency,
            "request_rate": args.request_rate,
            "num_warmups": args.num_warmups,
            "goodput": args.goodput,
            "percentile_metrics": args.percentile_metrics,
            "metric_percentiles": args.metric_percentiles,
            "result_dir": args.result_dir,
            "result_filename": args.result_filename,
            "extra_args": args.extra_args,
        }

    cmd = generate_command(params)
    print(cmd)


if __name__ == "__main__":
    main()
