#!/usr/bin/env python3
"""
Generate fake (randomly initialized) model weights for HuggingFace Diffusers pipelines.

Strategy:
  1. Download ONLY metadata files (config.json, model_index.json, tokenizer, scheduler)
     from HuggingFace, excluding large weight files (*.safetensors, *.bin).
  2. Instantiate each sub-model from its config with random weights.
  3. Save the full pipeline via save_pretrained(), producing files structurally
     identical to the original model.

This allows testing model loading, pipeline construction, and inference code
without downloading tens of GB of real weights.
"""

import argparse
import json
import os
import sys
import shutil
import traceback
from pathlib import Path
from typing import Optional


def download_metadata_only(model_id: str, local_dir: str, proxy: Optional[str] = None):
    from huggingface_hub import snapshot_download

    os.makedirs(local_dir, exist_ok=True)

    endpoint = proxy or os.environ.get("HF_ENDPOINT", None)

    print(f"Downloading metadata (no weights) from: {model_id}")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        ignore_patterns=[
            "*.safetensors",
            "*.bin",
            "*.msgpack",
            "*.h5",
            "*.ot",
            "*.onnx",
            "assets/*",
        ],
        endpoint=endpoint,
    )
    print(f"Metadata downloaded to: {local_dir}")


def generate_from_local_metadata(
    metadata_dir: str, output_dir: str, dtype_str: str = "bfloat16"
):
    import torch
    import diffusers
    import transformers

    torch_dtype = getattr(torch, dtype_str)

    model_index_path = os.path.join(metadata_dir, "model_index.json")
    if not os.path.exists(model_index_path):
        print(f"Error: model_index.json not found in {metadata_dir}", file=sys.stderr)
        return 1

    with open(model_index_path, "r") as f:
        model_index = json.load(f)

    pipeline_class_name = model_index["_class_name"]
    print(f"Pipeline: {pipeline_class_name}")

    os.makedirs(output_dir, exist_ok=True)

    # Copy model_index.json
    shutil.copy2(model_index_path, os.path.join(output_dir, "model_index.json"))

    component_keys = [k for k in model_index.keys() if not k.startswith("_")]

    for comp_name in component_keys:
        comp_info = model_index[comp_name]
        if not isinstance(comp_info, list) or len(comp_info) != 2:
            continue

        library_name, class_name = comp_info
        comp_metadata_dir = os.path.join(metadata_dir, comp_name)
        comp_output_dir = os.path.join(output_dir, comp_name)

        print(f"\n--- {comp_name}: {library_name}.{class_name} ---")

        if (
            class_name
            in ("Qwen2Tokenizer", "Qwen2TokenizerFast", "PreTrainedTokenizerFast")
            or "tokenizer" in comp_name.lower()
        ):
            if os.path.exists(comp_metadata_dir):
                shutil.copytree(comp_metadata_dir, comp_output_dir, dirs_exist_ok=True)
                print(f"  Copied tokenizer files")
            continue

        if "scheduler" in comp_name.lower() or "Scheduler" in class_name:
            if os.path.exists(comp_metadata_dir):
                shutil.copytree(comp_metadata_dir, comp_output_dir, dirs_exist_ok=True)
                print(f"  Copied scheduler config")
            continue

        # For model components (transformer, vae, text_encoder), instantiate from config
        config_path = os.path.join(comp_metadata_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"  Warning: no config.json found, skipping")
            continue

        try:
            if library_name == "diffusers":
                model_class = getattr(diffusers, class_name, None)
                if model_class is None:
                    model_class = getattr(diffusers.models, class_name, None)
                if model_class is None:
                    print(f"  Warning: {class_name} not found in diffusers, skipping")
                    continue

                config = model_class.load_config(comp_metadata_dir)
                model = model_class.from_config(config).to(dtype=torch_dtype)
                model.save_pretrained(comp_output_dir)
                param_count = sum(p.numel() for p in model.parameters())
                print(f"  Generated: {param_count:,} parameters ({dtype_str})")
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            elif library_name == "transformers":
                model_class = getattr(transformers, class_name, None)
                if model_class is None:
                    print(
                        f"  Warning: {class_name} not found in transformers, skipping"
                    )
                    continue

                with open(config_path, "r") as f:
                    config_dict = json.load(f)

                config_class_name = config_dict.get("model_type", "")
                auto_config = transformers.AutoConfig.from_pretrained(
                    comp_metadata_dir, trust_remote_code=True
                )
                model = model_class._from_config(auto_config).to(dtype=torch_dtype)
                model.save_pretrained(comp_output_dir)

                # Copy extra metadata files (generation_config.json, etc.)
                for extra_file in Path(comp_metadata_dir).glob("*.json"):
                    if extra_file.name != "config.json":
                        dest = Path(comp_output_dir) / extra_file.name
                        if not dest.exists():
                            shutil.copy2(extra_file, dest)

                param_count = sum(p.numel() for p in model.parameters())
                print(f"  Generated: {param_count:,} parameters ({dtype_str})")
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            else:
                print(f"  Warning: unknown library '{library_name}', skipping")
        except Exception as e:
            print(f"  Error generating {comp_name}: {e}", file=sys.stderr)
            traceback.print_exc()
            continue

    print(f"\n=== Fake weights saved to: {output_dir} ===")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate fake model weights for Diffusers pipelines"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Sub-command: from-hub
    hub_parser = subparsers.add_parser(
        "from-hub", help="Download metadata from HuggingFace and generate fake weights"
    )
    hub_parser.add_argument(
        "model_id", help="HuggingFace model ID (e.g., Qwen/Qwen-Image-2512)"
    )
    hub_parser.add_argument(
        "-o", "--output-dir", required=True, help="Output directory"
    )
    hub_parser.add_argument(
        "--proxy", help="HuggingFace proxy URL (e.g., https://hf-mirror.com)"
    )
    hub_parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )

    # Sub-command: from-local
    local_parser = subparsers.add_parser(
        "from-local", help="Generate fake weights from local metadata directory"
    )
    local_parser.add_argument(
        "metadata_dir",
        help="Local directory with model metadata (config.json, model_index.json)",
    )
    local_parser.add_argument(
        "-o", "--output-dir", required=True, help="Output directory"
    )
    local_parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "from-hub":
        metadata_dir = args.output_dir + "_metadata"
        download_metadata_only(args.model_id, metadata_dir, args.proxy)
        result = generate_from_local_metadata(metadata_dir, args.output_dir, args.dtype)
        shutil.rmtree(metadata_dir, ignore_errors=True)
        return result

    elif args.command == "from-local":
        return generate_from_local_metadata(
            args.metadata_dir, args.output_dir, args.dtype
        )


if __name__ == "__main__":
    sys.exit(main())
