#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch  # type: ignore[import-not-found]
from transformers import T5EncoderModel, T5Tokenizer  # type: ignore[import-not-found]


DEFAULT_SEQUENCE = "[NLU]MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPQ"


def default_weights_dir() -> Path:
    return Path(os.environ.get("ANKH3_LARGE_PATH", "./weights/Ankh3_large"))


def has_torch_npu() -> bool:
    try:
        import torch_npu  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_device(device: str = "auto") -> torch.device:
    normalized = str(device).lower()
    if normalized == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu:0")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if normalized == "npu":
        normalized = "npu:0"
    elif normalized == "cuda":
        normalized = "cuda:0"
    return torch.device(normalized)


def infer_torch_dtype(device: torch.device, dtype: str = "auto") -> torch.dtype:
    normalized = str(dtype).lower()
    if normalized == "auto":
        if device.type == "npu":
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[normalized]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify local Ankh3_large weights on Ascend NPU")
    parser.add_argument("--weights-dir", default=str(default_weights_dir()))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    required_files = ["config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json"]
    missing = [name for name in required_files if not (weights_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {weights_dir}: {missing}")

    device = resolve_device(args.device)
    torch_dtype = infer_torch_dtype(device, args.dtype)
    tokenizer = T5Tokenizer.from_pretrained(weights_dir)
    model = T5EncoderModel.from_pretrained(weights_dir, torch_dtype=torch_dtype)
    model = model.eval().to(device)

    batch = tokenizer(args.sequence, add_special_tokens=True, return_tensors="pt")
    batch = {name: tensor.to(device) for name, tensor in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)

    report = {
        "torch_version": torch.__version__,
        "torch_npu_installed": has_torch_npu(),
        "resolved_device": str(device),
        "resolved_dtype": str(torch_dtype),
        "weights_dir": str(weights_dir),
        "last_hidden_state_shape": list(outputs.last_hidden_state.shape),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
