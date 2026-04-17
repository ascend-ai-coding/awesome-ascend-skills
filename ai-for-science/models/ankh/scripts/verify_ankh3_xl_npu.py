#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch  # type: ignore[import-not-found]
from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer  # type: ignore[import-not-found]


DEFAULT_SEQUENCE = "[NLU]MKTAYIAKQRQISFVKSHFSRQ"


def default_weights_dir() -> str:
    return os.environ.get("ANKH3_XL_PATH", "./weights/Ankh3_XL")


def has_torch_npu() -> bool:
    try:
        import torch_npu  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


def is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def resolve_device(device: str = "auto") -> torch.device:
    normalized = str(device).lower()
    if normalized == "auto":
        if is_npu_available():
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
        if str(device).startswith("npu"):
            return torch.bfloat16
        if str(device).startswith("cuda"):
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


def load_model(weights_dir: Path, task: str, torch_dtype: torch.dtype):
    tokenizer = T5Tokenizer.from_pretrained(weights_dir)
    if task == "encoder":
        model = T5EncoderModel.from_pretrained(weights_dir, torch_dtype=torch_dtype)
    else:
        model = T5ForConditionalGeneration.from_pretrained(weights_dir, torch_dtype=torch_dtype)
    return model.eval(), tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Ankh3_XL on Ascend NPU")
    parser.add_argument("--weights-dir", default=default_weights_dir())
    parser.add_argument("--task", choices=["encoder", "generation"], default="encoder")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    required_files = ["pytorch_model.bin.index.json", "spiece.model", "tokenizer_config.json"]
    missing = [name for name in required_files if not (weights_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {weights_dir}: {missing}")

    device = resolve_device(args.device)
    torch_dtype = infer_torch_dtype(device, args.dtype)
    model, tokenizer = load_model(weights_dir, args.task, torch_dtype)
    model = model.to(device)

    batch = tokenizer(args.sequence, add_special_tokens=True, return_tensors="pt")
    batch = {name: tensor.to(device) for name, tensor in batch.items()}

    report = {
        "torch_npu_installed": has_torch_npu(),
        "resolved_device": str(device),
        "resolved_dtype": str(torch_dtype),
        "task": args.task,
        "weights_dir": str(weights_dir),
    }

    with torch.no_grad():
        if args.task == "encoder":
            outputs = model(**batch)
            report["last_hidden_state_shape"] = list(outputs.last_hidden_state.shape)
        else:
            generated = model.generate(batch["input_ids"], max_length=batch["input_ids"].shape[1] + 8)
            report["generated_ids_shape"] = list(generated.shape)
            report["decoded"] = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
