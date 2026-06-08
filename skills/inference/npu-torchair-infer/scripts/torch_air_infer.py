#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal, model-agnostic torchair (NPU graph-mode) inference harness.

This script loads *any* HuggingFace model from ``--model_name_or_path`` and runs
it through one of three execution paths:

  * ``torchair``  : NPU graph mode via ``torchair.get_npu_backend`` + ``torch.compile``
  * ``eager``     : plain eager execution (NPU single-op, or CPU)

It is intentionally free of any model-specific hard-coding. SigLIP2 / DINOv3 are
only example validation cases; the same code path works for any encoder whose
forward accepts ``pixel_values`` and/or ``input_ids``.

IMPORTANT import order (graph mode only works if respected):
    import torch
    import torch_npu      # MUST be imported before torchair
    import torchair
"""

import argparse
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# --- Mandatory import order for NPU graph mode -----------------------------
# torch_npu registers the 'npu' backend/device; torchair must come after it.
try:
    import torch_npu  # noqa: F401  (registers torch.npu)
    _HAS_TORCH_NPU = True
except Exception as exc:  # pragma: no cover - depends on host
    torch_npu = None
    _HAS_TORCH_NPU = False
    _TORCH_NPU_IMPORT_ERROR = exc

_HAS_TORCHAIR = False
if _HAS_TORCH_NPU:
    try:
        import torchair  # noqa: F401  (MUST be imported after torch_npu)
        from torchair import patch_for_hcom

        _HAS_TORCHAIR = True
    except Exception as exc:  # pragma: no cover
        _TORCHAIR_IMPORT_ERROR = exc

from transformers import AutoConfig, AutoModel


# --------------------------------------------------------------------------- #
# Device / dtype helpers
# --------------------------------------------------------------------------- #
def npu_available() -> bool:
    return _HAS_TORCH_NPU and hasattr(torch, "npu") and torch.npu.is_available()


def resolve_device(requested: str, npu_id: int) -> Tuple[str, torch.device]:
    """Map a --device request to (kind, torch.device). 'auto' prefers NPU.

    An explicit ``--device npu`` that cannot be satisfied is a hard error: silently
    falling back to CPU would turn an intended CPU-vs-NPU comparison into a
    meaningless CPU-vs-CPU one.
    """
    if requested == "npu":
        if not npu_available():
            raise RuntimeError(
                "--device npu requested but NPU is unavailable "
                f"(torch_npu imported: {_HAS_TORCH_NPU}). Refusing to fall back to CPU; "
                "fix the environment (see references/environment.md) or pass --device cpu."
            )
        return "npu", torch.device(f"npu:{npu_id}")
    if requested == "auto" and npu_available():
        return "npu", torch.device(f"npu:{npu_id}")
    return "cpu", torch.device("cpu")


def resolve_dtype(name: Optional[str], device_kind: str) -> torch.dtype:
    if name is not None:
        return {"float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float32": torch.float32, "fp32": torch.float32}[name]
    # Sensible defaults: fp16 on NPU (matches deployment), fp32 on CPU.
    return torch.float16 if device_kind == "npu" else torch.float32


def sync(device_kind: str) -> None:
    if device_kind == "npu":
        torch.npu.synchronize()


# --------------------------------------------------------------------------- #
# Generic input synthesis (config-driven, no model-specific hard-coding)
# --------------------------------------------------------------------------- #
def _sub(config, name):
    return getattr(config, name, None)


def detect_modalities(config) -> Dict[str, Any]:
    """Inspect an arbitrary HF config to decide which dummy inputs are needed."""
    vision_cfg = _sub(config, "vision_config")
    text_cfg = _sub(config, "text_config")

    # Pure-vision models (e.g. ViT / DINOv3) expose image_size+patch_size at top.
    top_is_vision = _sub(config, "image_size") is not None and _sub(config, "patch_size") is not None
    has_vision = vision_cfg is not None or top_is_vision
    # Text present if there is a text sub-config, or a vocab on a non-vision model.
    has_text = text_cfg is not None or (_sub(config, "vocab_size") is not None and not top_is_vision)

    vsrc = vision_cfg if vision_cfg is not None else config
    tsrc = text_cfg if text_cfg is not None else config
    return {
        "has_vision": has_vision,
        "has_text": has_text,
        "image_size": getattr(vsrc, "image_size", 224) if has_vision else None,
        "num_channels": getattr(vsrc, "num_channels", 3) if has_vision else None,
        "vocab_size": getattr(tsrc, "vocab_size", 30522) if has_text else None,
        "max_pos": getattr(tsrc, "max_position_embeddings", 64) if has_text else None,
    }


def build_inputs(config, batch_size: int, dtype: torch.dtype, device: torch.device,
                 image_size: Optional[int], seq_len: Optional[int], seed: int) -> Dict[str, torch.Tensor]:
    """Create a fixed (seeded) batch of dummy inputs appropriate for the model."""
    torch.manual_seed(seed)
    spec = detect_modalities(config)
    inputs: Dict[str, torch.Tensor] = {}

    if spec["has_vision"]:
        hw = image_size or spec["image_size"] or 224
        c = spec["num_channels"] or 3
        inputs["pixel_values"] = torch.randn(batch_size, c, hw, hw, dtype=dtype)

    if spec["has_text"]:
        L = seq_len or min(spec["max_pos"] or 64, 64)
        vocab = spec["vocab_size"] or 30522
        inputs["input_ids"] = torch.randint(0, vocab, (batch_size, L), dtype=torch.long)
        inputs["attention_mask"] = torch.ones(batch_size, L, dtype=torch.long)

    if not inputs:
        raise ValueError(
            "Could not infer inputs from config (no vision_config/text_config/image_size/vocab_size). "
            "This model likely needs a custom input builder.")

    return {k: v.to(device) for k, v in inputs.items()}


# --------------------------------------------------------------------------- #
# Output handling
# --------------------------------------------------------------------------- #
def flatten_float_tensors(obj: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    """Recursively collect floating-point tensors from a model output."""
    out: Dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            out[prefix or "tensor"] = obj
        return out
    if isinstance(obj, dict):
        items = obj.items()
    elif hasattr(obj, "items"):  # ModelOutput behaves like a dict
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = [(str(i), v) for i, v in enumerate(obj)]
    else:
        return out
    for k, v in items:
        out.update(flatten_float_tensors(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #
def maybe_compile(model, backend: str, device_kind: str, compile_mode: Optional[str]):
    """Wrap the model with the torchair NPU backend when requested."""
    if backend != "torchair":
        return model, "eager"
    if device_kind != "npu":
        print("[WARN] torchair backend only meaningful on NPU; using eager.", file=sys.stderr)
        return model, "eager"
    if not _HAS_TORCHAIR:
        raise RuntimeError("torchair is not importable; cannot use graph mode.")

    config = torchair.CompilerConfig()
    if compile_mode:
        # torchair exposes a 'mode' knob (e.g. 'reduce-overhead'); guard with hasattr.
        if hasattr(config, "mode"):
            config.mode = compile_mode
        else:
            print(f"[WARN] torchair CompilerConfig has no 'mode'; ignoring {compile_mode!r}.",
                  file=sys.stderr)
    # Required for models that contain HCCL collective ops; harmless otherwise.
    try:
        patch_for_hcom()
    except Exception:
        pass
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    compiled = torch.compile(model, backend=npu_backend, dynamic=False)
    return compiled, "torchair(graph)"


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def percentiles(samples_ms: List[float]) -> Dict[str, float]:
    s = sorted(samples_ms)
    n = len(s)

    def pct(p):
        if n == 1:
            return s[0]
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return s[idx]

    return {
        "p50": statistics.median(s),
        "p95": pct(95),
        "p99": pct(99),
        "mean": statistics.fmean(s),
        "min": s[0],
        "max": s[-1],
    }


def run(model, inputs, device_kind, warmup, iterations):
    first_iter_ms = None
    with torch.inference_mode():
        # Warmup (first iteration captures compile/build overhead for graph mode).
        for i in range(max(warmup, 1)):
            t0 = time.perf_counter()
            out = model(**inputs)
            sync(device_kind)
            if i == 0:
                first_iter_ms = (time.perf_counter() - t0) * 1000.0

        samples: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            out = model(**inputs)
            sync(device_kind)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return out, first_iter_ms, samples


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Minimal generic torchair graph-mode inference for any HF model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_name_or_path", required=True,
                   help="HuggingFace repo id or local path (any vision / text / image-text encoder).")
    p.add_argument("--device", default="auto", choices=["auto", "npu", "cpu"],
                   help="Execution device. 'auto' prefers NPU and falls back to CPU.")
    p.add_argument("--npu_id", type=int, default=0, help="NPU ordinal to bind to.")
    p.add_argument("--backend", default="torchair", choices=["torchair", "eager"],
                   help="'torchair' = NPU graph mode; 'eager' = single-op / CPU eager.")
    p.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"],
                   help="Compute dtype. Default: fp16 on NPU, fp32 on CPU.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--image_size", type=int, default=None,
                   help="Override square image size; default taken from config.")
    p.add_argument("--seq_len", type=int, default=None,
                   help="Override text sequence length; default from config max_position_embeddings.")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations (>=3 recommended).")
    p.add_argument("--iterations", type=int, default=20, help="Timed iterations.")
    p.add_argument("--compile_mode", default=None,
                   help="Optional torchair CompilerConfig.mode (e.g. 'reduce-overhead').")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device_kind, device = resolve_device(args.device, args.npu_id)
    dtype = resolve_dtype(args.dtype, device_kind)

    print("=" * 72)
    print(f"model          : {args.model_name_or_path}")
    print(f"device         : {device} ({device_kind})")
    print(f"dtype          : {dtype}")
    print(f"requested bk   : {args.backend}")
    print(f"torch_npu      : {'yes' if _HAS_TORCH_NPU else 'NO'} | torchair: {'yes' if _HAS_TORCHAIR else 'NO'}")
    print("=" * 72)

    if device_kind == "npu":
        torch.npu.set_device(device)

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype if device_kind == "npu" else torch.float32)
    model = model.to(device).eval()

    inputs = build_inputs(config, args.batch_size, dtype if device_kind == "npu" else torch.float32,
                          device, args.image_size, args.seq_len, args.seed)
    for k, v in inputs.items():
        print(f"input[{k:14s}] shape={tuple(v.shape)} dtype={v.dtype}")

    model, backend_mode = maybe_compile(model, args.backend, device_kind, args.compile_mode)
    print(f"backend mode   : {backend_mode}")

    out, first_ms, samples = run(model, inputs, device_kind, args.warmup, args.iterations)

    for name, t in flatten_float_tensors(out).items():
        print(f"output[{name}] shape={tuple(t.shape)} dtype={t.dtype}")

    stats = percentiles(samples)
    print("-" * 72)
    print(f"first iter (compile+run): {first_ms:.2f} ms")
    print(f"steady latency  : p50={stats['p50']:.2f}  p95={stats['p95']:.2f}  "
          f"p99={stats['p99']:.2f}  mean={stats['mean']:.2f} ms  (n={len(samples)})")
    thr = (args.batch_size * 1000.0) / stats["p50"] if stats["p50"] > 0 else float("nan")
    print(f"throughput      : {thr:.2f} samples/sec @ p50")
    print("=" * 72)


if __name__ == "__main__":
    main()
