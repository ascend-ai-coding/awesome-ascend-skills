#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accuracy + performance benchmark across three execution backends.

Backends
--------
  * ``torchair``  : NPU graph mode (torch.compile + torchair.get_npu_backend)
  * ``npu_eager`` : NPU single-op eager (no compile)
  * ``cpu``       : CPU eager (fp32, used as the golden reference)

For every backend we run the *same* seeded inputs, collect the model outputs,
and report:
  * accuracy vs the CPU golden output: cosine similarity, max abs diff, rel err
  * a cross-check of torchair vs npu_eager (same device+dtype, should be tight)
  * performance: first-iteration (compile) cost, steady p50/p95 latency,
    throughput, and NPU peak memory.

Results are written as ``benchmark_results.json`` and a Markdown table.

Model-agnostic: nothing about SigLIP2 / DINOv3 (or any model) is hard-coded.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch

# --- Mandatory import order: torch_npu BEFORE torchair ----------------------
try:
    import torch_npu  # noqa: F401
    _HAS_TORCH_NPU = True
except Exception:
    torch_npu = None
    _HAS_TORCH_NPU = False

_HAS_TORCHAIR = False
if _HAS_TORCH_NPU:
    try:
        import torchair  # noqa: F401  (after torch_npu)
        from torchair import patch_for_hcom
        _HAS_TORCHAIR = True
    except Exception:
        pass

from transformers import AutoConfig, AutoModel


# --------------------------------------------------------------------------- #
# Generic helpers (shared design with torch_air_infer.py)
# --------------------------------------------------------------------------- #
def npu_available() -> bool:
    return _HAS_TORCH_NPU and hasattr(torch, "npu") and torch.npu.is_available()


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _sub(config, name):
    return getattr(config, name, None)


def detect_modalities(config) -> Dict[str, Any]:
    vision_cfg = _sub(config, "vision_config")
    text_cfg = _sub(config, "text_config")
    top_is_vision = _sub(config, "image_size") is not None and _sub(config, "patch_size") is not None
    has_vision = vision_cfg is not None or top_is_vision
    has_text = text_cfg is not None or (_sub(config, "vocab_size") is not None and not top_is_vision)
    vsrc = vision_cfg if vision_cfg is not None else config
    tsrc = text_cfg if text_cfg is not None else config
    return {
        "has_vision": has_vision, "has_text": has_text,
        "image_size": getattr(vsrc, "image_size", 224) if has_vision else None,
        "num_channels": getattr(vsrc, "num_channels", 3) if has_vision else None,
        "vocab_size": getattr(tsrc, "vocab_size", 30522) if has_text else None,
        "max_pos": getattr(tsrc, "max_position_embeddings", 64) if has_text else None,
    }


def build_cpu_inputs(config, batch_size, image_size, seq_len, seed) -> Dict[str, torch.Tensor]:
    """Build one fixed (seeded) set of inputs on CPU. Backends clone+cast from this."""
    torch.manual_seed(seed)
    spec = detect_modalities(config)
    inp: Dict[str, torch.Tensor] = {}
    if spec["has_vision"]:
        hw = image_size or spec["image_size"] or 224
        c = spec["num_channels"] or 3
        inp["pixel_values"] = torch.randn(batch_size, c, hw, hw, dtype=torch.float32)
    if spec["has_text"]:
        L = seq_len or min(spec["max_pos"] or 64, 64)
        inp["input_ids"] = torch.randint(0, spec["vocab_size"] or 30522, (batch_size, L), dtype=torch.long)
        inp["attention_mask"] = torch.ones(batch_size, L, dtype=torch.long)
    if not inp:
        raise ValueError("Could not infer inputs from config; a custom builder is needed.")
    return inp


def cast_inputs(cpu_inputs, device, float_dtype) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in cpu_inputs.items():
        v = v.clone()
        if v.is_floating_point():
            v = v.to(float_dtype)
        out[k] = v.to(device)
    return out


def flatten_float_tensors(obj: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            out[prefix or "tensor"] = obj
        return out
    if isinstance(obj, dict) or hasattr(obj, "items"):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = [(str(i), v) for i, v in enumerate(obj)]
    else:
        return out
    for k, v in items:
        out.update(flatten_float_tensors(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


def sync(device_kind):
    if device_kind == "npu":
        torch.npu.synchronize()


# --------------------------------------------------------------------------- #
# Accuracy metrics
# --------------------------------------------------------------------------- #
def compare(golden: torch.Tensor, cand: torch.Tensor) -> Dict[str, float]:
    a = golden.detach().to("cpu", torch.float64).flatten()
    b = cand.detach().to("cpu", torch.float64).flatten()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    diff = (a - b).abs()
    denom = a.abs().clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return {
        "cosine_sim": float(cos),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "max_rel_err": float((diff / denom).max().item()),
    }


# --------------------------------------------------------------------------- #
# Per-backend run
# --------------------------------------------------------------------------- #
def load_model(path, float_dtype, device, trust_remote_code):
    model = AutoModel.from_pretrained(path, trust_remote_code=trust_remote_code, torch_dtype=float_dtype)
    return model.to(device).eval()


def compile_torchair(model, compile_mode):
    config = torchair.CompilerConfig()
    if compile_mode and hasattr(config, "mode"):
        config.mode = compile_mode
    try:
        patch_for_hcom()
    except Exception:
        pass
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    return torch.compile(model, backend=npu_backend, dynamic=False)


def percentiles(samples_ms: List[float]) -> Dict[str, float]:
    s = sorted(samples_ms)
    n = len(s)
    pct = lambda p: s[min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))]  # noqa: E731
    return {"p50": statistics.median(s), "p95": pct(95), "p99": pct(99),
            "mean": statistics.fmean(s), "std": (statistics.pstdev(s) if n > 1 else 0.0),
            "min": s[0], "max": s[-1]}


def run_backend(name, args, cpu_inputs, config) -> Dict[str, Any]:
    """Execute one backend end-to-end and return a result record."""
    rec: Dict[str, Any] = {"backend": name, "status": "ok", "error": None}

    if name == "cpu":
        device_kind, device, float_dtype = "cpu", torch.device("cpu"), torch.float32
    else:
        if not npu_available():
            rec.update(status="skipped", error="NPU not available")
            return rec
        device_kind = "npu"
        device = torch.device(f"npu:{args.npu_id}")
        float_dtype = DTYPES[args.dtype]
        torch.npu.set_device(device)
        torch.npu.reset_peak_memory_stats()

    rec.update(device=str(device), dtype=str(float_dtype))

    try:
        model = load_model(args.model_name_or_path, float_dtype, device, args.trust_remote_code)
        inputs = cast_inputs(cpu_inputs, device, float_dtype)

        if name == "torchair":
            if not _HAS_TORCHAIR:
                raise RuntimeError("torchair not importable")
            model = compile_torchair(model, args.compile_mode)

        first_ms = None
        with torch.inference_mode():
            for i in range(max(args.warmup, 1)):
                t0 = time.perf_counter()
                out = model(**inputs)
                sync(device_kind)
                if i == 0:
                    first_ms = (time.perf_counter() - t0) * 1000.0
            samples = []
            for _ in range(args.iterations):
                t0 = time.perf_counter()
                out = model(**inputs)
                sync(device_kind)
                samples.append((time.perf_counter() - t0) * 1000.0)

        rec["outputs"] = {k: v.detach().to("cpu", torch.float32)
                          for k, v in flatten_float_tensors(out).items()}
        rec["perf"] = {"first_iter_ms": first_ms, **percentiles(samples),
                       "iterations": len(samples),
                       "throughput_sps": (args.batch_size * 1000.0) / statistics.median(samples)}
        if device_kind == "npu":
            rec["perf"]["peak_mem_mb"] = torch.npu.max_memory_allocated() / (1024 ** 2)
    except Exception as exc:  # graph-mode compile failures land here -> graceful
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rec["traceback"] = traceback.format_exc()
        print(f"[ERROR] backend={name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        if device_kind == "npu":
            try:
                torch.npu.empty_cache()
            except Exception:
                pass
    return rec


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def pick_primary(keys: List[str]) -> Optional[str]:
    for pref in ("last_hidden_state", "logits", "pooler_output", "image_embeds", "text_embeds"):
        for k in keys:
            if k.endswith(pref) or k == pref:
                return k
    return keys[0] if keys else None


def build_accuracy(results, cosine_threshold, atol):
    """Compare each NPU backend's outputs against the CPU golden."""
    golden = next((r for r in results if r["backend"] == "cpu" and r["status"] == "ok"), None)
    acc: Dict[str, Any] = {}
    if golden is None or "outputs" not in golden:
        return acc, None
    gkeys = list(golden["outputs"].keys())
    primary = pick_primary(gkeys)

    for r in results:
        if r["backend"] == "cpu" or r["status"] != "ok" or "outputs" not in r:
            continue
        per_out = {}
        for k in gkeys:
            if k in r["outputs"]:
                m = compare(golden["outputs"][k], r["outputs"][k])
                cos_ok = m["cosine_sim"] >= cosine_threshold
                abs_ok = m["max_abs_diff"] <= atol
                m["status"] = "PASS" if (cos_ok and abs_ok) else ("PASS_COSINE" if cos_ok else "FAIL")
                per_out[k] = m
        acc[r["backend"]] = {"primary_key": primary, "per_output": per_out}
    return acc, primary


def markdown_report(meta, results, accuracy, primary, cosine_threshold, atol) -> str:
    L = []
    L.append(f"# torchair migration benchmark: `{meta['model_name_or_path']}`\n")
    L.append(f"_Generated {meta['timestamp']}_\n")
    L.append("## Environment\n")
    for k in ("python", "torch", "torch_npu", "torchair", "transformers", "cann", "npu_name", "device_count"):
        L.append(f"- {k}: `{meta['env'].get(k)}`")
    L.append(f"\n- batch_size: `{meta['batch_size']}`  dtype(NPU): `{meta['dtype']}`  "
             f"warmup: `{meta['warmup']}`  iterations: `{meta['iterations']}`")
    L.append(f"- accuracy gate: cosine ≥ `{cosine_threshold}`, max_abs_diff ≤ `{atol}`\n")

    L.append("## Accuracy (vs CPU fp32 golden)\n")
    if accuracy:
        L.append(f"Primary output: `{primary}`\n")
        L.append("| Backend | Output | cosine_sim | max_abs_diff | max_rel_err | Status |")
        L.append("|---|---|---:|---:|---:|---|")
        for bk, info in accuracy.items():
            for okey, m in info["per_output"].items():
                L.append(f"| {bk} | `{okey}` | {m['cosine_sim']:.6f} | {m['max_abs_diff']:.3e} | "
                         f"{m['max_rel_err']:.3e} | {m['status']} |")
    else:
        L.append("_No CPU golden available; accuracy comparison skipped._")

    L.append("\n## Performance (steady-state, warmup excluded)\n")
    L.append("| Backend | Status | first_iter ms | p50 ms | p95 ms | p99 ms | throughput (sps) | peak_mem MB |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        if r["status"] == "ok":
            p = r["perf"]
            L.append(f"| {r['backend']} | ok | {p['first_iter_ms']:.2f} | {p['p50']:.2f} | "
                     f"{p['p95']:.2f} | {p['p99']:.2f} | {p['throughput_sps']:.2f} | "
                     f"{p.get('peak_mem_mb', float('nan')):.1f} |")
        else:
            L.append(f"| {r['backend']} | **{r['status']}** | - | - | - | - | - | - | "
                     f"<br>`{(r['error'] or '')[:160]}` |")

    # Cross-check: torchair vs npu_eager (same device+dtype).
    ta = next((r for r in results if r["backend"] == "torchair" and r["status"] == "ok"), None)
    ne = next((r for r in results if r["backend"] == "npu_eager" and r["status"] == "ok"), None)
    if ta and ne and primary in ta["outputs"] and primary in ne["outputs"]:
        m = compare(ne["outputs"][primary], ta["outputs"][primary])
        L.append("\n## Cross-check: torchair vs npu_eager (same dtype)\n")
        L.append(f"- `{primary}`: cosine_sim=`{m['cosine_sim']:.6f}`  "
                 f"max_abs_diff=`{m['max_abs_diff']:.3e}`  max_rel_err=`{m['max_rel_err']:.3e}`")
        L.append(f"- gate (max_abs_diff ≤ {atol}): **{'PASS' if m['max_abs_diff'] <= atol else 'CHECK'}**")
    return "\n".join(L) + "\n"


def collect_env():
    def ver(m):
        try:
            return __import__(m).__version__
        except Exception:
            return "n/a"
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "n/a") if _HAS_TORCH_NPU else "n/a",
        "torchair": "ships_with_torch_npu" if _HAS_TORCHAIR else "n/a",
        "transformers": ver("transformers"),
        "cann": os.environ.get("ASCEND_TOOLKIT_HOME", os.environ.get("ASCEND_HOME_PATH", "n/a")),
        "device_count": torch.npu.device_count() if npu_available() else 0,
        "npu_name": torch.npu.get_device_name(0) if npu_available() else "n/a",
    }
    return env


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Three-backend accuracy/performance benchmark for any HF model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--backends", default="torchair,npu_eager,cpu",
                   help="Comma list from {torchair,npu_eager,cpu}.")
    p.add_argument("--device", default="auto", choices=["auto", "npu", "cpu"],
                   help="Accepted for CLI ergonomics; the actual device per run is "
                        "implied by --backends (cpu vs npu_eager/torchair).")
    p.add_argument("--npu_id", type=int, default=0)
    p.add_argument("--dtype", default="float16", choices=list(DTYPES),
                   help="Compute dtype for NPU backends (CPU golden is always fp32).")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--compile_mode", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cosine_threshold", type=float, default=0.999)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--output_dir", default="benchmark_results")
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    cpu_inputs = build_cpu_inputs(config, args.batch_size, args.image_size, args.seq_len, args.seed)
    print("Fixed inputs:", {k: tuple(v.shape) for k, v in cpu_inputs.items()})

    results = []
    for bk in backends:
        print(f"\n>>> running backend: {bk}")
        results.append(run_backend(bk, args, cpu_inputs, config))

    accuracy, primary = build_accuracy(results, args.cosine_threshold, args.atol)

    meta = {
        "model_name_or_path": args.model_name_or_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batch_size": args.batch_size, "dtype": args.dtype,
        "warmup": args.warmup, "iterations": args.iterations,
        "env": collect_env(),
    }

    # JSON (outputs tensors stripped; keep metrics only).
    json_results = []
    for r in results:
        j = {k: v for k, v in r.items() if k not in ("outputs", "traceback")}
        json_results.append(j)
    payload = {"meta": meta, "results": json_results, "accuracy": accuracy,
               "primary_output": primary,
               "gate": {"cosine_threshold": args.cosine_threshold, "atol": args.atol}}
    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    md = markdown_report(meta, results, accuracy, primary, args.cosine_threshold, args.atol)
    md_path = os.path.join(args.output_dir, "benchmark_results.md")
    with open(md_path, "w") as f:
        f.write(md)

    print("\n" + md)
    print(f"\n[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")


if __name__ == "__main__":
    main()
