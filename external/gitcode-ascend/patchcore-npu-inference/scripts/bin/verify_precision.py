#!/usr/bin/env python
"""verify_precision.py — Validate NN search precision across CPU/NPU/Faiss.

Compares BaihuNN (CPU/NPU) vs FaissNN (CPU) results on a small synthetic
dataset to ensure L2 distance and index outputs match within tolerances.

Usage:
    python bin/verify_precision.py [--device cpu|npu:0]
"""

import argparse
import logging
import sys
import time

import numpy as np
import torch

from patchcore.baihu_nn import BaihuNN
from patchcore.common import FaissNN

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
LOGGER = logging.getLogger(__name__)


def make_data(
    n_index: int = 5000,
    n_query: int = 200,
    dim: int = 1024,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    index = rng.randn(n_index, dim).astype(np.float32)
    query = rng.randn(n_query, dim).astype(np.float32)
    # Normalize rows to unit length (simulates typical backbone features)
    index /= np.linalg.norm(index, axis=1, keepdims=True).clip(min=1e-8)
    query /= np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)
    return index, query


def compute_mre(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Relative Error (mask where |b| > eps)."""
    abs_b = np.abs(b)
    mask = abs_b > eps
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(a[mask] - b[mask]) / abs_b[mask]))


def run_comparison(
    index: np.ndarray,
    query: np.ndarray,
    k: int = 9,
    use_fp16: bool = True,
    device_str: str = "cpu",
) -> dict:
    """Run BaihuNN and FaissNN on the same data and compare results.

    Returns a dict with 'max_re' (max relative error),
    'index_accuracy' (fraction of identical indices),
    and timing info.
    """
    n_index, dim = index.shape
    n_query = query.shape[0]

    # ── BaihuNN (target) ──────────────────────────────────────────────
    on_npu = "npu" in device_str

    baihu = BaihuNN(on_npu=on_npu, device_id=0, batch_size=4096, use_fp16=use_fp16)
    baihu.fit(index)

    t0 = time.perf_counter()
    baihu_dist, baihu_idx = baihu.run(k, query)
    t_baihu = time.perf_counter() - t0

    LOGGER.info(
        "BaihuNN (%s, fp16=%s): %d × %d → %d-NN in %.2f ms",
        device_str,
        use_fp16,
        n_query,
        dim,
        k,
        t_baihu * 1000,
    )

    # ── FaissNN (reference) ───────────────────────────────────────────
    faiss = FaissNN(on_gpu=False, num_workers=4)
    faiss.fit(index)

    t0 = time.perf_counter()
    faiss_dist, faiss_idx = faiss.run(k, query)
    t_faiss = time.perf_counter() - t0

    LOGGER.info(
        "FaissNN  (cpu):           %d × %d → %d-NN in %.2f ms",
        n_query,
        dim,
        k,
        t_faiss * 1000,
    )

    # ── Compare ───────────────────────────────────────────────────────
    dist_re = compute_mre(baihu_dist, faiss_dist)
    idx_ok = float(np.mean(baihu_idx == faiss_idx))

    LOGGER.info("Distance MRE:        %.6f", dist_re)
    LOGGER.info("Index accuracy:      %.4f %%", idx_ok * 100)

    # Show a few sample mismatches (first 5 queries)
    mismatches = np.where((baihu_idx != faiss_idx).any(axis=1))[0][:5]
    if len(mismatches):
        LOGGER.info("Sample mismatches (query → BaihuNN / FaissNN indices):")
        for qid in mismatches:
            LOGGER.info(
                "  query %d: Baihu=%s  Faiss=%s",
                qid,
                baihu_idx[qid],
                faiss_idx[qid],
            )

    return {
        "device": device_str,
        "fp16": use_fp16,
        "dist_mre": dist_re,
        "index_accuracy": float(idx_ok),
        "t_baihu_ms": t_baihu * 1000,
        "t_faiss_ms": t_faiss * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Precision verification")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target device: 'cpu' or 'npu:0'",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="Feature dimension (default 1024 ≈ WideResNet50 output)",
    )
    parser.add_argument(
        "--n_index",
        type=int,
        default=5000,
        help="Number of index vectors (default 5000)",
    )
    parser.add_argument(
        "--n_query",
        type=int,
        default=200,
        help="Number of query vectors (default 200)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=9,
        help="Number of nearest neighbours (default 9)",
    )
    args = parser.parse_args()

    LOGGER.info(
        "Generating %d × %d index + %d queries (k=%d)...",
        args.n_index,
        args.dim,
        args.n_query,
        args.k,
    )
    index, query = make_data(args.n_index, args.n_query, args.dim)

    # ── FP32 accuracy test ───────────────────────────────────────
    LOGGER.info("\n═══ FP32 comparison ═══")
    fp32 = run_comparison(index, query, args.k, use_fp16=False, device_str=args.device)

    # ── FP16 accuracy test (only matters on NPU; on CPU it's a no-op) ─
    LOGGER.info("\n═══ FP16 comparison ═══")
    fp16 = run_comparison(index, query, args.k, use_fp16=True, device_str=args.device)

    # ── Summary ────────────────────────────────────────────────────────
    LOGGER.info("\n═══ Precision Summary ═══")
    LOGGER.info("FP32:  dist MRE=%.6f  idx_acc=%.2f%%", fp32["dist_mre"], fp32["index_accuracy"] * 100)
    LOGGER.info("FP16:  dist MRE=%.6f  idx_acc=%.2f%%", fp16["dist_mre"], fp16["index_accuracy"] * 100)

    # Acceptable thresholds
    failures = []
    if fp32["dist_mre"] > 1e-5:
        failures.append(f"FP32 MRE {fp32['dist_mre']:.2e} > 1e-5")
    if fp16["dist_mre"] > 1e-2:
        failures.append(f"FP16 MRE {fp16['dist_mre']:.2e} > 1e-2")
    if fp32["index_accuracy"] < 0.99:
        failures.append(f"FP32 index accuracy {fp32['index_accuracy']:.4f} < 0.99")

    if failures:
        LOGGER.warning("\n⚠ FAILURES: %s", "; ".join(failures))
        sys.exit(1)
    else:
        LOGGER.info("\n✓ All precision checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
