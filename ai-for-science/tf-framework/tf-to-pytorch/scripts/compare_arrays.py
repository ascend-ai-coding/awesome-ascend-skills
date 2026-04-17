#!/usr/bin/env python3
"""Compare arrays saved as .npy or .npz for TF-to-PyTorch validation."""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_payload(path: Path):
    if path.suffix == ".npy":
        return {path.stem: np.load(path)}
    if path.suffix == ".npz":
        data = np.load(path)
        return {key: data[key] for key in data.files}
    raise ValueError(f"unsupported file type: {path}")


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_flat = lhs.reshape(-1).astype(np.float64)
    rhs_flat = rhs.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(lhs_flat) * np.linalg.norm(rhs_flat)
    if denom == 0:
        return 1.0 if np.allclose(lhs_flat, rhs_flat) else 0.0
    return float(np.dot(lhs_flat, rhs_flat) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare arrays from TF and PyTorch outputs")
    parser.add_argument("reference", help="Reference .npy or .npz file")
    parser.add_argument("candidate", help="Candidate .npy or .npz file")
    parser.add_argument("--key", default=None, help="Specific key to compare when using .npz")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance")
    args = parser.parse_args()

    ref = load_payload(Path(args.reference))
    cand = load_payload(Path(args.candidate))

    if args.key:
        keys = [args.key]
    else:
        keys = sorted(set(ref) & set(cand))

    if not keys:
        print("[FAIL] no common keys to compare")
        return 1

    failures = 0
    for key in keys:
        if key not in ref or key not in cand:
            print(f"[FAIL] {key}: missing in one side")
            failures += 1
            continue

        lhs = np.asarray(ref[key])
        rhs = np.asarray(cand[key])
        if lhs.shape != rhs.shape:
            print(f"[FAIL] {key}: shape mismatch {lhs.shape} vs {rhs.shape}")
            failures += 1
            continue

        diff = np.abs(lhs - rhs)
        max_abs = float(diff.max()) if diff.size else 0.0
        mean_abs = float(diff.mean()) if diff.size else 0.0
        cosine = cosine_similarity(lhs, rhs)
        passed = np.allclose(lhs, rhs, atol=args.atol, rtol=args.rtol)
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] {key}: shape={lhs.shape}, max_abs={max_abs:.6e}, "
            f"mean_abs={mean_abs:.6e}, cosine={cosine:.8f}"
        )
        if not passed:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
