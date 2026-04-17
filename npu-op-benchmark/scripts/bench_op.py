#!/usr/bin/env python3
import argparse, json, math, os, random, statistics, sys, time
from pathlib import Path

def parse_args():
    """Parse command-line arguments for the benchmark runner."""
    p = argparse.ArgumentParser(
        description='Single-operator benchmark runner for Ascend NPU or CPU. '
                    'The bundled implementation currently includes repeat_interleave as an example.'
    )
    p.add_argument('--op', default='repeat_interleave', help='Operator name. The bundled example implementation currently supports repeat_interleave.')
    p.add_argument('--device', default='npu:0')
    p.add_argument('--dtype', default='float32')
    p.add_argument('--spatial_h', type=int, default=64)
    p.add_argument('--spatial_w', type=int, default=64)
    p.add_argument('--embed_dim', type=int, default=1024)
    p.add_argument('--temporal_size', type=int, default=16)
    p.add_argument('--repeat_dim', type=int, choices=(0,1,2), default=0)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--json_out')
    p.add_argument('--profile', action='store_true')
    return p.parse_args()

def percentile(v, p):
    """Return the linear-interpolated percentile from a sorted list."""
    if len(v) == 1:
        return v[0]
    r = (len(v)-1) * p
    lo, hi = math.floor(r), math.ceil(r)
    if lo == hi:
        return v[lo]
    w = r - lo
    return v[lo] * (1 - w) + v[hi] * w

def sync_if_needed(torch, use_npu):
    """Synchronize the NPU stream when benchmarking on Ascend devices."""
    if use_npu:
        torch.npu.synchronize()

def resolve_dtype(torch, name):
    """Resolve a dtype alias to a torch dtype object."""
    m = {'float16': torch.float16, 'fp16': torch.float16, 'float32': torch.float32, 'fp32': torch.float32, 'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16}
    return m[name.lower()]

def validate_positive_args(args):
    """Reject non-positive benchmark dimensions and loop counts."""
    values = {
        'spatial_h': args.spatial_h,
        'spatial_w': args.spatial_w,
        'embed_dim': args.embed_dim,
        'temporal_size': args.temporal_size,
        'iters': args.iters,
        'warmup': args.warmup,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f'--{name} must be a positive integer, got {value}')

def main():
    """Run the benchmark and optionally write the JSON result to disk."""
    args = parse_args()
    validate_positive_args(args)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    try:
        import torch
    except ImportError as exc:
        print(f'ERROR: failed to import torch: {exc}', file=sys.stderr)
        return 1
    use_npu = args.device.startswith('npu')
    torch_npu_version = None
    if use_npu:
        try:
            import torch_npu
        except ImportError as exc:
            print(f'ERROR: failed to import torch_npu: {exc}', file=sys.stderr)
            return 1
        torch_npu_version = getattr(torch_npu, '__version__', 'unknown')
    dtype = resolve_dtype(torch, args.dtype)
    if use_npu and not torch.npu.is_available():
        print('ERROR: torch.npu.is_available() is False.', file=sys.stderr)
        return 1
    device = torch.device(args.device)
    if args.op != 'repeat_interleave':
        raise ValueError(f"Unsupported --op '{args.op}'. Currently supported: repeat_interleave")
    x = torch.randn(1, args.spatial_h * args.spatial_w, 3 * args.embed_dim // 4, device=device, dtype=dtype)
    def run_once():
        return x.repeat_interleave(args.temporal_size, dim=args.repeat_dim, output_size=x.shape[args.repeat_dim] * args.temporal_size)
    for _ in range(args.warmup):
        run_once(); sync_if_needed(torch, use_npu)
    lat = []
    out_shape = None
    for _ in range(args.iters):
        sync_if_needed(torch, use_npu)
        s = time.perf_counter_ns()
        y = run_once()
        sync_if_needed(torch, use_npu)
        e = time.perf_counter_ns()
        lat.append((e-s)/1_000_000.0)
        out_shape = list(y.shape)
    vals = sorted(lat)
    result = {
        'op': args.op,
        'device': args.device,
        'input_shape': list(x.shape),
        'output_shape': out_shape,
        'repeat_dim': args.repeat_dim,
        'avg_ms': statistics.fmean(lat),
        'min_ms': min(lat),
        'p50_ms': percentile(vals, 0.5),
        'p95_ms': percentile(vals, 0.95),
        'max_ms': max(lat),
        'torch_version': torch.__version__,
        'torch_npu_version': torch_npu_version,
    }
    print(json.dumps(result, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
