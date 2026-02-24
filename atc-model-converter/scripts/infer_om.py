#!/usr/bin/env python3
"""
OM Model Inference Script using ais_bench

This script provides Python API-based inference for Ascend OM models.
Requires: ais_bench and aclruntime packages (from ais-bench_workload)

Installation:
    # Source compilation (recommended for latest version)
    git clone https://gitee.com/ascend/tools.git
    cd tools/ais-bench_workload/tool/ais_bench
    pip3 wheel ./backend/ -v  # Build aclruntime
    pip3 wheel ./ -v          # Build ais_bench
    pip3 install ./aclruntime-*.whl ./ais_bench-*.whl

Usage:
    python3 infer_om.py --model model.om --input input.npy
    python3 infer_om.py --model model.om --batch-size 4 --warmup 3 --loop 10
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

try:
    from ais_bench.infer.interface import InferSession
except ImportError:
    print("Error: ais_bench package not installed.")
    print(
        "Install from: https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench"
    )
    sys.exit(1)


def get_device_id():
    """Get available NPU device ID."""
    # Default to device 0, can be overridden
    return int(os.environ.get("ASCEND_DEVICE_ID", 0))


def load_input_data(input_path, input_shape=None, dtype=np.float32):
    """Load input data from file or generate random data."""
    if input_path and os.path.exists(input_path):
        if input_path.endswith(".npy"):
            return np.load(input_path)
        elif input_path.endswith(".bin"):
            return (
                np.fromfile(input_path, dtype=dtype).reshape(input_shape)
                if input_shape
                else np.fromfile(input_path, dtype=dtype)
            )
        else:
            raise ValueError(f"Unsupported input format: {input_path}")
    else:
        # Generate random input if no file provided
        if input_shape:
            return np.random.randn(*input_shape).astype(dtype)
        else:
            raise ValueError("Either input file or input shape must be provided")


def print_model_info(session):
    """Print model input/output information."""
    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)

    print("\nInputs:")
    for i, inp in enumerate(session.get_inputs()):
        print(f"  [{i}] {inp.name}")
        print(f"      Shape: {inp.shape}")
        print(f"      Type:  {inp.datatype}")
        print(f"      Size:  {inp.size} bytes")

    print("\nOutputs:")
    for i, out in enumerate(session.get_outputs()):
        print(f"  [{i}] {out.name}")
        print(f"      Shape: {out.shape}")
        print(f"      Type:  {out.datatype}")
        print(f"      Size:  {out.size} bytes")
    print("=" * 60 + "\n")


def infer_static(session, inputs, warmup=1, loop=1):
    """Run static shape inference with warmup and timing."""
    # Warmup runs
    for _ in range(warmup):
        _ = session.infer(inputs, mode="static")

    # Reset timing info
    session.reset_summaryinfo()

    # Actual inference runs
    outputs = None
    for _ in range(loop):
        outputs = session.infer(inputs, mode="static")

    return outputs


def infer_pipeline(session, inputs_list, mode="static"):
    """Run pipeline inference for multiple inputs (better throughput)."""
    return session.infer_pipeline(inputs_list, mode=mode)


def main():
    parser = argparse.ArgumentParser(description="OM Model Inference using ais_bench")
    parser.add_argument("--model", required=True, help="Path to OM model file")
    parser.add_argument("--input", help="Path to input file (.npy or .bin)")
    parser.add_argument("--input-shape", help="Input shape (e.g., '1,3,640,640')")
    parser.add_argument(
        "--input-name", default=None, help="Input tensor name (for reference)"
    )
    parser.add_argument("--output", help="Path to save output (.npy)")
    parser.add_argument("--device", type=int, default=None, help="NPU device ID")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for random input"
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument(
        "--loop", type=int, default=1, help="Number of inference iterations"
    )
    parser.add_argument(
        "--mode",
        default="static",
        choices=["static", "dymbatch", "dymhw", "dymdims", "dymshape"],
        help="Model type",
    )
    parser.add_argument("--info", action="store_true", help="Print model info only")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Get device ID
    device_id = args.device if args.device is not None else get_device_id()

    # Create inference session
    print(f"Loading model: {args.model}")
    print(f"Device ID: {device_id}")

    session = InferSession(device_id=device_id, model_path=args.model, debug=args.debug)

    # Print model info
    print_model_info(session)

    if args.info:
        session.free_resource()
        return

    # Prepare input data
    input_shape = None
    if args.input_shape:
        input_shape = tuple(map(int, args.input_shape.split(",")))
    elif session.get_inputs():
        # Use model's input shape
        inp = session.get_inputs()[0]
        input_shape = tuple(inp.shape)

    # Get input dtype from model
    dtype = np.float32
    if session.get_inputs():
        inp = session.get_inputs()[0]
        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "int32": np.int32,
            "int64": np.int64,
            "uint8": np.uint8,
        }
        dtype = dtype_map.get(str(inp.datatype), np.float32)

    # Load or generate input
    if args.input:
        inputs = load_input_data(args.input, input_shape, dtype)
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
    else:
        if input_shape:
            print(f"Generating random input with shape: {input_shape}")
            inputs = [np.random.randn(*input_shape).astype(dtype)]
        else:
            print("Error: Cannot generate input without shape info")
            sys.exit(1)

    # Run inference
    print(f"Running inference (warmup={args.warmup}, loop={args.loop})...")

    outputs = infer_static(session, inputs, warmup=args.warmup, loop=args.loop)

    # Print performance summary
    summary = session.summary()
    if summary.exec_time_list:
        times = np.array(summary.exec_time_list)
        print("\n" + "=" * 60)
        print("Performance Summary")
        print("=" * 60)
        print(f"  Inference count: {len(times)}")
        print(f"  Latency (ms):")
        print(f"    Min:    {times.min():.3f}")
        print(f"    Max:    {times.max():.3f}")
        print(f"    Mean:   {times.mean():.3f}")
        print(f"    Median: {np.median(times):.3f}")
        print(f"    P99:    {np.percentile(times, 99):.3f}")
        print(f"  Throughput: {1000 / times.mean():.1f} FPS")
        print("=" * 60 + "\n")

    # Print output info
    if outputs:
        print("Output shapes:")
        for i, out in enumerate(outputs):
            print(f"  [{i}] {out.shape}")

    # Save output if requested
    if args.output and outputs:
        output_dir = os.path.dirname(args.output) or "."
        os.makedirs(output_dir, exist_ok=True)

        if len(outputs) == 1:
            np.save(args.output, outputs[0])
            print(f"Output saved to: {args.output}")
        else:
            for i, out in enumerate(outputs):
                out_path = args.output.replace(".npy", f"_{i}.npy")
                np.save(out_path, out)
                print(f"Output [{i}] saved to: {out_path}")

    # Cleanup
    session.free_resource()
    print("Done.")


if __name__ == "__main__":
    main()
